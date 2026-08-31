#!/usr/bin/env python3
"""FastAPI route tests via TestClient with the heavy pipeline mocked."""

import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import server as srv  # noqa: E402  (after path insert)
from fastapi.testclient import TestClient  # noqa: E402  (needs httpx)


def _client(**server_config_overrides):
    # Reset global config for isolation
    srv.SERVER_CONFIG.update(
        {
            "api_key": "test-key",
            "lemonade_url": "http://127.0.0.1:13305",
            "default_device": "cpu",
            "hf_token": None,
        }
    )
    srv.SERVER_CONFIG.update(server_config_overrides)
    srv.JOBS.clear()
    return TestClient(srv.app)


class TestHealthAndModels(unittest.TestCase):
    def test_health_ok(self):
        c = _client()
        with mock.patch("server.ensure_npu_whisper", return_value=True):
            r = c.get("/health")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["npu_backend"], "connected")

    def test_models_requires_auth(self):
        c = _client(api_key="secret")
        self.assertEqual(c.get("/v1/models").status_code, 401)
        self.assertEqual(
            c.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code, 200
        )

    def test_progress_unknown_job_404(self):
        c = _client(api_key=None)
        r = c.get("/v1/progress/nope")
        self.assertEqual(r.status_code, 404)

    def test_progress_known_job(self):
        c = _client(api_key=None)
        srv.JOBS["abc"] = {"stage": "diarize", "detail": "test", "pct": None, "ts": 0}
        r = c.get("/v1/progress/abc")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["stage"], "diarize")


class TestTranscriptionRoutes(unittest.TestCase):
    def _wav_bytes(self, sr=16000, secs=1):
        import struct
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(struct.pack("<" + "h" * sr * secs, *([0] * sr * secs)))
        return buf.getvalue()

    @mock.patch("server.process_pipeline")
    @mock.patch("server.librosa_or_sf_load", return_value=([0.0] * 16000, 16000))
    def test_verbose_json_verbose(self, mock_load, mock_pipeline):
        # librosa_or_sf_load return is (y, sr) where y is list/array with len
        import numpy as np

        mock_load.return_value = (np.zeros(16000, dtype=float), 16000)
        mock_pipeline.return_value = [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_00",
                "text": "hello",
                "latency_ms": 10,
                "backend": "NPU (XDNA 2)",
            },
        ]
        c = _client(api_key=None)
        r = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", self._wav_bytes(), "audio/wav")},
            data={"response_format": "verbose_json", "diarize": "true", "job_id": "jid1"},
        )
        self.assertEqual(r.status_code, 200, r.text[:500])
        body = r.json()
        self.assertEqual(body["text"], "hello")
        self.assertEqual(body["segments"][0]["speaker"], "SPEAKER_00")
        # Progress should be stashed
        self.assertIn("jid1", srv.JOBS)
        self.assertEqual(srv.JOBS["jid1"].get("stage"), "done")

    def test_oversized_upload_413(self):
        c = _client(api_key=None)
        # Patch the size limit to 1 byte so our tiny wav trips it
        with mock.patch("server.MAX_UPLOAD_BYTES", 1):
            r = c.post(
                "/v1/audio/transcriptions",
                files={"file": ("big.wav", self._wav_bytes(), "audio/wav")},
                data={"diarize": "true"},
            )
            self.assertEqual(r.status_code, 413)

    def test_unsupported_format_422(self):
        c = _client(api_key=None)
        with mock.patch(
            "server.process_pipeline",
            side_effect=RuntimeError("Could not decode audio file 'x' — unsupported format"),
        ):
            r = c.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", self._wav_bytes(), "audio/wav")},
                data={"diarize": "true"},
            )
            self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
