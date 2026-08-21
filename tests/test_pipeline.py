#!/usr/bin/env python3
"""Unit tests for pure pipeline helpers in transcribe_diarize.py (stdlib unittest only)."""

import io
import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import soundfile as sf  # noqa: E402

import transcribe_diarize as td  # noqa: E402


class TestFormatTimestamp(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(td.format_timestamp(0.0), "00:00:00.000")

    def test_standard(self):
        self.assertEqual(td.format_timestamp(3661.5), "01:01:01.500")

    def test_srt_uses_comma(self):
        self.assertEqual(td.format_timestamp(1.234, srt_format=True), "00:00:01,234")


class TestDiscoverAudioFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for name in ("a.wav", "b.mp3", "c.txt", "d.flac"):
            (self.dir / name).write_bytes(b"x")
        (self.dir / "sub").mkdir()
        (self.dir / "sub" / "e.ogg").write_bytes(b"x")

    def tearDown(self):
        self.tmp.cleanup()

    def test_directory_recursive(self):
        found = td.discover_audio_files([str(self.dir)])
        names = [p.name for p in found]
        self.assertEqual(sorted(names), ["a.wav", "b.mp3", "d.flac", "e.ogg"])

    def test_single_file(self):
        found = td.discover_audio_files([str(self.dir / "a.wav")])
        self.assertEqual([p.name for p in found], ["a.wav"])

    def test_wildcard(self):
        found = td.discover_audio_files([str(self.dir / "*.wav")])
        self.assertEqual([p.name for p in found], ["a.wav"])

    def test_dedup_and_sort(self):
        found = td.discover_audio_files(
            [str(self.dir / "b.mp3"), str(self.dir / "a.wav"), str(self.dir / "b.mp3")]
        )
        self.assertEqual([p.name for p in found], ["a.wav", "b.mp3"])


class TestOutputTranscript(unittest.TestCase):
    RESULTS = [
        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00", "text": "Hello world"},
        {"start": 1.6, "end": 3.0, "speaker": "SPEAKER_01", "text": "Hi there"},
    ]

    def test_text_format(self):
        out = td.output_transcript(self.RESULTS, out_format="text")
        self.assertIn("[00:00:00.000 -> 00:00:01.500] SPEAKER_00: Hello world", out)

    def test_json_roundtrip(self):
        out = td.output_transcript(self.RESULTS, out_format="json")
        self.assertEqual(json.loads(out), self.RESULTS)

    def test_srt_structure(self):
        out = td.output_transcript(self.RESULTS, out_format="srt")
        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nSPEAKER_00: Hello world", out)

    def test_vtt_header(self):
        out = td.output_transcript(self.RESULTS, out_format="vtt")
        self.assertTrue(out.startswith("WEBVTT"))

    def test_file_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.txt"
            td.output_transcript(self.RESULTS, out_format="text", output_file=str(path))
            self.assertIn("Hello world", path.read_text(encoding="utf-8"))


class TestNpuCache(unittest.TestCase):
    def setUp(self):
        td._NPU_STATUS.update({"url": None, "available": False, "checked_at": 0.0})

    def test_result_is_cached(self):
        with mock.patch.object(td.requests, "get", return_value=mock.Mock(status_code=200)) as m:
            self.assertTrue(td.check_npu_available("http://x:1"))
            self.assertTrue(td.check_npu_available("http://x:1"))
        self.assertEqual(m.call_count, 1)

    def test_cache_expires(self):
        with mock.patch.object(td.requests, "get", return_value=mock.Mock(status_code=200)) as m:
            td.check_npu_available("http://x:1")
            td._NPU_STATUS["checked_at"] = time.time() - td._NPU_CACHE_TTL - 1
            td.check_npu_available("http://x:1")
        self.assertEqual(m.call_count, 2)

    def test_unreachable_is_false(self):
        with mock.patch.object(td.requests, "get", side_effect=Exception("down")):
            self.assertFalse(td.check_npu_available("http://x:1"))


class TestHfToken(unittest.TestCase):
    def test_explicit_token_wins(self):
        self.assertEqual(td.get_hf_token("abc"), "abc")

    def test_env_token(self):
        with mock.patch.dict("os.environ", {"HF_TOKEN": "envtok"}):
            self.assertEqual(td.get_hf_token(), "envtok")


class TestLibrosaOrSfLoad(unittest.TestCase):
    def test_loads_generated_wav_mono_16k(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            sf.write(str(path), [0.0, 0.1, -0.1] * 100, 16000)
            y, sr = td.librosa_or_sf_load(str(path))
            self.assertEqual(sr, 16000)
            self.assertEqual(y.ndim, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
