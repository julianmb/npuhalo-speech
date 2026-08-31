# Contributing to npuhalo-speech

## Development setup

```bash
git clone https://github.com/julianmb/npuhalo-speech.git
cd npuhalo-speech
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[dev,test]"
```

Launchers create the venv automatically; the manual steps above are for linting and tests.

## Checks before pushing

```bash
# Shell launchers
bash -n diarize.sh server.sh

# Lint & format
.venv/bin/ruff check scripts/ tests/
.venv/bin/ruff format --check scripts/ tests/

# Type check (basic, non-blocking)
.venv/bin/basedpyright scripts/

# Tests (no torch/GPU needed for the light suite)
.venv/bin/python -m unittest discover -s tests -v
# or
.venv/bin/pytest tests/ -v  # if pytest is installed
```

All three are enforced in CI (`.github/workflows/ci.yml`).

## Tests

* `tests/test_pipeline.py` — pure helpers (`format_timestamp`, `discover_audio_files`, `output_transcript`), NPU cache, and PyAV m4a fallback. Uses stdlib `unittest` only.
* `tests/test_server.py` — FastAPI routes via `TestClient` with the pipeline mocked. Requires `httpx`.

Run a single test file:

```bash
.venv/bin/python -m unittest tests.test_server -v
```

## Pull requests

* Keep `.sh` and `.bat` launchers in sync when changing CLI flags (see `AGENTS.md`).
* Never hard-require `/dev/accel/accel0`; the NPU path must degrade to local Whisper.
* Document the gated `pyannote/speaker-diarization-community-1` fallback (VAD+KMeans) if you touch diarization.
* One feature per PR; include tests for new behaviour.
