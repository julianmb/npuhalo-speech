# AGENTS.md — npuhalo-speech

## What this repo is
Standalone speech product: Whisper-v3-turbo transcription (XDNA 2 NPU via Lemonade, with Faster-Whisper fallback) + pyannote speaker diarization (CPU/GPU). Public repo: github.com/julianmb/npuhalo-speech. Part of the npuhalo research family.

## Verify before pushing
```bash
bash -n diarize.sh server.sh
python3 -m py_compile scripts/*.py
```

## Layout
- `diarize.sh` / `diarize.bat` — CLI: single file or batch directory
- `server.sh` / `server.bat` — OpenAI-compatible API + browser studio
- `scripts/transcribe_diarize.py` — pipeline (ASR + diarization, padding, prompt memory)
- `scripts/server.py` — FastAPI server, Web UI HTML, /v1/audio/transcriptions

## Gotchas
1. **NPU is optional:** pipeline auto-falls back to local Whisper if Lemonade/NPU is offline. Never hard-require `/dev/accel/accel0`.
2. **pyannote is gated:** `pyannote/speaker-diarization-community-1` needs an HF token. Without it, the pipeline falls back to VAD+KMeans — document that behavior.
3. **Cross-platform:** keep `.sh` and `.bat` launchers in sync when changing CLI flags.
4. **Venv:** launchers create `.venv` on first run; never commit it (gitignored).
