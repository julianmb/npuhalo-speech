# npuhalo-speech — NPU-Accelerated Speech Transcription & Speaker Diarization on AMD Strix Halo

[![Hardware](https://img.shields.io/badge/Hardware-AMD_Strix_Halo_(gfx1151)-ED1C24?logo=amd)](https://www.amd.com)
[![NPU](https://img.shields.io/badge/NPU-AMD_XDNA_2_50_TOPS-6A5ACD)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Heterogeneous speech pipeline for **AMD Strix Halo (Ryzen AI Max+ 395)**: **Whisper-v3-turbo on the 50 TOPS XDNA 2 NPU** + **pyannote diarization on CPU/GPU**, with an OpenAI-compatible API server and browser studio.

Part of the **[npuhalo](https://github.com/julianmb/npuhalo)** research family — this repo is the standalone, installable product for the speech work.

## What it does
- **Transcription:** Whisper-v3-turbo via Lemonade on `/dev/accel/accel0`, auto-fallback to Faster-Whisper on CPU/GPU. Selectable engine: Auto / NPU-only / CPU / GPU.
- **Diarization:** `pyannote/speaker-diarization-community-1` on CPU (default) or ROCm/CUDA, with a librosa + KMeans fallback.
- **Formats:** WAV, MP3, M4A/AAC (via PyAV), FLAC, OGG/WebM — no ffmpeg install needed.
- **Features:** 150ms acoustic boundary padding, rolling prompt context, batch/directory mode, speaker-attributed output (Text / JSON / SRT / VTT).
- **Web Studio:** login-gated UI with live progress stepper (upload → diarize → transcribe), per-turn NPU/CPU backend chips, talk-time breakdown, playback speed + keyboard review controls, search with match navigation, transcript history, and connection-drop recovery.

## What's new in v1.2.0
- Transcription Engine selector (`asr_backend`: `auto`/`npu`/`cpu`/`gpu`) across CLI, API, and Web Studio
- Live job progress: `/v1/progress/{job_id}` polling endpoint + progress stepper in the UI
- Compressed audio support (M4A/AAC/WebM) via PyAV — no system ffmpeg required
- Full-screen auth gate (API key validated once per device, stored locally)
- Review workflow: playback speed 0.75×–2×, keyboard shortcuts (`Space`, `←/→`, `J/K/L`), follow-playback auto-scroll
- Per-speaker talk-time percentages in the results summary
- Inline search: dims non-matching turns, match counter, Enter/Shift+Enter navigation
- Resilience: in-flight jobs recover after client network drops; retry button on failures; 500 MB upload guard
- App-style viewport-fit desktop layout

## Quick start (Linux)

```bash
git clone https://github.com/julianmb/npuhalo-speech.git
cd npuhalo-speech

# Single file, CPU diarization (default)
./diarize.sh meeting.wav

# GPU diarization, SRT subtitles
./diarize.sh meeting.wav --device rocm --format srt -o meeting.srt

# Hint the speaker count when you know it
./diarize.sh meeting.wav --num-speakers 2          # exact
./diarize.sh meeting.wav --min-speakers 2 --max-speakers 4  # range

# Batch a folder
./diarize.sh ./recordings/ --output-dir ./transcripts/ --format json
```

Windows: `diarize.bat meeting.wav` (same flags).

## API server + Web Studio

```bash
./server.sh                    # http://localhost:8000/  (Web UI) + /v1/audio/transcriptions
./server.sh --port 8080 --device rocm --api-key my-secret
```

Browser studio: drag-and-drop upload, live mic recording with visualizer, speaker renaming, keyword search, and SRT/JSON export.

OpenAI-compatible endpoint (example):

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer my-secret" \
  -F file=@meeting.wav -F model=whisper-1 -F diarize=true
```

Extra form fields beyond the OpenAI spec: `diarize` (bool, default `true`), `num_speakers`, `min_speakers`, `max_speakers`, `device` (`cpu`/`rocm`/`cuda`/`auto`). Response formats: `json`, `verbose_json`, `text`, `srt`, `vtt`.

## Development

```bash
bash -n diarize.sh server.sh          # validate launchers
python3 -m py_compile scripts/*.py    # syntax-check sources
python3 -m unittest discover -s tests # unit tests (no torch/GPU needed)
```

CI runs the same checks on every push and PR (see `.github/workflows/ci.yml`).

## Hardware

- **ASR:** 50 TOPS AMD XDNA 2 NPU via [Lemonade](https://github.com/lemonade-sdk/lemonade) (`whisper-v3-turbo-FLM`). If Lemonade/NPU is unavailable, falls back cleanly to local Faster-Whisper.
- **Diarization:** Zen 5 CPU by default (`--device cpu`), or `--device rocm` / `cuda` / `auto`.

## Requirements
See `requirements.txt` — `torch`, `pyannote.audio`, `faster-whisper`, `librosa`, `soundfile`, `fastapi`/`uvicorn`, etc. The launchers create a `.venv` automatically on first run.

## Related

Part of the **npuhalo** research family for AMD Strix Halo:

- **[npuhalo](https://github.com/julianmb/npuhalo)** — NPU + iGPU live verification, compression, and routing research on Strix Halo
- **[halofpx](https://github.com/julianmb/halofpx)** — Unified OpenAI-compatible LLM server: tuned ROCmFP4 quants, validated 262K context, vision, hot-swap model zoo (iGPU + NPU)
- **[q38rocm](https://github.com/julianmb/q38rocm)** — Qwen 3.8 27B ROCmFP4 deep-dive: up to 36 tok/s via MTP Speculation, TurboQuant & Mesa RADV Wave64
