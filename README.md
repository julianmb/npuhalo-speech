# npuhalo-speech — NPU-Accelerated Speech Transcription & Speaker Diarization on AMD Strix Halo

[![Hardware](https://img.shields.io/badge/Hardware-AMD_Strix_Halo_(gfx1151)-ED1C24?logo=amd)](https://www.amd.com)
[![NPU](https://img.shields.io/badge/NPU-AMD_XDNA_2_50_TOPS-6A5ACD)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Heterogeneous speech pipeline for **AMD Strix Halo (Ryzen AI Max+ 395)**: **Whisper-v3-turbo on the 50 TOPS XDNA 2 NPU** + **pyannote diarization on CPU/GPU**, with an OpenAI-compatible API server and browser studio.

Part of the **[npuhalo](https://github.com/julianmb/npuhalo)** research family — this repo is the standalone, installable product for the speech work.

## What it does
- **Transcription:** Whisper-v3-turbo via Lemonade on `/dev/accel/accel0`, auto-fallback to Faster-Whisper on CPU/GPU.
- **Diarization:** `pyannote/speaker-diarization-community-1` on CPU (default) or ROCm/CUDA, with a librosa + KMeans fallback.
- **Features:** 150ms acoustic boundary padding, rolling prompt context, batch/directory mode, speaker-attributed output (Text / JSON / SRT / VTT).

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
- **npuhalo** — NPU research (TTFT, routing, full system benchmarks)
- **halofpx** — Unified LLM server for Strix Halo
- **q38rocm** — Qwen 3.8 27B deep-dive
