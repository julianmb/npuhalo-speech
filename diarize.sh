#!/usr/bin/env bash
# ==============================================================================
# diarize.sh — Fast Heterogeneous Speech Transcription & Diarization
# Runs Whisper-v3-Turbo on AMD XDNA 2 NPU + pyannote on CPU or GPU
# Supports: Single audio files or Batch directory processing
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"
LEMONADE_PORT="13305"

# Print banner
echo "=============================================================================="
echo " 🎙️  STRIX HALO SPEECH PIPELINE (NPU ASR + CPU/GPU SPEAKER DIARIZATION)"
echo "=============================================================================="

# Show help if no arguments provided
if [ $# -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: ./diarize.sh <file_or_dir> [MORE_FILES...] [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  <file_or_dir>            Path to audio file(s) or a folder of recordings"
    echo ""
    echo "Options:"
    echo "  --device, -d <dev>       Diarization device: cpu (default), rocm, cuda, auto"
    echo "  --format <fmt>           Output format: text (default), json, srt, vtt"
    echo "  --output, -o <file>      Save transcript to destination file (single file)"
    echo "  --output-dir <dir>       Save transcripts to directory (batch mode)"
    echo "  --num-speakers <N>       Exact number of speakers if known"
    echo "  --language <lang>        Language code (e.g., 'en', 'es', 'zh', 'fr')"
    echo "  --hf-token <token>       Hugging Face access token for pyannote"
    echo ""
    echo "Examples:"
    echo "  ./diarize.sh meeting.wav"
    echo "  ./diarize.sh meeting.wav --device rocm"
    echo "  ./diarize.sh ./recordings/ --output-dir ./transcripts/ --format srt"
    echo "  ./diarize.sh file1.wav file2.mp3 --output-dir ./transcripts/ --format json"
    echo "=============================================================================="
    exit 0
fi

# 1. Check Python virtual environment
if [ ! -f "$PYTHON_BIN" ]; then
    echo "[*] Setting up Python virtual environment in .venv..."
    python3 -m venv "$VENV_DIR"
    "$PIP_BIN" install --upgrade pip wheel setuptools
    "$PIP_BIN" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    "$PIP_BIN" install -r "${SCRIPT_DIR}/requirements.txt"
fi

# 2. Check Lemonade server & Whisper NPU model
if command -v lemonade &> /dev/null; then
    if ! curl -s "http://127.0.0.1:${LEMONADE_PORT}/health" &> /dev/null; then
        echo "[!] Lemonade server not responding on port ${LEMONADE_PORT}. Starting lemond..."
        lemonade run &> /dev/null &
        sleep 3
    fi
    LOADED_MODELS=$(lemonade status 2>/dev/null || true)
    if ! echo "$LOADED_MODELS" | grep -q "whisper-v3-turbo-FLM"; then
        echo "[*] Loading Whisper-v3-turbo on AMD XDNA 2 NPU..."
        lemonade load whisper-v3-turbo-FLM &> /dev/null || true
    fi
fi

# 3. Execute transcription & diarization pipeline
"$PYTHON_BIN" "${SCRIPT_DIR}/scripts/transcribe_diarize.py" "$@"
