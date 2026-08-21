#!/usr/bin/env bash
# ==============================================================================
# server.sh — Launch OpenAI-Compatible Speech & Diarization Server
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"
LEMONADE_PORT="13305"

# Show help if requested
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: ./server.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --host <host>          Host address (default: 0.0.0.0)"
    echo "  --port, -p <port>      Port to listen on (default: 8000)"
    echo "  --api-key, -k <key>    Bearer API key for authentication"
    echo "  --no-auth              Disable API key authentication"
    echo "  --device, -d <dev>     Diarization device: cpu (default), rocm, cuda, auto"
    echo ""
    echo "Examples:"
    echo "  ./server.sh"
    echo "  ./server.sh --port 8080 --api-key my-secret-key"
    echo "  ./server.sh --device rocm"
    echo "=============================================================================="
    exit 0
fi

# 1. Check Python virtual environment & dependencies
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

# 3. Launch the API server
"$PYTHON_BIN" "${SCRIPT_DIR}/scripts/server.py" "$@"
