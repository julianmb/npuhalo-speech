@echo off
REM ==============================================================================
REM server.bat — Launch Windows Native OpenAI Speech Server & Web Studio
REM ==============================================================================
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%.venv
set PYTHON_BIN=%VENV_DIR%\Scripts\python.exe
set PIP_BIN=%VENV_DIR%\Scripts\pip.exe

if not exist "%PYTHON_BIN%" (
    echo [*] Setting up Python virtual environment in .venv...
    python -m venv "%VENV_DIR%"
    "%PIP_BIN%" install --upgrade pip wheel setuptools
    "%PIP_BIN%" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    "%PIP_BIN%" install -r "%SCRIPT_DIR%requirements.txt"
)

"%PYTHON_BIN%" "%SCRIPT_DIR%scripts\server.py" %*
