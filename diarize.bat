@echo off
REM ==============================================================================
REM diarize.bat — Windows Native Speech Transcription & Diarization
REM ==============================================================================
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%.venv
set PYTHON_BIN=%VENV_DIR%\Scripts\python.exe
set PIP_BIN=%VENV_DIR%\Scripts\pip.exe

echo ==============================================================================
echo  [92mStrix Halo Speech Pipeline (Windows Native)[0m
echo ==============================================================================

if "%~1"=="" (
    echo Usage: diarize.bat ^<audio_file^> [OPTIONS]
    echo.
    echo Examples:
    echo   diarize.bat meeting.wav
    echo   diarize.bat interview.mp3 --format srt -o interview.srt
    echo   diarize.bat podcast.m4a --num-speakers 2
    exit /b 0
)

if not exist "%PYTHON_BIN%" (
    echo [*] Setting up Python virtual environment in .venv...
    python -m venv "%VENV_DIR%"
    "%PIP_BIN%" install --upgrade pip wheel setuptools
    "%PIP_BIN%" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    "%PIP_BIN%" install -r "%SCRIPT_DIR%requirements.txt"
)

"%PYTHON_BIN%" "%SCRIPT_DIR%scripts\transcribe_diarize.py" %*
