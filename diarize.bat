@echo off
REM ==============================================================================
REM diarize.bat — Fast Heterogeneous Speech Transcription & Diarization
REM Runs Whisper-v3-Turbo on AMD XDNA 2 NPU + pyannote on CPU or GPU
REM Supports: Single audio files or Batch directory processing
REM ==============================================================================
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%.venv
set PYTHON_BIN=%VENV_DIR%\Scripts\python.exe
set PIP_BIN=%VENV_DIR%\Scripts\pip.exe
set LEMONADE_PORT=13305

echo ==============================================================================
echo  [92m STRIX HALO SPEECH PIPELINE (NPU ASR + CPU/GPU SPEAKER DIARIZATION)[0m
echo ==============================================================================

if "%~1"=="" (
    echo Usage: diarize.bat ^<file_or_dir^> [MORE_FILES...] [OPTIONS]
    echo.
    echo Arguments:
    echo   ^<file_or_dir^>            Path to audio file^(s^) or a folder of recordings
    echo.
    echo Options:
    echo   --device, -d ^<dev^>       Diarization device: cpu (default), rocm, cuda, auto
    echo   --format ^<fmt^>           Output format: text (default), json, srt, vtt
    echo   --output, -o ^<file^>      Save transcript to destination file (single file)
    echo   --output-dir ^<dir^>       Save transcripts to directory (batch mode)
    echo   --num-speakers ^<N^>       Exact number of speakers if known
    echo   --min-speakers ^<N^>       Minimum number of speakers (if exact count unknown)
    echo   --max-speakers ^<N^>       Maximum number of speakers (if exact count unknown)
    echo   --language ^<lang^>        Language code (e.g., 'en', 'es', 'zh', 'fr')
    echo   --hf-token ^<token^>       Hugging Face access token for pyannote
    echo.
    echo Examples:
    echo   diarize.bat meeting.wav
    echo   diarize.bat meeting.wav --device rocm
    echo   diarize.bat .\recordings\ --output-dir .\transcripts\ --format srt
    echo ==============================================================================
    exit /b 0
)

REM 1. Check Python virtual environment
if not exist "%PYTHON_BIN%" (
    echo [*] Setting up Python virtual environment in .venv...
    python -m venv "%VENV_DIR%"
    "%PIP_BIN%" install --upgrade pip wheel setuptools
    "%PIP_BIN%" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    "%PIP_BIN%" install -r "%SCRIPT_DIR%requirements.txt"
)

REM 2. Check Lemonade server & Whisper NPU model
where lemonade >nul 2>&1
if errorlevel 1 goto :run_pipeline

curl -s "http://127.0.0.1:%LEMONADE_PORT%/health" >nul 2>&1
if not errorlevel 1 goto :check_model

echo [!] Lemonade server not responding on port %LEMONADE_PORT%. Starting lemond...
start "" /b lemonade run >nul 2>&1
for /l %%i in (1,1,15) do (
    curl -s "http://127.0.0.1:%LEMONADE_PORT%/health" >nul 2>&1 && goto :check_model
    timeout /t 1 /nobreak >nul
)

:check_model
set "LOADED_MODELS="
for /f "delims=" %%i in ('lemonade status 2^>nul') do set "LOADED_MODELS=!LOADED_MODELS!%%i"
echo !LOADED_MODELS! | findstr /C:"whisper-v3-turbo-FLM" >nul
if errorlevel 1 (
    echo [*] Loading Whisper-v3-turbo on AMD XDNA 2 NPU...
    lemonade load whisper-v3-turbo-FLM >nul 2>&1
)

:run_pipeline
REM 3. Execute transcription & diarization pipeline
"%PYTHON_BIN%" "%SCRIPT_DIR%scripts\transcribe_diarize.py" %*
