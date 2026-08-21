@echo off
REM ==============================================================================
REM server.bat — Launch OpenAI-Compatible Speech & Diarization Server (Windows)
REM ==============================================================================
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%.venv
set PYTHON_BIN=%VENV_DIR%\Scripts\python.exe
set PIP_BIN=%VENV_DIR%\Scripts\pip.exe
set LEMONADE_PORT=13305

if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
goto :setup_venv

:show_help
echo Usage: server.bat [OPTIONS]
echo.
echo Options:
echo   --host ^<host^>          Host address (default: 0.0.0.0)
echo   --port, -p ^<port^>      Port to listen on (default: 8000)
echo   --api-key, -k ^<key^>    Bearer API key for authentication
echo   --no-auth              Disable API key authentication
echo   --device, -d ^<dev^>     Diarization device: cpu (default), rocm, cuda, auto
echo   --lemonade-url ^<url^>   Lemonade NPU API URL (default: http://127.0.0.1:13305)
echo   --hf-token ^<token^>     Hugging Face token for gated pyannote model
echo.
echo Examples:
echo   server.bat
echo   server.bat --port 8080 --api-key my-secret-key
echo   server.bat --device rocm
echo ==============================================================================
exit /b 0

:setup_venv
REM 1. Check Python virtual environment & dependencies
if exist "%PYTHON_BIN%" goto :check_lemonade

echo [*] Setting up Python virtual environment in .venv...
python -m venv "%VENV_DIR%"
"%PIP_BIN%" install --upgrade pip wheel setuptools
"%PIP_BIN%" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
"%PIP_BIN%" install -r "%SCRIPT_DIR%requirements.txt"

:check_lemonade
REM 2. Check Lemonade server & Whisper NPU model
where lemonade >nul 2>&1
if errorlevel 1 goto :run_server

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

:run_server
REM 3. Launch the API server
"%PYTHON_BIN%" "%SCRIPT_DIR%scripts\server.py" %*
