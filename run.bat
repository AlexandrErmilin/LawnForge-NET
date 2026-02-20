@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ---- Resolve project root ----
set "ROOT=%~dp0"
cd /d "%ROOT%"

REM ---- Config ----
set "VENV_DIR=%ROOT%.venv_temp_gui"
set "PYTHON_EXE=py -3.11"

REM ---- Create temp venv if missing ----
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Creating temporary virtual environment: "%VENV_DIR%"
    %PYTHON_EXE% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [WARN] Python 3.11 launcher was not found. Falling back to default python...
        set "PYTHON_EXE=python"
        %PYTHON_EXE% -m venv "%VENV_DIR%"
        if errorlevel 1 (
            echo [ERROR] Failed to create virtual environment.
            exit /b 1
        )
    )
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

REM ---- Ensure pip ----
"%VENV_PY%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Bootstrapping pip...
    "%VENV_PY%" -m ensurepip --upgrade
)

REM ---- Install/update dependencies ----
echo [INFO] Installing dependencies from requirements.txt...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)

"%VENV_PY%" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    exit /b 1
)

REM ---- Force CUDA 11.8 build of PyTorch by default ----
echo [INFO] Installing PyTorch with CUDA 11.8...
"%VENV_PY%" -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
"%VENV_PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [ERROR] Failed to install CUDA 11.8 PyTorch build.
    exit /b 1
)

echo [INFO] Checking PyTorch/CUDA...
"%VENV_PY%" -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"

REM ---- Start GUI ----
echo [INFO] Launching GUI...
"%VENV_PY%" "%ROOT%scripts\gui_app.py"
set "RC=%ERRORLEVEL%"

echo [INFO] GUI exited with code %RC%.
exit /b %RC%
