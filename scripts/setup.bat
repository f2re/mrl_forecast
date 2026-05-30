@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo   MRL Forecast Pro: Setup for Windows 11
echo ======================================================

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: 2. Create Virtual Environment
if not exist venv (
    echo Creating virtual environment (venv)...
    python -m venv venv
) else (
    echo Virtual environment (venv) already exists.
)

:: 3. Install Dependencies
echo Installing dependencies from requirements.txt...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. Create Directories
echo Creating directory structure...
if not exist data\raw\archive mkdir data\raw\archive
if not exist data\processed_archive mkdir data\processed_archive
if not exist models\real_checkpoints mkdir models\real_checkpoints
if not exist models\registry mkdir models\registry

echo ======================================================
echo   Setup Completed Successfully!
echo   You can now run the app using scripts\run_app.bat
echo ======================================================
pause
