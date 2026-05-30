@echo off
setlocal

set PORT=5005
set MODEL_PATH=models/real_checkpoints/best_model.pt

if not exist %MODEL_PATH% (
    echo Warning: Model %MODEL_PATH% not found.
    set MODEL_PATH=models/thin_checkpoints/best_model.pt
    echo Using standard model: !MODEL_PATH!
)

echo ======================================================
echo   Starting MRL Forecast Pro (Windows)
echo   Port: %PORT%
echo   Model: %MODEL_PATH%
echo ======================================================

set NOWCAST_MODEL_CHECKPOINT=%MODEL_PATH%
set PORT=%PORT%

call venv\Scripts\activate
python src/web_app.py

pause
