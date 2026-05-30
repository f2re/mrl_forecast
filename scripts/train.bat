@echo off
setlocal

set EPOCHS=%1
if "%EPOCHS%"=="" set EPOCHS=10

set BATCH_SIZE=%2
if "%BATCH_SIZE%"=="" set BATCH_SIZE=4

set LR=%3
if "%LR%"=="" set LR=1e-4

set DATA_DIRS=%4
if "%DATA_DIRS%"=="" (
    echo Error: Data directories not specified.
    exit /b 1
)

set VAL_SPLIT=%5
if "%VAL_SPLIT%"=="" set VAL_SPLIT=0.2

set LEAD_TIME=%6
if "%LEAD_TIME%"=="" set LEAD_TIME=4

echo === Обучение модели (Windows) ===
echo Epochs: %EPOCHS%, Batch: %BATCH_SIZE%, LR: %LR%
echo Lead Time: %LEAD_TIME% steps
echo Data: %DATA_DIRS%

call venv\Scripts\activate
python src/train_nowcasting_model.py --epochs "%EPOCHS%" --batch-size "%BATCH_SIZE%" --lr "%LR%" --data-dirs "%DATA_DIRS%" --val-split "%VAL_SPLIT%" --target-length "%LEAD_TIME%"

echo === Обучение завершено ===
pause
