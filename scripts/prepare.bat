@echo off
setlocal

set SEQ_LEN=%1
if "%SEQ_LEN%"=="" set SEQ_LEN=8

set ARCHIVE_DIR=%2

if "%ARCHIVE_DIR%"=="" (
    echo Error: Archive directory not specified.
    exit /b 1
)

echo === Подготовка датасета (Windows) ===
echo Длина последовательности: %SEQ_LEN%
echo Папка архива: %ARCHIVE_DIR%

call venv\Scripts\activate
python src/make_dataset.py --seq-len "%SEQ_LEN%" --archive-dir "%ARCHIVE_DIR%"

echo === Подготовка завершена ===
pause
