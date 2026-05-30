@echo off
setlocal

set STATION=%1
if "%STATION%"=="" set STATION=KOKX

set DATE=%2
if "%DATE%"=="" (
    for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value') do set dt=%%a
    set /a Y=!dt:~0,4!
    set /a M=1!dt:~4,2! - 100
    set /a D=1!dt:~6,2! - 101
    if !D! equ 0 (
        :: Simple yesterday calculation for Windows batch is complex, using today as fallback if empty
        set DATE=!dt:~0,4!-!dt:~4,2!-!dt:~6,2!
    ) else (
        if !D! lss 10 set D=0!D!
        if !M! lss 10 set M=0!M!
        set DATE=!Y!-!M!-!D!
    )
)

set COUNT=%3
if "%COUNT%"=="" set COUNT=50

echo === Запуск скачивания данных (Windows) ===
echo Станция: %STATION%
echo Дата: %DATE%
echo Количество файлов: %COUNT%

call venv\Scripts\activate
python src/download_archive.py --station "%STATION%" --date "%DATE%" --count "%COUNT%" --output data/raw/archive

echo === Скачивание завершено ===
pause
