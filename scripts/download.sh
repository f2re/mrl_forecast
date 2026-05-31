#!/bin/bash

# Скрипт для скачивания архивных данных NEXRAD.
# Использование: ./scripts/download.sh [STATION] [DATE] [COUNT]

STATION=${1:-KOKX}
if [ -n "$2" ]; then
    DATE="$2"
elif date -v-1d +%Y-%m-%d >/dev/null 2>&1; then
    DATE=$(date -v-1d +%Y-%m-%d)
else
    DATE=$(date -d "yesterday" +%Y-%m-%d)
fi
COUNT=${3:-50}

echo "=== Запуск скачивания данных ==="
echo "Станция: $STATION"
echo "Дата: $DATE"
echo "Количество файлов: $COUNT"

source venv/bin/activate
python3 src/download_archive.py --station "$STATION" --date "$DATE" --count "$COUNT" --output data/raw/archive

echo "=== Скачивание завершено ==="
