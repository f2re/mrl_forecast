#!/bin/bash

# Скрипт для скачивания архивных данных NEXRAD.
# Использование: ./scripts/download.sh [STATION] [DATE] [COUNT]

STATION=${1:-KOKX}
DATE=${2:-$(date -d "yesterday" +%Y-%m-%d)}
COUNT=${3:-50}

echo "=== Запуск скачивания данных ==="
echo "Станция: $STATION"
echo "Дата: $DATE"
echo "Количество файлов: $COUNT"

source venv/bin/activate
python3 src/download_archive.py --station "$STATION" --date "$DATE" --count "$COUNT" --output data/raw/archive

echo "=== Скачивание завершено ==="
