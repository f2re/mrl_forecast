#!/bin/bash

# Скрипт для подготовки датасета из сырых данных.
# Преобразует радарные данные в последовательности Numpy.

SEQ_LEN=${1:-8}
INPUT_DIR=${2:-"data/raw/archive"}
OUTPUT_DIR="data/processed_archive"

echo "=== Подготовка датасета ==="
echo "Длина последовательности: $SEQ_LEN"
echo "Входная папка: $INPUT_DIR"

source venv/bin/activate
python3 src/make_dataset.py --archive-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --seq-len "$SEQ_LEN"

echo "=== Подготовка завершена ==="
