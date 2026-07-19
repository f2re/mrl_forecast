#!/bin/bash

# Подготовка masked dataset из сырых радиолокационных данных.

set -e

SEQ_LEN=${1:-8}
INPUT_DIR=${2:-"data/raw/archive"}
GRID_PROFILE=${3:-canonical}
OUTPUT_DIR="data/processed_archive"

echo "=== Подготовка датасета ==="
echo "Длина последовательности: $SEQ_LEN"
echo "Входная папка: $INPUT_DIR"
echo "Профиль сетки: $GRID_PROFILE"

source venv/bin/activate
python3 src/make_dataset.py \
    --archive-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --seq-len "$SEQ_LEN" \
    --grid-profile "$GRID_PROFILE"

echo "=== Подготовка завершена ==="
