#!/bin/bash

# Подготовка masked dataset из сырых радиолокационных данных.

set -e

SEQ_LEN=${1:-8}
INPUT_DIR=${2:-"data/raw/archive"}
GRID_PROFILE=${3:-canonical}
TIME_STEP_MINUTES=${4:-15}
OUTPUT_DIR="data/processed_archive"

echo "=== Подготовка датасета ==="
echo "Длина последовательности: $SEQ_LEN"
echo "Входная папка: $INPUT_DIR"
echo "Профиль сетки: $GRID_PROFILE"
echo "Шаг времени: $TIME_STEP_MINUTES мин"

source venv/bin/activate
python3 src/make_dataset.py \
    --archive-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --seq-len "$SEQ_LEN" \
    --grid-profile "$GRID_PROFILE" \
    --time-step-minutes "$TIME_STEP_MINUTES"

echo "=== Подготовка завершена ==="
