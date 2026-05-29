#!/bin/bash

# Скрипт для обучения модели ConvLSTM.

EPOCHS=${1:-10}
BATCH_SIZE=${2:-4}
LR=${3:-1e-4}
DATA_DIR=${4:-"data/processed_archive"}

echo "=== Запуск обучения модели ==="
echo "Эпох: $EPOCHS"
echo "Размер батча: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Датасет: $DATA_DIR"

source venv/bin/activate
python3 src/train_nowcasting_model.py \
    --data-dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --output-dir models/registry

echo "=== Обучение завершено ==="
