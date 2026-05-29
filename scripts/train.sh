#!/bin/bash

# Скрипт для обучения модели ConvLSTM.

EPOCHS=${1:-10}
BATCH_SIZE=${2:-4}
LR=${3:-1e-4}

echo "=== Запуск обучения модели ==="
echo "Эпох: $EPOCHS"
echo "Размер батча: $BATCH_SIZE"
echo "Learning Rate: $LR"

source venv/bin/activate
python3 src/train_nowcasting_model.py \
    --data-dir data/processed_archive \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --output-dir models/real_checkpoints

echo "=== Обучение завершено ==="
