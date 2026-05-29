#!/bin/bash

# Скрипт для обучения модели ConvLSTM (Pro).

EPOCHS=${1:-10}
BATCH_SIZE=${2:-4}
LR=${3:-1e-4}
DATA_DIRS=${4:-"data/processed_archive"} # Может быть списком через запятую
VAL_SPLIT=${5:-0.2}
LEAD_TIME=${6:-4}

echo "=== Запуск продвинутого обучения ==="
echo "Эпох: $EPOCHS"
echo "Размер батча: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Датасеты: $DATA_DIRS"
echo "Валидация: $VAL_SPLIT"
echo "Заблаговременность: $LEAD_TIME шагов"

source venv/bin/activate
python3 src/train_nowcasting_model.py \
    --data-dirs "$DATA_DIRS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --val-split "$VAL_SPLIT" \
    --target-length "$LEAD_TIME" \
    --output-dir models/registry

echo "=== Обучение завершено ==="
