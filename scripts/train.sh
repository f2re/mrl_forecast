#!/bin/bash

# Обучение MRL-PhysEvolution или контрольной ConvLSTM.

set -e

EPOCHS=${1:-20}
BATCH_SIZE=${2:-1}
LR=${3:-1e-4}
DATA_DIRS=${4:-"data/processed_archive"}
VAL_SPLIT=${5:-0.2}
LEAD_STEPS=${6:-4}
ARCHITECTURE=${7:-phys-evolution}
INPUT_STEPS=${8:-4}

echo "=== Запуск обучения ==="
echo "Архитектура: $ARCHITECTURE"
echo "Эпох: $EPOCHS"
echo "Размер батча: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Датасеты: $DATA_DIRS"
echo "Validation split: $VAL_SPLIT"
echo "История: $INPUT_STEPS шагов"
echo "Прогноз: $LEAD_STEPS шагов"

source venv/bin/activate
python3 src/train_nowcasting_model.py \
    --data-dirs "$DATA_DIRS" \
    --architecture "$ARCHITECTURE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --val-split "$VAL_SPLIT" \
    --input-length "$INPUT_STEPS" \
    --target-length "$LEAD_STEPS" \
    --output-dir models/registry

echo "=== Обучение завершено ==="
