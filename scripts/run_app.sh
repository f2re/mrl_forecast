#!/bin/bash

set -e

PORT=${1:-5005}
MODEL_PATH=${2:-}

source venv/bin/activate

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH=$(find models/registry -mindepth 2 -maxdepth 2 -name best_model.pt -print 2>/dev/null | sort | tail -n 1)
fi

export PORT
if [ -n "$MODEL_PATH" ]; then
    export NOWCAST_MODEL_CHECKPOINT="$MODEL_PATH"
    echo "Модель: $MODEL_PATH"
else
    echo "Модель не выбрана. Её можно загрузить из реестра через интерфейс."
fi

echo "Порт: $PORT"
echo "Запуск job worker..."
python3 scripts/job_worker.py &
WORKER_PID=$!

echo "Фоновая проверка активных источников и тестового чтения файлов..."
mkdir -p data src/static
python3 scripts/source_access.py \
    --action probe \
    --source all \
    --active-only \
    --download-test \
    --limit 1 \
    --report-path src/static/source_health.json \
    > data/source_health.log 2>&1 &
PROBE_PID=$!

trap 'kill "$WORKER_PID" "$PROBE_PID" 2>/dev/null || true' EXIT INT TERM

echo "Запуск веб-приложения..."
python3 src/web_app.py
