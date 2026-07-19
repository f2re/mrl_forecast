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
trap 'kill "$WORKER_PID" 2>/dev/null || true' EXIT INT TERM

echo "Запуск веб-приложения..."
python3 src/web_app.py
