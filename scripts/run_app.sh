#!/bin/bash

# Скрипт для запуска веб-сервера приложения.

PORT=${1:-5005}
MODEL_PATH=${2:-models/real_checkpoints/best_model.pt}

# Проверка наличия модели. Если нет пользовательской, пробуем стандартную.
if [ ! -f "$MODEL_PATH" ]; then
    echo "Предупреждение: Модель $MODEL_PATH не найдена."
    MODEL_PATH="models/thin_checkpoints/best_model.pt"
    echo "Используется стандартная модель: $MODEL_PATH"
fi

echo "=== Запуск веб-приложения ==="
echo "Порт: $PORT"
echo "Модель: $MODEL_PATH"

export PORT="$PORT"
export NOWCAST_MODEL_CHECKPOINT="$MODEL_PATH"

source venv/bin/activate
python3 src/web_app.py
