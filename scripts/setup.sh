#!/bin/bash

# Базовая настройка MRL Forecast:
# проверка Python, создание виртуального окружения, установка зависимостей
# и подготовка рабочих каталогов.

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Ошибка: $PYTHON_BIN не найден. Установите Python 3.10 или новее."
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Требуется Python >= 3.10, обнаружен {sys.version.split()[0]}"
    )
print(f"Python: {sys.version.split()[0]}")
PY

echo "=== Настройка проекта MRL Forecast ==="

if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения venv..."
    "$PYTHON_BIN" -m venv venv
else
    echo "Виртуальное окружение venv уже существует."
fi

source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

mkdir -p \
    data/raw/archive \
    data/processed_archive \
    data/processed \
    data/predictions \
    data/exports \
    data/source_samples \
    data/logs \
    models/registry \
    models/checkpoints \
    src/static

echo "Каталоги данных и моделей подготовлены."
echo "Проверка окружения: python mrl.py doctor"
echo "Запуск интерфейса: bash scripts/run_app.sh"
echo "=== Настройка завершена успешно ==="
