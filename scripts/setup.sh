#!/bin/bash

# Скрипт для начальной настройки проекта:
# создание виртуального окружения, установка зависимостей и подготовка папок.

set -e # Остановка при любой ошибке

echo "=== Настройка проекта MRL Forecast ==="

# 1. Создание виртуального окружения, если оно не существует
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения venv..."
    python3 -m venv venv
else
    echo "Виртуальное окружение venv уже существует."
fi

# 2. Активация venv и установка зависимостей
echo "Установка зависимостей из requirements.txt..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Создание необходимых директорий для данных и моделей
echo "Создание структуры директорий..."
mkdir -p data/raw/archive
mkdir -p data/processed_archive
mkdir -p models/real_checkpoints

echo "=== Настройка завершена успешно ==="
