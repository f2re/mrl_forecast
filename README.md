# 🌦️ MRL Forecast: ИИ-Наукастинг Осадков

Этот проект представляет собой систему краткосрочного прогнозирования (наукастинга) метеорологических радиолокационных данных с использованием глубокого обучения (Convolutional LSTM). Система автоматически предсказывает перемещение и развитие зон осадков на ближайший час.

## ✨ Ключевые возможности

- **🧠 Нейросетевой движок**: Реализация многослойной сети ConvLSTM на PyTorch для пространственно-временного прогнозирования.
- **📡 Универсальные Дата-Адаптеры**:
  - **NOAA FTP (Live)**: Автоматическое скачивание и декодирование сырых данных NEXRAD Level III с публичных серверов США (через библиотеку `MetPy`).
  - **Локальные архивы**: Декодирование файлов формата **BUFR** (МРЛ/ДМРЛ) с помощью `eccodes`.
  - **Online API**: Поддержка глобальных радарных композитов (RainViewer).
- **🗺️ Профессиональная визуализация**: Рендеринг данных с наложением колец дальности радара (50–250 км) и стандартизированной цветовой шкалой отражаемости (dBZ).
- **🖥️ Современный Web UI**: Адаптивный дашборд (Bootstrap 5 + AJAX) с возможностью динамического выбора радаров и исторических срезов.

## 📂 Структура проекта

```text
mrl_forecast/
├── src/
│   ├── web_app.py                 # 🚀 Главный Flask веб-сервер
│   ├── adapters.py                # 🔌 Адаптеры данных (FTP, Local, API)
│   ├── nexrad_decoder.py          # 🇺🇸 Декодер бинарных файлов NEXRAD (MetPy)
│   ├── bufr_decoder.py            # 🇷🇺 Декодер формата WMO BUFR (eccodes)
│   ├── map_visualization.py       # 🗺️ Генерация карт и графиков dBZ
│   ├── train_nowcasting_model.py  # 🏋️ Скрипт обучения модели
│   ├── make_dataset.py            # 📦 Пайплайн сборки датасета
│   └── generate_dummy_data.py     # 🧪 Генератор синтетических данных
├── templates/
│   └── index.html                 # 🎨 Шаблон веб-интерфейса (Bootstrap 5)
├── data/
│   ├── raw/                       # Сырые файлы радара (.bufr, .last)
│   └── processed/                 # Обработанные Numpy последовательности
├── models/
│   └── checkpoints/               # Обученные веса модели (.pt)
├── docs/                          # Аналитические отчеты и планы
└── requirements.txt               # Зависимости Python
```

## ⚙️ Установка

1. **Клонируйте репозиторий**:
   ```bash
   git clone git@github.com:f2re/mrl_forecast.git
   cd mrl_forecast
   ```

2. **Создайте виртуальное окружение**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Для Windows: venv\Scripts\activate
   ```

3. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Примечание: Для macOS может потребоваться установка `eccodes` через Homebrew/MacPorts перед установкой python-пакета).*
## 🚀 Запуск и Использование

### 1. Веб-интерфейс (Оперативный режим)

Проект поставляется с преднастроенной логикой и может сразу работать с публичными серверами (NOAA/RainViewer):

```bash
export PORT=5005
export NOWCAST_MODEL_CHECKPOINT=models/thin_checkpoints/best_model.pt
python src/web_app.py
```
Откройте браузер по адресу `http://localhost:5005`. В интерфейсе вы сможете выбрать реальную метеостанцию (например, Нью-Йорк или Детройт) и получить ИИ-прогноз на основе свежих сканирований.

---

## 🧬 Полный цикл ИИ: От данных до прогноза

Ниже приведен пошаговый процесс обучения собственной модели на архивных данных NEXRAD (США).

### Шаг 1: Скачивание архивных данных
Используйте `src/download_archive.py` для загрузки сырых файлов Level II из облака AWS S3:
```bash
# Скачать 50 последних сканов для станции KOKX (Нью-Йорк) за конкретную дату
python src/download_archive.py --station KOKX --date 2024-05-20 --count 50 --output data/raw/archive_KOKX
```

### Шаг 2: Создание обучающего датасета
Преобразуйте сырые радарные данные в 2D-сетки и сформируйте временные последовательности (Numpy):
```bash
# --seq-len 8 означает 4 кадра истории + 4 кадра для прогноза
python src/make_dataset.py --archive-dir data/raw/archive_KOKX --output-dir data/processed_archive --seq-len 8
```

### Шаг 3: Обучение нейросети ConvLSTM
Запустите процесс обучения. Модель будет сохранять лучший чекпоинт в указанную папку:
```bash
python src/train_nowcasting_model.py \
    --data-dir data/processed_archive \
    --epochs 20 \
    --batch-size 4 \
    --lr 1e-4 \
    --output-dir models/real_checkpoints
```

### Шаг 4: Применение (Inference)
Запустите веб-приложение, указав путь к вашей новой модели:
```bash
export NOWCAST_MODEL_CHECKPOINT=models/real_checkpoints/best_model.pt
python src/web_app.py
```

---

## ⚙️ Установка

- [ ] Поддержка архитектуры TrajGRU для работы с вращательными движениями циклонов.
- [ ] Интеграция интерактивных карт (Leaflet.js).

## 📄 Лицензия
MIT License
