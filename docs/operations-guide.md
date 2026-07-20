# 🚀 Полное руководство по запуску и эксплуатации MRL Forecast

Это руководство описывает полный рабочий цикл проекта:

```text
установка → настройка доступа → проверка источников → скачивание →
канонизация → контроль выборки → обучение → quality gate →
инференс → экспорт → CPU benchmark → обслуживание
```

> ⚠️ MRL Forecast формирует **экспериментальный прогноз поля радиолокационной отражаемости**. Результат не является официальным прогнозом осадков, штормовым предупреждением или продукцией Росгидромета.

Связанные документы:

- [`../README.md`](../README.md) — краткий обзор и быстрые команды;
- [`source-access.md`](source-access.md) — подробности по источникам, токенам и регистрации;
- [`../plan/10-implementation-roadmap.md`](../plan/10-implementation-roadmap.md) — статус реализации и дальнейшие этапы.

---

## 🧭 1. Архитектура рабочего процесса

```text
┌─────────────────────────────┐
│ NOAA / DWD / FMI / OPERA    │
│ DMI / KNMI / WIS2 / local   │
└──────────────┬──────────────┘
               │ raw download
               ▼
┌─────────────────────────────┐
│ data/raw/archive/<SESSION>  │
│ исходный файл + metadata    │
│ SHA-256 + provenance        │
└──────────────┬──────────────┘
               │ canonicalize
               ▼
┌─────────────────────────────┐
│ data/processed_archive      │
│ reflectivity + quality masks│
│ timestamps + split groups   │
└──────────────┬──────────────┘
               │ train
               ▼
┌─────────────────────────────┐
│ models/registry/<MODEL_ID>  │
│ checkpoint + metrics        │
│ quality gate + model card   │
└──────────────┬──────────────┘
               │ infer
               ▼
┌─────────────────────────────┐
│ PNG / NPZ / JSON / NetCDF   │
│ прогноз + motion/evolution  │
│ quality masks + provenance  │
└─────────────────────────────┘
```

Основной целевой профиль:

```text
сетка                 512 × 512
разрешение            1 км
проекция              локальная AEQD
история               60 минут
шаг                    10 минут
вход                   6 сроков
прогноз                6 сроков
горизонт               60 минут
```

Legacy-профиль для существующих данных:

```text
история               4 × 15 минут
прогноз                4 × 15 минут
горизонт               60 минут
```

---

## 💻 2. Требования к оборудованию

### Эксплуатационный CPU-сервер

Минимально разумная конфигурация:

```text
CPU                    4 ядра x86_64
RAM                    16 ГБ
свободное место        от 20 ГБ для проверки
Python                 3.10+
```

Рекомендуемая конфигурация:

```text
CPU                    8–16 физических ядер
RAM                    32–64 ГБ
SSD                    от 500 ГБ
ОС                     Debian 11/12 или совместимая Astra Linux
```

### Машина обучения

```text
GPU                    NVIDIA, желательно 12–24 ГБ VRAM
RAM                    от 32 ГБ
SSD                    от 1 ТБ для многодневных архивов
```

Обучение возможно на CPU, но для сетки `512 × 512` это существенно медленнее. Для первичного smoke-test допустимо уменьшить число каналов модели.

---

## 📦 3. Установка системных зависимостей

### Debian / Ubuntu / Astra Linux

Названия отдельных пакетов могут отличаться между редакциями Astra Linux. Базовый набор:

```bash
sudo apt update
sudo apt install -y \
  git \
  python3 \
  python3-venv \
  python3-pip \
  build-essential \
  libeccodes-dev \
  libnetcdf-dev \
  libhdf5-dev \
  libproj-dev \
  proj-bin \
  libgeos-dev \
  gdal-bin \
  libgdal-dev \
  libopenblas-dev
```

Проверка версии Python:

```bash
python3 --version
```

Требуется Python `3.10` или новее.

### macOS с Homebrew

```bash
brew install python@3.11 eccodes netcdf hdf5 proj geos gdal
```

---

## 📥 4. Получение и обновление репозитория

Первичная установка:

```bash
git clone https://github.com/f2re/mrl_forecast.git
cd mrl_forecast
git checkout main
git pull --ff-only origin main
```

Проверка состояния:

```bash
git status --short
git branch --show-current
git log -1 --oneline
```

Ожидается:

```text
ветка                  main
рабочее дерево         без локальных изменений
```

Обновление существующей установки:

```bash
cd /path/to/mrl_forecast
git checkout main
git pull --ff-only origin main
source venv/bin/activate
python -m pip install -r requirements.txt
python mrl.py doctor
```

> 💡 Не выполняйте `git reset --hard`, если в каталоге имеются локальные настройки или собственные изменения, которые не сохранены отдельно.

---

## 🛠️ 5. Первичная настройка

Запустите штатный установщик:

```bash
bash scripts/setup.sh
```

Он выполняет:

1. проверку Python `3.10+`;
2. создание `venv`;
3. установку `requirements.txt`;
4. создание рабочих каталогов;
5. вывод следующих диагностических команд.

Ручной эквивалент:

```bash
python3 -m venv venv
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
  models/checkpoints
```

Проверка импорта PyTorch:

```bash
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
PY
```

### Установка CUDA-сборки PyTorch

`requirements.txt` содержит общий пакет `torch`. На GPU-машине после создания окружения установите сборку PyTorch, соответствующую версии драйвера и CUDA, командой из официального селектора PyTorch.

После установки обязательно проверьте:

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA не обнаружена'
print(torch.cuda.get_device_name(0))
PY
```

---

## 🔐 6. Переменные окружения и секреты

Основные переменные:

```bash
export PORT=5005
export DEBUG=false
export RADAR_DATA_DIR=data/processed
export NOWCAST_MODEL_CHECKPOINT=models/registry/<MODEL_ID>/best_model.pt
```

Для отдельного защищённого хранилища ключей:

```bash
export MRL_CREDENTIALS_FILE=/secure/path/mrl_credentials.json
```

Поддерживаемые секреты:

```text
KNMI_API_KEY
METEOGATE_API_KEY
METEOFRANCE_API_TOKEN
EARTHDATA_TOKEN
```

Приоритет чтения:

```text
1. переменная окружения
2. ~/.config/mrl_forecast/credentials.json
3. интерактивный скрытый ввод
```

Хранилище создаётся с правами `0600`. Проверка:

```bash
stat -c '%a %n' ~/.config/mrl_forecast/credentials.json
```

Ожидаемый режим:

```text
600 ~/.config/mrl_forecast/credentials.json
```

> 🔒 Не добавляйте токены в `.env`, README, issue, логи, `metadata.json` или командную строку CI.

---

## 🩺 7. Первичная диагностика

Активируйте окружение:

```bash
source venv/bin/activate
```

Проверка Python, модулей, директорий и свободного места:

```bash
python mrl.py doctor
```

Проверка NOAA:

```bash
python mrl.py doctor \
  --check-aws \
  --station KOKX \
  --date 2024-05-20
```

Проверка DWD:

```bash
python mrl.py doctor \
  --check-dwd \
  --dwd-station ess
```

Глубокая проверка декодирования одного DWD-файла:

```bash
python scripts/check_dwd_source.py \
  --station ess \
  --decode-one
```

Глубокая проверка NOAA Level II:

```bash
python scripts/check_aws_source.py \
  --station KOKX \
  --date 2024-05-20 \
  --decode-one
```

---

## 🌍 8. Настройка и проверка источников

### Список всех адаптеров

```bash
python mrl.py sources --action list
```

### Информация о конкретном источнике

```bash
python mrl.py sources \
  --action info \
  --source knmi-radar
```

Команда выводит:

```text
режим доступа
переменную с ключом
адрес регистрации
порядок получения доступа
статус автоматического скачивания
лицензию
```

### Настройка KNMI API key

```bash
python mrl.py sources \
  --action configure \
  --source knmi-radar
```

Ключ вводится скрыто. После настройки:

```bash
python mrl.py sources \
  --action probe \
  --source knmi-radar \
  --download-test \
  --limit 1
```

### Настройка OPERA/MeteoGate key

Базовый OPERA ORD может работать анонимно. Для маршрутов с ключом:

```bash
python mrl.py sources \
  --action configure \
  --source opera-ord
```

Если документация конкретного маршрута требует специальное имя HTTP-заголовка:

```bash
export METEOGATE_API_KEY_HEADER='<documented-header-name>'
```

### Проверка всех полностью автоматизированных источников

```bash
python mrl.py sources \
  --action probe \
  --source all \
  --active-only \
  --download-test \
  --limit 1
```

Результат:

```text
data/source_health.json
```

Просмотр:

```bash
python -m json.tool data/source_health.json | less
```

Основные статусы:

```text
available              каталог и тестовое чтение доступны
degraded               endpoint доступен, но подходящий файл не найден
credential_required    нужен ключ
manual_registration    требуется ручная регистрация или запрос
unavailable            endpoint или скачивание недоступны
probe_not_supported     автоматический probe не реализован
```

### Проверка одного источника

DMI:

```bash
python mrl.py sources \
  --action probe \
  --source dmi-radar \
  --station 06194 \
  --collection volume \
  --download-test
```

FMI:

```bash
python mrl.py sources \
  --action probe \
  --source fmi-s3 \
  --prefix '<verified-prefix>' \
  --download-test
```

OPERA:

```bash
python mrl.py sources \
  --action probe \
  --source opera-ord \
  --prefix '<country/station/prefix>' \
  --download-test
```

WIS2:

```bash
python mrl.py sources \
  --action probe \
  --source wis2-cache \
  --prefix '<verified-WIS2-prefix>' \
  --download-test
```

### Скачивание одного тестового файла

```bash
python mrl.py sources \
  --action sample \
  --source dmi-radar \
  --station 06194 \
  --collection volume \
  --output-dir data/source_samples
```

> ✅ Новый источник допускается к обучению только после проверки нескольких реальных файлов, полей, единиц, геометрии, timestamps, quality flags и лицензии.

---

## 📡 9. Скачивание исходных радиолокационных данных

### NOAA NEXRAD Level II

```bash
python mrl.py download \
  --source noaa \
  --station KOKX \
  --date 2024-05-20 \
  --count 100
```

### DWD ODIM HDF5

```bash
python mrl.py download \
  --source dwd \
  --station ess \
  --date 2026-07-19 \
  --count 200
```

### FMI public S3

Сначала определите рабочий prefix через probe/listing, затем:

```bash
python mrl.py download \
  --source fmi \
  --prefix '<verified-prefix>' \
  --count 100
```

### OPERA ORD

```bash
python mrl.py download \
  --source opera \
  --prefix '<country/station/prefix>' \
  --count 100
```

### DMI STAC API

```bash
python mrl.py download \
  --source dmi \
  --station 06194 \
  --collection volume \
  --date 2026-07-19 \
  --count 100
```

Допустимые коллекции:

```text
volume
pseudoCappi
composite
```

### KNMI

```bash
python mrl.py download \
  --source knmi \
  --dataset-name radar_volume_full_herwijnen \
  --dataset-version 1.0 \
  --count 100
```

### WIS2 Global Cache

```bash
python mrl.py download \
  --source wis2 \
  --prefix '<verified-WIS2-radar-prefix>' \
  --count 20
```

> ⚠️ WIS2 Global Cache содержит не только радиолокационные данные. Prefix должен быть сопоставлен с проверенной WIS2 discovery-записью.

---

## 🗂️ 10. Структура сырого архива

После скачивания создаётся каталог:

```text
data/raw/archive/<SOURCE>_<STATION>_<DATE>_<TIME>/
├── metadata.json
├── <raw-file-001>
├── <raw-file-002>
└── ...
```

Проверка последней сессии:

```bash
find data/raw/archive -mindepth 1 -maxdepth 1 -type d | sort | tail
```

Просмотр metadata:

```bash
python - <<'PY'
import json
from pathlib import Path

session = Path('data/raw/archive/<SESSION>')
meta = json.loads((session / 'metadata.json').read_text(encoding='utf-8'))
print(json.dumps(meta, indent=2, ensure_ascii=False))
PY
```

Обязательные признаки успешной загрузки:

```text
status                 completed
downloaded_count       > 0
files[].timestamp_utc  заполнен
files[].sha256          заполнен
source                  корректный source ID
```

Проверка контрольных сумм одного файла:

```bash
sha256sum data/raw/archive/<SESSION>/<FILE>
```

---

## 🧾 11. SQLite-каталог наблюдений

Перестроение каталога:

```bash
python mrl.py catalog rebuild
```

Сводка:

```bash
python mrl.py catalog summary
```

Фильтрация:

```bash
python mrl.py catalog list \
  --source dwd-open-data \
  --station ESS \
  --limit 20
```

Каталог по умолчанию:

```text
data/radar_catalog.sqlite3
```

Он используется как индекс. Исходные файлы и `metadata.json` остаются первичными артефактами.

Резервная копия каталога:

```bash
cp data/radar_catalog.sqlite3 \
  data/radar_catalog.$(date +%Y%m%d_%H%M%S).sqlite3
```

---

## 🧪 12. Подготовка canonical dataset

### NOAA: профиль 4 + 4 по 15 минут

```bash
python mrl.py prepare \
  --archive-dir data/raw/archive/<NOAA_SESSION> \
  --output-dir data/processed_archive \
  --seq-len 8 \
  --grid-profile canonical \
  --time-step-minutes 15
```

### Целевой профиль 6 + 6 по 10 минут

```bash
python mrl.py prepare \
  --archive-dir data/raw/archive/<ODIM_SESSION> \
  --output-dir data/processed_archive \
  --seq-len 12 \
  --grid-profile canonical \
  --time-step-minutes 10
```

Результат:

```text
data/processed_archive/<DATASET_ID>/
├── metadata.json
├── manifest.json
├── seq_0000.npz
├── seq_0001.npz
└── ...
```

Каждая последовательность содержит:

```text
reflectivity           [T,H,W]  dBZ
valid_mask             [T,H,W]
coverage_mask          [T,H,W]
clutter_mask           [T,H,W]
interpolation_weight   [T,H,W]
timestamps_utc         [T]
```

Эффективная маска:

```text
valid_mask
AND coverage_mask
AND NOT clutter_mask
AND interpolation_weight > 0
```

### Проверка датасета

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('data/processed_archive/<DATASET_ID>')
meta = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
first = root / manifest['sequences'][0]['file']

with np.load(first, allow_pickle=False) as data:
    print('metadata status:', meta.get('status'))
    print('sample format:', meta.get('sample_format'))
    print('sample count:', meta.get('sample_count'))
    print('split groups:', meta.get('split_group_count'))
    print('class counts:', meta.get('class_counts'))
    for name in (
        'reflectivity',
        'valid_mask',
        'coverage_mask',
        'clutter_mask',
        'interpolation_weight',
        'timestamps_utc',
    ):
        print(name, data[name].shape, data[name].dtype)
PY
```

Для допуска к обучению ожидается:

```text
metadata.status         completed
sample_count            > 0
split_group_count       желательно >= 5
source_training_allowed true или отсутствует для ранее проверенного источника
```

Если статус стал `unverified_source`, источник ещё не прошёл обязательную научную и лицензионную проверку.

---

## ⚖️ 13. Балансировка выборки

Классы событий:

```text
dry_valid
weak_echo
precipitation
convective
severe_core
invalid
```

По умолчанию train sampler формирует:

```text
50% dry_valid
50% сумма echo-классов
```

Validation и test **не балансируются** и сохраняют естественное распределение.

Отключение балансировки:

```bash
python mrl.py train \
  --data-dirs data/processed_archive/<DATASET_ID> \
  --architecture phys-evolution \
  --no-balanced-sampling
```

---

## 🧠 14. Обучение модели

### Smoke-test на одной эпохе

```bash
python mrl.py train \
  --data-dirs data/processed_archive/<DATASET_ID> \
  --architecture phys-evolution \
  --input-length 6 \
  --target-length 6 \
  --base-channels 8 \
  --hidden-channels 12 \
  --batch-size 1 \
  --epochs 1
```

Smoke-test проверяет работоспособность контура, но не качество прогноза.

### Стандартный запуск `MRL-PhysEvolution`

```bash
python mrl.py train \
  --data-dirs data/processed_archive/<DATASET_ID> \
  --architecture phys-evolution \
  --input-length 6 \
  --target-length 6 \
  --val-split 0.2 \
  --test-split 0.1 \
  --base-channels 16 \
  --hidden-channels 24 \
  --batch-size 1 \
  --lr 1e-4 \
  --epochs 20
```

### Обучение на нескольких совместимых датасетах

```bash
python mrl.py train \
  --data-dirs data/processed_archive/<DATASET_A>,data/processed_archive/<DATASET_B> \
  --architecture phys-evolution \
  --input-length 6 \
  --target-length 6 \
  --epochs 20
```

Все датасеты должны иметь одинаковые:

```text
pipeline_version
time_step_minutes
grid
```

### Контрольная ConvLSTM

```bash
python mrl.py train \
  --data-dirs data/processed_archive/<DATASET_ID> \
  --architecture convlstm \
  --input-length 6 \
  --target-length 6 \
  --hidden-channels 24 \
  --batch-size 1 \
  --epochs 20
```

### Практические рекомендации

| Параметр | Первый запуск | Рабочее обучение |
| --- | ---: | ---: |
| `epochs` | 1–3 | 20–100 |
| `batch-size` | 1 | 1–4, по памяти |
| `base-channels` | 8 | 16–24 |
| `hidden-channels` | 12 | 24–48 |
| `lr` | `1e-4` | `1e-4`, затем подбор |
| `val-split` | `0.2` | `0.15–0.2` |
| `test-split` | `0.1` | `0.1–0.15` |

Не увеличивайте каналы до проверки времени инференса на целевом CPU.

---

## 📈 15. Артефакты обучения

После запуска создаётся:

```text
models/registry/<MODEL_ID>/
├── best_model.pt
├── metadata.json
├── history.csv
└── learning_curve.png
```

Просмотр состояния:

```bash
python - <<'PY'
import json
from pathlib import Path

model = Path('models/registry/<MODEL_ID>')
meta = json.loads((model / 'metadata.json').read_text(encoding='utf-8'))
print(json.dumps({
    'status': meta.get('status'),
    'architecture': meta.get('model_architecture'),
    'horizon_minutes': meta.get('horizon_minutes'),
    'quality_gate_dataset': meta.get('quality_gate_dataset'),
    'metrics': meta.get('metrics'),
    'quality_gate_metrics': meta.get('quality_gate_metrics'),
}, indent=2, ensure_ascii=False))
PY
```

Статусы:

```text
training                  обучение выполняется
completed                 quality gate пройден
rejected_quality_gate     модель не превзошла baselines
failed                    обучение завершилось ошибкой
unverified_source         датасет источника не допущен к train
```

---

## ✅ 16. Проверка качества модели

Лучшая эпоха выбирается по validation loss. Финальный quality gate выполняется:

```text
на independent test       если временных групп достаточно
на validation fallback    для малых/legacy-датасетов
```

Модель должна иметь меньший masked MSE, чем:

```text
persistence
global shift advection
local block motion
```

Также не допускается `uniform_field_anomaly`.

Основные метрики:

| Метрика | Назначение |
| --- | --- |
| `model_mse` | ошибка модели по валидной области |
| `persistence_mse` | сохранение последнего кадра |
| `advection_mse` | глобальный перенос |
| `block_motion_mse` | локальное неоднородное перемещение |
| `CSI` | качество обнаружения зон выше порога |
| `POD` | доля обнаруженных событий |
| `FAR` | доля ложных тревог |
| `frequency_bias` | завышение/занижение частоты события |
| `ETS` | skill с поправкой на случайные совпадения |
| `FSS` | пространственное качество на разных масштабах |
| `max_dbz_error` | ошибка максимальной отражаемости |
| `area_bias` | ошибка площади зон 20/30/40 dBZ |

Быстрая проверка quality gate:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('models/registry/<MODEL_ID>')
meta = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
q = meta.get('quality_gate_metrics', {})
print('status:', meta.get('status'))
for name in ('model_mse', 'persistence_mse', 'advection_mse', 'block_motion_mse'):
    print(f'{name}:', q.get(name))
print('quality_gate_passed:', q.get('quality_gate_passed'))
print('uniform_field_anomaly:', q.get('uniform_field_anomaly'))
PY
```

> 🔬 Прохождение quality gate означает превосходство над реализованными baseline на данном test-наборе, но не является государственной аттестацией методики.

---

## 🖥️ 17. Запуск веб-интерфейса

### Автоматический запуск

```bash
bash scripts/run_app.sh
```

С указанием порта и модели:

```bash
bash scripts/run_app.sh \
  5005 \
  models/registry/<MODEL_ID>/best_model.pt
```

Скрипт запускает:

```text
1. job worker
2. фоновую проверку активных источников
3. Flask-приложение
```

Открыть:

```text
http://localhost:5005
```

### Раздельный запуск для диагностики

Терминал 1:

```bash
source venv/bin/activate
python mrl.py worker
```

Терминал 2:

```bash
source venv/bin/activate
python mrl.py serve \
  --model-path models/registry/<MODEL_ID>/best_model.pt \
  --port 5005
```

Разовый запуск worker:

```bash
python mrl.py worker --once
```

Состояния заданий:

```text
queued
running
cancelling
completed
failed
interrupted
```

> ⚠️ Текущий Flask server предназначен для локального исследовательского использования. Production WSGI-контур пока не реализован.

---

## 🎛️ 18. Порядок работы в интерфейсе

1. **Источники** — проверить доступность и инструкции регистрации.
2. **Архив** — загрузить NOAA или DWD либо проверить уже скачанные сессии.
3. **Каталог** — перестроить индекс после ручного добавления файлов.
4. **Датасеты** — выбрать архив, длину sequence, шаг и canonical grid.
5. **Обучение** — выбрать датасет, архитектуру и параметры.
6. **Задания** — контролировать очередь и журналы.
7. **Модели** — загрузить только модель со статусом `completed`.
8. **Прогноз** — выбрать источник и построить nowcast.
9. **Слои** — переключать отражаемость, перенос, рост, распад, uncertainty и quality masks.
10. **Экспорт** — сохранить NetCDF после успешного прогноза.

---

## 🌧️ 19. Терминальный инференс

NOAA AWS:

```bash
python mrl.py infer \
  --model-path models/registry/<MODEL_ID> \
  --source aws \
  --station KOKX \
  --output-dir data/predictions
```

Локальная quality-aware последовательность:

```bash
python mrl.py infer \
  --model-path models/registry/<MODEL_ID> \
  --source local \
  --local-dir data/processed \
  --output-dir data/predictions
```

DEMO-проверка интерфейса модели:

```bash
python mrl.py infer \
  --model-path models/registry/<MODEL_ID> \
  --source demo \
  --station DEMO
```

Результаты:

```text
data/predictions/
├── <station>_forecast.npz
├── <station>_forecast.json
├── <station>_history_*.png
├── <station>_forecast_*.png
├── <station>_motion_*.png
├── <station>_growth_*.png
├── <station>_decay_*.png
├── <station>_uncertainty_*.png
└── quality-mask PNG
```

---

## 🗺️ 20. Экспорт NetCDF

После прогноза в веб-интерфейсе используйте кнопку **«Экспорт NetCDF»**.

Файл содержит:

```text
reflectivity
valid_mask
coverage_mask
clutter_mask
interpolation_weight
valid_time_utc
lead_time_minutes
x / y
crs
model_id
pipeline_version
source
not_official_warning=true
```

Проверка файла:

```bash
python - <<'PY'
import xarray as xr

path = 'data/exports/<FILE>.nc'
with xr.open_dataset(path) as ds:
    print(ds)
    print(ds.attrs)
PY
```

---

## ⏱️ 21. CPU benchmark

```bash
python mrl.py benchmark \
  --model-path models/registry/<MODEL_ID> \
  --threads 8 \
  --warmup 1 \
  --repeats 5 \
  --save
```

Результат сохраняется рядом с моделью:

```text
models/registry/<MODEL_ID>/cpu_benchmark.json
```

Основные показатели:

```text
latency_ms.mean
latency_ms.p50
latency_ms.p95
max_rss_mb
threads
grid
input_length
target_length
```

Рекомендуется измерять минимум для:

```text
1 поток
половина физических ядер
все физические ядра
```

Не используйте число логических потоков как гарантированно оптимальное значение — benchmark должен определить фактический минимум p95.

---

## 🧹 22. Проверка кода и тесты

Синтаксис Python:

```bash
python -m py_compile mrl.py
python -m compileall -q src scripts tests
```

Shell-скрипты:

```bash
for file in scripts/*.sh; do
  bash -n "$file"
done
```

JavaScript:

```bash
node --check src/static/app.js
node --check src/static/source-health.js
```

Критические лёгкие тесты:

```bash
python -m unittest \
  tests.test_source_access \
  tests.test_metadata_access
```

Полный набор:

```bash
python -m unittest discover \
  -s tests \
  -p 'test_*.py'
```

> 💡 Полный набор требует установленных PyTorch, Py-ART, ecCodes, NetCDF и HDF5-зависимостей.

---

## 🔄 23. Обновление рабочей установки

Перед обновлением сохраните:

```text
~/.config/mrl_forecast/credentials.json
data/radar_catalog.sqlite3
models/registry/
нужные raw/processed архивы
```

Команды обновления:

```bash
cd /path/to/mrl_forecast
git status --short
git checkout main
git pull --ff-only origin main
source venv/bin/activate
python -m pip install -r requirements.txt
python mrl.py doctor
python mrl.py sources --action probe --source all --active-only --limit 1
```

После изменения схемы каталога:

```bash
python mrl.py catalog rebuild
```

---

## 💾 24. Резервное копирование

Минимальный backup моделей и служебных данных:

```bash
tar -czf mrl_control_$(date +%Y%m%d_%H%M%S).tar.gz \
  models/registry \
  data/radar_catalog.sqlite3 \
  data/source_health.json 2>/dev/null || true
```

Архивы МРЛ обычно слишком велики для общего tar-файла. Их лучше копировать инкрементально через `rsync`:

```bash
rsync -a --info=progress2 \
  data/raw/archive/ \
  /backup/mrl/raw/
```

---

## 🧯 25. Диагностика типовых ошибок

### `python3 -m venv` не работает

Установить пакет:

```bash
sudo apt install python3-venv
```

### Не импортируется `eccodes`

```bash
sudo apt install libeccodes-dev
source venv/bin/activate
python -m pip install --force-reinstall eccodes
```

Проверка:

```bash
python - <<'PY'
import eccodes
print('eccodes import OK')
PY
```

### Ошибка HDF5 / NetCDF / Py-ART

Проверьте системные библиотеки:

```bash
sudo apt install libhdf5-dev libnetcdf-dev libproj-dev libgeos-dev
python -m pip install --force-reinstall h5py netCDF4 arm-pyart
```

### `No files were downloaded`

Проверить:

```bash
python mrl.py sources --action info --source <SOURCE_ID>
python mrl.py sources --action probe --source <SOURCE_ID> --download-test
```

Возможные причины:

```text
неверный prefix
нет данных за выбранную дату
неверный station ID
просроченный API key
короткое окно архива
изменение endpoint или dataset version
```

### `unverified_source`

Данные скачаны, но источник не допущен к обучению. Требуется:

```text
проверить формат и поля
проверить dBZ units
проверить координаты станции
проверить timestamps
проверить quality masks
проверить лицензию
после этого изменить training_allowed в source capabilities
```

### `Dataset is too small`

Для независимого test желательно не менее пяти временных групп. Решение:

```text
скачать больше суток
увеличить число ситуаций
объединить совместимые датасеты
не уменьшать purge gap вручную без обоснования
```

### `rejected_quality_gate`

Модель не превзошла один из baseline. Не публиковать checkpoint как рабочий. Проверить:

```text
баланс классов
качество масок
число событий с сильным эхом
learning rate
число эпох
рост/распад в diagnostics
station/domain shift
```

### Модель не загружается в интерфейсе

Проверить:

```bash
cat models/registry/<MODEL_ID>/metadata.json
ls -l models/registry/<MODEL_ID>/best_model.pt
```

Рабочий статус должен быть:

```text
completed
```

### Worker не выполняет задания

Запустить отдельно:

```bash
python mrl.py worker
```

Проверить процессы:

```bash
ps aux | grep -E 'job_worker|web_app' | grep -v grep
```

### Порт занят

```bash
ss -ltnp | grep 5005
python mrl.py serve --port 5006
```

### CUDA не обнаружена

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

### Недостаточно места

```bash
df -h
du -sh data/raw/archive data/processed_archive models/registry
```

Удалять следует только явно ненужные сессии после сохранения metadata и проверки зависимых датасетов.

---

## 🇷🇺 26. Российские ДМРЛ

Текущий автоматический открытый архив исходных объёмов российских ДМРЛ не подтверждён.

Используются два направления:

1. WIS2 discovery и Global Cache — автоматический мониторинг появления открытых наборов;
2. `ncradar-cao` — профиль официального запроса в ЦАО.

Инструкция:

```bash
python mrl.py sources \
  --action info \
  --source ncradar-cao
```

В официальный запрос следует включать:

```text
WIGOS/radar ID
интервал дат
полные объёмы или elevation scans
DBZH / TH / VRADH / WRADH
ZDR / PHIDP / KDP / RHOHV
шаг дальности и азимута
quality masks
NetCDF/CfRadial, ODIM_H5 или FM 94 BUFR
условия использования
```

Meteoinfo и RainViewer используются только для визуального контроля и не преобразуются в количественный `dBZ`.

---

## 🛡️ 27. Требования безопасности и воспроизводимости

Обязательно сохранять:

```text
исходный файл
SHA-256
source ID
station ID
timestamp UTC
pipeline version
grid contract
quality masks
model ID
training dataset IDs
quality gate metrics
```

Запрещается:

```text
подменять timestamp файловым mtime
считать область вне покрытия отсутствием осадков
обучать на visual-only PNG/GIF как на dBZ
публиковать модель со статусом rejected_quality_gate
сохранять токены в репозитории
смешивать несовместимые grid/cadence/pipeline datasets
```

---

## 🧩 28. Полный пример: DWD → обучение → прогноз

```bash
# 1. Установка
bash scripts/setup.sh
source venv/bin/activate

# 2. Проверка
python mrl.py doctor --check-dwd --dwd-station ess
python scripts/check_dwd_source.py --station ess --decode-one

# 3. Скачивание
python mrl.py download \
  --source dwd \
  --station ess \
  --date 2026-07-19 \
  --count 240

# 4. Найти созданную сессию
find data/raw/archive -maxdepth 1 -type d -name 'DWD_*' | sort | tail -1

# 5. Подготовка 6 + 6 по 10 минут
python mrl.py prepare \
  --archive-dir data/raw/archive/<DWD_SESSION> \
  --seq-len 12 \
  --grid-profile canonical \
  --time-step-minutes 10

# 6. Smoke-test
python mrl.py train \
  --data-dirs data/processed_archive/<DATASET_ID> \
  --architecture phys-evolution \
  --input-length 6 \
  --target-length 6 \
  --base-channels 8 \
  --hidden-channels 12 \
  --epochs 1

# 7. Рабочее обучение
python mrl.py train \
  --data-dirs data/processed_archive/<DATASET_ID> \
  --architecture phys-evolution \
  --input-length 6 \
  --target-length 6 \
  --val-split 0.2 \
  --test-split 0.1 \
  --base-channels 16 \
  --hidden-channels 24 \
  --batch-size 1 \
  --epochs 20

# 8. Benchmark
python mrl.py benchmark \
  --model-path models/registry/<MODEL_ID> \
  --threads 8 \
  --save

# 9. Интерфейс
bash scripts/run_app.sh 5005 models/registry/<MODEL_ID>/best_model.pt
```

---

## ✅ 29. Контрольный список первого рабочего запуска

### Установка

- [ ] Python 3.10+ установлен;
- [ ] `scripts/setup.sh` завершился без ошибки;
- [ ] `python mrl.py doctor` возвращает `ok=true`;
- [ ] свободно не менее 20 ГБ.

### Источник

- [ ] `info` показывает ожидаемый режим доступа;
- [ ] обязательный ключ настроен;
- [ ] `probe --download-test` успешен;
- [ ] sample-файл скачан;
- [ ] timestamp и формат подтверждены.

### Датасет

- [ ] `metadata.status=completed`;
- [ ] quality-маски присутствуют;
- [ ] `sample_count > 0`;
- [ ] классы ситуаций не состоят только из `dry_valid`;
- [ ] источник допущен к обучению;
- [ ] есть независимые временные группы.

### Модель

- [ ] smoke-test прошёл;
- [ ] рабочее обучение завершено;
- [ ] `status=completed`;
- [ ] `quality_gate_passed=true`;
- [ ] модель лучше persistence и block motion;
- [ ] CPU p95 измерен.

### Эксплуатация

- [ ] worker запущен;
- [ ] интерфейс доступен;
- [ ] модель загружена;
- [ ] прогноз строится;
- [ ] quality layers отображаются;
- [ ] NetCDF открывается через xarray;
- [ ] токены отсутствуют в логах и metadata.

---

## 📚 30. Краткий справочник команд

```bash
# Установка
bash scripts/setup.sh

# Диагностика
python mrl.py doctor

# Источники
python mrl.py sources --action list
python mrl.py sources --action info --source knmi-radar
python mrl.py sources --action configure --source knmi-radar
python mrl.py sources --action probe --source all --active-only --download-test --limit 1

# Загрузка
python mrl.py download --source noaa --station KOKX --date 2024-05-20 --count 100
python mrl.py download --source dwd --station ess --date 2026-07-19 --count 200

# Каталог
python mrl.py catalog rebuild
python mrl.py catalog summary

# Датасет
python mrl.py prepare --archive-dir data/raw/archive/<SESSION> --seq-len 12 --time-step-minutes 10

# Обучение
python mrl.py train --data-dirs data/processed_archive/<DATASET_ID> --architecture phys-evolution --input-length 6 --target-length 6 --epochs 20

# Инференс
python mrl.py infer --model-path models/registry/<MODEL_ID> --source aws --station KOKX

# CPU benchmark
python mrl.py benchmark --model-path models/registry/<MODEL_ID> --threads 8 --save

# Интерфейс
bash scripts/run_app.sh
```
