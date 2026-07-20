# 🌦️ MRL Forecast Pro

Экспериментальная система краткосрочного прогноза поля радиолокационной отражаемости и зон радиоэха.

> ⚠️ Результат не является официальным прогнозом интенсивности осадков, штормовым предупреждением или продукцией Росгидромета.

## 📚 Документация

| Документ | Назначение |
| --- | --- |
| **[Полное руководство по запуску и эксплуатации](docs/operations-guide.md)** | установка, настройка, источники, загрузка, датасеты, обучение, quality gate, инференс, UI, benchmark и диагностика |
| [Доступ к международным источникам](docs/source-access.md) | API, S3, токены, регистрация, probe и правила допуска источников |
| [План реализации](plan/10-implementation-roadmap.md) | фактический статус этапов и дальнейшая разработка |

## 🚀 Быстрый старт

```bash
git clone https://github.com/f2re/mrl_forecast.git
cd mrl_forecast
git checkout main

bash scripts/setup.sh
source venv/bin/activate
python mrl.py doctor
bash scripts/run_app.sh
```

Интерфейс:

```text
http://localhost:5005
```

`scripts/run_app.sh` запускает:

```text
job worker
+ фоновую проверку активных источников
+ веб-интерфейс
```

## 🧭 Полный рабочий цикл

```text
источник → raw archive → canonical grid → quality-aware dataset →
train/validation/test → quality gate → model registry →
inference → PNG/NPZ/JSON/NetCDF → CPU benchmark
```

Рекомендуемый порядок первого запуска:

```bash
# 1. Проверка окружения
python mrl.py doctor

# 2. Проверка источников
python mrl.py sources \
  --action probe \
  --source all \
  --active-only \
  --download-test \
  --limit 1

# 3. Скачивание DWD
python mrl.py download \
  --source dwd \
  --station ess \
  --date 2026-07-19 \
  --count 200

# 4. Подготовка 6 + 6 сроков по 10 минут
python mrl.py prepare \
  --archive-dir data/raw/archive/<SESSION> \
  --seq-len 12 \
  --grid-profile canonical \
  --time-step-minutes 10

# 5. Обучение
python mrl.py train \
  --data-dirs data/processed_archive/<DATASET_ID> \
  --architecture phys-evolution \
  --input-length 6 \
  --target-length 6 \
  --val-split 0.2 \
  --test-split 0.1 \
  --batch-size 1 \
  --epochs 20

# 6. CPU benchmark
python mrl.py benchmark \
  --model-path models/registry/<MODEL_ID> \
  --threads 8 \
  --save
```

Подробные проверки и расшифровка каждого шага находятся в [руководстве оператора](docs/operations-guide.md).

## 🧠 Модель `MRL-PhysEvolution`

Модель явно разделяет перенос и эволюцию радиоэха:

```text
reflectivity + effective quality mask + range_norm
                         │
                         ▼
                lightweight encoder
                         │
                         ▼
                     ConvGRU
              ┌──────────┼───────────┐
              ▼          ▼           ▼
           Motion    Growth/Decay  Uncertainty
              └──────────┼───────────┘
                         ▼
             differentiable advection
                         ▼
              reflectivity forecast
```

Основной профиль:

```text
canonical grid          512 × 512
resolution              1 км
time step               10 минут
history                 6 сроков / 60 минут
forecast                6 сроков / 60 минут
```

Legacy baseline:

```text
4 входных + 4 выходных срока по 15 минут
```

## 🛰️ Источники данных

### Полностью автоматизированные

| Source ID | Данные | Доступ | Скачивание | Train по умолчанию |
| --- | --- | --- | ---: | ---: |
| `noaa-aws` | NEXRAD Level II | открытый S3 | ✅ | ✅ |
| `dwd-open-data` | DBZH ODIM HDF5 | прямой HTTP | ✅ | ✅ для проверенного продукта |
| `fmi-s3` | ODIM 2.3 HDF5 volumes | открытый S3 | ✅ | ⛔ до отдельного QC |
| `opera-ord` | ODIM HDF5/BUFR | API + 24h cache | ✅ | ⛔ до per-file QC |
| `dmi-radar` | volume/pseudo-CAPPI/composite | открытый STAC | ✅ | ⛔ до отдельного QC |
| `knmi-radar` | polarimetric HDF5 | API key | ✅ | ⛔ до отдельного QC |
| `wis2-cache` | BUFR/GRIB/NetCDF | открытый S3 | ✅ | ⛔ до проверки набора |

### Probe-only или ручной доступ

```text
WMO Weather Radar Database
Météo-France Package Radar
CEDA NIMROD
MeteoSwiss
GeoSphere Austria
AURA/NCI
MetService
Taiwan QPESUMS
NASA GPM-GV
ЦАО NCRadar
```

Список и режимы доступа:

```bash
python mrl.py sources --action list
```

Инструкция регистрации:

```bash
python mrl.py sources \
  --action info \
  --source knmi-radar
```

Безопасная настройка ключа:

```bash
python mrl.py sources \
  --action configure \
  --source knmi-radar
```

Секрет сохраняется вне репозитория:

```text
~/.config/mrl_forecast/credentials.json
```

с правами `0600`.

## 🛡️ Контракт качества данных

Формат `npz-radar-quality-v2`:

```text
reflectivity           [T,H,W]  dBZ
valid_mask             [T,H,W]
coverage_mask          [T,H,W]
clutter_mask           [T,H,W]
interpolation_weight   [T,H,W]
timestamps_utc         [T]
```

Эффективная область:

```text
valid_mask
AND coverage_mask
AND NOT clutter_mask
AND interpolation_weight > 0
```

Маски проходят через полный контур:

```text
adapter → RadarFrame → dataset → loss/metrics → runtime → UI → NetCDF
```

## ⚖️ Выборка и quality gate

Train sampler по умолчанию балансирует:

```text
50% dry_valid
50% сумма echo-классов
```

Validation и test сохраняют естественное распределение.

Модель публикуется только если:

```text
model MSE < persistence MSE
model MSE < global advection MSE
model MSE < local block-motion MSE
uniform_field_anomaly == false
```

Отчёт дополнительно содержит:

```text
CSI / POD / FAR
frequency bias / ETS
FSS
max dBZ error
area bias 20/30/40 dBZ
motion/growth/decay diagnostics
```

## 🎛️ Единый CLI

```bash
python mrl.py --help
```

Команды:

```text
doctor       проверка окружения
sources      список, регистрация, probe и sample-download
download     загрузка открытого архива
prepare      canonical dataset
train        ConvLSTM или MRL-PhysEvolution
infer        терминальный прогноз
catalog      SQLite-каталог наблюдений
benchmark    CPU p50/p95 и RAM
serve        веб-интерфейс
worker       локальная очередь заданий
```

## 🗂️ Основные каталоги

```text
data/raw/archive/          исходные файлы и metadata
data/processed_archive/    quality-aware datasets
data/predictions/          результаты CLI-инференса
data/exports/              NetCDF
models/registry/           модели и quality gate
models/checkpoints/        служебные checkpoints
data/radar_catalog.sqlite3 каталог наблюдений
```

## ✅ Проверки кода

```bash
python -m py_compile mrl.py
python -m compileall -q src scripts tests

for file in scripts/*.sh; do
  bash -n "$file"
done

node --check src/static/app.js
node --check src/static/source-health.js

python -m unittest \
  tests.test_source_access \
  tests.test_metadata_access
```

Полный набор:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## 🇷🇺 Российские ДМРЛ

Подтверждённого открытого долговременного архива сырых российских ДМРЛ-BUFR пока нет.

Используются:

```text
WIS2 discovery и Global Cache
+ профиль официального запроса ncradar-cao
```

```bash
python mrl.py sources \
  --action info \
  --source ncradar-cao
```

Meteoinfo и RainViewer остаются visual-only и не преобразуются в фиктивный количественный `dBZ`.

## ⚠️ Ограничения

- Выход модели — отражаемость `dBZ`, а не QPE.
- `MRL-PhysEvolution` — physics-guided nowcasting model, не полная атмосферная PINN.
- Прогноз инициации новой конвекции без спутника, молний и NWP ограничен.
- Горизонты более 60 минут по одной отражаемости имеют пониженную достоверность.
- Доступность endpoint не означает научную пригодность данных.
- Визуальные тайлы не допускаются к количественному обучению.
- Production WSGI, ONNX Runtime и квантизация пока не завершены.

## 📄 Лицензия

MIT License
