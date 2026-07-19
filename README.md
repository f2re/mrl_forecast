# 🌦️ MRL Forecast Pro

Экспериментальная система прогноза поля радиолокационной отражаемости и зон радиоэха. Результат не является официальным прогнозом интенсивности осадков или предупреждением об опасном явлении.

## Текущее состояние

Рабочий контур включает:

- canonical grid `512 × 512`, разрешение около `1 км`, локальная AEQD;
- исходные файлы без изменения и полный provenance;
- отдельные `reflectivity`, `valid_mask`, сроки UTC и quality metadata;
- количественные адаптеры NOAA NEXRAD Level II и DWD ODIM HDF5;
- WIS2 discovery;
- Meteoinfo и RainViewer только как визуальные источники;
- SQLite-каталог архивов, отдельных наблюдений и датасетов;
- классификацию сухих, слабых, осадочных и конвективных ситуаций;
- 50/50 dry/echo sampling только для train;
- purged chronological train/validation/test split;
- persistence, global shift и local block-motion baselines;
- CSI, POD, FAR, frequency bias, ETS, FSS и пространственные ошибки;
- ConvLSTM baseline;
- `MRL-PhysEvolution` с отдельными ветвями переноса, роста, распада и неопределённости;
- общий runtime для терминала и веба;
- SQLite job runner;
- адаптивный интерфейс в визуальном языке macOS/iOS;
- CPU benchmark p50/p95 и RAM.

Практический статус этапов: [`plan/10-implementation-roadmap.md`](plan/10-implementation-roadmap.md).

## Российские ДМРЛ

Подтверждённого открытого долговременного архива сырых российских ДМРЛ-BUFR пока не найдено. WIS2 используется для поиска открытых машинно-читаемых наборов. Meteoinfo и RainViewer не преобразуются в фиктивный количественный `dBZ`.

Российский источник получит `training_allowed` только после проверки:

- реального файла или открытого endpoint;
- единиц отражаемости;
- времени наблюдения;
- геометрии и координат;
- маски покрытия;
- условий использования.

## Быстрый старт

```bash
bash scripts/setup.sh
bash scripts/run_app.sh
```

Интерфейс: `http://localhost:5005`.

В интерфейсе доступны:

- загрузка NOAA и DWD;
- выбор временного шага и длины sequence;
- перестроение каталога;
- подготовка masked dataset;
- запуск ConvLSTM или MRL-PhysEvolution;
- очередь, отмена и журналы заданий;
- выбор модели;
- слои отражаемости, движения, роста, распада и неопределённости.

## Проверка окружения и источников

```bash
python scripts/doctor.py
python scripts/doctor.py --check-dwd --dwd-station ess
python scripts/check_aws_source.py --station KOKX --date 2024-05-20 --decode-one
python scripts/check_dwd_source.py --station ess --list-stations
python scripts/check_dwd_source.py --station ess --decode-one
python scripts/check_open_radar_sources.py --source all
python scripts/check_open_radar_sources.py --source wis2 --limit 100
```

Перед включением DWD-станции в обучение обязателен успешный `--decode-one` на целевой машине.

## Загрузка исходных данных

### NOAA NEXRAD Level II

```bash
bash scripts/download.sh KOKX 2024-05-20 100
```

### DWD ODIM HDF5

```bash
bash scripts/download_dwd.sh ess 2026-07-19 200
```

Оба загрузчика сохраняют исходные файлы, метаданные, SHA-256 и индексируют наблюдения в SQLite.

## Каталог наблюдений

```bash
python scripts/catalog.py rebuild
python scripts/catalog.py summary
python scripts/catalog.py list --source dwd-open-data --station ESS --limit 20
```

Каталог по умолчанию: `data/radar_catalog.sqlite3`.

Он хранит:

- источник и станцию;
- срок UTC;
- путь к исходному файлу;
- формат и размер;
- SHA-256;
- QC и provenance;
- prepared datasets и распределение классов.

## Подготовка датасета

### NOAA: 4 + 4 срока по 15 минут

```bash
bash scripts/prepare.sh \
  8 \
  data/raw/archive/<NOAA_SESSION> \
  canonical \
  15
```

### DWD/целевой 10-минутный профиль: 6 + 6 сроков

```bash
bash scripts/prepare.sh \
  12 \
  data/raw/archive/<DWD_SESSION> \
  canonical \
  10
```

Результат `.npz` содержит:

```text
reflectivity    [T,H,W]
valid_mask      [T,H,W]
timestamps_utc  [T]
```

В manifest сохраняются event class, статистика и трёхчасовая `split_group`.

## Обучение

```bash
bash scripts/train.sh \
  20 \
  1 \
  1e-4 \
  data/processed_archive/<DATASET_ID> \
  0.2 \
  6 \
  phys-evolution \
  6
```

Для контрольной модели замените `phys-evolution` на `convlstm`.

Расширенный запуск:

```bash
python src/train_nowcasting_model.py \
  --data-dirs data/processed_archive/<DATASET_ID> \
  --architecture phys-evolution \
  --input-length 6 \
  --target-length 6 \
  --val-split 0.2 \
  --test-split 0.1 \
  --batch-size 1 \
  --epochs 20
```

Выбор лучшей эпохи выполняется по validation. Окончательный quality gate выполняется на независимых test-блоках, если датасет содержит не менее пяти временных групп. Для старых или малых наборов metadata явно указывает `validation_fallback`.

Модель публикуется только если её masked MSE лучше:

- persistence;
- global shift advection;
- local block motion;

и отсутствует uniform-field anomaly.

## Терминальный инференс

```bash
python src/run_inference.py \
  --model-path models/registry/<MODEL_ID> \
  --source aws \
  --station KOKX \
  --output-dir data/predictions
```

Команда использует тот же `ModelRuntime`, grid contract и cadence, что и веб-приложение.

## CPU benchmark

```bash
python scripts/benchmark_cpu.py \
  --model-path models/registry/<MODEL_ID> \
  --threads 8 \
  --warmup 1 \
  --repeats 5 \
  --save
```

Отчёт содержит p50/p95/mean latency и max RSS. Синтетическое эхо используется только для измерения производительности, не для оценки качества.

## Ключевые файлы

```text
src/radar_contract.py       единый формат и capabilities
src/radar_pipeline.py       canonical gridding
src/radar_catalog.py        SQLite-каталог
src/dwd_source.py           DWD ODIM HDF5 adapter
src/open_sources.py         WIS2/Meteoinfo/RainViewer
src/make_dataset.py         masked sequences и split groups
src/event_catalog.py        классы ситуаций и балансировка
src/forecast_quality.py     baselines и метрики
src/phys_evolution.py       перенос + рост/распад
src/model_runtime.py        общий inference runtime
src/jobs.py                 очередь заданий
src/web_app.py              API
src/static/                 интерфейс
```

## Ограничения

- Выход — отражаемость `dBZ`, а не QPE.
- `MRL-PhysEvolution` — physics-guided nowcasting model, не полная атмосферная PINN.
- Прогноз инициации новой конвекции без спутника, молний и NWP ограничен.
- Горизонты более 60 минут по одной отражаемости имеют пониженную достоверность.
- Визуальные тайлы не используются для количественного обучения.
- DWD-продукт и каждая новая станция должны пройти decode/QC-проверку.
- Модели и датасеты с разными grid, cadence или pipeline version не смешиваются.
- ONNX/ONNX Runtime и квантизация ещё не включены в рабочий контур.

## Лицензия

MIT License
