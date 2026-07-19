# MRL Forecast Pro

Экспериментальная система прогноза поля радиолокационной отражаемости и зон радиоэха. Результат не является официальным прогнозом интенсивности осадков или предупреждением об опасном явлении.

Практический статус этапов: [`plan/10-implementation-roadmap.md`](plan/10-implementation-roadmap.md).

## Текущий рабочий контур

- canonical grid `512 × 512`, разрешение около `1 км`, локальная AEQD;
- исходные файлы без изменения, SHA-256 и provenance;
- количественные адаптеры NOAA NEXRAD Level II и DWD ODIM HDF5;
- WIS2 discovery;
- Meteoinfo и RainViewer только как visual-only источники;
- SQLite-каталог архивов, отдельных сроков и датасетов;
- quality-aware `.npz` с отражаемостью, сроками и масками;
- классификация `dry_valid`, `weak_echo`, `precipitation`, `convective`, `severe_core`, `invalid`;
- dry/echo 50/50 sampling только в train;
- purged chronological train/validation/test split;
- persistence, global shift и local block-motion baselines;
- CSI, POD, FAR, frequency bias, ETS, FSS и пространственные ошибки;
- ConvLSTM baseline;
- `MRL-PhysEvolution` с переносом, ростом, распадом и неопределённостью;
- общий `ModelRuntime` для терминала и веба;
- SQLite job runner;
- адаптивный интерфейс в визуальном языке macOS/iOS;
- NetCDF с CRS, сроками, provenance и quality-масками;
- CPU benchmark p50/p95 и RAM;
- единый CLI `python mrl.py ...`.

## Quality-маски

Новый формат последовательности `npz-radar-quality-v2` содержит:

```text
reflectivity           [T,H,W]  dBZ
valid_mask             [T,H,W]  фактически допустимые пиксели
coverage_mask          [T,H,W]  геометрическое покрытие радиолокатора
clutter_mask           [T,H,W]  исключённые помехи
interpolation_weight   [T,H,W]  относительная уверенность 0…1
timestamps_utc         [T]
```

Эффективная область обучения и инференса вычисляется как:

```text
valid_mask AND coverage_mask AND NOT clutter_mask AND interpolation_weight > 0
```

Если источник не предоставляет количественный вес интерполяции, используется явно маркированный бинарный fallback по валидности. Такой fallback не трактуется как реальная ошибка интерполяции.

Маски проходят через:

```text
адаптер → RadarFrame → dataset → loss/metrics → runtime → UI → NetCDF
```

В интерфейсе доступны отдельные слои валидности, покрытия, помех и веса интерполяции. В NetCDF отражаемость вне `valid_mask` записывается как missing value.

## Российские ДМРЛ

Подтверждённого открытого долговременного архива сырых российских ДМРЛ-BUFR пока не найдено. WIS2 используется для поиска открытых машинно-читаемых наборов. Meteoinfo и RainViewer не преобразуются в фиктивный количественный `dBZ`.

Российский источник получит `training_allowed` только после проверки:

- реального файла или открытого endpoint;
- единиц отражаемости;
- времени наблюдения;
- геометрии и координат;
- маски покрытия и QC;
- условий использования.

## Быстрый старт

```bash
bash scripts/setup.sh
bash scripts/run_app.sh
```

Интерфейс: `http://localhost:5005`.

Job worker при запуске приложения поднимается отдельным процессом. Для ручного запуска:

```bash
python mrl.py worker
```

## Единый CLI

Общая справка:

```bash
python mrl.py --help
```

Проверка окружения:

```bash
python mrl.py doctor
python mrl.py doctor --check-aws --station KOKX --date 2024-05-20
python mrl.py doctor --check-dwd --dwd-station ess
```

Поиск открытых источников:

```bash
python mrl.py sources --source all
python mrl.py sources --source wis2 --limit 100
```

Загрузка NOAA:

```bash
python mrl.py download \
  --source noaa \
  --station KOKX \
  --date 2024-05-20 \
  --count 100
```

Загрузка DWD:

```bash
python mrl.py download \
  --source dwd \
  --station ess \
  --date 2026-07-19 \
  --count 200
```

Каталог:

```bash
python mrl.py catalog rebuild
python mrl.py catalog summary
python mrl.py catalog list --source dwd-open-data --station ESS --limit 20
```

Подготовка NOAA-профиля `4 + 4` по 15 минут:

```bash
python mrl.py prepare \
  --archive-dir data/raw/archive/<NOAA_SESSION> \
  --seq-len 8 \
  --grid-profile canonical \
  --time-step-minutes 15
```

Подготовка целевого профиля `6 + 6` по 10 минут:

```bash
python mrl.py prepare \
  --archive-dir data/raw/archive/<DWD_SESSION> \
  --seq-len 12 \
  --grid-profile canonical \
  --time-step-minutes 10
```

Обучение `MRL-PhysEvolution`:

```bash
python mrl.py train \
  --data-dirs data/processed_archive/<DATASET_ID> \
  --architecture phys-evolution \
  --input-length 6 \
  --target-length 6 \
  --val-split 0.2 \
  --test-split 0.1 \
  --batch-size 1 \
  --epochs 20
```

Контрольная ConvLSTM запускается с `--architecture convlstm`.

Терминальный инференс:

```bash
python mrl.py infer \
  --model-path models/registry/<MODEL_ID> \
  --source aws \
  --station KOKX \
  --output-dir data/predictions
```

CPU benchmark:

```bash
python mrl.py benchmark \
  --model-path models/registry/<MODEL_ID> \
  --threads 8 \
  --warmup 1 \
  --repeats 5 \
  --save
```

Запуск интерфейса:

```bash
python mrl.py serve \
  --model-path models/registry/<MODEL_ID> \
  --port 5005
```

## Выборка и quality gate

Выбор лучшей эпохи выполняется по validation. Окончательный quality gate выполняется на независимых test-блоках, если архив содержит достаточно временных групп. Для небольших и legacy-наборов metadata явно указывает `validation_fallback`.

Модель публикуется только если её masked MSE лучше:

- persistence;
- global shift advection;
- local block motion;

и отсутствует uniform-field anomaly.

Validation и test сохраняют естественное распределение ситуаций. Искусственная балансировка применяется только к train sampler.

## Проверка источников

Перед включением DWD-станции в обучение обязателен успешный decode/QC-check:

```bash
python scripts/check_dwd_source.py --station ess --decode-one
```

Для NOAA:

```bash
python scripts/check_aws_source.py \
  --station KOKX \
  --date 2024-05-20 \
  --decode-one
```

## Ключевые файлы

```text
mrl.py                       единый CLI
src/radar_contract.py        единый формат и capabilities
src/radar_pipeline.py        canonical gridding и quality masks
src/radar_catalog.py         SQLite-каталог
src/dwd_source.py            DWD ODIM HDF5 adapter
src/open_sources.py          WIS2/Meteoinfo/RainViewer
src/make_dataset.py          quality-aware sequences и split groups
src/datasets.py              effective mask для обучения
src/event_catalog.py         классы ситуаций и балансировка
src/forecast_quality.py      baselines и метрики
src/phys_evolution.py        перенос + рост/распад
src/model_runtime.py         общий inference runtime
src/export_utils.py          NetCDF + quality masks
src/jobs.py                  очередь заданий
src/web_app.py               API
src/static/                  интерфейс
```

## Ограничения

- Выход — отражаемость `dBZ`, а не QPE.
- `MRL-PhysEvolution` — physics-guided nowcasting model, не полная атмосферная PINN.
- Прогноз инициации новой конвекции без спутника, молний и NWP ограничен.
- Горизонты более 60 минут по одной отражаемости имеют пониженную достоверность.
- Визуальные тайлы не используются для количественного обучения.
- DWD-продукт и каждая новая станция должны пройти decode/QC-проверку.
- Модели и датасеты с разными grid, cadence или pipeline version не смешиваются.
- Текущая forecast quality geometry консервативно повторяет маски последнего наблюдаемого срока.
- ONNX/ONNX Runtime и квантизация ещё не включены в рабочий контур.

## Лицензия

MIT License
