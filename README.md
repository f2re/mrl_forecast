# MRL Forecast Pro

Экспериментальная система прогноза поля радиолокационной отражаемости и зон радиоэха. Результат не является официальным прогнозом интенсивности осадков или предупреждением об опасном явлении.

Документы:

- [`plan/10-implementation-roadmap.md`](plan/10-implementation-roadmap.md) — текущий статус этапов;
- [`docs/source-access.md`](docs/source-access.md) — источники, проверки доступа, токены и регистрация.

## Текущий рабочий контур

- canonical grid `512 × 512`, разрешение около `1 км`, локальная AEQD;
- исходные файлы без изменения, SHA-256 и provenance;
- NOAA NEXRAD Level II и DWD ODIM HDF5;
- OPERA ORD, FMI S3, DMI STAC, KNMI API и WIS2 Global Cache;
- отдельные профили доступа для Météo-France, CEDA, MeteoSwiss, GeoSphere, AURA, NASA и ЦАО NCRadar;
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

## Быстрый старт

```bash
bash scripts/setup.sh
bash scripts/run_app.sh
```

Интерфейс: `http://localhost:5005`.

При запуске `scripts/run_app.sh` в фоне выполняется проверка полностью автоматизированных источников. Для каждого источника проверяются:

1. доступность API или S3-каталога;
2. возможность получить список файлов;
3. чтение первых 1024 байт тестового файла;
4. наличие обязательного API key.

Обезличенный отчёт сохраняется в `src/static/source_health.json` и отображается на странице «Источники». Токены и query-параметры подписанных URL в отчёт не попадают.

## Единый CLI

```bash
python mrl.py --help
```

Основные команды:

```text
doctor
sources
download
prepare
train
infer
catalog
benchmark
serve
worker
```

## Международные источники

### Список адаптеров

```bash
python mrl.py sources --action list
```

### Автопроверка каталога и скачивания

```bash
python mrl.py sources \
  --action probe \
  --source all \
  --active-only \
  --download-test \
  --limit 1
```

Отчёт по умолчанию: `data/source_health.json`.

Проверка одного источника:

```bash
python mrl.py sources \
  --action probe \
  --source dmi-radar \
  --station 06194 \
  --collection volume \
  --download-test
```

### Порядок регистрации

```bash
python mrl.py sources --action info --source knmi-radar
python mrl.py sources --action info --source opera-ord
python mrl.py sources --action info --source meteofrance-radar
python mrl.py sources --action info --source ncradar-cao
```

### Безопасный ввод API key

```bash
python mrl.py sources --action configure --source knmi-radar
```

Ключ вводится скрыто. По умолчанию он сохраняется в:

```text
~/.config/mrl_forecast/credentials.json
```

Файл получает права `0600`. Переменные окружения имеют приоритет над файлом:

```text
KNMI_API_KEY
METEOGATE_API_KEY
METEOFRANCE_API_TOKEN
EARTHDATA_TOKEN
```

Путь к хранилищу можно переопределить:

```bash
export MRL_CREDENTIALS_FILE=/secure/path/mrl_credentials.json
```

## Скачивание исходных данных

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

### FMI open S3

```bash
python mrl.py download \
  --source fmi \
  --prefix '<prefix>' \
  --count 100
```

### OPERA ORD 24-hour cache

```bash
python mrl.py download \
  --source opera \
  --prefix '<country/station/prefix>' \
  --count 100
```

### DMI radar API

```bash
python mrl.py download \
  --source dmi \
  --station 06194 \
  --collection volume \
  --date 2026-07-19 \
  --count 100
```

### KNMI Open Data API

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
  --prefix '<verified-WIS2-prefix>' \
  --count 20
```

Новые downloader-адаптеры создают стандартный `raw_data` archive:

- исходные файлы без изменения;
- URL и remote file ID;
- timestamp UTC;
- формат и access profile;
- размер и SHA-256;
- ошибки отдельных файлов;
- запись в SQLite-каталоге.

## Quality-маски

Формат `npz-radar-quality-v2` содержит:

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
valid_mask AND coverage_mask AND NOT clutter_mask AND interpolation_weight > 0
```

Маски проходят через:

```text
adapter → RadarFrame → dataset → loss/metrics → runtime → UI → NetCDF
```

Если источник не предоставляет количественный interpolation weight, используется явно маркированный бинарный fallback по валидности.

## Подготовка датасета

NOAA-профиль `4 + 4` по 15 минут:

```bash
python mrl.py prepare \
  --archive-dir data/raw/archive/<NOAA_SESSION> \
  --seq-len 8 \
  --grid-profile canonical \
  --time-step-minutes 15
```

Целевой профиль `6 + 6` по 10 минут:

```bash
python mrl.py prepare \
  --archive-dir data/raw/archive/<ODIM_SESSION> \
  --seq-len 12 \
  --grid-profile canonical \
  --time-step-minutes 10
```

Зарегистрированные ODIM HDF5 архивы используют один Py-ART decoder и общий canonical gridding. Timestamp берётся из download metadata; файловый `mtime` не принимается.

Новые международные источники по умолчанию не получают автоматический допуск к обучению. Перед использованием требуется проверка полей, единиц, станции, геометрии, quality flags и лицензии.

## Обучение

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

Лучшая эпоха выбирается по validation. Окончательный quality gate выполняется на независимых test-блоках, если архив содержит достаточно временных групп. Для небольших и legacy-наборов metadata явно указывает `validation_fallback`.

Модель должна превзойти:

- persistence;
- global shift advection;
- local block motion;

и не иметь uniform-field anomaly.

## Инференс и CPU benchmark

```bash
python mrl.py infer \
  --model-path models/registry/<MODEL_ID> \
  --source aws \
  --station KOKX
```

```bash
python mrl.py benchmark \
  --model-path models/registry/<MODEL_ID> \
  --threads 8 \
  --warmup 1 \
  --repeats 5 \
  --save
```

## Российские ДМРЛ

Подтверждённого открытого долговременного архива сырых российских ДМРЛ-BUFR пока нет. WIS2 используется для discovery, а Meteoinfo и RainViewer остаются visual-only.

`ncradar-cao` хранит порядок официального запроса в ЦАО. Российский источник получит `training_allowed` только после проверки:

- реального файла или открытого endpoint;
- единиц отражаемости;
- времени наблюдения;
- геометрии и координат;
- масок покрытия и QC;
- условий использования.

## Ограничения

- Выход — отражаемость `dBZ`, а не QPE.
- `MRL-PhysEvolution` — physics-guided nowcasting model, не полная атмосферная PINN.
- Прогноз инициации новой конвекции без спутника, молний и NWP ограничен.
- Горизонты более 60 минут по одной отражаемости имеют пониженную достоверность.
- Доступность источника не равна научной пригодности данных.
- Визуальные тайлы не используются для количественного обучения.
- ONNX/ONNX Runtime и квантизация ещё не включены в рабочий контур.

## Лицензия

MIT License
