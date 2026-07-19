# Доступ к международным источникам МРЛ

## Назначение

Модуль источников решает четыре разные задачи и не смешивает их:

1. описание семантики и условий доступа;
2. проверка доступности каталога;
3. тестовое чтение первых 1024 байт файла;
4. полное скачивание с SHA-256 и provenance.

Наличие доступного URL не означает автоматического допуска данных к обучению. Новый источник получает `training_allowed=true` только после проверки единиц, полей, геометрии, сроков, quality flags, лицензии и нескольких реальных файлов.

## Безопасность токенов

Секреты читаются в следующем порядке:

1. переменная окружения;
2. пользовательский файл `~/.config/mrl_forecast/credentials.json`;
3. интерактивный скрытый ввод через `getpass`.

Файл создаётся с правами `0600`. Значения токенов не выводятся в CLI, API, health-report или интерфейс.

Путь можно переопределить:

```bash
export MRL_CREDENTIALS_FILE=/secure/path/mrl_credentials.json
```

## Реализованные автоматические адаптеры

| Source ID | Данные | Доступ | Probe | Скачать | Статус для train |
| --- | --- | --- | --- | --- | --- |
| `noaa-aws` | NEXRAD Level II | открытый AWS S3 | отдельный NOAA check | да | разрешён после текущего QC |
| `dwd-open-data` | DBZH ODIM HDF5 sweep | прямой HTTP | да | да | разрешён для проверенного продукта |
| `fmi-s3` | однорадарные ODIM 2.3 HDF5 volumes | открытый S3 | да | да | требует отдельного QC |
| `opera-ord` | ODIM HDF5/BUFR, single-site и composite | анонимный API/S3, ключ опционален | да | да | требует per-file QC/licence |
| `dmi-radar` | ODIM HDF5 `volume`, `pseudoCappi`, `composite` | открытый STAC API | да | да | требует отдельного QC |
| `knmi-radar` | HDF5 polarimetric volumes/archive | обязательный API key | да | да | требует отдельного QC |
| `wis2-cache` | BUFR/GRIB/NetCDF WIS2 core data | открытый 24-hour S3 cache | да | да | discovery до проверки набора |

ODIM HDF5 архивы от DWD/FMI/DMI/KNMI сохраняют исходный source ID и могут быть переданы в общий Py-ART canonical pipeline. Для OPERA конкретный файл сначала должен быть классифицирован как ODIM HDF5 или BUFR.

## Источники с probe-only или ручным доступом

В реестре присутствуют:

- `wmo-radar-db` — глобальный каталог радаров, без файлов наблюдений;
- `meteofrance-radar` — учётная запись и подписка на Package Radar;
- `ceda-nimrod` — CEDA account и dataset permission;
- `meteoswiss-radar` — открытые STAC/ODIM продукты, downloader ещё не реализован;
- `geosphere-radar` — открытый короткий архив, Data Hub downloader ещё не реализован;
- `aura-nci` — NCI account и project membership;
- `metservice-radar` — SFTP после запроса;
- `taiwan-qpesums` — API/grid adapter запланирован;
- `nasa-gpm-gv` — Earthdata account и collection access;
- `ncradar-cao` — официальный запрос в ЦАО, публичный anonymous endpoint не подтверждён.

Для таких источников probe подтверждает доступность landing/registration page, но возвращает `can_download=false`.

## Команды диагностики

Список источников и условий:

```bash
python mrl.py sources --action list
```

Информация и порядок регистрации:

```bash
python mrl.py sources --action info --source knmi-radar
python mrl.py sources --action info --source ncradar-cao
```

Проверка всех полностью автоматизированных адаптеров:

```bash
python mrl.py sources \
  --action probe \
  --source all \
  --active-only \
  --download-test \
  --limit 1
```

Результат сохраняется в `data/source_health.json`. При запуске через `scripts/run_app.sh` такой probe выполняется автоматически, а обезличенный отчёт доступен на странице «Источники».

Проверка одного источника:

```bash
python mrl.py sources \
  --action probe \
  --source dmi-radar \
  --station 06194 \
  --collection volume \
  --download-test
```

Проверка конкретного S3 prefix:

```bash
python mrl.py sources \
  --action probe \
  --source fmi-s3 \
  --prefix '<prefix>' \
  --download-test
```

## Регистрация и ключи

### KNMI

1. Зарегистрировать KNMI Data Platform account; если самостоятельная регистрация недоступна, направить на `opendata@knmi.nl` имя, организацию и цель использования.
2. После входа открыть API Catalog.
3. Выбрать Open Data API и запросить API key.
4. Скопировать ключ при показе.
5. Настроить локально:

```bash
python mrl.py sources --action configure --source knmi-radar
```

Для полной выгрузки dataset требуется отдельный bulk key с указанием dataset name/version.

По умолчанию используется:

```text
dataset_name    radar_volume_full_herwijnen
dataset_version 1.0
```

### OPERA/MeteoGate

Анонимный доступ и 24-hour cache используются без ключа. Ключ нужен для повышенных лимитов или маршрутов, где он явно требуется.

```bash
python mrl.py sources --action configure --source opera-ord
```

Имя HTTP header задаётся только по документации выбранного маршрута:

```bash
export METEOGATE_API_KEY_HEADER='<documented-header-name>'
```

### Météo-France

1. Создать учётную запись в API portal.
2. Подписаться на Données Publiques Paquet Radar.
3. Создать токен.
4. Сохранить:

```bash
python mrl.py sources --action configure --source meteofrance-radar
```

Автоматический Package Radar downloader пока не активирован; статус останется `probe_only`.

### NASA Earthdata

1. Создать Earthdata Login.
2. Принять условия выбранной GPM-GV/GHRC collection.
3. Создать token, если коллекция поддерживает token access.
4. Сохранить:

```bash
python mrl.py sources --action configure --source nasa-gpm-gv
```

### ЦАО NCRadar

Токен не запрашивается, поскольку открытый API не подтверждён. Требуется официальный запрос, включающий:

- radar/WIGOS ID;
- интервал дат;
- полные volumes или elevation scans;
- DBZH, TH, VRADH, WRADH;
- ZDR, PHIDP, KDP, RHOHV при наличии;
- range/azimuth resolution;
- quality masks;
- формат NetCDF/CfRadial, ODIM_H5 или FM 94 BUFR;
- условия использования.

## Полное скачивание

FMI:

```bash
python mrl.py download \
  --source fmi \
  --prefix '<prefix>' \
  --count 100
```

OPERA 24-hour cache:

```bash
python mrl.py download \
  --source opera \
  --prefix '<country/station/prefix>' \
  --count 100
```

DMI:

```bash
python mrl.py download \
  --source dmi \
  --station 06194 \
  --collection volume \
  --date 2026-07-19 \
  --count 100
```

KNMI:

```bash
python mrl.py download \
  --source knmi \
  --dataset-name radar_volume_full_herwijnen \
  --dataset-version 1.0 \
  --count 100
```

WIS2 cache:

```bash
python mrl.py download \
  --source wis2 \
  --prefix '<verified-WIS2-prefix>' \
  --count 20
```

Каждая сессия создаёт `metadata.json`, сохраняет исходные файлы без изменения, записывает SHA-256, URL, remote file ID, срок и access profile, затем индексируется в SQLite-каталоге.

## Переход к обучению

Перед включением нового источника:

1. выполнить `probe --download-test`;
2. скачать несколько файлов;
3. проверить декодирование, поля и timestamps;
4. проверить station coordinates и projection;
5. оценить coverage/clutter/interpolation masks;
6. построить небольшой canonical dataset;
7. сравнить статистику dBZ и spatial spectrum с уже проверенным источником;
8. только после этого изменить `training_allowed`.
