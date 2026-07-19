# 🌦️ MRL Forecast Pro: экспериментальный прогноз отражаемости МРЛ

## Статус реализации

Проект находится в активной модернизации. Текущий рабочий контур использует 15-минутный baseline `radar-grid-v2-15min`, одновременно формируется новый единый слой данных для сетки `512 x 512` с разрешением `1 км`.

Результат трактуется строго как **экспериментальный прогноз поля радиолокационной отражаемости и зон радиоэха**, а не как официальный прогноз интенсивности осадков или предупреждение об опасных явлениях.

Синтетические данные доступны только через отдельный `DEMO`-режим и не должны использоваться для production-обучения.

Основные документы:

- [`plan/README.md`](plan/README.md) — текущее состояние и ближайшие задачи;
- [`plan/10-implementation-roadmap.md`](plan/10-implementation-roadmap.md) — практический план реализации.

## Реализовано

- **ConvLSTM baseline** с temporal split, persistence/advection comparison и quality gate.
- **Час истории** в текущем профиле: четыре входных срока с шагом 15 минут.
- **Masked dataset format**: новые последовательности `.npz` содержат `reflectivity`, `valid_mask` и `timestamps_utc`.
- **Masked training/evaluation**: невалидные пиксели исключаются из loss и пороговых метрик.
- **Canonical radar contract**: целевая сетка `512 x 512`, `1 км`, локальная AEQD, явные маски покрытия и помех.
- **Source capabilities и registry**: источник явно сообщает, пригоден ли он для количественного обучения или только для визуализации.
- **Открытые источники**:
  - NOAA AWS — количественный референсный источник;
  - WIS2 — поиск открытых радарных наборов;
  - Meteoinfo и RainViewer — визуальные источники, не используемые как количественный `dBZ`.
- **NetCDF export** с CRS, `lead_time_minutes`, `valid_time_utc`, provenance и `not_official_warning=true`.

## Важное ограничение по российским ДМРЛ

В проекте пока нет подтверждённого открытого долговременного архива сырых российских ДМРЛ-BUFR. WIS2 подключён как discovery-контур, Meteoinfo и RainViewer — как визуальный контроль. Любой российский источник получит статус `training_allowed` только после проверки реального файла, единиц отражаемости, времени наблюдения, геометрии, маски покрытия и условий использования.

## Структура проекта

```text
mrl_forecast/
├── plan/
├── scripts/
│   ├── setup.sh
│   ├── download.sh
│   ├── prepare.sh
│   ├── train.sh
│   ├── doctor.py
│   ├── check_aws_source.py
│   └── check_open_radar_sources.py
├── src/
│   ├── config.py
│   ├── radar_pipeline.py
│   ├── radar_contract.py
│   ├── source_registry.py
│   ├── open_sources.py
│   ├── datasets.py
│   ├── losses.py
│   ├── make_dataset.py
│   ├── train_nowcasting_model.py
│   ├── forecast_quality.py
│   ├── web_app.py
│   └── ...
├── templates/
├── tests/
├── data/
└── models/
```

## Быстрый старт

```bash
bash scripts/setup.sh
bash scripts/run_app.sh
```

Веб-интерфейс: `http://localhost:5005`.

Проверка окружения:

```bash
python scripts/doctor.py
```

Проверка NOAA AWS:

```bash
python scripts/check_aws_source.py --station KOKX --date 2024-05-20 --decode-one
```

Проверка открытых discovery/visual источников:

```bash
python scripts/check_open_radar_sources.py --source all
python scripts/check_open_radar_sources.py --source wis2 --limit 100
```

## Терминальный цикл baseline

### 1. Скачивание архива

```bash
bash scripts/download.sh KOKX 2024-05-20 100
```

Данные сохраняются в `data/raw/archive/<ID_СЕССИИ>`.

### 2. Создание masked dataset

```bash
bash scripts/prepare.sh 8 data/raw/archive/<ID_СЕССИИ>
```

Новые последовательности сохраняются в `.npz`. Неизвестный срок наблюдения не подменяется файловым `mtime`.

### 3. Обучение baseline-модели

```bash
bash scripts/train.sh 20 4 1e-4 data/processed_archive/<ID_ДАТАСЕТА>
```

Чекпоинты и метаданные сохраняются в `models/registry/`. Текущая ConvLSTM остаётся контрольной моделью; целевая архитектура `MRL-PhysEvolution` будет отдельно прогнозировать перенос, рост, распад и неопределённость радиоэха.

## Ограничения

- Выход модели — отражаемость в `dBZ`, а не измеренная интенсивность осадков.
- Проект не является официальным источником штормовых предупреждений.
- Модели и датасеты без совместимого `pipeline_version` считаются legacy.
- Модель со статусом `training`, `failed` или `rejected_quality_gate` нельзя выбирать для operational-инференса.
- Горизонты более 60 минут по одной отражаемости имеют пониженную экспериментальную достоверность.
- Визуальные растровые источники не должны использоваться для количественного обучения без доказанной шкалы и provenance.

## Лицензия

MIT License
