# 🌦️ MRL Forecast Pro: экспериментальный прогноз отражаемости МРЛ

## Статус реализации

Проект содержит рабочий исследовательский контур подготовки данных, обучения и инференса. Текущий baseline `radar-grid-v2-15min` сохранён для совместимости, а новые датасеты по умолчанию строятся в canonical-профиле `radar-grid-v3-1km`: `512 x 512`, разрешение около `1 км`, локальная AEQD.

Результат трактуется строго как **экспериментальный прогноз поля радиолокационной отражаемости и зон радиоэха**, а не как официальный прогноз интенсивности осадков или предупреждение об опасных явлениях.

Синтетические данные доступны только через отдельный `DEMO`-режим и не должны использоваться для production-обучения.

Основные документы:

- [`plan/README.md`](plan/README.md) — текущее состояние и ближайшие задачи;
- [`plan/10-implementation-roadmap.md`](plan/10-implementation-roadmap.md) — практический план реализации.

## Реализовано

- **Единый radar contract**: canonical grid, явные маски валидности, покрытия, помех и интерполяции, provenance.
- **Masked dataset**: `.npz` содержит `reflectivity`, `valid_mask`, `timestamps_utc`; legacy `.npy` читается только для совместимости.
- **Контроль времени**: неизвестный срок наблюдения не подменяется файловым `mtime`.
- **Поиск ситуаций**: `dry_valid`, `weak_echo`, `precipitation`, `convective`, `severe_core`, `invalid`.
- **Балансировка train**: сухие и echo-ситуации выбираются с суммарной вероятностью 50/50; validation остаётся естественной.
- **ConvLSTM baseline**: temporal split, persistence/global-advection comparison и quality gate.
- **MRL-PhysEvolution**: отдельные ветви переноса, роста, распада и неопределённости, differentiable advection и physics-guided loss.
- **Общий model runtime** для CPU/GPU, веб-интерфейса и терминального инференса.
- **Диагностические слои**: отражаемость, перенос, рост, распад и неопределённость.
- **SQLite job runner**: скачивание, подготовка датасета и обучение запускаются из UI или терминала.
- **Адаптивный интерфейс** в визуальном языке macOS/iOS с тёмной темой и мобильной навигацией.
- **Открытые источники**:
  - NOAA AWS — количественный референсный источник;
  - WIS2 — discovery открытых машинно-читаемых наборов;
  - Meteoinfo и RainViewer — визуальные источники, не используемые как количественный `dBZ`.
- **NetCDF export** с CRS, `lead_time_minutes`, `valid_time_utc`, provenance и `not_official_warning=true`.

## Важное ограничение по российским ДМРЛ

В проекте пока нет подтверждённого открытого долговременного архива сырых российских ДМРЛ-BUFR. WIS2 подключён как discovery-контур, Meteoinfo и RainViewer — как визуальный контроль. Российский источник получит статус `training_allowed` только после проверки реального файла, единиц отражаемости, времени наблюдения, геометрии, маски покрытия и условий использования.

## Структура проекта

```text
mrl_forecast/
├── plan/
├── scripts/
│   ├── setup.sh
│   ├── download.sh
│   ├── prepare.sh
│   ├── train.sh
│   ├── run_app.sh
│   ├── job_worker.py
│   ├── doctor.py
│   ├── check_aws_source.py
│   └── check_open_radar_sources.py
├── src/
│   ├── radar_contract.py
│   ├── radar_pipeline.py
│   ├── source_registry.py
│   ├── open_sources.py
│   ├── datasets.py
│   ├── event_catalog.py
│   ├── convlstm.py
│   ├── phys_evolution.py
│   ├── losses.py
│   ├── model_runtime.py
│   ├── jobs.py
│   ├── make_dataset.py
│   ├── train_nowcasting_model.py
│   ├── run_inference.py
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

Проверка окружения и источников:

```bash
python scripts/doctor.py
python scripts/check_aws_source.py --station KOKX --date 2024-05-20 --decode-one
python scripts/check_open_radar_sources.py --source all
python scripts/check_open_radar_sources.py --source wis2 --limit 100
```

## Терминальный цикл

### 1. Скачивание архива

```bash
bash scripts/download.sh KOKX 2024-05-20 100
```

Данные сохраняются в `data/raw/archive/<ID_СЕССИИ>`.

### 2. Создание canonical masked dataset

```bash
bash scripts/prepare.sh 8 data/raw/archive/<ID_СЕССИИ> canonical
```

Восемь сроков соответствуют четырём входным и четырём целевым кадрам в текущем 15-минутном профиле. Для проверенного 10-минутного источника следует использовать последовательность из 12 сроков и параметры обучения `6 + 6`.

### 3. Обучение MRL-PhysEvolution

```bash
bash scripts/train.sh \
  20 \
  1 \
  1e-4 \
  data/processed_archive/<ID_ДАТАСЕТА> \
  0.2 \
  4 \
  phys-evolution \
  4
```

Контрольная ConvLSTM запускается заменой `phys-evolution` на `convlstm`.

### 4. Терминальный инференс

```bash
python src/run_inference.py \
  --model-path models/registry/<ID_МОДЕЛИ> \
  --source aws \
  --station KOKX \
  --output-dir data/predictions
```

Команда использует тот же runtime, grid contract, cadence и архитектуру модели, что и веб-приложение. Для `source=local` поддерживаются доверенные `.npz` с масками и `timestamps_utc`.

## Ограничения

- Выход модели — отражаемость в `dBZ`, а не измеренная интенсивность осадков.
- Проект не является официальным источником штормовых предупреждений.
- `MRL-PhysEvolution` является physics-guided nowcasting model, а не полной атмосферной PINN.
- Модели и датасеты без совместимого `pipeline_version`, шага времени и grid contract не смешиваются.
- Модель со статусом `training`, `failed` или `rejected_quality_gate` нельзя выбирать для рабочего инференса.
- Горизонты более 60 минут по одной отражаемости имеют пониженную экспериментальную достоверность.
- Визуальные растровые источники не используются для количественного обучения без доказанной шкалы и provenance.
- До обучения на российских ДМРЛ необходимы реальные открытые fixtures и отдельная верификация BUFR-дескрипторов.

## Лицензия

MIT License
