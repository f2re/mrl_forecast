# 🌦️ MRL Forecast Pro: экспериментальный прогноз отражаемости МРЛ

## Статус реализации

Проект находится в активной модернизации. Новый контур разработки переводит систему на 15-минутный временной контракт `radar-grid-v2-15min` и трактует результат строго как **экспериментальный прогноз поля радиолокационной отражаемости**, а не как официальный прогноз интенсивности осадков или предупреждение об опасных явлениях.

Синтетические данные доступны только через отдельный `DEMO`-режим и не должны использоваться для production-обучения. Модели и датасеты со старым `radar-grid-v1` считаются legacy после перехода на 15-минутный контракт.

Основной план и ограничения: [`plan/README.md`](plan/README.md).

## Ключевые возможности

- **Нейросетевой baseline**: ConvLSTM с temporal split, сравнением с persistence/advection baseline и quality gate.
- **15-минутный контракт прогноза**: целевые сроки прогноза `+15`, `+30`, `+45`, `+60` минут и далее.
- **Версионированный pipeline**: текущий контракт `radar-grid-v2-15min`, продукт `lowest_elevation_reflectivity`, единицы `dBZ`.
- **Экспорт**: NetCDF с CRS, `lead_time_minutes`, `valid_time_utc`, provenance и явным статусом `not_official_warning=true`.
- **Data lifecycle**: архив, подготовка датасета, обучение, реестр моделей, инференс, экспорт.
- **DEMO-режим**: явно маркированные синтетические кадры только для проверки UI.

## Структура проекта

```text
mrl_forecast/
├── plan/                      # Актуальный поэтапный план разработки
├── scripts/                   # Скрипты автоматизации
│   ├── setup.sh
│   ├── download.sh
│   ├── prepare.sh
│   ├── train.sh
│   ├── doctor.py
│   └── check_aws_source.py
├── src/
│   ├── config.py              # Центральный 15-минутный контракт
│   ├── radar_pipeline.py      # Версионированный radar-grid pipeline
│   ├── web_app.py             # Flask backend
│   ├── train_nowcasting_model.py
│   ├── forecast_quality.py
│   ├── export_utils.py
│   ├── make_dataset.py
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

Откройте браузер: `http://localhost:5005`.

Проверка окружения:

```bash
python scripts/doctor.py
```

Опциональная AWS-проверка:

```bash
python scripts/check_aws_source.py --station KOKX --date 2024-05-20 --decode-one
```

## Терминальный цикл работы

### 1. Скачивание архива

```bash
bash scripts/download.sh KOKX 2024-05-20 100
```

Данные сохраняются в `data/raw/archive/<ID_СЕССИИ>`.

### 2. Создание датасета

```bash
bash scripts/prepare.sh 8 data/raw/archive/<ID_СЕССИИ>
```

После перехода на `radar-grid-v2-15min` старые 10-минутные датасеты не следует смешивать с новыми.

### 3. Обучение baseline-модели

```bash
bash scripts/train.sh 20 4 1e-4 data/processed_archive/<ID_ДАТАСЕТА>
```

Чекпоинты и метаданные сохраняются в `models/registry/`.

## Ограничения

- Выход модели — отражаемость в `dBZ`, а не измеренная интенсивность осадков.
- Проект не является официальным источником штормовых предупреждений.
- Модели и датасеты без совместимого `pipeline_version` считаются legacy.
- Модель со статусом `training`, `failed` или `rejected_quality_gate` нельзя выбирать для operational-инференса.
- Горизонты 2–3 часа по одной отражаемости должны маркироваться как пониженная/экспериментальная достоверность.
- Поддержка российских МРЛ/ДМРЛ остается отдельным контуром до получения реальных fixtures и подтверждения источника данных.

## Лицензия

MIT License
