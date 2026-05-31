# Этап 01. Контракт времени 15 минут

## Цель

Перевести весь проект с исторически сложившегося 10-минутного шага на явный 15-минутный контракт прогноза.

Целевая постановка:

```text
вход: последние кадры отражаемости МРЛ
выход: прогноз отражаемости на +15, +30, +45, +60 ... минут
основной горизонт: 1 час
расширенные горизонты: 2 и 3 часа с пониженной/экспериментальной достоверностью
```

---

## 1. Что уже реализовано

1. В `RadarPipelineConfig` есть поле `time_step_minutes`, но сейчас default равен `10`.
2. В `map_visualization.generate_sequence_plots()` есть аргумент `interval_minutes`, но default равен `10`.
3. В `export_utils.save_forecast_to_netcdf()` есть аргумент `interval_minutes`, но default равен `10`.
4. В `web_app.py` forecast labels и `LAST_FORECAST` формируются с предположением о 10 минутах.
5. В тестах есть проверки, ожидающие `time_step_minutes == 10`.

---

## 2. Что нужно сделать

### 2.1. Добавить единый конфиг

Создать файл `src/config.py`:

```python
"""Central project constants for radar nowcasting."""

FORECAST_STEP_MINUTES = 15
DEFAULT_INPUT_LENGTH = 4
DEFAULT_TARGET_LENGTH = 4
MAX_DBZ = 70.0
MIN_DBZ = 0.0

SUPPORTED_TARGET_LENGTHS = {
    "1h": 4,
    "2h": 8,
    "3h": 12,
}
```

### 2.2. Обновить pipeline

В `src/radar_pipeline.py`:

1. Импортировать `FORECAST_STEP_MINUTES`.
2. Изменить default:

```python
time_step_minutes: int = FORECAST_STEP_MINUTES
```

3. Обновить `PIPELINE_VERSION`.

Рекомендуемый вариант:

```python
PIPELINE_VERSION = "radar-grid-v2-15min"
```

Если нужно сохранить совместимость со старыми моделями, в loader явно маркировать `radar-grid-v1` как legacy и не смешивать с новыми датасетами.

### 2.3. Обновить подготовку датасета

В `src/make_dataset.py`:

1. `regular_frame_segments()` должен использовать `step_minutes=15` из pipeline metadata.
2. Добавить параметр CLI:

```bash
--time-step-minutes 15
```

или жестко брать значение из `RadarPipelineConfig`.

3. Если исходные кадры имеют шаг 5–10 минут, выбрать ближайшие к 15-минутной сетке.
4. Добавить в metadata:

```json
{
  "time_step_minutes": 15,
  "resampling_policy": "nearest_with_tolerance",
  "resampling_tolerance_minutes": 4
}
```

### 2.4. Обновить обучение

В `src/train_nowcasting_model.py`:

1. В metadata модели записывать:

```json
"forecast_step_minutes": 15
```

2. Для target length использовать смысловые горизонты:

```text
1h -> 4 шага
2h -> 8 шагов
3h -> 12 шагов
```

3. В CLI добавить `--horizon {1h,2h,3h}` или оставить `--target-length`, но в UI показывать часы и минуты, а не абстрактные шаги.

### 2.5. Обновить web/API

В `src/web_app.py`:

1. Вызов `generate_sequence_plots()` должен передавать `interval_minutes=FORECAST_STEP_MINUTES`.
2. Forecast labels должны строиться через `FORECAST_STEP_MINUTES`, а не через `10`.
3. `LAST_FORECAST` должен хранить:

```python
"forecast_step_minutes": FORECAST_STEP_MINUTES
```

4. API `/api/predict` должен вернуть:

```json
{
  "forecast_step_minutes": 15,
  "horizon_minutes": 60,
  "base_time": "...",
  "lead_times_minutes": [15, 30, 45, 60]
}
```

### 2.6. Обновить визуализацию

В `src/map_visualization.py`:

1. Default `interval_minutes` заменить на `FORECAST_STEP_MINUTES`.
2. Подписи должны быть:

```text
История T-45 мин, T-30 мин, T-15 мин, Сейчас
Прогноз ИИ T+15 мин, T+30 мин, T+45 мин, T+60 мин
```

### 2.7. Обновить NetCDF export

В `src/export_utils.py`:

1. Default `interval_minutes` заменить на `FORECAST_STEP_MINUTES`.
2. Координату `lead_time` переименовать или продублировать как `lead_time_minutes`.
3. Добавить global attrs:

```json
{
  "product": "experimental_radar_reflectivity_nowcast",
  "forecast_step_minutes": 15,
  "not_official_warning": "true"
}
```

---

## 3. Тесты

Добавить `tests/test_time_contract.py`.

Минимальные проверки:

1. `RadarPipelineConfig().time_step_minutes == 15`.
2. `generate_sequence_plots(..., interval_minutes=15)` формирует корректные lead labels.
3. `save_forecast_to_netcdf(..., interval_minutes=15)` пишет `lead_time_minutes = [15, 30, ...]`.
4. `/api/predict` возвращает `forecast_step_minutes = 15`.
5. В коде не осталось критичных hardcoded `10` для forecast step, кроме legacy/test comments.

---

## 4. Критические замечания

1. Нельзя просто заменить подписи времени. Старые датасеты с 10-минутной дискретностью становятся legacy.
2. После изменения `PIPELINE_VERSION` старые модели не должны загружаться как совместимые.
3. Если исходные радары имеют шаг около 8–10 минут, надо явно решить: ресемплинг к 15 минутам или обучение на фактическом шаге. Для текущей задачи выбран 15-минутный контракт.
4. Если после ресемплинга мало последовательностей, модель на 2–3 часа не обучать до накопления архива.

---

## 5. Критерий завершения

Этап завершен, если:

1. Во всех новых metadata указан `time_step_minutes=15`.
2. API, PNG labels и NetCDF показывают сроки +15/+30/+45/+60.
3. Старые 10-минутные модели/датасеты не считаются совместимыми с новым pipeline.
4. Есть тесты на 15-минутный временной контракт.
5. Первый прогноз на 1 час формально соответствует целевой постановке по времени.