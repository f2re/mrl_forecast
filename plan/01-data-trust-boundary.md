# Этап 01. Контракт времени 15 минут

## Статус

**Частично реализовано в коде.** Выполнен базовый инфраструктурный переход на 15-минутный контракт:

- добавлен `src/config.py`;
- введен `FORECAST_STEP_MINUTES = 15`;
- `RadarPipelineConfig` переведен на 15 минут;
- `PIPELINE_VERSION` обновлен до `radar-grid-v2-15min`;
- demo-последовательности строятся с 15-минутным шагом;
- NetCDF export по умолчанию пишет 15-минутные lead times;
- визуализация использует 15-минутный interval по умолчанию;
- training metadata и checkpoint metadata сохраняют `forecast_step_minutes` и `horizon_minutes`;
- Flask API `/api/predict` возвращает `forecast_step_minutes`, `lead_times_minutes`, `horizon_minutes`, `confidence_by_lead`, `product` и warnings;
- README обновлен под экспериментальный 15-минутный reflectivity-nowcast;
- добавлены/обновлены тесты временного контракта.

Остаток этапа: пересобрать реальные датасеты под `radar-grid-v2-15min`, убедиться, что старые модели/датасеты `radar-grid-v1` не используются как совместимые, и вернуть безопасный background job runner вместо прямого запуска задач из Flask.

---

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

1. В `RadarPipelineConfig` есть поле `time_step_minutes`, теперь default должен быть `15`.
2. В `map_visualization.generate_sequence_plots()` есть аргумент `interval_minutes`, теперь default должен быть `15`.
3. В `export_utils.save_forecast_to_netcdf()` есть аргумент `interval_minutes`, теперь default должен быть `15`.
4. В `web_app.py` forecast labels и `LAST_FORECAST` должны использовать `FORECAST_STEP_MINUTES`.
5. В тестах есть проверки 15-минутного контракта.

---

## 2. Что нужно сделать

### 2.1. Единый конфиг

Файл `src/config.py` должен содержать центральные константы:

```python
FORECAST_STEP_MINUTES = 15
DEFAULT_INPUT_LENGTH = 4
DEFAULT_TARGET_LENGTH = 4
MAX_DBZ = 70.0
MIN_DBZ = 0.0
```

### 2.2. Pipeline

В `src/radar_pipeline.py`:

```python
PIPELINE_VERSION = "radar-grid-v2-15min"
time_step_minutes: int = FORECAST_STEP_MINUTES
```

Старые `radar-grid-v1` модели и датасеты не смешивать с новыми.

### 2.3. Dataset

В `src/make_dataset.py`:

1. `regular_frame_segments()` должен использовать `step_minutes=15` из pipeline metadata.
2. Если исходные кадры имеют шаг 5–10 минут, выбрать ближайшие к 15-минутной сетке.
3. В metadata хранить `time_step_minutes=15` и resampling policy.

### 2.4. Training

В `src/train_nowcasting_model.py`:

1. В metadata модели записывать `forecast_step_minutes=15`.
2. В checkpoint записывать `forecast_step_minutes=15`.
3. В metadata хранить `horizon_minutes = target_length * 15`.

### 2.5. Web/API

В `src/web_app.py`:

1. Вызов `generate_sequence_plots()` должен передавать `interval_minutes=FORECAST_STEP_MINUTES`.
2. Forecast labels должны строиться через `FORECAST_STEP_MINUTES`.
3. `LAST_FORECAST` должен хранить `forecast_step_minutes`.
4. API `/api/predict` должен вернуть `forecast_step_minutes`, `horizon_minutes`, `lead_times_minutes`, `product`, `warnings`.

### 2.6. Visualization

В `src/map_visualization.py`:

1. Default `interval_minutes` должен быть `FORECAST_STEP_MINUTES`.
2. Подписи должны соответствовать `T+15`, `T+30`, `T+45`, `T+60`.

### 2.7. NetCDF export

В `src/export_utils.py`:

1. Default `interval_minutes` должен быть `FORECAST_STEP_MINUTES`.
2. Координата `lead_time_minutes` должна содержать `[15, 30, ...]`.
3. Global attrs должны содержать `product`, `forecast_step_minutes`, `not_official_warning`.

---

## 3. Тесты

Минимальные проверки:

1. `RadarPipelineConfig().time_step_minutes == 15`.
2. `generate_sequence_plots(..., interval_minutes=15)` работает.
3. `save_forecast_to_netcdf(..., interval_minutes=15)` пишет `lead_time_minutes = [15, 30, ...]`.
4. `/api/predict` возвращает `forecast_step_minutes = 15`.
5. Старые hardcoded `10` не используются как forecast step.

---

## 4. Критические замечания

1. Нельзя просто заменить подписи времени. Старые датасеты с 10-минутной дискретностью становятся legacy.
2. После изменения `PIPELINE_VERSION` старые модели не должны загружаться как совместимые.
3. Если после ресемплинга мало последовательностей, модель на 2–3 часа не обучать до накопления архива.

---

## 5. Критерий завершения

Этап завершен, если:

1. Во всех новых metadata указан `time_step_minutes=15`.
2. API, PNG labels и NetCDF показывают сроки +15/+30/+45/+60.
3. Старые 10-минутные модели/датасеты не считаются совместимыми с новым pipeline.
4. Есть тесты на 15-минутный временной контракт.
5. Первый прогноз на 1 час формально соответствует целевой постановке по времени.

---

## 6. Известный технический долг после первой реализации

1. Web background tasks временно отключены в Flask API до внедрения безопасного job runner. Скрипты `scripts/*.sh` остаются доступными вручную.
2. Реальные датасеты надо пересобрать; старые `radar-grid-v1` нельзя считать совместимыми.
3. Следующий обязательный этап — `valid_mask` в dataset/loss/metrics, см. [02-unified-radar-pipeline.md](02-unified-radar-pipeline.md).