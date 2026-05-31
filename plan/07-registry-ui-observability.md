# Этап 07. Registry, API, UI и доверие прогноза

## Цель

Сделать выдачу прогноза честной для пользователя: каждый результат должен иметь источник, базовый срок, lead time, версию pipeline, версию модели, статус качества и уровень доверия.

UI не должен показывать экспериментальную карту так, будто это официальный прогноз осадков или штормовое предупреждение.

---

## 1. Что уже реализовано

1. Есть model registry в `models/registry`.
2. Есть metadata модели.
3. Есть статусы `training`, `completed`, `rejected_quality_gate`, `failed`.
4. `web_app.is_model_usable()` не должен пропускать незавершенные модели.
5. `/api/model/details/<model_id>` возвращает metadata и learning curve.
6. `/api/predict` возвращает изображения history/forecast и diagnostics.
7. Uniform-field forecast отклоняется с HTTP 422.
8. Demo mode явно возвращает `source_status=demo`.

---

## 2. Проблема

Сейчас UI/API не полностью выражают метеорологическую достоверность:

1. 10-минутная временная логика должна быть заменена на 15 минут.
2. Нет явного `horizon_minutes`.
3. Нет `lead_times_minutes`.
4. Нет confidence labels для 1/2/3 часов.
5. Недостаточно явно указано, что прогноз — это поле отражаемости, а не официальный прогноз осадков.
6. Не показывается, какие baseline-метрики прошла модель.
7. Пользователь не видит ограничения: один канал отражаемости, без ветра/температуры/влажности/NWP.

---

## 3. API contract

### 3.1. `/api/predict`

Ответ должен содержать:

```json
{
  "product": "experimental_radar_reflectivity_nowcast",
  "units": "dBZ",
  "base_time_utc": "2026-05-31T12:00:00Z",
  "forecast_step_minutes": 15,
  "horizon_minutes": 60,
  "lead_times_minutes": [15, 30, 45, 60],
  "pipeline_version": "radar-grid-v2-15min",
  "model_id": "model_...",
  "model_architecture": "convlstm_baseline|mrl_physlite",
  "source_status": "observed|demo|...",
  "confidence_by_lead": ["normal_experimental", "normal_experimental", "normal_experimental", "normal_experimental"],
  "diagnostics": {},
  "warnings": []
}
```

### 3.2. Confidence policy

Для любых моделей по одной отражаемости:

```text
0–60 мин: normal_experimental
60–120 мин: reduced
120–180 мин: experimental_low_confidence
>180 мин: unsupported
```

### 3.3. Warnings

Добавлять предупреждения:

```text
- source_is_demo
- model_is_legacy
- pipeline_is_legacy
- low_valid_fraction
- forecast_rejected_uniform_field
- long_horizon_low_confidence
- reflectivity_only_no_nwp
- not_official_warning
```

---

## 4. UI changes

### 4.1. Название продукта

Заменить:

```text
Прогноз осадков
```

на:

```text
Экспериментальный прогноз отражаемости МРЛ
```

### 4.2. Карточка прогноза

Показывать:

```text
Базовый срок: 31.05.2026 12:00 UTC
Шаг прогноза: 15 мин
Горизонт: 60 мин
Модель: model_x / MRL-PhysLite
Pipeline: radar-grid-v2-15min
Источник: observed/local/aws/demo
Статус: экспериментальный продукт, не официальное предупреждение
```

### 4.3. Lead time labels

Каждый кадр:

```text
T+15 мин
T+30 мин
T+45 мин
T+60 мин
```

Для длинных сроков:

```text
T+75 мин — пониженная достоверность
T+135 мин — экспериментальный сценарий
```

### 4.4. Model card in UI

Для выбранной модели показывать:

```text
architecture
training datasets
pipeline_version
time_step_minutes
input_length
target_length
horizon_minutes
quality_gate_status
baseline comparison
known limitations
```

### 4.5. Source panel

Показывать:

```text
station
source
last_observation_time
age_minutes
valid_fraction
cadence_status
mask coverage
```

---

## 5. Registry rules

Модель доступна для инференса только если:

1. `status` в `completed` или `published`.
2. Есть `best_model.pt`.
3. `pipeline_version` совместим с текущим pipeline.
4. `forecast_step_minutes` совпадает с текущим контрактом.
5. `quality_gate_passed = true`.
6. Есть baseline metrics.

Модель с `radar-grid-v1` после перехода на `radar-grid-v2-15min` должна быть помечена как legacy.

---

## 6. Файлы

Изменить:

```text
src/web_app.py
src/metadata_utils.py
src/map_visualization.py
templates/index.html
src/export_utils.py
```

Добавить при необходимости:

```text
src/confidence.py
src/model_card.py
```

---

## 7. Тесты

Создать/обновить:

```text
tests/test_web_app.py
tests/test_registry.py
tests/test_confidence.py
tests/test_time_contract.py
```

Проверки:

1. `/api/predict` возвращает `forecast_step_minutes=15`.
2. `/api/predict` возвращает `lead_times_minutes=[15,30,45,60]` для 1h.
3. Demo forecast содержит warning `source_is_demo`.
4. 2h/3h forecast содержит reduced/experimental confidence labels.
5. Legacy model нельзя выбрать как compatible production model.
6. Rejected model не отображается как usable.
7. UI labels не используют формулировку `официальный прогноз`.

---

## 8. Критические замечания

1. Хорошая картинка без статуса качества — эксплуатационный риск.
2. Нельзя скрывать, что вход — только отражаемость.
3. Нельзя смешивать forecast horizon и target_length в UI. Пользователь должен видеть минуты.
4. Нельзя использовать demo/source_failed/missing как observed.

---

## 9. Критерий завершения

Этап завершен, если:

1. API возвращает полный metadata contract прогноза.
2. UI показывает base time, lead time, model, source, pipeline, confidence.
3. 2–3-часовые прогнозы имеют сниженный статус доверия.
4. Legacy/rejected/training models нельзя выбрать как operational.
5. Пользователь видит, что продукт экспериментальный и не является официальным предупреждением.