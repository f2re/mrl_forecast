# Этап 00. Текущее состояние и долги после аудита

## Назначение

Этот файл фиксирует, что уже реализовано в проекте, что можно считать закрытым, а что осталось техническим и метеорологическим долгом после перехода к задаче: **прогноз поля отражаемости МРЛ на 1–3 часа с шагом 15 минут**.

Старый baseline-аудит по NOAA/KOKX сохраняет историческую ценность, но больше не является главным планом работ. Главный фокус теперь — физическая непротиворечивость, маски качества, 15-минутный временной контракт и новая легкая physics-guided модель.

---

## 1. Уже реализовано

### 1.1. Доверие к источникам

В коде есть структуры:

- `RadarFrame`;
- `RadarSequence`;
- `status`;
- `qc`;
- `provenance`;
- `RadarSourceError`;
- `RadarDecodeError`.

Синтетические кадры вынесены в отдельный `DemoRadarAdapter`. Это правильная граница доверия: demo-данные не должны попадать в production-датасеты и operational-инференс без явной маркировки.

### 1.2. Версионированный pipeline

Есть базовый контракт:

```text
pipeline_version = radar-grid-v1
product = lowest_elevation_reflectivity
units = dBZ
grid = 256 x 256
radius = 250 km
crs = local_aeqd
time_step_minutes = 10
```

Контракт надо сохранить как legacy, но для новой задачи создать/обновить до 15-минутного режима.

### 1.3. Датасеты и metadata

Уже есть:

- `manifest.json`;
- dataset metadata;
- список кадров;
- список sequence;
- decode error count;
- regular segment count;
- проверка pipeline version при обучении.

Это не надо переписывать с нуля. Надо расширить metadata под `valid_mask`, `forecast_step_minutes=15`, качество временного ресемплинга и split.

### 1.4. Модель и quality gate

Есть текущая ConvLSTM baseline-модель, функции:

- persistence forecast;
- global shift advection forecast;
- threshold metrics по dBZ;
- uniform-field detector;
- quality gate против persistence/advection.

Это использовать как baseline-контур. Новая модель должна сравниваться с ним, а не заменять его вслепую.

### 1.5. Визуализация

Уже реализованы:

- AEQD → Web Mercator;
- ориентация `origin='lower'`;
- range rings;
- azimuth labels;
- dBZ legend;
- timestamp annotation;
- отказ от неизвестных координат станции.

Оставшийся долг: передавать реальный 15-минутный интервал и показывать маски/статус достоверности.

### 1.6. Export и диагностика

Есть:

- NetCDF export через `xarray`;
- CRS variable;
- `lead_time`;
- `pipeline_version`;
- `model_id`;
- `source`;
- `scripts/doctor.py`;
- `scripts/check_aws_source.py`.

Долг: заменить default 10 минут, добавить `valid_time_utc`, `lead_time_minutes`, экспериментальный статус, mask/provenance и более строгие тесты.

---

## 2. Главные незакрытые дефекты

| Дефект | Почему критично | Где исправлять |
| --- | --- | --- |
| 10-минутный шаг зашит в pipeline, visualization, export, tests | Целевая постановка требует 15 минут; иначе forecast labels и NetCDF неверны | `radar_pipeline.py`, `web_app.py`, `map_visualization.py`, `export_utils.py`, tests |
| `valid_mask` не используется в обучении | Модель путает `нет данных` и `нет эха` | dataset, model input, loss, metrics |
| BUFR decoder использует отдельный `griddata` path | Domain shift между BUFR и Py-ART/Barnes pipeline | `bufr_decoder.py`, `radar_pipeline.py` |
| `mtime` используется как fallback timestamp | Файловая дата не равна сроку МРЛ-наблюдения | `make_dataset.py`, `adapters.py`, metadata parser |
| ConvLSTM + MSE сглаживает ядра | Недостоверно для сильной конвекции и lead time 1–3 часа | новая модель и loss |
| Global shift baseline слишком слабый | ИИ надо сравнивать с нормальной адвекцией | `baselines.py`, `verification.py` |
| Прогноз 2–3 часа не маркирован как низкодостоверный | Одной отражаемости недостаточно для инициации конвекции | API/UI/export |

---

## 3. Новые инженерные решения

### 3.1. Новый временной контракт

Ввести единый конфиг:

```python
FORECAST_STEP_MINUTES = 15
DEFAULT_INPUT_LENGTH = 4
DEFAULT_TARGET_LENGTH = 4
SUPPORTED_TARGET_LENGTHS = {
    "1h": 4,
    "2h": 8,
    "3h": 12,
}
MAX_DBZ = 70.0
MIN_DBZ = 0.0
```

### 3.2. Новый data contract

Минимальный sample для обучения должен содержать:

```text
x_reflectivity: [T, 1, H, W]
x_valid_mask:  [T, 1, H, W]
x_range_norm:  [T, 1, H, W] или static [1, H, W]
y_reflectivity: [T_out, 1, H, W]
y_valid_mask:  [T_out, 1, H, W]
metadata: station, timestamps, source, pipeline_version
```

### 3.3. Новый model contract

Новая модель не заменяет baseline сразу. Она добавляется как отдельная архитектура:

```text
MRL-PhysLite = MotionHead + DifferentiableAdvection + Residual ConvGRU
```

---

## 4. Проверки завершения этапа 00

Этап 00 считается закрытым, если:

1. В `plan/README.md` указан новый порядок этапов.
2. Все файлы в `plan/` отражают актуальное состояние, а не только исторический NOAA-план.
3. В плане явно перечислено, что уже реализовано и что остается долгом.
4. Первый следующий PR определен как переход на `FORECAST_STEP_MINUTES=15`.

---

## 5. Следующий этап

Переходить к [01-data-trust-boundary.md](01-data-trust-boundary.md): единый 15-минутный временной контракт.