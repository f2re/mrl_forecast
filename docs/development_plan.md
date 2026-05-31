# План разработки физически обоснованного прогноза МРЛ-данных

Документ задает практический план доработки текущего проекта `mrl_forecast` под задачу прогноза зон радиолокационной отражаемости МРЛ на 1–3 часа вперед с дискретностью 15 минут при ограничениях: входные данные — только уровень отражаемости МРЛ, вычисления — CPU, RAM 16 ГБ.

План не предполагает немедленное внедрение тяжелых DGMR/LDCast/диффузионных моделей. Для текущих ресурсов целевая архитектура — легкая физически-информированная модель `MRL-PhysLite`: дифференцируемая адвекция + остаточная эволюция + легкий ConvGRU/U-Net-блок + строгая верификация против persistence и advection baseline.

---

## 0. Критические выводы аудита

### 0.1. Что в проекте уже правильно

1. Введен версионированный контракт обработки `radar-grid-v1`.
2. В данных явно указаны продукт и единицы: `lowest_elevation_reflectivity`, `dBZ`.
3. Есть структура `RadarFrame` с `valid_mask`, `qc`, `provenance`, `status`.
4. Есть quality gate: модель должна быть лучше persistence и advection baseline.
5. Есть отказ от публикации модели, если она не прошла quality gate.
6. Есть базовая ConvLSTM-реализация и web-инференс.
7. Есть NetCDF export, который можно сделать основным форматом выдачи результата.

### 0.2. Что физически и методически неверно или недостаточно

1. В проекте зашит 10-минутный шаг, а целевая постановка требует 15 минут.
2. `dBZ` нельзя напрямую трактовать как сохраняемую физическую массу или интенсивность осадков.
3. Нельзя считать физический штраф адвекции на логарифмической величине `dBZ` без преобразования в линейный proxy.
4. Недопустимо смешивать `нет эха`, `нет данных`, `замаскированный пиксель`, `край радиуса`, `сектор блокировки` в один ноль.
5. В обучении сейчас используется только поле отражаемости, без `valid_mask` как входного канала и без маски в loss.
6. BUFR-декодер и основной Py-ART pipeline используют разные подходы к гридированию.
7. `mtime` файла не должен использоваться как метеорологический срок наблюдения.
8. Простая ConvLSTM + MSE склонна к размытию и недооценке сильных ядер отражаемости.
9. Глобальный integer-shift baseline не является полноценным optical-flow nowcasting.
10. Прогноз 2–3 часа при одном поле отражаемости должен маркироваться как продукт пониженной достоверности.

### 0.3. Целевая формулировка продукта

Корректное название продукта:

> Экспериментальный прогноз поля радиолокационной отражаемости / зон радиоэха МРЛ.

Некорректное название без дополнительной калибровки:

> Точный прогноз осадков / интенсивности осадков.

Если в будущем будет выполнено Z–R-преобразование, оно должно быть явно подписано как расчетная оценка, а не как измеренная интенсивность осадков.

---

## 1. Целевая архитектура системы

### 1.1. Общая схема

```text
BUFR / локальные сетки / архив
        |
        v
Единый ingestion + metadata parser
        |
        v
QC + valid_mask + range_mask + unified gridding
        |
        v
radar-grid-v2-15min dataset
        |
        +--> persistence baseline
        +--> advection / optical-flow baseline
        +--> текущая ConvLSTM baseline
        +--> новая MRL-PhysLite model
        |
        v
Валидация по lead time и порогам dBZ
        |
        v
Model registry + quality gate
        |
        v
Web/API inference + NetCDF export + визуализация
```

### 1.2. Новая модель `MRL-PhysLite`

Модель должна состоять из трех блоков:

1. `MotionNet`: оценивает поле переноса `u, v`.
2. `DifferentiableAdvection`: переносит последнее наблюдаемое поле вперед.
3. `ResidualNet / ConvGRU`: оценивает рост, распад и деформацию радиоэха.

Итоговая идея:

```text
forecast_proxy(t + lead) = advect(last_proxy, u, v, lead) + residual_growth_decay
forecast_dbz = inverse_proxy_transform(forecast_proxy)
```

Где `proxy` — линейная или квазилинейная величина, полученная из dBZ. Не применять закон сохранения напрямую к dBZ.

---

## 2. Этапы разработки

## Этап 1. Зафиксировать новый контракт времени и данных

### Цель

Перевести проект с неявного 10-минутного шага на явный конфиг 15 минут и подготовить основу для прогнозов на 1, 2 и 3 часа.

### Файлы

- `src/radar_pipeline.py`
- `src/web_app.py`
- `src/export_utils.py`
- `src/train_nowcasting_model.py`
- `scripts/train.sh`
- `scripts/prepare.sh`
- `templates/index.html`
- `docs/report.md`

### Работы

1. Создать единый конфиг, например `src/config.py`:

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

2. В `RadarPipelineConfig` заменить `time_step_minutes=10` на значение из конфига.

3. В `web_app.py` заменить все ручные `+10 минут` на `FORECAST_STEP_MINUTES`.

4. В `export_utils.py` и вызове `save_forecast_to_netcdf` передавать реальный интервал 15 минут.

5. В `scripts/train.sh` и `scripts/prepare.sh` явно передавать `--time-step-minutes 15` или брать из конфига.

6. В UI заменить термин `lead_time` на понятный `forecast_horizon` или `target_length` с вариантами:
   - 1 час: 4 срока;
   - 2 часа: 8 сроков;
   - 3 часа: 12 сроков.

### Критические замечания

- Нельзя просто поменять подписи на картинках. Датасеты должны быть сформированы с 15-минутной дискретностью.
- Если исходные МРЛ-кадры приходят чаще, например 5 или 10 минут, нужен ресемплинг/выбор ближайших кадров к регулярной 15-минутной сетке.
- Если пропуск больше заданного допуска, последовательность не должна попадать в обучение.

### Критерии завершения

- В metadata датасета записано `time_step_minutes: 15`.
- В metadata модели записано `forecast_step_minutes: 15`.
- В API forecast labels показывают `+15`, `+30`, `+45`, `+60` минут.
- NetCDF содержит корректные forecast times с шагом 15 минут.
- Все тесты, завязанные на прежний 10-минутный шаг, обновлены.

---

## Этап 2. Развести `нет эха` и `нет данных`

### Цель

Сделать `valid_mask` полноценной частью обучения и инференса.

### Файлы

- `src/radar_pipeline.py`
- `src/adapters.py`
- `src/train_nowcasting_model.py`
- новый файл `src/datasets.py`
- новый файл `src/losses.py`
- тесты `tests/test_masks.py`

### Работы

1. В датасете возвращать не только `x` и `y`, но и маски:

```python
return {
    "x": x_tensor,              # [T, C, H, W]
    "y": y_tensor,              # [T, 1, H, W]
    "x_mask": x_mask_tensor,    # [T, 1, H, W]
    "y_mask": y_mask_tensor,    # [T, 1, H, W]
}
```

2. Входные каналы модели:

```text
channel 0: normalized reflectivity
channel 1: valid_mask
channel 2: range_norm
optional channel 3: static radar coverage mask
```

3. Добавить `MaskedHuberLoss`:

```python
def masked_huber_loss(pred, target, mask):
    raw = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)
```

4. Все continuous-метрики считать только по валидным пикселям.

5. Визуализацию невалидных зон показывать отдельной прозрачной/серой маской, а не как 0 dBZ.

### Критические замечания

- Заполнение невалидных пикселей нулем допустимо только как технический placeholder, но loss и метрики должны использовать `valid_mask`.
- На краях радиуса и в секторах блокировки модель не должна получать штраф за «неправильный прогноз».
- Для operational-интерфейса надо показывать пользователю, где прогноз построен по недостоверному входу.

### Критерии завершения

- Модель принимает минимум 2 канала: reflectivity + valid_mask.
- Loss не штрафует модель за невалидные пиксели.
- В metadata датасета есть `valid_fraction` по каждому кадру или агрегированно по датасету.
- Есть тест: если половина таргета замаскирована, loss считается только по незамаскированной половине.
- В UI/экспорте невалидные зоны не отображаются как метеорологическое отсутствие осадков.

---

## Этап 3. Унифицировать BUFR и gridding pipeline

### Цель

Исключить разные статистические свойства сеток между обучением и инференсом.

### Файлы

- `src/bufr_decoder.py`
- `src/radar_pipeline.py`
- `src/adapters.py`
- новый файл `src/metadata_parser.py`
- тесты `tests/test_ingestion.py`

### Работы

1. BUFR-декодер должен возвращать не только `grid`, а структуру:

```python
@dataclass
class DecodedRadar:
    values: np.ndarray
    valid_mask: np.ndarray
    timestamp_utc: datetime.datetime
    station: str
    source_product: str
    provenance: dict
```

2. Убрать использование `mtime` как основного срока наблюдения.

3. Реализовать порядок определения времени:
   - BUFR metadata;
   - имя файла по утвержденному шаблону;
   - sidecar JSON/metadata;
   - только как последний fallback — `mtime`, но со статусом `timestamp_source="file_mtime_fallback"`.

4. Привести `MRLBufrDecoder` к единому контракту `RadarPipeline.frame_from_grid(...)`.

5. В `bufr_decoder.py` не использовать `griddata` как альтернативный основной путь, если основной pipeline использует Barnes2. Если Py-ART не подходит для конкретного BUFR, надо явно документировать fallback и маркировать его в metadata.

6. Добавить контроль геометрии:
   - радиус обзора;
   - разрешение сетки;
   - CRS/локальная проекция;
   - центр радара;
   - станция;
   - высота/угол, если есть.

### Критические замечания

- Два разных метода гридирования создают искусственный domain shift.
- Сетка должна быть одинаковой на обучении, валидации и инференсе.
- Все преобразования должны быть воспроизводимы по metadata.

### Критерии завершения

- Любой подготовленный `.npy/.npz/.nc` содержит или сопровождается metadata с `pipeline_version`, `time_step_minutes`, `grid`, `station`, `source`, `timestamp_source`.
- Один и тот же BUFR-файл при повторной обработке дает идентичную сетку и идентичные metadata.
- В тестах проверяется, что shape, units, valid_mask и timestamp не теряются.
- Подготовка датасета отказывается от файлов без надежного timestamp, если не включен явный `--allow-mtime-fallback`.

---

## Этап 4. Добавить нормальные baseline-модели

### Цель

Получить честный эталон качества до внедрения новой ИИ-модели.

### Файлы

- `src/forecast_quality.py`
- новый файл `src/baselines.py`
- `src/train_nowcasting_model.py`
- новый файл `src/evaluate.py`
- тесты `tests/test_baselines.py`

### Baseline-уровни

1. `Eulerian persistence` — повтор последнего кадра.
2. `Global shift advection` — текущий быстрый baseline.
3. `Block matching / local cross-correlation` — локальный motion baseline.
4. Опционально: интеграция `pySTEPS`, если зависимости допустимы на целевой машине.

### Работы

1. Вынести persistence и advection из `forecast_quality.py` в `baselines.py`.

2. Добавить локальный block-matching baseline:
   - разбить поле на блоки;
   - искать локальное смещение в окне;
   - сгладить поле смещений;
   - выполнить semi-Lagrangian extrapolation.

3. Метрики baseline считать тем же кодом, что и метрики модели.

4. В quality gate добавить порог:

```text
модель допускается, если:
- MSE/Huber лучше persistence;
- MSE/Huber лучше global shift;
- CSI/FSS на порогах 20/30 dBZ не хуже baseline;
- нет uniform-field anomaly;
- нет систематического исчезновения сильных ядер.
```

### Критические замечания

- Нельзя считать модель хорошей только потому, что она снизила MSE.
- Для МРЛ-прогноза baseline должен быть сильным: optical flow/advection часто очень трудно обойти на первых 30–60 минутах.
- Если ИИ не обгоняет baseline хотя бы на независимой выборке, его нельзя публиковать как основной прогноз.

### Критерии завершения

- Команда `python src/evaluate.py --dataset ... --baselines persistence,global_shift,block_motion` формирует отчет.
- Отчет содержит метрики по lead time: 15/30/45/60/90/120/180 мин.
- Отчет содержит threshold metrics по dBZ: 5/10/20/30/40.
- Baseline-прогнозы можно визуально сравнить с прогнозом ИИ.
- Quality gate использует baseline-метрики, а не только validation loss.

---

## Этап 5. Перепроектировать модель: `MRL-PhysLite`

### Цель

Заменить ConvLSTM-only подход на легкую физически-информированную модель, пригодную для CPU/16 ГБ.

### Файлы

- новый файл `src/models/physlite.py`
- новый файл `src/models/components.py`
- новый файл `src/losses.py`
- `src/train_nowcasting_model.py`
- `src/web_app.py`
- тесты `tests/test_model_shapes.py`

### Архитектура

```text
Input: [B, T, C, H, W]
  C = reflectivity_norm + valid_mask + range_norm

Encoder2D per frame
  -> temporal ConvGRU
  -> MotionHead: [B, target_T, 2, H, W]
  -> ResidualHead: [B, target_T, 1, H, W]

last_frame_proxy
  -> DifferentiableAdvection(motion)
  -> + residual
  -> clamp/activation
  -> output forecast_dbz_norm
```

### Минимальные параметры для CPU

```text
input grid: 256x256
base channels: 16
ConvGRU hidden: 16 или 24
encoder depth: 3
target_length: 4 для основного обучения; 8/12 отдельными экспериментами
batch_size: 1–2 на CPU
mixed precision: не обязательно
num_workers: 0–2
```

### Proxy-преобразование dBZ

Минимальный вариант:

```python
z_linear = torch.pow(10.0, dbz / 10.0)
z_proxy = torch.log1p(z_linear / scale)
```

Важно: физический штраф считать на `z_proxy` или `z_linear`, а не напрямую на `dBZ`.

### Loss

Составная функция:

```text
L = L_masked_huber
  + lambda_grad * L_gradient
  + lambda_struct * L_ssim_or_laplacian
  + lambda_adv * L_advection_residual
  + lambda_heavy * L_heavy_echo
```

Практические стартовые веса:

```text
lambda_grad = 0.05
lambda_struct = 0.05
lambda_adv = 0.10
lambda_heavy = 0.50
```

Эти значения не считать окончательными. Они должны подбираться по валидации.

### Критические замечания

- Не называть модель PINN в строгом смысле, если она не решает физические уравнения атмосферы.
- Корректный термин: physics-guided / physics-informed radar nowcasting.
- Не делать 3-часовой прогноз основной моделью на первом этапе. Сначала довести 1 час.
- Не использовать GAN/диффузию до появления GPU и большого локального архива.

### Критерии завершения

- Модель принимает `[B, T, C, H, W]` и возвращает `[B, target_T, 1, H, W]`.
- Есть unit-тесты на shape для target_length 4, 8, 12.
- Inference одного прогноза 256×256×4 на CPU занимает приемлемое время, целевой ориентир — до 30 секунд.
- Модель не падает при полностью пустом поле и при частично замаскированном поле.
- Forecast values ограничены физически допустимым диапазоном `[0, 70] dBZ` или явно заданным диапазоном проекта.

---

## Этап 6. Перестроить обучение

### Цель

Сделать обучение воспроизводимым, leakage-free и пригодным для малых ресурсов.

### Файлы

- `src/train_nowcasting_model.py`
- новый файл `src/datasets.py`
- новый файл `src/training.py`
- новый файл `src/evaluate.py`
- `models/registry/*/metadata.json`

### Работы

1. Разделить датасет строго по времени, а не случайно.

2. Исключить leakage между train и validation через пересекающиеся sliding windows.

3. Добавить режимы обучения:

```text
--horizon 1h -> target_length 4
--horizon 2h -> target_length 8
--horizon 3h -> target_length 12
```

4. Для CPU включить early stopping:

```text
patience: 5 эпох
min_delta: 0.001
max_epochs по умолчанию: 30
```

5. Добавить weighted sampling, чтобы пустые поля не доминировали:
   - sample без эха;
   - слабые осадки;
   - умеренные зоны;
   - сильные ядра `>= 30 dBZ`.

6. Сохранять в metadata:
   - список датасетов;
   - станции;
   - период;
   - число sample;
   - долю пустых/слабых/сильных случаев;
   - `time_step_minutes`;
   - `input_length`, `target_length`;
   - baseline metrics;
   - model metrics;
   - quality gate decision.

### Критические замечания

- Нельзя валидировать на окнах, соседних с обучающими, если они используют почти те же кадры.
- Нельзя публиковать модель только по train/val loss.
- Для разных сезонов качество будет разным. Желательно иметь seasonal split.

### Критерии завершения

- При запуске обучения создается полная папка модели в `models/registry/model_*`.
- В metadata есть `status: completed` только при прохождении quality gate.
- Если модель хуже baseline, статус `rejected_quality_gate`.
- Есть `history.csv`, `learning_curve.png`, `metrics.json` или единый metadata с метриками.
- Обучение можно повторить с тем же seed и получить сопоставимые результаты.

---

## Этап 7. Расширить верификацию

### Цель

Перейти от MSE-only проверки к метеорологически значимой валидации.

### Файлы

- `src/forecast_quality.py`
- новый файл `src/verification.py`
- новый файл `src/evaluate.py`
- `templates/index.html`

### Метрики

Обязательные:

1. MSE/MAE по lead time.
2. CSI по порогам 5/10/20/30/40 dBZ.
3. POD по порогам.
4. FAR по порогам.
5. Bias/frequency bias по порогам.
6. FSS или neighborhood CSI для учета пространственного сдвига.
7. Доля uniform-field anomaly.
8. Ошибка максимума dBZ: `max_dbz_error`.
9. Сохранность площади сильного эха: area error для `>=30 dBZ`.

Желательные:

1. Object-based метрики: смещение центра масс зоны `>=20/30 dBZ`.
2. Split/merge diagnostic для ячеек.
3. Метрики по радиальным зонам от радара.
4. Метрики отдельно для пустых, стратиформных и конвективных случаев.

### Критические замечания

- MSE может улучшаться при ухудшении сильных осадков.
- CSI зависит от частоты события, поэтому его надо смотреть вместе с POD и FAR.
- Для предупреждений важен не только пиксельный error, но и положение, площадь и максимум ядра.

### Критерии завершения

- `evaluate.py` формирует markdown/html/json отчет.
- В отчете есть таблица по lead time и порогам.
- В отчете есть сравнение минимум четырех методов: persistence, global_shift, block_motion, AI model.
- Для модели в registry сохраняется полный evaluation report.
- UI показывает не только картинки, но и краткую диагностику качества модели.

---

## Этап 8. Обновить web-инференс и API

### Цель

Сделать operational-интерфейс честным: прогноз должен иметь срок, статус, достоверность, источник, модель и предупреждения.

### Файлы

- `src/web_app.py`
- `templates/index.html`
- `src/map_visualization.py`
- `src/export_utils.py`

### Работы

1. В API `/api/predict` вернуть расширенный JSON:

```json
{
  "base_time": "...",
  "forecast_step_minutes": 15,
  "horizon_minutes": 60,
  "source_status": "observed",
  "model_id": "...",
  "pipeline_version": "radar-grid-v2-15min",
  "confidence_by_lead": ["normal", "normal", "reduced", "reduced"],
  "diagnostics": {...},
  "warnings": [...]
}
```

2. Для сроков больше 60 минут показывать маркировку:

```text
60–120 мин: пониженная достоверность
120–180 мин: экспериментальный прогноз
```

3. В UI заменить формулировку:

```text
Прогноз осадков
```

на:

```text
Экспериментальный прогноз поля отражаемости МРЛ
```

4. Добавить слой/подпись `валидность входных данных`.

5. В карточке модели показывать:
   - model_id;
   - horizon;
   - training dataset;
   - quality gate status;
   - baseline comparison.

6. Если источник `demo`, прогноз должен быть явно помечен как демо и не экспортироваться как operational.

### Критические замечания

- Пользователь не должен принять экспериментальный ИИ-прогноз за официальное штормовое предупреждение.
- Дата/срок должны быть понятны: базовый срок, lead time, UTC/местное время.
- Нельзя скрывать факт, что прогноз построен по одному каналу отражаемости.

### Критерии завершения

- Визуализация показывает `base time` и `lead time` для каждого кадра.
- API возвращает `forecast_step_minutes=15`.
- Forecast frames подписаны `+15`, `+30`, `+45`, `+60` и далее.
- При `demo`-источнике UI явно пишет `DEMO`.
- При `uniform_field_anomaly` прогноз отклоняется или маркируется как непригодный.

---

## Этап 9. Обновить NetCDF/export

### Цель

Сделать экспорт пригодным для последующего анализа и аудита.

### Файлы

- `src/export_utils.py`
- `src/web_app.py`
- тесты `tests/test_export.py`

### Обязательные metadata в NetCDF

```text
product = experimental_radar_reflectivity_nowcast
units = dBZ
source_product = lowest_elevation_reflectivity или фактический продукт
model_id
model_type
pipeline_version
forecast_step_minutes = 15
base_time_utc
station_id
station_lat
station_lon
grid_crs
radius_km
input_length
target_length
quality_gate_status
not_official_warning = true
```

### Работы

1. Добавить координату `lead_time_minutes`.
2. Добавить координату `valid_time_utc`.
3. Добавить переменную `valid_mask` или `forecast_valid_mask`, если применимо.
4. Добавить global attributes с предупреждением об экспериментальном статусе.
5. Проверить, что экспорт не использует жесткий `interval_minutes=10`.

### Критические замечания

- NetCDF без metadata почти бесполезен для метеорологического аудита.
- Экспорт должен позволять восстановить, какой моделью и из каких входов получен прогноз.

### Критерии завершения

- Открытие NetCDF через `xarray.open_dataset()` показывает корректные координаты времени.
- В файле есть `lead_time_minutes=[15,30,45,60,...]`.
- В файле есть model/pipeline provenance.
- Тест проверяет, что 15-минутный шаг не потерян.

---

## Этап 10. Документация и регламент использования

### Цель

Сделать проект понятным для разработчика, метеоролога и проверяющего.

### Файлы

- `README.md`
- `docs/report.md`
- `docs/model_card.md`
- `docs/data_contract.md`
- `docs/validation_protocol.md`
- `docs/development_plan.md`

### Работы

1. Создать `docs/data_contract.md`:
   - формат входа;
   - сетка;
   - единицы;
   - маски;
   - time step;
   - допустимые пропуски.

2. Создать `docs/model_card.md`:
   - назначение модели;
   - входы;
   - выходы;
   - ограничения;
   - known failure modes;
   - training data;
   - validation metrics.

3. Создать `docs/validation_protocol.md`:
   - разбиение train/val/test;
   - метрики;
   - пороги допуска;
   - правила публикации модели.

4. В `README.md` добавить честное предупреждение:

```text
Проект формирует экспериментальный прогноз поля радиолокационной отражаемости и не является официальным предупреждением об опасных явлениях погоды.
```

### Критические замечания

- Документация должна запрещать использование неподтвержденной модели как operational warning source.
- Любые публичные выводы должны сопровождаться версией модели и результатами локальной валидации.

### Критерии завершения

- В репозитории есть отдельные документы по данным, модели и валидации.
- Новый разработчик может по README подготовить датасет, обучить baseline, обучить модель, запустить evaluation и web-интерфейс.
- Ограничения модели явно указаны в UI и документации.

---

## 3. Рекомендуемая очередность работ

### Спринт 1: исправить контракт времени и metadata

1. Ввести `FORECAST_STEP_MINUTES=15`.
2. Обновить pipeline metadata.
3. Обновить web labels и NetCDF export.
4. Добавить тесты на 15-минутную шкалу.

Результат: проект больше не выдает 10-минутный прогноз под видом 15-минутного.

### Спринт 2: маски и единый ingestion

1. Добавить `valid_mask` в dataset.
2. Убрать `mtime` как основной timestamp.
3. Унифицировать BUFR/gridding.
4. Добавить тесты на timestamp, mask, metadata.

Результат: обучение и инференс используют метеорологически корректный вход.

### Спринт 3: baseline и верификация

1. Вынести baseline-модели.
2. Добавить block-motion baseline.
3. Добавить evaluation report.
4. Усилить quality gate.

Результат: появляется честная шкала сравнения качества.

### Спринт 4: MRL-PhysLite model

1. Реализовать MotionNet.
2. Реализовать differentiable advection.
3. Реализовать ResidualNet/ConvGRU.
4. Реализовать masked weighted loss.
5. Обучить 1h-модель.

Результат: новая физически-информированная модель на 1 час.

### Спринт 5: горизонты 2–3 часа

1. Обучить 2h-вариант.
2. Обучить 3h-вариант только как экспериментальный.
3. Добавить confidence labels.
4. Сравнить degradation по lead time.

Результат: честный прогноз 1–3 часа с явной достоверностью.

### Спринт 6: UI/API/export hardening

1. Обновить интерфейс.
2. Добавить model card в UI.
3. Добавить warning labels.
4. Доработать NetCDF metadata.
5. Подготовить документацию.

Результат: проект готов к демонстрации и локальному тестированию.

---

## 4. Definition of Done для всей версии

Версия считается завершенной, если выполнены все условия:

1. Везде используется 15-минутный forecast step.
2. Источник времени наблюдения не основан на `mtime`, кроме явно разрешенного fallback.
3. Датасет содержит или сопровождается `valid_mask`.
4. Loss и метрики используют `valid_mask`.
5. Есть минимум три baseline: persistence, global shift, local/block motion.
6. Новая модель лучше baseline на независимой выборке по заранее заданным метрикам.
7. Качество показано отдельно по lead time и порогам dBZ.
8. 1-часовой прогноз имеет основной статус, 2–3 часа маркируются как сниженная/экспериментальная достоверность.
9. UI и NetCDF явно пишут, что это экспериментальный прогноз отражаемости МРЛ.
10. Модель не публикуется, если не прошла quality gate.
11. В model registry сохранены metadata, метрики, графики, версия pipeline и параметры обучения.
12. Документация позволяет воспроизвести подготовку данных, обучение, оценку и инференс.

---

## 5. Запрещенные упрощения

1. Не менять только подписи времени без пересборки датасета.
2. Не считать `dBZ` физически сохраняемой массой.
3. Не обучать по MSE-only и не принимать модель по одному validation loss.
4. Не смешивать `нет данных` и `нет осадков`.
5. Не использовать demo-данные как доказательство качества.
6. Не объявлять 3-часовой прогноз надежным без локальной валидации.
7. Не называть результат официальным прогнозом или предупреждением.
8. Не сравнивать новую модель только с persistence; нужен хотя бы один advection/optical-flow baseline.
9. Не публиковать модель без metadata и model card.
10. Не использовать разные методы гридирования на обучении и в эксплуатации.

---

## 6. Минимальный набор новых файлов

```text
src/config.py
src/datasets.py
src/losses.py
src/baselines.py
src/verification.py
src/evaluate.py
src/models/__init__.py
src/models/components.py
src/models/physlite.py
docs/data_contract.md
docs/model_card.md
docs/validation_protocol.md
tests/test_time_contract.py
tests/test_masks.py
tests/test_baselines.py
tests/test_model_shapes.py
tests/test_export.py
```

---

## 7. Практический первый Pull Request

Первый PR должен быть небольшим и безопасным:

1. Добавить `src/config.py`.
2. Заменить 10-минутные константы на `FORECAST_STEP_MINUTES`.
3. Обновить `RadarPipelineConfig.time_step_minutes`.
4. Обновить подписи forecast в `web_app.py`.
5. Обновить `interval_minutes` в NetCDF export.
6. Добавить тест `test_time_contract.py`.
7. Обновить README/документацию с указанием 15-минутного шага.

Критерий приемки первого PR:

```text
При forecast_step_minutes=15 система формирует сроки +15/+30/+45/+60,
пишет этот шаг в metadata и NetCDF, а тесты подтверждают отсутствие
жестко зашитого 10-минутного интервала в ключевых местах инференса.
```
