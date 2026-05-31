# Этап 04. Бейзлайны прогноза: persistence, advection, block-motion

## Цель

Сделать честный набор baseline-моделей, с которыми будет сравниваться любая нейросеть. До прохождения этого этапа нельзя утверждать, что новая ML-модель дает метеорологическую пользу.

Старый смысл файла — AWS ingestion — в основном закрыт базовой реализацией. AWS/NEXRAD остается отладочным источником, но текущий этап теперь посвящен качеству baseline-прогноза.

---

## 1. Что уже реализовано

1. `persistence_forecast(history, output_steps)` повторяет последний наблюденный кадр.
2. `advection_forecast(history, output_steps, search_radius=6)` реализует глобальный integer-shift по двум последним кадрам.
3. `threshold_metrics_by_lead_time()` считает hits/misses/false alarms, CSI, POD, FAR.
4. `evaluate_model_quality()` сравнивает модель с persistence и текущим advection baseline.
5. `quality_gate_passes()` не пропускает модель, если она хуже persistence/advection по MSE или дает uniform-field anomaly.

---

## 2. Проблема

Текущий `advection_forecast` полезен как быстрый smoke baseline, но он не является полноценным радарным nowcasting baseline:

```text
previous frame -> global integer shift -> forecast
```

Он не описывает:

- локально разные скорости движения;
- деформацию фронта;
- вращение/сдвиг конвективной линии;
- разделение и слияние ячеек;
- разные скорости на разных участках поля.

Если ИИ сравнивается только с таким слабым baseline, его качество будет переоценено.

---

## 3. Целевой набор baseline

### 3.1. Baseline A: Eulerian persistence

```text
forecast[t+k] = last_observed_frame
```

Назначение: нижний обязательный baseline. Модель хуже persistence не допускается.

### 3.2. Baseline B: global shift advection

Текущая реализация. Оставить как быстрый baseline и regression test.

### 3.3. Baseline C: local block-motion advection

Новый обязательный baseline.

Алгоритм:

1. Разбить поле на блоки, например 32×32 или 48×48 пикселей.
2. Для каждого блока найти локальное смещение между двумя/тремя последними кадрами.
3. Ограничить максимальное смещение физически разумным диапазоном.
4. Сгладить поле смещений.
5. Перенести последнее поле на каждый lead time.
6. Сохранить поле движения `u, v` как диагностический продукт.

### 3.4. Baseline D: optional pySTEPS-like optical flow

Если зависимости допустимы, добавить pySTEPS или совместимый optical-flow backend. Если нет — оставить как optional research path.

---

## 4. Работы

### 4.1. Вынести baseline в отдельный модуль

Создать `src/baselines.py`:

```python
class ForecastBaseline(Protocol):
    name: str
    def predict(self, history: np.ndarray, output_steps: int) -> np.ndarray: ...
```

Реализации:

```text
PersistenceBaseline
GlobalShiftAdvectionBaseline
BlockMotionAdvectionBaseline
```

### 4.2. Добавить block-motion baseline

Минимальная реализация:

```python
def estimate_block_motion(previous, current, block_size=32, search_radius=8):
    ...
```

Важно:

- считать motion только по валидным пикселям;
- игнорировать почти пустые блоки;
- ограничивать шумные векторы;
- сглаживать motion field;
- сохранять diagnostics.

### 4.3. Обновить evaluation

Создать `src/evaluate.py`.

Команда:

```bash
python src/evaluate.py \
  --dataset data/processed_archive/<dataset_id> \
  --model models/registry/<model_id> \
  --baselines persistence,global_shift,block_motion \
  --output reports/evaluation_<model_id>.json
```

Отчет должен содержать:

```text
MSE/MAE by lead time
CSI/POD/FAR by dBZ threshold and lead time
area bias by threshold
max_dbz_error by lead time
valid_fraction by lead time
uniform_field_anomaly
baseline ranking
quality_gate_passed
```

### 4.4. Усилить quality gate

Новая модель допускается только если:

1. Лучше persistence по masked loss.
2. Не хуже global shift по основным метрикам.
3. Лучше или сопоставима с block-motion baseline на горизонте 15–60 минут.
4. Не ухудшает CSI/FAR для порогов 20/30 dBZ.
5. Не дает uniform-field anomaly.
6. Не занижает максимумы отражаемости систематически.

---

## 5. Тесты

Создать/обновить:

- `tests/test_baselines.py`;
- `tests/test_forecast_quality.py`;
- `tests/test_evaluate.py`.

Минимальные проверки:

1. Persistence повторяет последний кадр.
2. Global shift переносит одиночную ячейку в ожидаемом направлении.
3. Block-motion переносит две ячейки с разными локальными смещениями.
4. Empty field не вызывает NaN/Inf.
5. Metrics считаются отдельно по lead time.
6. Quality gate отклоняет модель хуже block-motion baseline.

---

## 6. Критические замечания

1. Baseline должен быть сильным. Слабый baseline делает оценку ИИ недостоверной.
2. Не принимать модель только по MSE: она может улучшить MSE и одновременно уничтожить сильные ядра.
3. Для МРЛ-наукастинга spatial displacement error важен не меньше пиксельной ошибки.
4. Для 2–3 часов baseline будет деградировать, но это не означает, что ИИ автоматически надежен.

---

## 7. Критерий завершения

Этап завершен, если:

1. Есть отдельный модуль `baselines.py`.
2. Есть минимум три baseline: persistence, global shift, block-motion.
3. Есть команда `evaluate.py`, формирующая отчет по lead time и dBZ thresholds.
4. Quality gate сравнивает модель не только с persistence, но и с advection/block-motion.
5. Результаты baseline сохраняются в model registry metadata.