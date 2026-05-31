# Этап 02. Маски качества: разделить `нет эха` и `нет данных`

## Цель

Сделать `valid_mask` полноценной частью датасета, входа модели, функции потерь, метрик, визуализации и экспорта.

Это критический метеорологический этап. Для МРЛ ноль/слабый dBZ, отфильтрованный clutter, край радиуса, сектор блокировки и отсутствие данных — разные состояния. Модель не должна учиться считать их одним и тем же классом.

---

## 1. Что уже реализовано

1. В `RadarFrame` уже есть `valid_mask`.
2. `RadarPipeline.frame_from_grid()` умеет извлекать mask из masked array и считает `masked_pixels`, `valid_fraction`, `min_dbz`, `max_dbz`, `mean_dbz`.
3. `RadarSequence.stack()` возвращает только `frame.data`.
4. `RadarSequenceDataset` в `train_nowcasting_model.py` загружает только `.npy` с отражаемостью и не получает mask.
5. `frame_from_grid()` технически заполняет невалидные пиксели нулем, сохраняя mask отдельно. Это допустимо только при условии, что mask дальше используется.

---

## 2. Проблема

Сейчас модель получает массив вида:

```text
[T, H, W] reflectivity
```

Она не знает, где:

- реальное отсутствие радиоэха;
- пиксель вне зоны обзора;
- masked pixel;
- пропуск данных;
- отфильтрованное неметеорологическое эхо;
- сектор с плохим качеством.

Итог: loss и метрики штрафуют модель за области, где наблюдения недостоверны или отсутствуют.

---

## 3. Целевой data contract

Каждый training sample должен иметь минимум:

```text
x_reflectivity: [T_in, 1, H, W]
x_valid_mask:  [T_in, 1, H, W]
x_range_norm:  [T_in, 1, H, W] или [1, H, W]
y_reflectivity: [T_out, 1, H, W]
y_valid_mask:  [T_out, 1, H, W]
```

Рекомендуемый формат хранения после этапа:

```text
seq_0000.npz
  x_reflectivity / reflectivity
  valid_mask
  timestamps_utc
  station
  source_files
```

Если временно сохраняется `.npy`, рядом должен быть mask-файл или mask должен быть восстановим из manifest. Но целевой формат — `.npz` или Zarr.

---

## 4. Работы

### 4.1. Dataset builder

В `src/make_dataset.py`:

1. При сохранении последовательности сохранять не только `frame.data`, но и `frame.valid_mask`.
2. Для новых датасетов использовать `.npz`:

```python
np.savez_compressed(
    output_dir / filename,
    reflectivity=np.stack([frame.data for frame in selected], axis=0),
    valid_mask=np.stack([frame.valid_mask for frame in selected], axis=0),
    timestamps_utc=np.array([frame.timestamp_utc.isoformat() for frame in selected]),
)
```

3. В metadata добавить:

```json
{
  "contains_valid_mask": true,
  "mask_policy": "loss_and_metrics_use_target_valid_mask"
}
```

### 4.2. Dataset loader

Создать `src/datasets.py` и перенести туда `RadarSequenceDataset`.

Новый loader должен:

1. Читать `.npz` и legacy `.npy`.
2. Для legacy `.npy` выставлять `legacy_no_mask=true` и строить mask как `np.isfinite(data)` только для обратной совместимости.
3. Возвращать dict, а не tuple:

```python
{
    "x": x,
    "y": y,
    "x_mask": x_mask,
    "y_mask": y_mask,
    "metadata": metadata,
}
```

4. Добавить канал mask во вход модели:

```python
x_channels = torch.cat([x_reflectivity, x_mask, range_norm], dim=channel_axis)
```

### 4.3. Loss

Создать `src/losses.py`.

Минимальная функция:

```python
def masked_huber_loss(pred, target, mask):
    raw = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)
```

Добавить weighted variant для сильной отражаемости:

```python
weight = 1.0 + alpha * (target_dbz >= 30.0)
loss = (raw * mask * weight).sum() / (mask * weight).sum().clamp_min(1.0)
```

### 4.4. Метрики

В `src/forecast_quality.py` или новом `src/verification.py`:

1. Все continuous метрики принимать optional mask.
2. Threshold metrics считать только по валидным пикселям target.
3. Добавить в отчеты:

```text
valid_fraction_by_lead
masked_fraction_by_lead
```

### 4.5. Визуализация

В `src/map_visualization.py`:

1. Не показывать masked area как 0 dBZ.
2. Добавить отдельную визуальную политику:
   - прозрачность для `no echo`;
   - серый/штриховой слой для `no data`, если надо показать область невалидности.

### 4.6. API/Export

В `web_app.py` и `export_utils.py`:

1. В API вернуть `valid_fraction` входных кадров.
2. В NetCDF добавить переменную `valid_mask`, если она доступна.
3. В global attrs указать mask policy.

---

## 5. Тесты

Создать `tests/test_masks.py`.

Минимальные тесты:

1. `RadarPipeline.frame_from_grid()` сохраняет mask.
2. Dataset builder сохраняет `valid_mask` в `.npz`.
3. Dataset loader возвращает `x_mask` и `y_mask`.
4. `masked_huber_loss()` игнорирует невалидные пиксели.
5. Threshold metrics не считают hits/misses/false alarms по невалидным пикселям.
6. Legacy `.npy` помечается как `legacy_no_mask`.

---

## 6. Критические замечания

1. Нельзя удалять `valid_mask` ради простоты обучения.
2. Нельзя считать заполненный нулем masked pixel реальным отсутствием осадков.
3. Нельзя сравнивать новую masked-loss модель со старой MSE-моделью без единого evaluation protocol.
4. Если у входного кадра слишком маленький `valid_fraction`, кадр или вся последовательность должны быть отклонены.

---

## 7. Критерий завершения

Этап завершен, если:

1. Новые датасеты содержат `valid_mask`.
2. Модель получает mask как входной канал.
3. Loss и метрики используют `y_valid_mask`.
4. UI/API/NetCDF не смешивают `no data` и `no echo`.
5. Есть unit-тесты на mask pipeline.
6. Старые `.npy` датасеты явно помечаются legacy и не используются для новой production-модели без предупреждения.