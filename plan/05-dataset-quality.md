# Этап 05. Датасеты 15 минут, QC, split и sampling

## Цель

Пересобрать обучающие датасеты под новый 15-минутный контракт и сделать их пригодными для честной оценки модели.

Датасет должен быть не просто набором `.npy`, а проверяемым метеорологическим архивом с manifest, QC, масками, временной шкалой, независимыми split и описанием состава погодных ситуаций.

---

## 1. Что уже реализовано

1. `make_dataset.py` создает dataset directory.
2. Сохраняется `metadata.json`.
3. Сохраняется `manifest.json`.
4. Есть `regular_frame_segments()`.
5. Есть защита от части temporal leakage через `temporal_split_indices()`.
6. Dataset metadata проверяется при обучении.

---

## 2. Что нужно изменить

### 2.1. Новый формат sample

Перейти от:

```text
seq_0000.npy -> [T, H, W]
```

к:

```text
seq_0000.npz
  reflectivity: [T, H, W]
  valid_mask: [T, H, W]
  timestamps_utc: [T]
  source_files: [T]
```

Legacy `.npy` оставить только для чтения старых экспериментов.

### 2.2. Новый sequence length

Для 1 часа при шаге 15 минут:

```text
input_length = 4
target_length = 4
sequence_length = 8
```

Для 2 часов:

```text
input_length = 4 или 6
target_length = 8
sequence_length = 12 или 14
```

Для 3 часов:

```text
input_length = 4 или 6
target_length = 12
sequence_length = 16 или 18
```

Сначала реализовать и валидировать 1h. 2h/3h — только после накопления достаточного архива.

### 2.3. Resampling policy

Если исходные кадры чаще/нерегулярнее 15 минут:

1. Построить целевую сетку времени от первого валидного кадра.
2. Для каждого target timestamp выбрать ближайший observation в пределах tolerance.
3. Если ближайшего кадра нет — разрывать segment.
4. В manifest записывать фактическое отклонение:

```json
"time_offset_seconds": 87
```

Рекомендуемый старт:

```text
resampling_policy = nearest_with_tolerance
resampling_tolerance_minutes = 4
```

### 2.4. QC-фильтры

Отклонять кадр или sequence, если:

- `valid_fraction` ниже порога;
- есть NaN/Inf после preprocessing;
- поле почти константно;
- max dBZ за пределами допустимого диапазона;
- timestamp ненадежен;
- source status не `observed`;
- frame получен из `demo`;
- frame имеет legacy/no-mask статус, если строится production dataset.

Рекомендуемые стартовые пороги:

```text
min_valid_fraction = 0.70
max_dbz_allowed = 80
min_dbz_allowed = -20 для raw, 0 для clipped ML input
uniform_std_threshold_dbz = 0.25
```

### 2.5. Split policy

Делить данные до нарезки sliding windows.

Приоритетный порядок:

1. По датам/дням.
2. По событиям.
3. По станциям.
4. Только затем по sequence index с gap, если других данных мало.

Запрещено: случайно перемешивать соседние окна, которые отличаются одним кадром.

### 2.6. Sampling policy

Из-за доминирования пустых пикселей и сухих кадров нужно учитывать состав выборки.

В QC report считать:

```text
fraction pixels >= 5 dBZ
fraction pixels >= 10 dBZ
fraction pixels >= 20 dBZ
fraction pixels >= 30 dBZ
fraction dry frames
fraction convective-like frames
```

Для обучения добавить weighted sampling или balanced batch policy:

```text
dry / weak / moderate / strong echo cases
```

---

## 3. Работы

### 3.1. Новый `src/datasets.py`

Содержит:

```text
RadarSequenceDataset
load_sequence_npz
load_legacy_npy
build_range_channel
collate_fn
```

### 3.2. Новый QC report

Создать `src/dataset_qc.py` или реализовать в `make_dataset.py`.

Выход:

```text
metadata.json
manifest.json
qc_report.json
split_manifest.json
```

### 3.3. Split manifest

Формат:

```json
{
  "split_policy": "by_day_or_temporal_gap",
  "train": ["seq_0000.npz"],
  "validation": ["seq_0100.npz"],
  "test": ["seq_0200.npz"],
  "source_frame_overlap_check": "passed"
}
```

### 3.4. Совместимость с train script

`train_nowcasting_model.py` должен использовать новый dataset loader, но уметь явно отказать старому dataset, если режим production.

---

## 4. Тесты

Создать/обновить:

- `tests/test_dataset_pipeline.py`;
- `tests/test_dataset_qc.py`;
- `tests/test_time_contract.py`;
- `tests/test_masks.py`.

Проверки:

1. Новый dataset сохраняет `.npz` с reflectivity и valid_mask.
2. `qc_report.json` содержит распределения по dBZ thresholds.
3. Split не содержит пересекающихся source timestamps.
4. Sequence с временным разрывом отклоняется.
5. Sequence с demo frame отклоняется.
6. Sequence с `valid_fraction < threshold` отклоняется.
7. Legacy `.npy` читается только в legacy mode.

---

## 5. Критические замечания

1. После честного split метрики могут резко ухудшиться. Это не регрессия, а исправление оценки.
2. Датасет из одного дня одного радара не доказывает устойчивость модели.
3. NOAA/NEXRAD и российские МРЛ/ДМРЛ нельзя смешивать до унификации продукта, маски, шкалы и metadata.
4. Для 3-часового прогноза нужен значительно больший архив и отдельная валидация по режимам погоды.

---

## 6. Критерий завершения

Этап завершен, если:

1. Новые датасеты строятся с `time_step_minutes=15`.
2. Каждый sample содержит reflectivity, valid_mask, timestamps и source_files.
3. Есть `qc_report.json` и `split_manifest.json`.
4. Нет пересечения исходных кадров между train/validation/test.
5. Dataset loader возвращает mask и range channel.
6. Training script умеет обучаться на новом dataset format.
7. Старые dataset/model с `radar-grid-v1` помечаются legacy.