# Этап 03. Единый BUFR / gridding / timestamp pipeline

## Цель

Убрать расхождение между разными путями подготовки МРЛ-данных. Один и тот же исходный файл должен проходить через один и тот же контракт обработки независимо от того, используется он для архива, обучения, live-инференса, API или теста.

---

## 1. Что уже реализовано

1. Основной `RadarPipeline` использует Py-ART `grid_from_radars()` и `weighting_function=Barnes2`.
2. `RadarPipelineConfig` хранит grid shape, radius, CRS и metadata.
3. `MRLBufrDecoder` существует и декодирует BUFR через `eccodes`.
4. `MRLBufrDecoder` сейчас сам переводит polar → Cartesian через `scipy.interpolate.griddata(method='linear')`.
5. `LocalDirectoryAdapter` для `.bufr` использует `MRLBufrDecoder`, а для `.npy/.npz` напрямую грузит массив.
6. `make_dataset.py` для NEXRAD берет timestamp из имени файла, но fallback — `mtime`.

---

## 2. Проблема

Сейчас есть два разных gridding path:

```text
NEXRAD / Py-ART path:
raw radar -> Py-ART -> Barnes2 -> RadarFrame

BUFR path:
BUFR -> eccodes arrays -> scipy.griddata linear -> ndarray -> RadarFrame
```

Это создает разные статистические свойства полей, а значит model domain shift. Для ИИ-прогноза это критичнее, чем выбор между ConvLSTM и ConvGRU.

---

## 3. Целевое решение

### 3.1. Единый объект декодирования

BUFR-декодер должен возвращать не просто `np.ndarray`, а структурированный объект:

```python
@dataclass
class DecodedRadarObservation:
    reflectivity: np.ndarray
    valid_mask: np.ndarray
    timestamp_utc: datetime.datetime
    station: str
    product: str
    units: str
    coordinate_system: str
    provenance: dict
    qc: dict
```

### 3.2. Единый pipeline entrypoint

Добавить метод уровня pipeline:

```python
RadarPipeline.process_decoded_observation(decoded: DecodedRadarObservation) -> RadarFrame
```

Если BUFR уже приходит как регулярная сетка, pipeline должен только проверить контракт и mask. Если BUFR приходит в полярных координатах, gridding должен быть согласован с `RadarPipelineConfig`.

### 3.3. Правило timestamp

Порядок получения времени наблюдения:

1. BUFR metadata / descriptor.
2. Имя файла по утвержденному шаблону.
3. Sidecar metadata JSON.
4. Только при явном флаге `--allow-mtime-fallback` — `mtime`.

Fallback через `mtime` должен записывать:

```json
"timestamp_source": "file_mtime_fallback"
```

и такие данные не должны попадать в production-датасет без явного разрешения.

---

## 4. Работы

### 4.1. BUFR decoder

В `src/bufr_decoder.py`:

1. Разделить чтение BUFR descriptors и gridding.
2. Не выполнять неявную линейную интерполяцию как основной production path.
3. Возвращать mask и metadata.
4. Сохранять имена реально найденных BUFR descriptors:

```json
"bufr_descriptors": ["reflectivity", "bearing", "range"]
```

5. Если descriptor неизвестен — возвращать явную ошибку, а не `continue` без диагностики.

### 4.2. Metadata parser

Создать `src/metadata_parser.py`:

Функции:

```python
def parse_timestamp_from_filename(path: Path) -> ParsedTimestamp: ...
def parse_station_from_filename(path: Path) -> str | None: ...
def require_observation_timestamp(...): ...
```

Для каждого timestamp сохранять source:

```text
bufr_metadata
filename
sidecar_metadata
file_mtime_fallback
```

### 4.3. Dataset builder

В `src/make_dataset.py`:

1. Заменить `_timestamp_from_path()` на metadata parser.
2. Добавить CLI-флаг:

```bash
--allow-mtime-fallback
```

3. По умолчанию запрещать mtime fallback.
4. В manifest записывать `timestamp_source` для каждого frame.

### 4.4. Local adapter

В `src/adapters.py`:

1. Для локальных `.bufr` использовать новый decoding contract.
2. Для `.npz` требовать наличие `reflectivity` и желательно `valid_mask`.
3. Для `.npy` считать файл legacy и возвращать warning в provenance.

### 4.5. География

Существующую AEQD-визуализацию оставить. Добавить контроль, что station coordinates берутся из:

1. metadata файла;
2. station catalog;
3. явного пользовательского параметра.

Неизвестная станция должна давать ошибку, а не fallback.

---

## 5. Тесты

Создать/обновить:

- `tests/test_ingestion.py`;
- `tests/test_decoders.py`;
- `tests/test_radar_pipeline.py`;
- `tests/test_map_visualization.py`.

Минимальные проверки:

1. BUFR без reflectivity descriptor дает понятную ошибку.
2. Timestamp из filename парсится с timezone UTC.
3. При отсутствии timestamp и без `--allow-mtime-fallback` dataset builder падает.
4. При включенном fallback metadata содержит `timestamp_source=file_mtime_fallback`.
5. `.npz` с `valid_mask` сохраняет mask до `RadarFrame`.
6. Unknown station в визуализации дает ошибку.
7. Контрольный marker на севере отображается выше центра.

---

## 6. Критические замечания

1. Разные gridding algorithms в train и inference запрещены для production-модели.
2. Старые `.npy` без metadata можно использовать только как legacy/debug.
3. Нельзя молча пропускать неизвестные BUFR descriptors: это скрывает поврежденный источник.
4. Временная метка наблюдения — часть метеорологического продукта. Файловый `mtime` не является надежным сроком.

---

## 7. Критерий завершения

Этап завершен, если:

1. BUFR, NEXRAD и local `.npz` приводятся к единому `RadarFrame` contract.
2. Gridding method и timestamp source записываются в provenance.
3. Старый `scipy.griddata` путь либо удален из production, либо явно маркирован как fallback.
4. Dataset builder не использует `mtime` без явного разрешения.
5. Тесты подтверждают сохранность timestamp, station, mask, units и pipeline metadata.