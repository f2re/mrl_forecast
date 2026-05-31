# Этап 09. Российские МРЛ/ДМРЛ как отдельный ingestion track

## Цель

Подключить российские МРЛ/ДМРЛ как отдельный проверяемый источник данных, не смешивая его с NOAA/NEXRAD до доказанной совместимости продукта, геометрии, временной шкалы, масок и metadata.

Этот этап нельзя закрыть без реальных разрешенных fixtures российских данных.

---

## 1. Текущее состояние

1. В проекте есть `MRLBufrDecoder`.
2. В `info.md` перечислены возможные источники и инструменты: ВНИИГМИ-МЦД/ФГИС ЕГФД, сеть ДМРЛ Росгидромета, Гидрометцентр, WMO OSCAR, ODIM HDF5, BUFR, wradlib, Py-ART, ecCodes.
3. Реального подтвержденного production fixture российского BUFR/ODIM HDF5 в коде не зафиксировано.
4. Наличие BUFR-декодера не равно поддержке российских ДМРЛ.
5. Текущий рабочий источник — NOAA/NEXRAD, его нельзя использовать как доказательство качества на российских МРЛ.

---

## 2. Что нужно получить до реализации

Для каждого российского источника требуется:

```text
source_name
правовой статус доступа
условия использования
формат файла
пример реального файла
station id
station coordinates
product type
units
timestamp source
range / resolution
update cadence
mask / nodata semantics
```

Минимальный набор fixtures:

```text
1 dry/no echo case
1 stratiform precipitation case
1 convective case
1 problematic QC case if available
```

---

## 3. Целевые форматы

Приоритет поддержки:

1. ODIM HDF5, если доступен.
2. BUFR с подтвержденными descriptors.
3. NetCDF/CF или CF-Radial, если источник может быть конвертирован.
4. `.vol` или другие proprietary formats — только отдельным adapter с явной документацией.

---

## 4. Реализация

### 4.1. Source adapters

Создать отдельные адаптеры:

```text
RussianBufrAdapter
RussianOdimH5Adapter
RussianLocalArchiveAdapter
```

Не смешивать их с NOAA adapters.

### 4.2. Station catalog

Создать проверяемый каталог:

```text
data/static/radar_stations_ru.json
```

Поля:

```json
{
  "station_id": "...",
  "name": "...",
  "lat": 0.0,
  "lon": 0.0,
  "altitude_m": 0.0,
  "radar_type": "...",
  "source": "...",
  "verified": true
}
```

### 4.3. Product compatibility

Для каждого источника определить, что именно используется:

```text
lowest elevation reflectivity
CMAX
CAPPI
composite
rain rate product
```

Нельзя обучать одну модель на смеси `lowest_elevation_reflectivity` и `composite/CMAX`, если это не отражено в product channel/metadata.

### 4.4. Conversion to common contract

Каждый российский источник должен приводиться к:

```text
RadarFrame.data
RadarFrame.valid_mask
RadarFrame.timestamp_utc
RadarFrame.station
RadarFrame.product
RadarFrame.qc
RadarFrame.provenance
```

и далее использовать тот же 15-минутный dataset/pipeline contract.

---

## 5. QC для российских МРЛ

Минимальный QC checklist:

1. Ground clutter.
2. Anomalous propagation.
3. Beam blockage.
4. Range-dependent quality degradation.
5. Attenuation, если диапазон/частота радара чувствительны.
6. Невалидные сектора.
7. Биологические/неметеорологические цели.
8. Неполные обзоры.
9. Неправильные/дублирующиеся сроки.
10. Единицы и шкала dBZ.

Если доступна только отражаемость без dual-pol, QC будет ограниченным. Это надо явно писать в metadata и model card.

---

## 6. Валидация совместимости с NOAA-контуром

До смешивания источников в обучении выполнить отчет:

```text
NOAA/NEXRAD vs Russian MRL/DMRL comparison
- product type
- grid geometry
- dBZ distribution
- valid fraction
- cadence
- range coverage
- seasonal/event distribution
- QC flags
```

Если различия существенные, обучать отдельную модель или использовать source-specific embedding/channel.

---

## 7. Тесты

Создать:

```text
tests/test_russian_bufr_adapter.py
tests/test_russian_odim_adapter.py
tests/test_station_catalog_ru.py
```

Проверки:

1. Реальный fixture декодируется.
2. Timestamp берется из metadata, не из `mtime`.
3. Station coordinates подтверждены.
4. Units = dBZ или явно конвертированы.
5. `valid_mask` не пустая и не полностью true без причины.
6. Unknown descriptor дает явную ошибку.
7. Визуальная карта совпадает с эталонным продуктом источника.
8. Source-specific metadata записывается в provenance.

---

## 8. Критические замечания

1. Не объявлять поддержку Росгидромета без разрешенного источника и fixtures.
2. Не парсить публичные картинки как production ML-датасет без проверки правового статуса и шкалы.
3. Не смешивать российские и NOAA-данные до валидации совместимости.
4. Не использовать данные, если невозможно восстановить срок, станцию, продукт и единицы.
5. Не обещать прогноз интенсивности осадков без калибровки и/или дождемеров.

---

## 9. Критерий завершения

Этап завершен, если для каждого заявленного российского источника есть:

1. Документированный и разрешенный доступ.
2. Реальные fixtures.
3. Тестируемый adapter.
4. Подтвержденные station metadata.
5. Единый `RadarFrame` output.
6. QC report.
7. Визуальная сверка с эталоном.
8. Решение: обучать общую модель, отдельную модель или source-specific модель.

До выполнения этих условий российский контур остается в статусе `planned`, а не `supported`.