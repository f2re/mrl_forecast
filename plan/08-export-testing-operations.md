# Этап 08. Export, тесты и эксплуатационный hardening

## Цель

Закрыть сквозной контур от входного МРЛ-кадра до экспортируемого прогноза так, чтобы результат можно было проверить, воспроизвести и корректно интерпретировать вне веб-интерфейса.

---

## 1. Что уже реализовано

1. Есть `save_forecast_to_netcdf()`.
2. NetCDF содержит переменную `reflectivity` с `units=dBZ`.
3. Есть `crs` с `grid_mapping_name=azimuthal_equidistant`.
4. Есть координата `lead_time`.
5. Есть attrs: `station`, `base_time`, `pipeline_version`, `model_id`, `source`.
6. Есть `scripts/doctor.py`.
7. Есть `scripts/check_aws_source.py`.
8. Есть unit-тесты на export, radar pipeline, dataset pipeline, map visualization, forecast quality, web app.

---

## 2. Незакрытые дефекты

1. В `export_utils.py` default `interval_minutes=10`.
2. Атрибут `description` сейчас звучит как `AI Precipitation Nowcast`, что методически слишком широко для отражаемости.
3. Нет явного `product=experimental_radar_reflectivity_nowcast`.
4. Нет `not_official_warning=true`.
5. Нет `forecast_step_minutes=15`.
6. Нет отдельной координаты `lead_time_minutes`.
7. Нет `valid_time_utc` как явной координаты или attrs.
8. Нет mask variable.
9. `doctor.py` проверяет окружение, но не проверяет совместимость active model/pipeline/time step.

---

## 3. NetCDF contract

### 3.1. Обязательные координаты

```text
time или valid_time_utc: фактические сроки прогноза UTC
lead_time_minutes: [15, 30, 45, 60, ...]
y: local AEQD meters
x: local AEQD meters
```

### 3.2. Обязательные переменные

```text
reflectivity(time, y, x), units=dBZ
valid_mask(time, y, x), optional but required if available
crs
```

### 3.3. Обязательные global attrs

```json
{
  "product": "experimental_radar_reflectivity_nowcast",
  "description": "Experimental radar reflectivity nowcast; not an official warning",
  "units": "dBZ",
  "station": "...",
  "base_time_utc": "...",
  "forecast_step_minutes": 15,
  "pipeline_version": "radar-grid-v2-15min",
  "model_id": "...",
  "model_architecture": "...",
  "source": "...",
  "quality_gate_status": "passed|failed|unknown",
  "not_official_warning": "true",
  "reflectivity_only_no_nwp": "true"
}
```

---

## 4. `doctor.py` hardening

Добавить проверки:

1. Наличие `src/config.py` и `FORECAST_STEP_MINUTES=15`.
2. Совместимость активной модели с текущим `PIPELINE_VERSION`.
3. Наличие writable каталогов:
   - `data/raw`;
   - `data/processed_archive`;
   - `models/registry`;
   - `data/exports`.
4. Наличие хотя бы одного usable model или понятное предупреждение.
5. Наличие хотя бы одного dataset с текущим pipeline или понятное предупреждение.
6. Проверка свободного места.
7. Опциональная проверка AWS через `--check-aws`.
8. Проверка, что legacy 10-minute модель не выбрана как production.

---

## 5. Тесты

Создать/обновить:

```text
tests/test_export_utils.py
tests/test_time_contract.py
tests/test_doctor.py
tests/test_integration_smoke.py
```

Минимальные проверки:

1. NetCDF содержит `forecast_step_minutes=15`.
2. NetCDF содержит `lead_time_minutes=[15,30]` для двух кадров.
3. NetCDF description не называет продукт полноценным precipitation forecast.
4. NetCDF содержит `not_official_warning=true`.
5. `xarray.open_dataset()` открывает файл без ошибки.
6. `doctor.py` возвращает JSON и корректный exit code.
7. `doctor.py` предупреждает о legacy pipeline mismatch.

---

## 6. CI / локальные проверки

Минимальная команда локального разработчика:

```bash
python -m pytest tests/test_time_contract.py \
  tests/test_masks.py \
  tests/test_export_utils.py \
  tests/test_forecast_quality.py \
  tests/test_model_shapes.py
```

Полная проверка:

```bash
python -m pytest
python scripts/doctor.py
```

Live smoke отдельно:

```bash
python scripts/doctor.py --check-aws --station KOKX --date 2024-05-20
python scripts/check_aws_source.py --station KOKX --date 2024-05-20 --decode-one
```

Live smoke не должен быть обязательным для быстрых unit-тестов.

---

## 7. Эксплуатационные правила

1. Любая ошибка источника, модели или export должна иметь отдельный код/сообщение.
2. Не использовать `print()` как основной logging в production path.
3. Сохранять structured logs для ingestion, training, inference, export.
4. Не удалять legacy-модели автоматически; маркировать их несовместимыми.
5. Export из demo-mode должен быть либо запрещен, либо явно помечен `source=demo` и `not_operational=true`.

---

## 8. Критические замечания

1. NetCDF без provenance непригоден для метеорологического аудита.
2. `lead_time` в минутах должен совпадать с PNG labels и API.
3. Если в файле нет station coordinates, GIS-интеграция становится сомнительной.
4. Нельзя оставлять термин `AI Precipitation Nowcast`, если выход — отражаемость dBZ.

---

## 9. Критерий завершения

Этап завершен, если:

1. NetCDF открывается через xarray и содержит корректные CRS/time/provenance attrs.
2. Forecast step в export равен 15 минутам.
3. Export содержит экспериментальный статус и предупреждение, что это не официальный прогноз.
4. `doctor.py` проверяет окружение, каталоги, pipeline/model compatibility и опционально AWS.
5. Быстрые unit-тесты и fixture integration tests проходят локально.
6. Ошибки source/model/export диагностируются раздельно.