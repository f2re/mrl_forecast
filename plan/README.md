# Актуальный план разработки MRL Forecast

Этот каталог содержит рабочий план доработки проекта после аудита метеорологической состоятельности. Старый план по базовой модернизации NOAA/NEXRAD в основном закрыт; текущая задача — привести проект к физически более корректному прогнозу поля радиолокационной отражаемости МРЛ на 1–3 часа с шагом 15 минут при ограничениях CPU и RAM 16 ГБ.

Целевой продукт формулируется строго как **экспериментальный прогноз поля радиолокационной отражаемости / зон радиоэха МРЛ**, а не как официальный прогноз интенсивности осадков и не как штормовое предупреждение.

---

## 1. Что уже реализовано и не должно переписываться без причины

| Блок | Текущий статус | Комментарий |
| --- | --- | --- |
| Граница доверия к данным | Частично закрыто | Есть `RadarFrame`, `RadarSequence`, `status`, `qc`, `provenance`, отдельный `DemoRadarAdapter`; ошибки источника не должны превращаться в синтетику. |
| Версионированный pipeline | Частично закрыто | Есть `radar-grid-v1`, `product=lowest_elevation_reflectivity`, `units=dBZ`, единая сетка 256×256, радиус 250 км. |
| NOAA/AWS ingestion | Базово закрыто | Есть `configure_public_aws_region()`, фильтрация `_MDM`, `check_aws_source.py`. Это оставить как отладочный/референсный источник. |
| География визуализации | Базово закрыто | Есть AEQD → Web Mercator, north-up overlay, range rings, azimuth lines, отказ от неизвестной станции. |
| Датасет и manifest | Частично закрыто | Есть `manifest.json`, metadata, regular segment selection и защита от части temporal leakage. |
| ConvLSTM baseline | Частично закрыто | Есть текущая ConvLSTM, persistence/advection comparison, threshold metrics, uniform-field gate. |
| Registry/UI | Частично закрыто | Модель с `training`/`failed`/`rejected_quality_gate` не должна использоваться как operational. |
| NetCDF export | Частично закрыто | Есть CRS и `lead_time`, но нужно заменить 10-минутный default и усилить metadata. |

---

## 2. Главные незакрытые дефекты

1. В коде всё еще доминирует 10-минутный шаг, а целевая постановка требует 15 минут.
2. `dBZ` используется как обычное числовое поле; физические штрафы нельзя считать напрямую на логарифмической шкале.
3. `valid_mask` хранится в `RadarFrame`, но не используется как вход модели и как mask в loss.
4. `нет эха`, `нет данных`, `masked pixel`, `край радиуса`, `отфильтрованный clutter` фактически смешиваются при обучении.
5. BUFR-путь использует отдельную интерполяцию `scipy.griddata`, а основной pipeline — Py-ART/Barnes2. Это создает domain shift.
6. В отдельных местах timestamp всё еще может браться из `mtime` файла.
7. Текущий advection baseline — глобальный integer shift, а не полноценный локальный optical-flow/block-motion baseline.
8. ConvLSTM + MSE недостаточна для 1–3 часов: она сглаживает сильные ядра и неявно усредняет сценарии.
9. 2–3-часовой прогноз по одному каналу отражаемости должен маркироваться как пониженная/экспериментальная достоверность.
10. UI и экспорт должны явно писать, что это экспериментальный прогноз отражаемости, не официальный прогноз опасных явлений.

---

## 3. Новый порядок выполнения

| Приоритет | Документ | Смысл этапа | Статус |
| --- | --- | --- | --- |
| P0 | [00-audit-baseline.md](00-audit-baseline.md) | Зафиксировать текущее реализованное состояние и долги | Актуализировано |
| P0 | [01-data-trust-boundary.md](01-data-trust-boundary.md) | Перевести весь проект на контракт времени 15 минут | К выполнению |
| P0 | [02-unified-radar-pipeline.md](02-unified-radar-pipeline.md) | Развести `нет эха` и `нет данных`, протащить `valid_mask` в dataset/loss | К выполнению |
| P0 | [03-geospatial-rendering.md](03-geospatial-rendering.md) | Унифицировать BUFR, timestamp, gridding и геометрию | К выполнению |
| P1 | [04-aws-ingestion.md](04-aws-ingestion.md) | Усилить baseline: persistence, global shift, block-motion/optical-flow | К выполнению |
| P1 | [05-dataset-quality.md](05-dataset-quality.md) | Пересобрать датасеты под 15 минут, QC, split, sampling | К выполнению |
| P1 | [06-model-training.md](06-model-training.md) | Реализовать `MRL-PhysLite`: motion + advection + residual ConvGRU | К выполнению |
| P1 | [07-registry-ui-observability.md](07-registry-ui-observability.md) | Registry, API, UI, confidence labels, model card | К выполнению |
| P2 | [08-export-testing-operations.md](08-export-testing-operations.md) | NetCDF, тесты, `doctor`, эксплуатационный hardening | К выполнению |
| P2 | [09-russian-dmrl-track.md](09-russian-dmrl-track.md) | Реальные российские МРЛ/ДМРЛ как отдельный источник | Ожидает fixtures/источник |

---

## 4. Целевая модель

Рабочее имя: `MRL-PhysLite`.

Не делать сразу DGMR/LDCast/диффузию: для CPU и RAM 16 ГБ это нерационально как первый production-контур. Нужна легкая physics-guided модель:

```text
input: reflectivity_norm + valid_mask + range_norm
        |
        v
encoder + ConvGRU temporal core
        |
        +--> MotionHead: u, v
        +--> ResidualHead: growth/decay
        |
        v
DifferentiableAdvection(last_proxy, u, v) + residual
        |
        v
forecast dBZ на +15/+30/+45/+60 ... минут
```

Физический регуляризатор считать на линейном proxy отражаемости, а не напрямую на `dBZ`.

---

## 5. Общий Definition of Done

Версия считается готовой к демонстрации только если:

1. Везде используется `FORECAST_STEP_MINUTES=15`.
2. Dataset, model metadata, API, PNG labels и NetCDF показывают один и тот же 15-минутный шаг.
3. `valid_mask` используется в dataset, входе модели, loss и метриках.
4. `mtime` не является основным источником срока наблюдения.
5. BUFR и operational inference используют один контракт гридирования или явно маркированный fallback.
6. Есть baseline: persistence, global shift, local/block motion или pySTEPS-like optical flow.
7. Модель сравнивается с baseline по lead time и порогам dBZ.
8. Модель не публикуется, если хуже baseline или дает uniform-field anomaly.
9. Горизонты 2–3 часа имеют маркировку пониженной/экспериментальной достоверности.
10. UI и экспорт называют продукт прогнозом отражаемости, а не официальным прогнозом осадков.
11. NetCDF содержит CRS, `lead_time_minutes`, `valid_time_utc`, model/pipeline provenance и экспериментальный статус.
12. Документация позволяет новому разработчику воспроизвести подготовку данных, обучение, evaluation и inference.

---

## 6. Первый актуальный PR

Первым PR делать только безопасную инфраструктурную правку:

1. Добавить `src/config.py`.
2. Ввести `FORECAST_STEP_MINUTES = 15`.
3. Заменить жестко заданные 10 минут в pipeline, web labels, map visualization и NetCDF export.
4. Добавить тесты на `+15/+30/+45/+60`.
5. Обновить metadata dataset/model/export.

После этого пересобрать датасеты. Старые датасеты и модели с 10-минутным контрактом считать legacy.