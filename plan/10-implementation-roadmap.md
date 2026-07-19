# Этап 10. План практической реализации

## Цель

Перевести проект к воспроизводимой системе подготовки радиолокационных данных и экспериментального прогноза радиоэха. Приоритет — работающий результат без избыточной архитектуры.

Принципы:

- **KISS** — минимум обязательных сущностей;
- **DRY** — терминал и веб используют одну прикладную логику;
- **SOLID** — источник, доступ, декодирование, гридирование, датасет и модель разделены;
- небольшое число проверок на критические контракты;
- изменения выполняются последовательно в `main`.

## Зафиксированные решения

1. Canonical grid: `512 × 512`, `1 км`, local AEQD, радиус около `256 км`.
2. Исходные файлы сохраняются без изменения; canonical frame является производным продуктом.
3. `нет эха`, `нет данных`, покрытие, помехи и интерполяция разделены.
4. `mtime` не используется как время наблюдения.
5. Основной вход модели содержит не менее `60 минут` истории.
6. Целевой профиль: `6` входных и `6` выходных сроков с шагом `10 минут`.
7. Профиль `4 × 15 минут` сохраняется как baseline.
8. `MRL-PhysEvolution` отдельно прогнозирует перенос, рост, распад и неопределённость.
9. Обучение запускается из CLI и интерфейса через один job runner.
10. Доступность URL не означает автоматический допуск данных к обучению.
11. Секреты хранятся вне репозитория и не публикуются в health-report.

## Этапы

### P0. Контракт данных и quality masks — реализовано

- canonical contract и source capabilities;
- `npz-radar-quality-v2`;
- `reflectivity`, `valid_mask`, `coverage_mask`, `clutter_mask`, `interpolation_weight`, `timestamps_utc`;
- effective mask в loss, verification и runtime;
- quality masks в UI и NetCDF;
- legacy `.npy` только для совместимости;
- неизвестный timestamp завершает обработку ошибкой;
- DEMO cadence согласуется с cadence модели.

Оставшийся долг: подключать реальные clutter/interpolation fields конкретных продуктов. Пока источник их не предоставляет, применяется явно маркированный бинарный fallback.

### P1. Источники, доступ и каталог — основной контур реализован

Полностью автоматизированы:

- NOAA NEXRAD Level II;
- DWD ODIM HDF5;
- FMI public S3;
- OPERA ORD API и anonymous 24-hour cache;
- DMI Radar STAC API;
- KNMI Open Data API с API key;
- WIS2 Global Cache.

Реализовано:

- общий `RemoteRadarFile`;
- `SourceProbeResult`;
- открытый HTTP и unsigned S3;
- byte-range download test;
- полное скачивание с потоковым SHA-256;
- единый raw archive metadata;
- SQLite-каталог архивов и наблюдений;
- автоматическая индексация;
- concurrent health probe;
- startup probe полностью автоматизированных источников;
- dashboard health-report;
- удаление signed URL query и credential fields из отчётов;
- secure credential store `0600`;
- env override;
- интерактивный скрытый ввод токена;
- порядок регистрации в capabilities и UI;
- generic ODIM archive → canonical dataset path;
- вычисляемая блокировка датасетов от источников с `training_allowed=false`.

Probe-only/manual profiles:

- WMO Weather Radar Database;
- Météo-France Package Radar;
- CEDA NIMROD;
- MeteoSwiss;
- GeoSphere Austria;
- AURA/NCI;
- MetService;
- Taiwan QPESUMS;
- NASA GPM-GV;
- ЦАО NCRadar.

Документация: [`docs/source-access.md`](../docs/source-access.md).

Оставшийся долг P1:

1. проверить реальные prefix/schema для FMI, OPERA и WIS2;
2. выполнить sample download DMI/KNMI;
3. реализовать MeteoSwiss STAC и GeoSphere Data Hub downloader;
4. получить открытый российский raw DMRL/BUFR endpoint либо официальный доступ NCRadar.

### P2. События и выборка — реализовано с fallback

- классы `dry_valid`, `weak_echo`, `precipitation`, `convective`, `severe_core`, `invalid`;
- пороговые площади и тенденции;
- dry/echo 50/50 только в train;
- validation/test сохраняют естественное распределение;
- трёхчасовые chronological groups;
- purged boundaries между train, validation и test;
- выбор эпохи по validation;
- финальный quality gate по independent test;
- `validation_fallback` для малых/legacy-наборов.

Оставшийся долг: многодневные синоптические группы и station holdout.

### P3. Baseline и verification — рабочий набор реализован

- persistence;
- global shift;
- local block motion;
- masked MSE;
- CSI, POD, FAR;
- frequency bias и ETS;
- FSS;
- max dBZ error;
- area bias `20/30/40 dBZ`;
- uniform-field gate.

Quality gate требует превосходства над persistence, global shift и block motion.

### P4. `MRL-PhysEvolution` — рабочая реализация

```text
reflectivity + effective quality mask + range_norm
              |
              v
       lightweight encoder
              |
              v
          ConvGRU core
       /      |       \
  Motion   Growth/Decay  Uncertainty
       \      |       /
     differentiable advection
              |
              v
      reflectivity forecast
```

Реализованы physics-guided loss, диагностические карты и горизонт `0–60 минут`.

Следующий шаг: реальное обучение на многодневном canonical archive и анализ ошибок по типам ситуаций.

### P5. Job runner, CLI и интерфейс — реализовано

- SQLite jobs table и один worker;
- download/prepare/train/catalog jobs;
- единый CLI `python mrl.py`;
- команды `doctor`, `sources`, `download`, `prepare`, `train`, `infer`, `catalog`, `benchmark`, `serve`, `worker`;
- source list/info/probe/configure/sample;
- NOAA/DWD/FMI/OPERA/DMI/KNMI/WIS2 downloads;
- адаптивный интерфейс и dark mode;
- source health и registration guidance;
- слои отражаемости, движения, роста, распада, uncertainty и quality masks;
- HTML/CSS/JavaScript разделены.

### P6. CPU deployment — начато

Реализован CPU benchmark:

- p50/p95/mean latency;
- max RSS;
- число потоков;
- grid, history, horizon и architecture.

Остаются:

- ONNX export;
- ONNX Runtime CPU;
- проверка квантизации;
- отдельный inference dependency set;
- production WSGI;
- ограничение checkpoint path во всех терминальных сценариях.

## Следующий рабочий срез

1. Выполнить live probe и sample download FMI/OPERA/DMI/KNMI/WIS2 на целевой машине.
2. Проверить реальный ODIM decode и source-specific fields.
3. Одобрить только проверенные adapters для train.
4. Построить многодневный canonical archive.
5. Обучить `MRL-PhysEvolution`, выполнить independent test и CPU benchmark.
6. Реализовать MeteoSwiss/GeoSphere downloader.
7. Подключить реальные clutter/interpolation fields.
8. Продолжить поиск российского DMRL/BUFR/NCRadar доступа без подмены визуальными тайлами.
9. После подтверждения качества перейти к ONNX/ONNX Runtime.
