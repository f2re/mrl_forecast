# Этап 10. План практической реализации

## Цель

Перевести проект от демонстрационного ConvLSTM-контура к воспроизводимой системе подготовки радиолокационных данных и экспериментального прогноза радиоэха. Приоритет — работающий результат без избыточной архитектуры.

Принципы реализации:

- **KISS**: минимум обязательных сущностей и зависимостей;
- **DRY**: терминал и веб-интерфейс используют одну прикладную логику;
- **SOLID**: источник, декодирование, гридирование, датасет и модель разделены;
- небольшое число проверок на критические контракты вместо большого набора хрупких тестов;
- изменения выполняются последовательно в `main`.

## Зафиксированные решения

1. Canonical grid: `512 × 512`, `1 км`, локальная AEQD, радиус около `256 км`.
2. Исходные файлы сохраняются без изменения; canonical frame является производным продуктом.
3. `нет эха`, `нет данных`, покрытие, помехи и интерполяция разделены.
4. `mtime` не используется как штатное время МРЛ.
5. Основной вход модели содержит не менее `60 минут` истории.
6. Целевой российский профиль: `6` входных и `6` выходных сроков с шагом `10 минут`.
7. Профиль `4 × 15 минут` сохраняется как совместимый baseline.
8. `MRL-PhysEvolution` отдельно прогнозирует перенос, рост, распад и неопределённость.
9. Обучение запускается из терминала и интерфейса через один job runner.
10. Интерфейс адаптивный, в визуальном языке macOS/iOS, без копирования фирменных элементов Apple.

## Источники данных

### Количественные

- NOAA NEXRAD Level II — legacy и canonical adapters;
- DWD Open Data — ODIM HDF5 DBZH adapter, raw archive downloader и UI job;
- локальные ODIM HDF5/BUFR/NPZ — только при проверенном provenance;
- российские ДМРЛ-BUFR — после получения проверенного открытого файла или endpoint.

### Discovery и визуальный контроль

- WIS2 Global Discovery Catalogue;
- Meteoinfo — visual-only;
- RainViewer — visual-only, короткий оперативный архив.

Источник не получает `training_allowed`, пока не подтверждены единицы, геометрия, время, маски качества и условия использования.

## Этапы

### P0. Контракт данных и адаптеры — реализовано

- canonical contract и capabilities;
- сетка `512 × 512 / 1 км`;
- формат `npz-radar-quality-v2`;
- `reflectivity`, `valid_mask`, `coverage_mask`, `clutter_mask`, `interpolation_weight`, `timestamps_utc`;
- effective mask используется в loss и verification;
- маски проходят через local adapter и operational runtime;
- NetCDF содержит quality masks;
- UI отображает отдельные quality layers;
- legacy `.npy` читается только для совместимости;
- фактический шаг источника используется при формировании последовательностей;
- неизвестный timestamp завершает обработку ошибкой;
- DEMO cadence может быть согласован с cadence активной модели.

Оставшийся долг P0: подключать реальные clutter/interpolation quality fields конкретных радарных продуктов. Пока источник их не предоставляет, применяется явно маркированный бинарный fallback.

### P1. Открытые источники и каталог — реализован основной контур

- NOAA AWS downloader с фильтрацией `_MDM`, SHA-256 и provenance;
- DWD ODIM HDF5 adapter и raw downloader;
- WIS2 discovery;
- Meteoinfo/RainViewer как visual-only;
- SQLite-каталог архивов, отдельных сроков и датасетов;
- индекс по источнику, станции и времени;
- хранение SHA-256, QC и provenance;
- CLI управления каталогом;
- автоматическая индексация после download и prepare;
- перестроение каталога из интерфейса;
- source-specific NOAA/DWD ingest jobs в UI.

Оставшийся долг: найти и проверить открытый российский raw DMRL/BUFR endpoint.

### P2. События и выборка — реализовано с fallback

- классы `dry_valid`, `weak_echo`, `precipitation`, `convective`, `severe_core`, `invalid`;
- пороговые площади и тенденции;
- dry/echo 50/50 только в train;
- validation/test сохраняют естественное распределение;
- sequences объединяются в трёхчасовые chronological groups;
- между train, validation и test удаляется пограничная группа;
- выбор эпохи выполняется по validation;
- окончательный quality gate — по независимому test;
- для старых или слишком малых датасетов применяется явно маркированный `validation_fallback` с overlap gap.

Оставшийся долг: расширить event grouping на многодневные синоптические эпизоды и station holdout.

### P3. Baseline и верификация — реализован рабочий набор

- persistence;
- global shift advection;
- локальный coarse block motion;
- masked MSE;
- CSI, POD, FAR;
- frequency bias и ETS;
- FSS на нескольких пространственных масштабах;
- ошибка максимальной отражаемости;
- bias площади зон `20/30/40 dBZ`;
- uniform-field gate.

Quality gate требует превосходства над persistence, global shift и block motion. Следующий возможный baseline — сторонний optical flow/pySTEPS после проверки вычислительной цены.

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
- NOAA и DWD в одной форме загрузки;
- отмена, журналы и восстановление статусов;
- общий `ModelRuntime` для веба и терминала;
- единый CLI `python mrl.py`;
- команды `doctor`, `sources`, `download`, `prepare`, `train`, `infer`, `catalog`, `benchmark`, `serve`, `worker`;
- адаптивный интерфейс и тёмная тема;
- экраны источников, архива, каталога, обучения, заданий и моделей;
- слои отражаемости, движения, роста, распада, неопределённости и quality masks;
- HTML, CSS и JavaScript разделены.

### P6. CPU deployment — начато

Реализован CPU benchmark, который фиксирует:

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
- запрет произвольного checkpoint path во всех терминальных сценариях.

## Следующий рабочий срез

1. Выполнить end-to-end DWD decode/download/prepare на целевой машине и зафиксировать QC.
2. Построить многодневный canonical archive для обучения.
3. Обучить `MRL-PhysEvolution`, выполнить независимый test quality gate и CPU benchmark.
4. Подключить реальные source-specific clutter/interpolation fields там, где они доступны.
5. Реализовать ONNX/ONNX Runtime после подтверждения качества PyTorch-модели.
6. Добавить station holdout и многодневное event grouping.
7. Продолжить поиск открытого российского DMRL/BUFR endpoint через WIS2 без подмены визуальными тайлами.
