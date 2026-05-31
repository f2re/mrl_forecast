# Этап 8. Экспорт, тесты и эксплуатация

## Цель

Закрыть сквозную проверку системы: от источника до экспортируемого прогноза.

## NetCDF

### Проблема

Текущий экспорт содержит оси `x`, `y`, но не описывает полноценный CRS и
географическую привязку. Для интеграции с GIS и внешними моделями этого
недостаточно.

### Задачи

1. Сделать экспорт CF-compliant.
2. Добавить:
   - grid mapping variable;
   - CRS;
   - station coordinates;
   - UTC time units;
   - lead time;
   - units `dBZ`;
   - source;
   - pipeline version;
   - model id;
   - dataset provenance;
   - QC flags.
3. Проверить чтение файла через xarray.
4. Проверить импорт в GIS-инструменте.

## Автоматические тесты

### Unit

- декодеры;
- QC;
- resampling;
- преобразования CRS;
- метрики;
- uniform field detector;
- metadata schema.

### Integration

- fixture raw file -> grid;
- grid -> dataset;
- dataset -> training smoke;
- checkpoint -> inference;
- inference -> PNG;
- inference -> NetCDF;
- Flask API inventory and model details.

### Live smoke

- AWS список сканов;
- AWS скачивание одного файла;
- AWS декодирование одного файла;
- проверка возраста live-данных.

Live smoke запускать отдельно от быстрых unit-тестов.

## Эксплуатационные задачи

1. Добавить команду `scripts/doctor.sh`.
2. Проверять:
   - Python и зависимости;
   - доступность AWS;
   - writable каталоги;
   - наличие pipeline metadata;
   - совместимость модели;
   - возраст последних данных;
   - свободное место.
3. Добавить структурированные логи.
4. Убрать отладочные `print()` из Flask startup.
5. Исправить macOS-несовместимость `date -d` в `scripts/download.sh`.
6. Зафиксировать версии критичных зависимостей.
7. Добавить CI:
   - compile;
   - unit;
   - integration fixtures;
   - optional scheduled AWS smoke.

## Definition of Done

- `doctor.sh` дает понятный итоговый статус.
- Unit и fixture integration тесты проходят локально и в CI.
- Scheduled live smoke подтверждает AWS.
- NetCDF открывается в xarray и содержит корректную географию.
- Ошибки источника, модели и экспорта диагностируются отдельно.

