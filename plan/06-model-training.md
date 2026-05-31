# Этап 06. Новая модель `MRL-PhysLite`

## Цель

Добавить физически более корректную и вычислительно легкую модель прогноза отражаемости МРЛ, пригодную для CPU и RAM 16 ГБ.

Целевая модель не должна называться полноценной атмосферной PINN. Корректная формулировка: **physics-guided radar nowcasting model**. Она использует физически мотивированную адвекцию и остаточную эволюцию, но не решает полную систему уравнений атмосферы.

---

## 1. Что уже реализовано

1. Есть ConvLSTM baseline в `train_nowcasting_model.py`.
2. ConvLSTM поддерживает `input_length`, `target_length`, `hidden_channels`.
3. Есть авторегрессионная prediction phase.
4. Есть MSE training loop.
5. Есть quality gate против persistence/advection.
6. Есть сохранение checkpoint и metadata.

---

## 2. Почему текущей ConvLSTM недостаточно

1. MSE-only обучение сглаживает поля и занижает сильные ядра.
2. Модель не видит `valid_mask`.
3. Модель не имеет явного блока переноса радиоэха.
4. В одном и том же hidden state смешиваются перенос, рост, распад и шум.
5. Для 1–3 часов по одной отражаемости модель должна явно различать:
   - перенос уже существующего эха;
   - остаточный рост/затухание;
   - неопределенность длинных lead times.

---

## 3. Целевая архитектура

Рабочее имя: `MRL-PhysLite`.

```text
Input [B, T, C, H, W]
  C:
    0 reflectivity_norm
    1 valid_mask
    2 range_norm
        |
        v
FrameEncoder CNN
        |
        v
TemporalCore ConvGRU
        |
        +--> MotionHead -> [B, T_out, 2, H, W]
        |
        +--> ResidualHead -> [B, T_out, 1, H, W]
        |
        v
DifferentiableAdvection(last_proxy, motion)
        |
        v
advected_proxy + residual_proxy
        |
        v
forecast_reflectivity_norm [B, T_out, 1, H, W]
```

---

## 4. Физически корректная переменная для регуляризации

Нельзя считать физический advective residual напрямую на `dBZ`, потому что `dBZ` — логарифмическая шкала.

Использовать proxy:

```python
def dbz_to_proxy(dbz):
    z_linear = torch.pow(10.0, dbz / 10.0)
    return torch.log1p(z_linear / scale)
```

или более простой нормированный proxy для первого этапа, но в документации явно указать, что это не масса воды и не QPE.

---

## 5. Компоненты модели

### 5.1. `src/models/components.py`

Реализовать:

```text
ConvBlock
FrameEncoder
ConvGRUCell
ConvGRU
MotionHead
ResidualHead
DifferentiableAdvection
```

### 5.2. `DifferentiableAdvection`

Использовать `torch.nn.functional.grid_sample`.

Требования:

1. Поддержка batch.
2. Поддержка нескольких lead times.
3. Нормировка координат в диапазон `[-1, 1]`.
4. Явная обработка out-of-domain через padding.
5. Unit-test: одиночная ячейка переносится на заданный пиксель.

### 5.3. `src/models/physlite.py`

Класс:

```python
class MRLPhysLite(nn.Module):
    def __init__(self, input_channels=3, base_channels=16, hidden_channels=24, output_steps=4):
        ...

    def forward(self, x):
        ...
        return forecast, diagnostics
```

Diagnostics:

```python
{
  "motion": motion,
  "residual": residual,
  "advected": advected,
}
```

---

## 6. Loss

Создать `src/losses.py`.

Базовый состав:

```text
L = L_masked_weighted_huber
  + lambda_grad * L_gradient
  + lambda_adv * L_advection_residual
  + lambda_heavy * L_heavy_echo
```

Стартовые веса:

```text
lambda_grad = 0.05
lambda_adv = 0.10
lambda_heavy = 0.50
```

### 6.1. Heavy echo weighting

Сильные зоны не должны исчезать ради улучшения MSE.

```python
weight = 1.0 + alpha20 * (target_dbz >= 20) + alpha30 * (target_dbz >= 30)
```

### 6.2. Gradient loss

Нужен для сохранения границ зон радиоэха.

Считать только по валидной области или с осторожным mask erosion.

### 6.3. Advection residual loss

Стимулирует модель не заменять физический перенос произвольной генерацией.

Не делать этот штраф слишком жестким: рост/затухание конвекции реальны.

---

## 7. Training strategy

### 7.1. Первый запуск

Только 1 час:

```bash
python src/train_nowcasting_model.py \
  --architecture physlite \
  --horizon 1h \
  --data-dirs data/processed_archive/<dataset> \
  --batch-size 1 \
  --epochs 30
```

### 7.2. CPU-настройки

```text
base_channels = 16
hidden_channels = 16 или 24
batch_size = 1–2
num_workers = 0–2
early_stopping_patience = 5
```

### 7.3. Model selection

Выбирать модель не по `val_loss` alone, а по composite score:

```text
masked_loss
CSI@20/30 dBZ
FAR@20/30 dBZ
max_dbz_error
area_bias@20/30 dBZ
comparison with block-motion baseline
```

---

## 8. Горизонты 2–3 часа

2h/3h не делать основным продуктом до закрытия 1h.

Правила:

```text
0–60 мин: основной экспериментальный nowcast
60–120 мин: пониженная достоверность
120–180 мин: экспериментальный сценарий, не предупреждение
```

В model metadata хранить:

```json
"confidence_policy": {
  "0_60_min": "normal_experimental",
  "60_120_min": "reduced",
  "120_180_min": "experimental_low_confidence"
}
```

---

## 9. Файлы

Создать:

```text
src/models/__init__.py
src/models/components.py
src/models/physlite.py
src/losses.py
```

Изменить:

```text
src/train_nowcasting_model.py
src/web_app.py
src/forecast_quality.py или src/verification.py
scripts/train.sh
templates/index.html
```

---

## 10. Тесты

Создать:

```text
tests/test_model_shapes.py
tests/test_differentiable_advection.py
tests/test_losses.py
```

Проверки:

1. `MRLPhysLite` принимает `[B,T,3,H,W]`.
2. Output shape `[B,T_out,1,H,W]` для `T_out=4,8,12`.
3. Output bounded в `[0,1]` или документированном диапазоне.
4. Advection переносит synthetic blob в ожидаемом направлении.
5. Masked loss игнорирует невалидные пиксели.
6. Heavy echo weight увеличивает штраф за ошибку в сильном ядре.
7. Empty input не дает NaN.

---

## 11. Критические замечания

1. Не удалять ConvLSTM: она остается baseline.
2. Не внедрять DGMR/LDCast до появления GPU и большого локального архива.
3. Не заявлять физическое сохранение массы по dBZ.
4. Не публиковать PhysLite, если она не обгоняет baseline.
5. Не обучать 3h как основной продукт до стабильного качества 1h.

---

## 12. Критерий завершения

Этап завершен, если:

1. `MRLPhysLite` реализована отдельным модулем.
2. Есть differentiable advection и motion diagnostics.
3. Loss использует mask и heavy echo weighting.
4. 1h-модель обучается на новом 15-минутном masked dataset.
5. Evaluation report показывает сравнение с persistence/global shift/block-motion.
6. Model registry сохраняет architecture, loss config, pipeline version, horizon и confidence policy.
7. Модель не публикуется без прохождения quality gate.