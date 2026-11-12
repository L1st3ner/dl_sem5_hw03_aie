## Вариант B4. Multi-Task Learning: классификация + pseudo-NDVI на EuroSAT (`b4_multitask`)

Вам нужно задеплоить на борту спутника две модели: классификатор типа земной поверхности (лес, поле, вода, застройка — 10 классов) и регрессор для оценки индекса растительности по RGB-снимку. Бортовой компьютер не потянет два отдельных ViT одновременно, так как ограничены память, вычисления и энергопотребление. Решено использовать один общий backbone, который считает признаки один раз, и две лёгкие задачные головы поверх него. В этом задании вы возьмёте fine-tuned ViT-Tiny из задания 4.2, удалите его классификационную голову и прикрепите две новые: классификацию land cover и регрессию индекса растительности.

> Классический обзор multi-task learning в глубоком обучении приведен в [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/abs/1706.05098) (Sebastian Ruder, 2017). Ваша архитектура (общий backbone + две головы) — это канонический пример hard parameter sharing из раздела 3.

**Использует из основной части:** fine-tuned модель из задания 4.2 (чекпоинт `results/model.pt`), timm'овский `vit_tiny_patch16_224` в качестве backbone-архитектуры, EuroSAT как источник изображений.

### Структура файлов и что нужно реализовать

Все файлы задания располагаются в директории `bonus/b4_multitask/`, запускающий скрипт — `scripts/train_multitask.py`:

| Файл | Уже реализовано в шаблоне | Нужно реализовать |
|------|---------------------------|-------------------|
| `_model_utils.py` | Весь файл | — |
| `_ndvi.py` | Весь файл | — |
| `_train_utils.py` | Весь файл | — |
| `multitask_dataset.py` | `EuroSATMultiTaskDataset.__init__` | `__len__`, `__getitem__` |
| `multitask_loss.py` | `MultiTaskLoss.__init__` | `MultiTaskLoss.forward` |
| `multitask_model.py` | — | `MultiTaskViT.__init__`, `forward` |
| `train_multitask.py` | — | `train_multitask` |
| `scripts/train_multitask.py` | Весь файл | — |

> Перед запуском B4 необходимо, чтобы задание 4.2 было успешно выполнено и чекпоинт `results/model.pt` был сохранён. Скрипт `scripts/train_multitask.py` загружает этот чекпоинт как **инициализацию** backbone'а. Если чекпоинта нет, обучение не запустится.

---

### Задание B4.1. EuroSATMultiTaskDataset

Реализуйте методы `__len__` и `__getitem__` в классе `EuroSATMultiTaskDataset` (файл `bonus/b4_multitask/multitask_dataset.py`). Метод `__init__` уже реализован и сохраняет `self._dataset`, `self._indices`, `self._transform`.

**Сигнатура:**

```python
class EuroSATMultiTaskDataset(Dataset):
    def __init__(
        self,
        dataset,                    # HuggingFace Dataset или list[dict]
        indices,                    # какие строки входят в этот сплит
        transform=None,             # Callable[PIL.Image → FloatTensor (3,H,W)]
    ):
        ...  # готово

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ...
```

**Требования:**
- `__len__` возвращает число индексов в этом сплите (длину `self._indices`).
- `__getitem__` возвращает словарь с тремя ключами:
  - `"image"`: `FloatTensor (3, H, W)` — изображение после `self._transform` (или просто `TF.to_tensor` если transform отсутствует);
  - `"cls_target"`: `LongTensor ()` (0-мерный скаляр) — метка класса из `sample["label"]`;
  - `"reg_target"`: `FloatTensor ()` (0-мерный скаляр) — pseudo-NDVI, посчитанный через `compute_pseudo_ndvi` из `_ndvi.py`.
- pseudo-NDVI должен считаться от **ненормализованного** `[0, 1]` тензора. Если `self._transform` содержит `Normalize` (как в `scripts/train_multitask.py`), то применять его **до** вычисления NDVI нельзя — NDVI от нормализованных значений (которые могут быть отрицательными после вычитания среднего) даст мусор. Правильный порядок:
  1. `raw_tensor = TF.to_tensor(image_pil)` — чистый `[0, 1]` тензор.
  2. `ndvi = compute_pseudo_ndvi(raw_tensor)` — считаем NDVI от него.
  3. `image_tensor = self._transform(image_pil)` — отдельно применяем полный transform к PIL-изображению для модели (он уже содержит и `ToTensor`, и `Normalize`).
  4. Возвращаем `image_tensor` и скаляр `ndvi`.
- `sample["image"]` — это `PIL.Image`; `sample["label"]` — `int`. Доступ: `self._dataset[self._indices[idx]]`.

> NDVI (Normalized Difference Vegetation Index) — стандартный индекс растительности в дистанционном зондировании, определяемый как $(\text{NIR} - \text{Red}) / (\text{NIR} + \text{Red})$. Настоящий NDVI требует ближнего инфракрасного канала (NIR), которого нет в RGB-снимках. Краткое объяснение физики NDVI приведено в [Measuring Vegetation](https://science.nasa.gov/earth/earth-observatory/measuring-vegetation-ndvi-evi/). Аппроксимация NDVI из RGB в виде pseudo-NDVI — это вынужденный компромисс для бортовых камер, у которых нет NIR-сенсора.

**Датасет для тестов:** автотесты используют список из 8 синтетических PIL-изображений 64×64 со случайными `uint8`-пикселями (без `transform`). Проверяется: структура словаря и типы, диапазон `reg_target` ∈ [−1, 1], детерминированность.

---

### Задание B4.2. MultiTaskViT

Реализуйте класс `MultiTaskViT` в файле `bonus/b4_multitask/multitask_model.py` — обёртку над общим backbone'ом с двумя задачными головами.

**Сигнатура:**

```python
class MultiTaskViT(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,     # forward(x) → (B, d_model)
        d_model: int,
        num_classes: int = 10,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]: ...
```

**Требования:**
- В `__init__`:
  - Вызовите `super().__init__()`.
  - При `d_model < 1` или `num_classes < 1` бросьте `ValueError`.
  - Сохраните `backbone` как атрибут модуля (`self.backbone = backbone`) — это важно, иначе `nn.Module` не зарегистрирует его параметры и backbone не будет обучаться.
  - Создайте две независимые линейные головы: `self.cls_head = nn.Linear(d_model, num_classes)` и `self.reg_head = nn.Linear(d_model, 1)`.
- В `forward`:
  - `features = self.backbone(x)` — ожидается `(B, d_model)`.
  - Верните словарь с ключами `"cls_logits"` (форма `(B, num_classes)`) и `"reg_pred"` (форма `(B, 1)`).

**Датасет:** не требуется. Автотесты используют синтетический `_TinyBackbone` и случайные входы `torch.rand(4, 3, 32, 32)`. Проверяются: структура выхода, конечность значений, общий градиентный путь через backbone, численное совпадение с эталоном на одинаковых весах.

---

### Задание B4.3. MultiTaskLoss

Реализуйте метод `forward` класса `MultiTaskLoss` в файле `bonus/b4_multitask/multitask_loss.py` (метод `__init__` с валидацией `alpha ∈ [0, 1]` уже реализован).

**Сигнатура:**

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha: float = 0.5) -> None: ...  # готово

    def forward(
        self,
        cls_logits: torch.Tensor,   # (B, C), сырые логиты
        reg_pred: torch.Tensor,     # (B, 1)
        cls_target: torch.Tensor,   # (B,), torch.long
        reg_target: torch.Tensor,   # (B,), torch.float
    ) -> torch.Tensor:              # 0-мерный скаляр
```

**Формула:**
$$\mathcal{L} = \alpha \cdot \operatorname{CE}(\text{cls\textunderscore logits}, \text{cls\textunderscore target}) + (1 - \alpha) \cdot \operatorname{MSE}(\text{reg\textunderscore pred}, \text{reg\textunderscore target})$$

**Требования:**
- CE-часть: `F.cross_entropy(cls_logits, cls_target)` — функция принимает **сырые логиты** (без softmax) и `int`-метки.
- MSE-часть: `F.mse_loss(reg_pred.squeeze(-1), reg_target)` — `squeeze(-1)` обязателен, чтобы привести `(B, 1)` к `(B,)`, иначе broadcast `(B, 1)` vs `(B,)` даст матрицу `(B, B)` и loss будет бессмысленным.
- Итог: `alpha * ce + (1 - alpha) * mse`.

**Крайние случаи:**
- `alpha=0` → loss равен чистой `F.mse_loss(reg_pred.squeeze(-1), reg_target)`.
- `alpha=1` → loss равен чистой `F.cross_entropy(cls_logits, cls_target)`.

> В начале обучения CE и MSE могут сильно различаться (в 20-200 раз). В этом случае в обучении возникает перекос -- оптимизатор фактически учит только одну компоненту. Это известная проблема multi-task learning — loss balancing. В продакшне применяют динамическое взвешивание, либо нормализуют каждую loss по её начальному масштабу.

> Метод динамического взвешивания loss'ов через неопределённость задач описан в [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) (Kendall, Gal & Cipolla, 2018, CVPR). Идея в том, чтобывместо ручного подбора $\alpha$ модель сама учит скаляры $\sigma_1, \sigma_2$ для каждой задачи, автоматически балансируя градиенты. В задании мы фиксируем $\alpha$, но в продакшне подход Kendall et al. является де-факто стандартом.

**Датасет:** не требуется. Автотесты используют синтетические тензоры `cls_logits = torch.randn(8, 10)`, `reg_pred = torch.randn(8, 1)`, `cls_target = torch.randint(0, 10, (8,))`, `reg_target = torch.randn(8)`. Проверяются: совпадение с чистым MSE при `alpha=0`, с чистым CE при `alpha=1`, скалярность выхода, численное совпадение с эталоном по сетке `alpha ∈ {0.0, 0.2, 0.5, 0.8, 1.0}`, валидация `alpha` вне `[0, 1]`.

---

### Задание B4.4. Training loop `train_multitask`

Реализуйте функцию `train_multitask` в файле `bonus/b4_multitask/train_multitask.py`.

**Сигнатура:**

```python
def train_multitask(
    model: nn.Module,           # MultiTaskViT
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    alpha: float = 0.5,
    device: str = "cpu",
) -> dict[str, list[float]]:
```

**Схема обучения:**
1. Валидация: при `epochs < 1` или `lr <= 0` бросьте `ValueError`.
2. `model.to(device)`.
3. Настроить:
   - `optimizer = torch.optim.Adam(model.parameters(), lr=lr)` — простой Adam без weight decay и scheduler'а: multi-task обучение само по себе сложное, добавлять ещё один фактор нежелательно.
   - `criterion = MultiTaskLoss(alpha=alpha)`.
4. В цикле по эпохам:
   - `model.train()`.
   - Итерация по `train_loader`. Каждый батч — словарь с ключами `"image"`, `"cls_target"`, `"reg_target"`:
     - Перенести три тензора на `device`.
     - `out = model(images)` → словарь с `"cls_logits"` и `"reg_pred"`.
     - `loss = criterion(out["cls_logits"], out["reg_pred"], cls_t, reg_t)`.
     - `optimizer.zero_grad(set_to_none=True)` → `loss.backward()` → `optimizer.step()`.
     - Накапливайте взвешенный `running_loss += loss.item() * images.size(0)`.
   - `epoch_train_loss = running_loss / total_samples`.
   - `val_metrics = evaluate_multitask(model, val_loader, device=device)` из `_train_utils.py`.
   - Добавьте три значения в `history`: `train_loss`, `val_cls_acc`, `val_reg_mse`.
5. Верните словарь `history` с ключами `"train_loss"`, `"val_cls_acc"`, `"val_reg_mse"`, каждый — список длины `epochs`.

**Датасет:** на данном этапе не используется. Для автотестов используется 16 синтетических PIL-изображений 32×32, обёрнутых в `EuroSATMultiTaskDataset` и `DataLoader` с `_TinyBackbone`-моделью. Проверяется: структура и длина `history`, конечность всех значений.

---

### Запуск и артефакты

Для запуска используйте Colab или локальный GPU:
```bash
python -m scripts.train_multitask --epochs 10
```

**Ожидаемое время на T4:** 8–12 минут.

Подумайте, какую экономию по параметрам даёт multi-task подход по сравнению с двумя отдельными моделями? А по инференс-времени на batch?

> Практический разбор совмещения multi-task learning и edge deployment приведен в [MTL-Split: Multi-Task Learning for Edge Devices using Split Computing](https://arxiv.org/abs/2407.05982) (Capogrosso et al., DAC 2024).

**Для сдачи приложите к репозиторию** содержимое папки `results/bonus/b4_multitask/` (все три артефакта), полученное после реального запуска. Автотесты проверяют существование `metrics.json` с ключами `cls_accuracy`, `reg_mse`, `alpha`, загружаемость `multitask_model.pt` как state-dict с ключами `cls_head.weight` и `reg_head.weight`, и `cls_accuracy > 0.5`.