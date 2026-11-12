## Вариант B2. Knowledge Distillation: сжатие ViT для edge-deployment (`b2_distillation`)

После задания 4.2 у вас есть fine-tuned ViT-Tiny на EuroSAT на 5.5M параметров и test accuracy ≥ 90%. Модель нужно задеплоить на edge-устройстве, но 5.5M слишком много, нужна модель поменьше. Один из стандартных подходов — обучить «мелкого» student с помощью soft targets большой модели (teacher). Этот приём называется **knowledge distillation** и лежит в основе таких production-моделей, как DistilBERT, TinyBERT, MobileViT.

**Использует из основной части:** fine-tuned модель и data pipeline из задания 4.2 (`scripts/train_vit.py`, функция `get_dataloaders`, чекпоинт `results/model.pt`).

> Оригинальная статья по теме бонусного задания — [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton, Vinyals & Dean, 2015). 
### Структура файлов и что нужно реализовать

Все файлы задания располагаются в директории `bonus/b2_distillation/`, запускающий скрипт — `scripts/train_distillation.py`:

| Файл | Уже реализовано в шаблоне | Нужно реализовать |
|------|---------------------------|-------------------|
| `_teacher_loader.py` | Весь файл | — |
| `_train_utils.py` | Весь файл | — |
| `student_vit.py` | — | `build_student_vit` |
| `distillation_loss.py` | `DistillationLoss.__init__` | `DistillationLoss.forward` |
| `train_distill.py` | — | `train_student_with_distillation`, `train_student_baseline` |
| `scripts/train_distillation.py` | Весь файл | — |

> Перед запуском B2 необходимо, чтобы задание 4.2 было успешно выполнено и чекпоинт `results/model.pt` был сохранён. `load_teacher` из `_teacher_loader.py` зеркалирует архитектуру из `scripts/train_vit.py`, поэтому любые изменения архитектуры в 4.2 приведут к ошибке загрузки. По той же причине `scripts/train_distillation.py` импортирует `get_dataloaders` из `scripts/train_vit.py` — убедитесь, что эта функция реализована корректно.

---

### Задание B2.1. Student ViT

Реализуйте функцию `build_student_vit` в файле `bonus/b2_distillation/student_vit.py`, собирающую «мелкую» ViT-модель для обучения с нуля через дистилляцию.

**Сигнатура:**

```python
def build_student_vit(
    num_classes: int,
    num_blocks: int = 3,
) -> nn.Module:
```

**Требования:**
- Создайте базовую архитектуру ``vit_tiny_patch16_224`` через ``timm.create_model`` с **`pretrained=False`** и ``num_classes=num_classes`` (параметр `num_classes` автоматически создаст голову `nn.Linear` нужной размерности).
- **Критично:** ``pretrained=False`` обязательно. Student обучается с нуля через дистилляцию — использование предобученных весов смешало бы вклад дистилляции и вклад предобучения, сделав baseline-сравнение (задание B2.3) бессмысленным.
- Обрежьте стек transformer-блоков: в timm они лежат в ``model.blocks`` как `nn.Sequential` из 12 `Block`-модулей. Замените его на новый `nn.Sequential` из первых `num_blocks` элементов:
  ```python
  model.blocks = nn.Sequential(*list(model.blocks)[:num_blocks])
  ```
- При `num_classes < 1` или `num_blocks` вне диапазона `[1, 12]` бросьте `ValueError`.

**Датасет:** не требуется. Автотесты проверяют форму выхода `(B, num_classes)` на случайном входе `(2, 3, 224, 224)`, точное число блоков в `model.blocks`, диапазон числа параметров (1.0M–2.0M для `num_blocks=3`), отсутствие предобученных весов (разные seeds → разные параметры) и валидацию аргументов.

---

### Задание B2.2. Distillation Loss

Реализуйте метод `forward` класса `DistillationLoss` в файле `bonus/b2_distillation/distillation_loss.py` (метод `__init__` с валидацией уже реализован).

**Сигнатура:**

```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0) -> None: ...  # готово

    def forward(
        self,
        student_logits: torch.Tensor,   # (B, C), требует grad
        teacher_logits: torch.Tensor,   # (B, C), без grad (detached)
        labels: torch.Tensor,           # (B,), torch.long
    ) -> torch.Tensor:
```

**Формула (Hinton et al., 2015):**
$$\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}\!\left(\sigma(z_s / T) \,\|\, \sigma(z_t / T)\right) + (1 - \alpha) \cdot \text{CE}(z_s, y)$$

где $z_s, z_t$ — логиты student и teacher, $\sigma$ — softmax, $T$ — температура, $y$ — истинные метки.

**Роль множителя $T^2$:** при повышении температуры softmax становится «размытее», а градиенты KL-члена масштабируются как $\propto 1/T^2$. Умножение на $T^2$ возвращает им масштаб, сравнимый с градиентами CE-члена — это позволяет менять $T$ без необходимости подбирать $\alpha$ заново.

> Функция [`F.kl_div`](https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html) в PyTorch принимает **log-probabilities** первым аргументом и **probabilities** вторым (по умолчанию), а также имеет неочевидное поведение параметра `reduction`. Рекомендуется внимательно ознакомиться с секцией *Parameters* в документации, чтобы избежать проблем с неправильным порядоком аргументов или `reduction`.

**Требования:**
- KL-часть считайте через ``F.kl_div`` с параметром ``reduction="batchmean"`` (математически корректная нормировка — усреднение по размеру батча). Первым аргументом в ``F.kl_div`` передаются **log-probabilities** студента (``F.log_softmax(... / T)``), вторым — **probabilities** teacher'а (``F.softmax(... / T)``).
- CE-часть считайте через ``F.cross_entropy(student_logits, labels)`` — функция принимает **сырые логиты** (без softmax) и int-метки.
- Не забудьте множитель $T^2$ перед KL-членом.
- Валидация форм: если `student_logits.shape != teacher_logits.shape`, или `labels` не 1D, или размер батча в `labels` не совпадает с `student_logits.size(0)` — бросьте `ValueError`.

**Крайние случаи:**
- `alpha=0` → loss равен чистой `F.cross_entropy(student_logits, labels)` (без KL-члена).
- `alpha=1` → loss равен `T² · KL(...)` (без CE-члена).

> **Примечание:** `teacher_logits` должны быть detached вызывающим кодом (через `torch.no_grad()` при прогоне teacher или `.detach()`) — иначе градиенты потекут в teacher и обновят его веса, что противоречит самой идее дистилляции.

**Датасет:** не требуется. Автотесты проверяют совпадение loss'а с чистым CE при `alpha=0`, совпадение с $T^2 \cdot \text{KL}$ при `alpha=1`, корректное масштабирование по $T^2$ на нескольких температурах, скалярность выхода, прохождение градиентов до `student_logits`, численное совпадение с эталоном по сетке `(alpha, T)` и валидацию аргументов.

---

### Задание B2.3. Training loops: с дистилляцией и baseline

Реализуйте две функции обучения в файле `bonus/b2_distillation/train_distill.py`.

#### Часть A. Обучение с дистилляцией

```python
def train_student_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    alpha: float = 0.5,
    temperature: float = 4.0,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, list[float]]:
```

**Схема обучения:**
1. Зафиксировать seed: ``set_seed(seed)`` из `_train_utils.py`.
2. Перенести `student` на `device`, teacher — на `device` + `eval()` + отключить градиенты (`requires_grad = False` на всех параметрах teacher'а). Teacher **не обучается**.
3. Настроить:
   - `criterion_train = DistillationLoss(alpha=alpha, temperature=temperature)`
   - `criterion_eval = nn.CrossEntropyLoss()` (валидация обычным CE: интересует качество классификации, а не величина training-loss).
   - `optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.01)`
   - `scheduler = CosineAnnealingLR(optimizer, T_max=epochs)`
4. В цикле по эпохам:
   - `student.train()`, итерация по `train_loader`:
     - Прогнать `images` через teacher в `torch.no_grad()` → `teacher_logits`.
     - Прогнать `images` через student → `student_logits`.
     - `loss = criterion_train(student_logits, teacher_logits, labels)`.
     - Шаг оптимизатора. Накапливайте взвешенный `running_loss += loss.item() * batch_size`.
   - `scheduler.step()` после эпохи.
   - `val_loss, val_acc = evaluate(student, val_loader, criterion_eval, device)`.
   - Добавить три значения в `history`.
5. Вернуть словарь `history` с ключами `"train_loss"`, `"val_loss"`, `"val_acc"`, каждый — список длины `epochs`.

При `epochs < 1` бросьте `ValueError`.

> Самый известный production-пример дистилляции — [DistilBERT](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019): 6-слойный student, обученный на 12-слойном BERT, сохраняет 97% качества при 40% параметров и 60% скорости.

#### Часть B. Baseline-обучение

```python
def train_student_baseline(
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, list[float]]:
    ...
```

**Требования:**
- Полностью повторите training-recipe из части A (тот же `AdamW` с `weight_decay=0.01`, тот же `CosineAnnealingLR`, тот же `set_seed(seed)`), но:
  - Уберите teacher из пайплайна.
  - Замените `DistillationLoss` на `nn.CrossEntropyLoss` в training-loss.
- Формат возвращаемого `history` — идентичен части A.
- При `epochs < 1` бросьте `ValueError`.

> **Зачем два почти одинаковых training loop'а?** Чтобы корректно сравнить вклад именно дистилляционного сигнала, нужно изолировать его от всех остальных факторов: оптимизатор, scheduler, инициализация, число эпох должны быть идентичными. Любое различие в результатах можно будет приписать именно дистилляции. Такой controlled baseline используется для корректного сравнения любого нового метода обучения.

**Датасет:** для автотестов не требуется — тесты используют tiny-модель (`AdaptiveAvgPool2d + Linear(3, 10)`) и синтетические данные `torch.randn(8, 3, 224, 224)`. Проверяются: успешное выполнение 1 эпохи для обеих функций, правильная длина `history`, конечность всех значений (отсутствие NaN/Inf).

---

### Запуск и артефакты

Запуск (выполняется в Colab с GPU или локально при наличии GPU):
```bash
python -m scripts.train_distillation --epochs 10 --batch-size 64 --alpha 0.5 --temperature 4.0
```

Скрипт `scripts/train_distillation.py` **полностью предоставлен**: он загружает teacher через `load_teacher`, собирает dataloaders через `get_dataloaders` из задания 4.2, вызывает ваши `train_student_with_distillation` и `train_student_baseline` на одном и том же student-архитектурном классе (оба с `num_blocks=3`), и сохраняет в `results/bonus/b2_distillation/`:

- `metrics_distill.json` — конфигурация, число параметров student'а, test loss/accuracy и `history` для обоих режимов (distill и baseline), времена обучения.
- `student_model.pt` — `state_dict` student'а, обученного **с дистилляцией**.
- `loss_curves.png` — train/val loss для обоих режимов.
- `accuracy_curves.png` — val accuracy для обоих режимов (главный визуальный результат эксперимента).

**Ожидаемое время на T4:** ~30–35 минут.

Автотесты проверяют, что distilled student достигает test accuracy > 50% — это грубая проверка того, что обучение действительно запускалось и сходилось (случайное угадывание на EuroSAT даёт ~10%).

> Однако не стоит воспринимать дистилляцию как универсальный вариант. Систематическое исследование того, *когда* дистилляция действительно помогает, а когда baseline не уступает приведено в [Does Knowledge Distillation Really Work?](https://arxiv.org/abs/2106.05945) (Stanton et al., 2021, NeurIPS). 

**Для сдачи приложите к репозиторию** содержимое папки `results/bonus/b2_distillation/` (все четыре артефакта), полученное после реального запуска. Автотесты проверяют существование `metrics_distill.json` с нужной структурой, загружаемость `student_model.pt` как state-dict с характерными для ViT ключами (`blocks.0.attn.qkv.weight`) и accuracy > 50% для distilled student.