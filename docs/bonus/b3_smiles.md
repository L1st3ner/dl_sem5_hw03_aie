## Вариант B3. Предсказание Property по SMILES: Transformer vs baseline на маленьком датасете (`b3_smiles_property`)

В computational chemistry одна из базовых задач — это предсказание физико-химических свойств молекулы по её структурной формуле в формате SMILES. Используя `SmilesTokenizer` (задание 1.2) и `TransformerEncoder` (задание 3.2), соберите end-to-end пайплайн предсказания log-растворимости молекулы в воде (logS): токенизация → Transformer encoder → masked mean pooling → линейная регрессионная голова. Параллельно обучите простой baseline по тому же рецепту (Adam, MSELoss, одинаковое число эпох). Сравните обе модели на отложенной test-выборке по MAE, RMSE и R².

> Датасет ESOL для этого варианта взят из статьи [ESOL: Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x) (Delaney, 2004, JCIM). Место ESOL в современном ландшафте молекулярных бенчмарков (MoleculeNet) и обзор подходов к предсказанию свойств по SMILES приведены в [MoleculeNet: A Benchmark for Molecular Machine Learning](https://arxiv.org/abs/1703.00564) (Wu et al., 2018).

**Использует из основной части:** `SmilesTokenizer` из задания 1.2, `TokenEmbeddingPipeline` из задания 1.5 (неявно, внутри `TransformerEncoder`), `TransformerEncoderBlock` и `TransformerEncoder` из заданий 3.1–3.2.

### Структура файлов и что нужно реализовать

Все файлы задания располагаются в директории `bonus/b3_smiles/`, запускающий скрипт — `scripts/train_smiles.py`:

| Файл | Уже реализовано в шаблоне | Нужно реализовать |
|------|---------------------------|-------------------|
| `data/esol.csv` | Датасет | — |
| `_esol_data.py` | Весь файл | — |
| `_train_utils.py` | Весь файл  | — |
| `smiles_dataset.py` | `ESOLDataset.__init__` | `ESOLDataset.__len__`, `ESOLDataset.__getitem__` |
| `smiles_model.py` | — | `SmilesTransformerRegressor`, `BagOfTokensBaseline` |
| `train_smiles.py` | — | `train_regressor` |
| `scripts/train_smiles.py` | Весь файл | — |

> **Замечание о сплите:** `load_esol` делит датасет на три части — train (~80%), val (~10%), test (~10%). Val-сплит используется для мониторинга сходимости (кривые на `loss_curves.png`), test-сплит — только для финальной оценки в `metrics.json`. На test не производится подбор гиперпараметров и ранняя остановка, поэтому метрики на нём корректно отражают обобщающую способность модели.

> **Замечание о словаре:** `SmilesTokenizer` в `scripts/train_smiles.py` обучается (`fit`) на полном корпусе SMILES до разбиения на сплиты. Так как набор токенов — свойство грамматики SMILES, а не конкретной выборки, утечкой данных это не является. 

---

### Задание B3.1. Датасет `ESOLDataset`

Реализуйте методы `__len__` и `__getitem__` в классе `ESOLDataset` (файл `bonus/b3_smiles/smiles_dataset.py`).

**Сигнатура:**

```python
class ESOLDataset(Dataset):
    def __init__(self, smiles: list[str], targets: list[float], tokenizer):
        # уже реализовано: self.smiles, self.targets, self.tokenizer

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

**Требования:**
- `__len__` возвращает число молекул в датасете.
- `__getitem__` токенизирует SMILES-строку `self.smiles[idx]` через `self.tokenizer.encode(..., add_special=True)` (добавляет `<bos>` и `<eos>`) и возвращает кортеж:
  - `input_ids`: `LongTensor [L]` — одномерный тензор id-шников токенов,
  - `target`: `FloatTensor []` — **нуль-мерный** скаляр с z-нормализованным таргетом (нормализация уже выполнена в `load_esol`, студенту ничего пересчитывать не нужно).
- Таргеты в `self.targets` уже являются `float`'ами — просто оберните в `torch.tensor(..., dtype=torch.float)`.
- Если `tokenizer.encode` вернул пустой список (битая SMILES), бросьте `ValueError` с указанием проблемной строки **и** индекса — это сильно упростит отладку.

**Датасет:** `bonus/b3_smiles/data/esol.csv` — ESOL (Delaney, 2004), 1128 молекул, колонки `smiles` и `measured log solubility in mols per litre`. Загружается и делится на train/val/test через `load_esol` (уже реализовано). Автотесты проверяют корректность длины сплитов, форму и dtype возвращаемых тензоров, присутствие специальных токенов, обработку пустой токенизации.

---

### Задание B3.2. Модели для регрессии

Реализуйте два класса в файле `bonus/b3_smiles/smiles_model.py`: главную модель на основе Transformer'а и простой baseline. Обе модели должны иметь одинаковую сигнатуру `forward`, чтобы их можно было подставлять в одну и ту же функцию обучения.

#### B3.2a. `SmilesTransformerRegressor`

**Сигнатура:**

```python
class SmilesTransformerRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 2,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
```

**Требования:**
- Архитектура: `TransformerEncoder` (из задания 3.2) → masked mean pooling → `nn.Linear(d_model, 1)` без финальной активации (регрессия).
- **Критично:** `TransformerEncoder` уже содержит внутри `TokenEmbeddingPipeline` (эмбеддинги + позиционное кодирование), поэтому отдельный `nn.Embedding` в `__init__` создавать не нужно. Передайте `vocab_size`, `d_model`, `max_len`, `pad_idx` прямо в конструктор `TransformerEncoder`.
- Masked mean pooling: свернуть выход энкодера `[B, L, d_model]` в `[B, d_model]` по размерности длины, усредняя **только по непаддинговым позициям**:
  ```python
  mask = attention_mask.unsqueeze(-1).to(hidden.dtype)       # [B, L, 1]
  pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
  ```
  Приведение маски к `hidden.dtype` важно: без него умножение `long × float` упадёт с type mismatch.
- `forward` возвращает `FloatTensor [B, 1]` — z-нормализованную предсказанную log-растворимость.

#### B3.2b. `BagOfTokensBaseline`

**Сигнатура:**

```python
class BagOfTokensBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        pad_idx: int = 0,
    ):
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

**Требования:**
- Архитектура: `nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)` → сумма по размерности длины (`sum(dim=1)`) → `nn.Linear(embed_dim, 1)`. `padding_idx` гарантирует, что эмбеддинг паддинга — нулевой вектор, который не влияет на сумму, — поэтому `attention_mask` можно игнорировать.
- При `vocab_size < 1` или `embed_dim < 1` бросьте `ValueError`.
- Baseline умышленно игнорирует порядок токенов: `CCO` и `OCC` дадут одинаковое предсказание.
- `forward` возвращает `FloatTensor [B, 1]` — той же формы, что у `SmilesTransformerRegressor`.

**Датасет:** не требуется. Автотесты для обеих моделей проверяют форму выхода `(B, 1)` на случайных входах, число обучаемых параметров в разумном диапазоне, инициализацию (разные seeds → разные веса), валидацию аргументов, инвариантность baseline к перестановке токенов.

---

### Задание B3.3. Функция обучения `train_regressor`

Реализуйте универсальную функцию обучения регрессора в файле `bonus/b3_smiles/train_smiles.py` — она должна одинаково работать и с Transformer'ом, и с baseline'ом.

**Сигнатура:**

```python
def train_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, list[float]]:
```

**Требования:**
- Оптимизатор: `Adam(model.parameters(), lr=lr)`. Loss: `nn.MSELoss()`.
- В начале функции перенесите модель на `device` (`model.to(device)`). Каждый батч тоже переносите на `device` внутри цикла.
- Каждый батч из `train_loader`/`val_loader` — словарь с ключами `input_ids`, `attention_mask`, `targets` (см. `esol_collate_fn`).
- `loss = criterion(preds.squeeze(-1), targets)`.
- Используйте `optimizer.zero_grad(set_to_none=True)` — это чуть быстрее классического `zero_grad()` и даёт более понятные ошибки при отладке.
- Train-loss за эпоху считается как средневзвешенное по числу примеров:
  ```python
  running_sq += loss.item() * targets.numel()
  epoch_train_mse = running_sq / total_samples
  ```
- Val-loss считается через готовую функцию `evaluate_mse(model, val_loader, device=device)` из `_train_utils.py` — она возвращает MSE на валидации в z-единицах.
- При `epochs < 1` или `lr <= 0` бросьте `ValueError`.
- Возврат: словарь `{"train_loss": [...], "val_loss": [...]}`, каждый список длины `epochs`. Именно эта структура ожидается скриптом `scripts/train_smiles.py` для сохранения в `metrics.json["train_history"]` и построения `loss_curves.png`.

> Почему таргет нормализуют и почему статистики считают только по train кратко и с примерами описано в [документации scikit-learn: StandardScaler](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling) и в разделе [Common pitfalls: data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

**Датасет:** ESOL. Автотесты проверяют корректность сигнатуры, монотонность убывания train-loss на смоук-тесте (5 эпох), валидацию аргументов, корректную обработку `device="cpu"`, восстановление `model.train()` после внутреннего вызова `evaluate_mse` (который переключает модель в `eval`-режим).

---

### Запуск и артефакты (задание B3.4)

Запустите готовый скрипт из корня репозитория или в Google Colab. Скрипт обучает обе модели по одинаковому рецепту, оценивает на test-сплите и сохраняет артефакты.

```bash
python -m scripts.train_smiles --epochs 50
```

В папке `results/bonus/b3_smiles/` должны появиться:

| Файл | Содержание |
|------|-----------|
| `metrics.json` | MSE и R² в z-единицах, RMSE и MAE в единицах logS, число параметров, размеры сплитов, `config` запуска, `train_history` для обеих моделей |
| `transformer_model.pt` | `state_dict` обученного трансформера (должен содержать ключ `head.weight` линейной регрессионной головы) |
| `loss_curves.png` | Кривые train/val MSE для обеих моделей |

**Время запуска:** 2–4 минуты на CPU, <1 минуты на T4. GPU не обязателен.

**Автотесты** проверяют существование всех трёх файлов, структуру ключей в `metrics.json` (включая новые `rmse_logS`, `mae_logS`, `n_train`, `n_val`, `n_test`, `config`), загружаемость `transformer_model.pt` через `torch.load`, порог `transformer_r2 > 0`.

Посмотрите на `loss_curves.png` и количество параметров у модели: подумайте, в каких случаях имеет смысл использовать Transformer, а когда достаточно baseline'а?

> Систематическое сравнение Transformer'ов с простыми моделями на небольших табличных датасетах приведено в [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) (Gorishniy et al., 2021, NeurIPS).