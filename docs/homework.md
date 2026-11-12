# Разбалловка

<table>
  <thead>
    <tr><th>Раздел</th><th>Задание</th><th>Максимальный балл</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="7"><b>1. Токенизация и эмбеддинги</b></td>
        <td>1.1</td><td align="right">2.5</td></tr>
    <tr><td>1.2</td>  <td align="right">1.5</td></tr>
    <tr><td>1.3a</td> <td align="right">1.0</td></tr>
    <tr><td>1.3b</td> <td align="right">0.5</td></tr>
    <tr><td>1.4</td>  <td align="right">0.5</td></tr>
    <tr><td>1.5</td>  <td align="right">1.5</td></tr>
    <tr><td><i>Итого</i></td><td align="right"><i>7.5</i></td></tr>
    <tr><td rowspan="3"><b>2. Attention</b></td>
        <td>2.1</td><td align="right">2.5</td></tr>
    <tr><td>2.2</td>  <td align="right">2.5</td></tr>
    <tr><td><i>Итого</i></td><td align="right"><i>5.0</i></td></tr>
    <tr><td rowspan="3"><b>3. Transformer</b></td>
        <td>3.1</td><td align="right">2.0</td></tr>
    <tr><td>3.2</td>  <td align="right">2.0</td></tr>
    <tr><td><i>Итого</i></td><td align="right"><i>4.0</i></td></tr>
    <tr><td rowspan="4"><b>4. Vision Transformer</b></td>
        <td>4.1</td><td align="right">1.0</td></tr>
    <tr><td>4.2</td>  <td align="right">1.0</td></tr>
    <tr><td>4.3</td>  <td align="right">1.5</td></tr>
    <tr><td><i>Итого</i></td><td align="right"><i>3.5</i></td></tr>
    <tr><td colspan="2"><b>Всего (основная часть)</b></td>
        <td align="right"><b>20.0</b></td></tr>
    <tr><td colspan="2">Бонус (B1–B4)</td>
        <td align="right">+4.0</td></tr>
  </tbody>
</table>

# Раздел 1. Токенизация и эмбеддинги

> Обзорный разбор всех основных алгоритмов токенизации можно найти в [главе 6 Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter6/1).

## Практическая часть

Все классы и функции раздела располагаются в файлах `modules/tokenization.py` и `modules/embeddings.py`.

---

### Задание 1.1. Byte Pair Encoding

Реализуйте класс `BPETokenizer` в файле `modules/tokenization.py`, следующий алгоритму BPE.

> Пошаговый разбор алгоритма BPE с визуализацией слияний можно найти в [Hugging Face NLP Course, §6.5](https://huggingface.co/learn/nlp-course/chapter6/5).

**Сигнатуры:**

```python
class BPETokenizer:
    END_OF_WORD = "</w>"

    def fit(self, corpus: list[str], vocab_size: int) -> None: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...

    @property
    def vocab(self) -> dict[str, int]: ...
```

**Требования:**

- Метод `fit` обучает токенизатор на корпусе `corpus` до достижения размера словаря `vocab_size`. Если частотных пар больше нет, обучение завершается досрочно. При `vocab_size < 1` бросьте `ValueError`.
- Метод `encode` токенизирует произвольную строку, последовательно применяя сохранённые при обучении слияния в том же порядке. Если в тексте встречается символ, отсутствующий в словаре (OOV), бросьте `KeyError`.
- Метод `decode` восстанавливает строку из последовательности идентификаторов. Для текстов из обучающего корпуса должно выполняться `decode(encode(text)) == text` с точностью до нормализации пробелов (множественные пробелы и пробелы по краям не сохраняются). При неизвестном идентификаторе бросьте `KeyError`.
- Токен конца слова `</w>` используется для корректного восстановления границ слов при декодировании.
- При вызове `encode` или `decode` до `fit` бросьте `RuntimeError`.
- Свойство `vocab` возвращает **копию** текущего словаря `{токен: id}`.

**Подсказка:** В шаблоне уже реализованы вспомогательные методы (`_build_word_freqs`, `_init_splits`, `_base_vocab`, `_compute_pair_freqs`, `_merge_pair`, `_tokenize_word`, `_check_fitted`). Изучите их docstring'и и используйте при реализации.

**Датасет:** не требуется. Автотесты используют синтетические корпуса с контролируемой статистикой пар.

---

### Задание 1.2. SMILES-токенизатор

Реализуйте класс `SmilesTokenizer` в файле `modules/tokenization.py` — regex-based токенизатор для молекулярных формул в формате SMILES.

> SMILES — строковое представление молекулярной структуры ([Wikipedia](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system), [формальная спецификация Daylight](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)). Специализированный токенизатор необходим, потому что наивное разбиение по символам разрушает двухбуквенные элементы (`Br`, `Cl`) и атомы в скобках (`[nH]`, `[C@@H]`), что делает строку химически невалидной.

**Правила токенизации** (в порядке убывания приоритета):

1. Любая подстрока в квадратных скобках — один токен: `[nH]`, `[C@@H]`, `[O-]`.
2. Двухбуквенные обозначения элементов `Br`, `Cl` — один токен.
3. Остальные символы (буквы, цифры, скобки, `=`, `#`, `/`, `\`, `+`, `-`, `.`, `@`, `%`) — отдельные токены.

**Сигнатуры:**

```python
class SmilesTokenizer:
    SPECIAL_TOKENS: ClassVar[list[str]] = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def tokenize(self, smiles: str) -> list[str]: ...
    def fit(self, smiles_list: list[str]) -> None: ...
    def encode(self, smiles: str, add_special: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int], skip_special: bool = True) -> str: ...

    @property
    def vocab(self) -> dict[str, int]: ...

    @property
    def vocab_size(self) -> int: ...
```

**Требования:**

- Метод `tokenize` разбивает SMILES-строку на токены согласно правилам выше. В шаблоне уже предоставлен скомпилированный regex `_SMILES_REGEX` — используйте его.
- При построении словаря (`fit`) первыми четырьмя идентификаторами (0–3) должны быть специальные токены в порядке, заданном `SPECIAL_TOKENS`. Остальные уникальные токены из корпуса добавляются в **лексикографическом** порядке. Не забудьте заполнить как прямой (`_token_to_idx`), так и обратный (`_idx_to_token`) словари.
- При `add_special=True` метод `encode` оборачивает последовательность в `<bos> ... <eos>`.
- Неизвестные при кодировании токены заменяются на `<unk>`.
- Метод `decode` конкатенирует токены **без разделителей** (SMILES — непрерывная строка). При `skip_special=True` из вывода удаляются все специальные токены. Неизвестные идентификаторы молча пропускаются.
- Для корректно токенизированных строк из обучающего корпуса `decode(encode(smiles)) == smiles`.
- При вызове `encode` или `decode` до `fit` бросьте `RuntimeError`.
- Свойства `vocab` и `vocab_size` возвращают копию словаря и его размер соответственно.

**Датасет:** не требуется. Автотесты используют небольшой фиксированный набор SMILES-строк, покрывающий все правила токенизации.

---

### Задание 1.3. Позиционное кодирование

Реализуйте оба варианта позиционного кодирования в файле `modules/embeddings.py`.

**Часть A. Синусоидальное кодирование.**

```python
def sinusoidal_positional_encoding(
    max_len: int,
    d_model: int,
) -> torch.Tensor:
    ...
```

Формулы:
$$PE(p, 2k) = \sin\!\Bigl(\frac{p}{10000^{2k/d_{\text{model}}}}\Bigr), \qquad PE(p, 2k+1) = \cos\!\Bigl(\frac{p}{10000^{2k/d_{\text{model}}}}\Bigr)$$

где $p$ — позиция ($0, 1, \ldots, \text{max_len} - 1$), $k$ — индекс пары измерений.

**Требования:**
- Вычисление должно быть полностью **векторизовано** средствами `torch`. Циклы по позициям или координатам запрещены.
- Корректно обрабатывайте случай **нечётного** `d_model`: число косинусных столбцов на 1 меньше числа синусных.
- При `max_len < 1` или `d_model < 1` бросьте `ValueError`.

**Часть B. Обучаемое кодирование.**

```python
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

**Требования:**
- `forward` прибавляет к входному тензору позиционные эмбеддинги для позиций $0, 1, \ldots, \text{seq\_len}-1$.
- При `seq_len > max_len` бросьте `ValueError`.
- При `max_len < 1` или `d_model < 1` в конструкторе бросьте `ValueError`.
- Позиционная матрица должна быть обучаемым параметром (`nn.Parameter` или `nn.Embedding`).

**Датасет:** не требуется.

---

### Задание 1.4. Collate-функция

Реализуйте функцию `collate_fn` в файле `modules/embeddings.py`, выполняющую сборку батча с паддингом.

> О том, как `collate_fn` встраивается в `DataLoader` и зачем нужен padding, кратко написано в [документации PyTorch](https://pytorch.org/docs/stable/data.html#working-with-collate-fn).

```python
def collate_fn(
    sequences: list[list[int]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

**Требования:**
- Все последовательности паддятся до длины **наибольшей** последовательности в батче.
- Пустые последовательности внутри списка допустимы: для них вся строка заполняется `pad_id`, а маска — `False`.
- На пустом входе (`sequences == []`) бросьте `ValueError`.

**Датасет:** не требуется.

---

### Задание 1.5. Входной эмбеддинг-пайплайн

Реализуйте класс `TokenEmbeddingPipeline` в файле `modules/embeddings.py`, объединяющий token-эмбеддинг, позиционное кодирование, нормализацию и dropout. Этот модуль будет использован в разделе 3 как входной блок Transformer encoder.

```python
class TokenEmbeddingPipeline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        pos_encoding: str = "sinusoidal",  # "sinusoidal" | "learnable"
        dropout: float = 0.1,
        layer_norm: bool = True,
        pad_idx: int = 0,
    ) -> None: ...

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(batch_size, seq_len) -> (batch_size, seq_len, d_model)"""
        ...
```

**Порядок операций в `forward`:**

1. Token-эмбеддинг (`nn.Embedding` с `padding_idx`) → масштабирование на $\sqrt{d_{\text{model}}}$.
2. Прибавление позиционного кодирования.
3. Layer Normalization (если включена).
4. Dropout.

**Требования:**
- В зависимости от `pos_encoding`:
  - `"sinusoidal"`: используйте `sinusoidal_positional_encoding` и зарегистрируйте результат как буфер (`register_buffer`, `persistent=False`). В `forward` берите срез до `seq_len`.
  - `"learnable"`: используйте `LearnablePositionalEmbedding`.
- При `layer_norm=True` — `nn.LayerNorm(d_model)`, при `False` — `nn.Identity()`.
- При недопустимом значении `pos_encoding` бросьте `ValueError`.
- При `seq_len > max_len` в `forward` бросьте `ValueError`.

**Датасет:** не требуется.

# Раздел 2. Механизм внимания (Attention)

> Эволюцию механизмов внимания подробно разбирает Lilian Weng в посте [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/). Визуальное пошаговое объяснение scaled dot-product и multi-head attention можно найти в [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) Джея Аламмара (секции *Encoder* и *Decoder*).

---

## Практическая часть

Все функции и классы раздела располагаются в файле `modules/attention.py`.

---

### Задание 2.1. Scaled Dot-Product Attention

Реализуйте функцию `scaled_dot_product_attention`, вычисляющую внимание по формуле из статьи *Attention Is All You Need*:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Сигнатура:**

```python
def scaled_dot_product_attention(
    query: torch.Tensor,     # (..., seq_len_q, d_k)
    key: torch.Tensor,       # (..., seq_len_k, d_k)
    value: torch.Tensor,     # (..., seq_len_k, d_v)
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

**Требования:**

- Размерность $d_k$ определяется как `query.size(-1)`.
- Масштабирование логитов на $\sqrt{d_k}$ выполняется **до** применения маски.
- При `mask is not None` маска имеет тип `torch.bool`: `True` — позиция **разрешена**, `False` — **запрещена**. Допустимы формы `(..., seq_len_q, seq_len_k)` и `(..., 1, seq_len_k)` (широковещание по оси запросов).
- Запрещённые позиции заполняются значением `torch.finfo(scores.dtype).min` **до** softmax, чтобы на выходе softmax они получили нулевой вес.
- Функция должна поддерживать **произвольное число ведущих измерений** (например, `(batch, num_heads, seq_len, d_k)` для многоголового внимания). Используйте `torch.matmul` и отрицательные индексы осей (`-1`, `-2`) в `transpose`, а не фиксированные индексы.
- Функция возвращает **оба** тензора: `output` (взвешенную сумму значений) и `weights` (матрицу весов после softmax).
- Градиенты должны корректно течь через все операции.

**Датасет:** не требуется. Автотесты используют случайные тензоры с фиксированным seed и проверяют численное совпадение с эталонной реализацией, корректность работы маски (нулевые веса на запрещённых позициях, нормировка оставшихся) и форму выходов.

---

### Задание 2.2. Multi-Head Attention

Реализуйте класс `MultiHeadAttention` как `nn.Module`.

**Сигнатура:**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None: ...

    def forward(
        self,
        query: torch.Tensor,   # (batch_size, seq_len_q, d_model)
        key: torch.Tensor,     # (batch_size, seq_len_k, d_model)
        value: torch.Tensor,   # (batch_size, seq_len_k, d_model)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

**Требования:**

- `d_model` должен делиться на `num_heads`. Иначе — `ValueError`.
- Сохраните `self.d_model`, `self.num_heads` и `self.d_k = d_model // num_heads`.
- Четыре обучаемые линейные проекции **с `bias=True`**: `self.w_q`, `self.w_k`, `self.w_v`, `self.w_o` — каждая `nn.Linear(d_model, d_model, bias=True)`.
- Разделение на головы реализуется через `reshape` (`.view()`) и `transpose`, **не** через создание `num_heads` отдельных `nn.Linear`. Формы при разделении: `(B, L, d_model) → (B, L, num_heads, d_k) → (B, num_heads, L, d_k)`.
- Если `mask is not None`, добавьте ось голов через `mask.unsqueeze(1)` для корректного broadcasting по `(B, 1, ...) → (B, num_heads, ...)`.
- Внутри вызывайте `scaled_dot_product_attention` из задания 2.1.
- При сборке голов выполните обратный reshape: `(B, num_heads, L_q, d_k) → (B, L_q, num_heads, d_k) → (B, L_q, d_model)`. Не забудьте `.contiguous()` перед финальным `.view()`.
- `forward` возвращает **только** выходной тензор (без весов внимания).
- Поддержка как self-attention (`query = key = value`), так и cross-attention (разные последовательности).

**Датасет:** не требуется. Автотесты проверяют:

- форму выходного тензора,
- корректную валидацию `d_model % num_heads == 0`,
- что количество обучаемых параметров соответствует $4d^2 + 4d$ (четыре `nn.Linear` с bias),
- что выход для случая `num_heads=1` численно совпадает с результатом `scaled_dot_product_attention` с теми же весами,
- что маскирование паддинга работает (идентичные последовательности с разным количеством паддинга дают идентичные выходы на непаддинговых позициях).

# Раздел 3. Transformer

> Классификация архитектур по типу отображения (one-to-one, seq-to-seq и т.д.) наглядно показана в [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). Различия между pre-norm и post-norm, и почему современные модели (GPT-2, ViT, LLaMA) используют именно pre-norm, разбираются в статье [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).

## Практическая часть

Все классы раздела располагаются в файле `modules/transformer.py`. Модуль импортирует `MultiHeadAttention` из `modules.attention` и `TokenEmbeddingPipeline` из `modules.embeddings`.

---

### Задание 3.1. Transformer Encoder Block

Реализуйте класс `TransformerEncoderBlock` с **pre-norm** архитектурой.

**Сигнатура:**

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,                 # (batch_size, seq_len, d_model)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

**Архитектура блока (pre-norm):**

Каждый подблок использует residual connection вида `x = residual + Sublayer(LayerNorm(residual))`, где `LayerNorm` применяется **до** подблока, а не после.

**1. Self-attention подблок:**
```
residual = x
x = residual + Dropout(MultiHeadAttention(LayerNorm(x), ..., mask))
```
- `query = key = value = LayerNorm(x)` (self-attention).
- `MultiHeadAttention.forward` возвращает **один тензор** (не кортеж).
- Dropout применяется к выходу MHA **перед** сложением с residual.

**2. Feed-forward подблок:**
```
residual = x
x = residual + Linear(Dropout(GELU(Dropout(Linear(LayerNorm(x))))))
```
- Структура FFN: `Linear(d_model, d_ff) → GELU → Dropout → Linear(d_ff, d_model) → Dropout`.
- Итого **три** `nn.Dropout` слоя в блоке (один после MHA, два в FFN), все с `p=dropout`.

**Требования:**

- Используйте `MultiHeadAttention` из `modules.attention` и `nn.GELU()` в качестве активации.
- Маска передаётся в `MultiHeadAttention` без изменений.
- Все линейные слои — с `bias=True` (по умолчанию в `nn.Linear`).

**Датасет:** не требуется. Автотесты проверяют:

- форму выходного тензора,
- количество обучаемых параметров,
- что при нулевых весах подслоёв блок ведёт себя как тождественное отображение (residual connection),
- что градиенты текут через все компоненты,
- численное совпадение с эталонной реализацией на фиксированном входе.

---

### Задание 3.2. Transformer Encoder

Соберите полный encoder из `TokenEmbeddingPipeline` (раздел 1) и стека `TransformerEncoderBlock`.

**Сигнатура:**

```python
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None: ...

    def forward(
        self,
        input_ids: torch.Tensor,                    # (batch_size, seq_len)
        attention_mask: torch.Tensor | None = None,  # (batch_size, seq_len), bool
    ) -> torch.Tensor:
        ...
```

**Архитектура:** `TokenEmbeddingPipeline → N × TransformerEncoderBlock → LayerNorm`.

**Требования:**

- **Входной слой** (`self.embedding`): `TokenEmbeddingPipeline` с `pos_encoding="sinusoidal"`, `layer_norm=True` и остальными параметрами из сигнатуры.
- **Стек блоков** (`self.layers`): `nn.ModuleList` из `num_layers` экземпляров `TransformerEncoderBlock`.
- **Финальная нормализация** (`self.final_norm`): `nn.LayerNorm(d_model)` после стека блоков.
- **Преобразование маски:** если `attention_mask` имеет форму `(B, L)` (2D), расширьте до `(B, 1, L)` через `.unsqueeze(1)` для broadcasting по осям голов и запросов в MHA. Если `attention_mask is None`, передавайте `None` во все блоки.
- Все блоки используют одну и ту же маску (маска не меняется от слоя к слою).

**Датасет:** не требуется. Автотесты проверяют:

- форму выходного тензора `(B, L, d_model)`,
- что изменение `num_layers` корректно меняет число блоков,
- что `attention_mask` влияет на выход (разные маски → разные выходы),
- что паддинг-позиции не влияют на выходы реальных токенов,
- численное совпадение с эталонной реализацией.

# Раздел 4. Vision Transformer (ViT)

## Практическая часть

### Задание 4.1. Patch Embedding

Реализуйте класс `PatchEmbedding` в файле `modules/vit.py`, превращающий изображение в последовательность патч-эмбеддингов.

**Сигнатура:**

```python
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        d_model: int,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

**Требования:**

- Реализуйте проекцию через **один `nn.Conv2d`** с `kernel_size = stride = patch_size` и `padding = 0`. Такая свёртка одновременно разрезает изображение на непересекающиеся патчи и проецирует каждый в пространство размерности `d_model`.
- После свёртки преобразуйте выход `(B, d_model, H/P, W/P)` в `(B, N, d_model)` через `flatten` по пространственным осям и `transpose`.
- Если `H` или `W` не делится на `patch_size`, бросьте `ValueError` с информативным сообщением (включите в текст реальные размеры и остатки).
- Сохраните `patch_size`, `in_channels`, `d_model` как публичные атрибуты.

**Датасет:** не требуется. Автотесты используют случайные тензоры и проверяют:

- форму выхода,
- корректное вычисление `num_patches`,
- `ValueError` при неделимых размерах,
- численное совпадение с эталонной реализацией.

---

### Задание 4.2. Fine-tuning предобученной ViT на классификации спутниковых снимков

В задании 4.1 вы реализовали ключевой компонент ViT — Patch Embedding. Вместе с `TransformerEncoderBlock` из раздела 3 и `LearnablePositionalEmbedding` из раздела 1 у вас есть все строительные блоки для полной архитектуры Vision Transformer. Однако обучение ViT с нуля на реальных данных потребовало бы сотен GPU-часов и миллионов размеченных изображений. В production так не делают: берут модель, **предобученную** на большом датасете (ImageNet), и **дообучают** на целевой задаче — это называется *transfer learning*.

В этом задании вы выполните fine-tuning предобученной ViT-Tiny из библиотеки `timm` на датасете **EuroSAT** — классификация типов земного покрова по спутниковым снимкам Sentinel-2. Архитектура `vit_tiny_patch16_224` в `timm` устроена ровно так, как описано в теоретической части: patch embedding → `[CLS]`-токен → positional embedding → стек transformer encoder blocks → LayerNorm → classification head. Каждый из этих компонентов вы реализовали вручную в разделах 1–4. Теперь вы увидите, как эта же архитектура, но с весами, обученными на 1.28 млн изображений ImageNet, решает реальную задачу за считанные минуты.

#### Что уже реализовано в шаблоне

В `scripts/train_vit.py` для вас **уже написаны**:

- Класс-адаптер `HFImageDataset` (обёртка HuggingFace Dataset → PyTorch Dataset).
- Функция `_build_transforms(img_size)`, возвращающая `(train_transform, eval_transform)`.
- Функция `set_seed(seed)` для фиксации случайности.
- Функции `evaluate()` и `collect_predictions()` для инференса.
- Функции `save_loss_curve()` и `save_confusion_matrix()` для построения графиков.
- Функция `main()` и CLI-парсер — полный pipeline, запускающий все три реализуемые вами функции и сохраняющий артефакты.

**Вам нужно реализовать три функции:** `get_pretrained_vit`, `get_dataloaders`, `train`.

#### Датасет

EuroSAT — 27 000 RGB-патчей спутниковых снимков Sentinel-2 (10 м/пиксель), 10 классов:

| Класс | Описание |
|-------|----------|
| AnnualCrop | Сельхозугодья (однолетние культуры) |
| Forest | Лес |
| HerbaceousVegetation | Травянистая растительность |
| Highway | Автомагистрали |
| Industrial | Промышленные зоны |
| Pasture | Пастбища |
| PermanentCrop | Многолетние культуры (сады, виноградники) |
| Residential | Жилая застройка |
| River | Реки |
| SeaLake | Моря и озёра |

Изображения 64×64, при загрузке масштабируются до 224×224 (стандартный вход ViT-Tiny). Датасет сбалансирован (~2 700 изображений на класс).

**Загрузка:** `load_dataset("timm/eurosat-rgb")` → сплиты `"train"` (16 200), `"validation"` (5 400), `"test"` (5 400).

#### Сигнатуры и требования

**`get_pretrained_vit(num_classes, freeze_backbone=True) → nn.Module`**

1. Загрузите модель: `timm.create_model("vit_tiny_patch16_224", pretrained=True)`.
2. Замените классификационную голову: `model.head = nn.Linear(model.head.in_features, num_classes)`.
3. Если `freeze_backbone=True`, установите `requires_grad=False` для всех параметров, **имя которых не начинается с `"head."`**.
4. Выведите в stdout количество обучаемых и общих параметров:
   ```
   Total parameters:     5,537,286
   Trainable parameters: 1,930 (0.035%)
   ```

**`get_dataloaders(batch_size=64, img_size=224, num_workers=2) → (train_loader, val_loader, test_loader)`**

1. Загрузите датасет через `load_dataset("timm/eurosat-rgb")`.
2. Постройте transforms с помощью готовой функции `_build_transforms(img_size)` (она уже содержит нужные ImageNet-нормализации, `InterpolationMode.BILINEAR` и аугментации; флипы по вертикали допустимы, т.к. у спутниковых снимков нет «естественной» ориентации).
3. Оберните каждый сплит в `HFImageDataset` с соответствующим transform.
4. Создайте `DataLoader` для каждого сплита:
   - train: `shuffle=True`, `drop_last=True`
   - val/test: `shuffle=False`
   - `pin_memory=torch.cuda.is_available()` для всех

**`train(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cuda", seed=42) → dict`**

1. Зафиксируйте seed: `set_seed(seed)` (функция уже реализована).
2. Перенесите модель на `device`.
3. Настройте:
   - `criterion = nn.CrossEntropyLoss()`
   - `optimizer = AdamW` **только по обучаемым параметрам** (`p.requires_grad == True`), `lr=lr`, `weight_decay=0.01`
   - `scheduler = CosineAnnealingLR(optimizer, T_max=epochs)`
4. Цикл по эпохам (от 1 до `epochs`):
   - `model.train()`, итерация по `train_loader`: forward → loss → backward → step. Накапливайте взвешенный `running_loss`.
   - `scheduler.step()` после эпохи.
   - Вызовите `evaluate(model, val_loader, criterion, device)` → `(val_loss, val_acc)`.
   - Сохраните три значения в `history`.
5. Верните словарь:
   ```python
   {
       "train_loss": [float, ...],  # len == epochs
       "val_loss":   [float, ...],
       "val_acc":    [float, ...],
   }
   ```

#### Общие требования

- **Interpolation:** во всех `Resize` используется `InterpolationMode.BILINEAR` (уже в `_build_transforms`).
- **Переключение режимов:** `model.train()` перед обучением, `model.eval()` перед валидацией/тестированием. При вычислении метрик используйте `torch.no_grad()` / `torch.inference_mode()` (в `evaluate` это уже сделано).
- **Обучение:** 10 эпох, batch_size=64. Ожидаемое время на T4: **10–15 минут**.
- **Ожидаемая test accuracy: ≥ 90 %.**

> **Примечание о переносимости паттерна заморозки.** В ViT используется LayerNorm, у которого нет running statistics (running mean/var), поэтому заморозка через `requires_grad=False` работает корректно. Если в будущем вы перенесёте этот паттерн на архитектуры с BatchNorm (ResNet, EfficientNet, ConvNeXt), замороженные BatchNorm-слои в режиме `model.train()` продолжат обновлять running statistics, что приведёт к деградации качества. В таких случаях необходимо либо перевести замороженные слои в `eval()` режим, либо установить `layer.track_running_stats = False`.

#### Запуск и артефакты

Запуск из командной строки:
```bash
python -m scripts.train_vit --epochs 10 --batch-size 64 --lr 1e-3
```

Либо из ноутбука `dl_sem5_lab03_aie.ipynb` (запускает тот же `main()` в Colab с GPU).

`main()` автоматически сохранит в `results/`:

- `metrics.json` — метрики по эпохам, финальные test loss/accuracy, количество параметров, время обучения, конфигурация.
- `loss_curve.png` — графики train/val loss и val accuracy.
- `confusion_matrix.png` — матрица ошибок на тестовом наборе.
- `model.pt` — `state_dict` обученной модели.

**Для сдачи приложите к репозиторию** содержимое папки `results/` (все четыре артефакта), полученное после реального запуска.

#### Вопрос для размышления

Изучите confusion matrix. Какие пары классов модель путает чаще всего? Почему эти классы визуально похожи на спутниковых снимках?