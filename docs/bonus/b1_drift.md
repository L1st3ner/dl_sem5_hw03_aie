## Вариант B1. Embedding Drift Detection (`b1_drift`)

Вы задеплоили Transformer/ViT как feature extractor для downstream-системы (поиск, рекомендации, retrieval). Через время метрики падают, анализ показал, что распределение эмбеддингов на входе сместилось. Нужна система мониторинга, детектирующая drift по потоку эмбеддингов без ground truth.

**Использует из основной части:** `TransformerEncoder` (задание 3.2).

> Таксономия сдвигов распределения с примерами из практики описана в обзоре [A Unifying View on Dataset Shift in Classification](https://www.researchgate.net/publication/220602504_A_unifying_view_on_dataset_shift_in_classification) (Moreno-Torres et al., 2012). Более прикладной и короткий разбор типов дрифта с визуализациями можно найти в [этой статье](https://www.evidentlyai.com/ml-in-production/data-drift) от Evidently AI, а про дрифт в эмбеддингах можно почитать в [этой статье](https://www.evidentlyai.com/blog/embedding-drift-detection).
### Структура файлов и что нужно реализовать

Все файлы задания располагаются в директории `bonus/b1_drift/`:

| Файл | Уже реализовано в шаблоне | Нужно реализовать |
|------|---------------------------|-------------------|
| `stream_generator.py` | Весь файл | — |
| `drift_metrics.py` | `_pairwise_squared_distances`, `_quantile_resample` | `mmd_rbf`, `cosine_centroid_distance`, `sliced_wasserstein_distance` |
| `embedding_extractor.py` | — | `extract_embeddings` |
| `drift_detector.py` | `DriftDetector.__init__`, `DriftDetector._compute_distance` | `DriftDetector.fit`, `DriftDetector.detect` |

> **Примечание:** Данный вариант не требует обучения на GPU и генерации артефактов в `results/` — проверка выполняется полностью автотестами. Для локальной отладки используйте готовый генератор `SyntheticDriftStream` из `bonus/b1_drift/stream_generator.py`.

---

### Задание B1.1. Метрики расстояния между распределениями

Реализуйте три непараметрические метрики расстояния между двумя выборками эмбеддингов $X \in \mathbb{R}^{N \times D}$ и $Y \in \mathbb{R}^{M \times D}$ в файле `bonus/b1_drift/drift_metrics.py`.

#### Часть A. Maximum Mean Discrepancy с RBF-ядром

> Теоретическое обоснование MMD как статистического теста можно найти в [A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html) (Gretton et al., 2012, JMLR).

```python
def mmd_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    gamma: float | None = None,
) -> torch.Tensor:
```

**Формулы:**
$$\text{MMD}^2(X, Y) = \mathbb{E}[k(x, x')] - 2\,\mathbb{E}[k(x, y)] + \mathbb{E}[k(y, y')], \qquad k(a, b) = \exp\!\bigl(-\gamma \|a - b\|^2\bigr)$$

где матожидания оцениваются как средние по всем парам матриц ядра $K_{XX}, K_{XY}, K_{YY}$.

**Требования:**
- Для вычисления матриц квадратов евклидовых расстояний используйте готовую вспомогательную функцию `_pairwise_squared_distances(a, b)`.
- При `gamma is None` используйте **median heuristic**:
  1. Объедините выборки по строкам: $Z = \text{concat}(X, Y) \in \mathbb{R}^{(N+M) \times D}$.
  2. Вычислите матрицу попарных квадратов расстояний в $Z$.
  3. Возьмите медиану элементов **строго верхнего треугольника** (`torch.triu(..., diagonal=1)`).
  4. Положите $\gamma = \frac{1}{2 \cdot \text{median}}$. Если медиана равна `0.0` (все точки совпали), используйте значение по умолчанию `gamma = 1.0`.
- Проверяйте входные тензоры: если `X` или `Y` не 2D, их размерности признаков `D` не совпадают, или хотя бы одна выборка пуста (`size(0) < 1`), бросьте `ValueError`.
- Возвращайте скалярный (0-d) тензор.

#### Часть B. Косинусное расстояние между центроидами

```python
def cosine_centroid_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
```

**Формула:**
$$d(X, Y) = 1 - \frac{\mu_X \cdot \mu_Y}{\|\mu_X\|\,\|\mu_Y\|}, \qquad \mu_X = \frac{1}{N}\sum_{i=1}^N x_i, \quad \mu_Y = \frac{1}{M}\sum_{j=1}^M y_j$$

**Требования:**
- Для вычисления косинусного сходства используйте `F.cosine_similarity(mu_x, mu_y, dim=0, eps=1e-12)`.
- Валидация входных тензоров аналогична `mmd_rbf` (2D, совпадение `D`, непустые выборки → иначе `ValueError`).

#### Часть C. Sliced Wasserstein Distance

> Идея проецирования многомерных распределений на случайные одномерные направления и усреднения Wasserstein-расстояний по проекциям описана в [Sliced Wasserstein Distance for Learning Gaussian Mixture Models](https://ar5iv.labs.arxiv.org/html/1711.05376) (Kolouri, Rohde, Hoffmann, 2018). Связь с задачей оптимального транспорта и визуализация одномерного Wasserstein описана в книге [Computational Optimal Transport](https://optimaltransport.github.io/book/) (Peyré & Cuturi, 2019).

```python
def sliced_wasserstein_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_projections: int = 100,
    seed: int | None = None,
) -> torch.Tensor:
```

**Алгоритм Монте-Карло оценки Sliced Wasserstein-1:**
1. Сгенерируйте `n_projections` случайных единичных векторов $\theta_k \in S^{D-1}$: сэмплируйте матрицу из $\mathcal{N}(0, I)$ формы `(n_projections, D)` и нормируйте каждую строку на её L2-норму (`clamp_min(1e-12)`). Если `seed is not None`, создайте локальный `torch.Generator(device=X.device).manual_seed(int(seed))`, не затрагивая глобальный RNG.
2. Спроецируйте обе выборки на все направления: $P_X = X \Theta^\top \in \mathbb{R}^{N \times K}$, $P_Y = Y \Theta^\top \in \mathbb{R}^{M \times K}$.
3. Отсортируйте проекции вдоль оси сэмплов (`dim=0`).
4. Если размеры выборок различаются ($N \neq M$), приведите обе отсортированные матрицы к общей равномерной квантильной сетке `torch.linspace(0.0, 1.0, steps=max(N, M))` с помощью готовой вспомогательной функции `_quantile_resample`.
5. Вычислите среднее модуля разности между выровненными отсортированными проекциями по всем элементам и направлениям.

**Требования:**
- Валидация тензоров аналогична предыдущим метрикам + проверка `n_projections >= 1` (иначе `ValueError`).
- Тензоры направлений и квантильной сетки должны создаваться на том же `device` и с тем же `dtype`, что и `X`.

**Датасет:** не требуется. Автотесты проверяют свойства метрик (равенство нулю на одинаковых выборках, симметричность, неотрицательность, воспроизводимость по `seed`, работу на выборках разного размера) и численное совпадение с эталоном.

---

### Задание B1.2. Извлечение эмбеддингов из TransformerEncoder

> Про сравнение стратегий пулинга для Transformer-эмбеддингов можно почитать в [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019).

Реализуйте функцию `extract_embeddings` в файле `bonus/b1_drift/embeddings_extractor.py`, превращающую батч токенов в матрицу векторных представлений фиксированной размерности (паттерн из `sentence-transformers`).

**Сигнатура:**

```python
def extract_embeddings(
    model: TransformerEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str = "mean",
) -> torch.Tensor:
```

**Требования:**
- Проверьте, что `pooling` принимает значение `"mean"` или `"cls"`, иначе бросьте `ValueError`.
- Перед прогоном сохраните исходный режим модели (`was_training = model.training`), переведите её в `model.eval()`, а после завершения **обязательно восстановите** исходный режим в блоке `finally` (если модель была в `train()`).
- Выполняйте прямой проход внутри `torch.no_grad()`, а результат верните через `.detach()`, чтобы гарантировать `requires_grad=False`.
- Режимы пулинга:
  - `"cls"`: верните скрытое состояние первого токена `hidden[:, 0, :]`.
  - `"mean"`: усредните скрытые состояния только по непаддинговым позициям с использованием `attention_mask` (приведите маску к `hidden.dtype`, умножьте на `hidden`, просуммируйте по `dim=1` и поделите на число реальных токенов с `clamp_min(1.0)` для защиты от деления на ноль).

**Датасет:** не требуется. Автотесты проверяют формы выходов, инвариантность `"mean"`-пулинга к количеству паддинга, отсутствие градиентов и восстановление режима `model.training`.

---

### Задание B1.3. Детектор дрифта на основе скользящего окна

Реализуйте методы `fit` и `detect` класса `DriftDetector` в файле `bonus/b1_drift/drift_detector.py` (методы `__init__` и `_compute_distance` уже реализованы).

**Сигнатура:**

```python
class DriftDetector:
    def __init__(
        self,
        metric: str = "mmd",
        window_size: int = 50,
        threshold: float | None = None,
        step: int = 1,
        metric_kwargs: dict[str, Any] | None = None,
    ) -> None: ...  # уже реализовано

    def fit(self, reference: torch.Tensor) -> None:

    def detect(self, stream: torch.Tensor) -> dict[str, Any]:
```

**Требования к `fit(reference)`:**
- Проверьте, что `reference` — 2D-тензор `(N_ref, D)` и `N_ref >= self.window_size` (иначе `ValueError`).
- Сохраните референсную выборку в `self._reference`.
- Если `self.threshold is not None`, запишите `self._fitted_threshold = float(self.threshold)`.
- Если `self.threshold is None`, вычислите **адаптивный порог** по референсной выборке:
  1. Разбейте `reference` на `n_windows = N_ref // window_size` непересекающихся окон длины `window_size` (хвост длиной `< window_size` отбрасывается). Если `n_windows < 2` (то есть `N_ref < 2 * window_size`), бросьте `ValueError`.
  2. Вычислите расстояния между всеми парами окон `(i, j)` для `0 <= i < j < n_windows` с помощью `self._compute_distance(windows[i], windows[j])`.
  3. Установите порог по правилу трёх сигм:
     $$\text{threshold} = \mu_d + 3\,\sigma_d$$
     где $\sigma_d$ — стандартное отклонение со смещением (`unbiased=False`), или `0.0`, если пара всего одна (`n_windows == 2`).

**Требования к `detect(stream)`:**
- Если детектор не обучен (`self._reference is None` или `self._fitted_threshold is None`), бросьте `RuntimeError`.
- Проверьте, что `stream` — 2D-тензор `(T, D)` и его размерность признаков `D` совпадает с `self._reference.size(1)` (иначе `ValueError`).
- Если поток короче окна (`T < self.window_size`), верните `{"drift_detected": False, "drift_index": None, "distances": [], "threshold": self._fitted_threshold}`.
- Пройдите скользящим окном `stream[start : start + window_size]` для `start = 0, step, 2*step, ...` пока `start + window_size <= T`:
  - Для каждого окна вычислите расстояние до **всей** референсной выборки `self._reference` через `self._compute_distance(window, self._reference)` и добавьте в список `distances`.
  - Если `distance > self._fitted_threshold` (строго больше) и дрифт ещё не был зафиксирован, запомните `drift_index = start`.
- Верните словарь с четырьмя ключами:
  ```python
  {
      "drift_detected": bool,        # True, если хотя бы одно окно превысило порог
      "drift_index": int | None,     # start первого окна с превышением или None
      "distances": list[float],      # расстояния для всех проверенных окон
      "threshold": float,            # использованный порог (self._fitted_threshold)
  }
  ```

**Датасет:** не требуется. Автотесты используют `SyntheticDriftStream` и проверяют отсутствие ложных срабатываний на стационарном потоке, обнаружение дрифта для сценариев `mean_shift`, `variance_change` и `component_swap`, работу явного и адаптивного порогов и формат выходного словаря.