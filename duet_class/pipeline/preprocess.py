import numpy as np
from collections import defaultdict
from typing import Any, Iterable, Optional, Tuple, Union


def balance_windows(
    X: np.ndarray,
    y: np.ndarray,
    max_ratio: Union[int, float] = 1.0,
    exclude_labels: Optional[Union[Any, Iterable[Any]]] = None,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Балансирует набор окон по классам через downsampling (без oversampling).

    Идея:
      - Пусть m = размер самого маленького класса (после исключений).
      - Для каждого класса i оставляем n_i = min(count_i, floor(max_ratio * m)) сэмплов.
      - Тем самым гарантируется, что max(count_after) <= max_ratio * min(count_after).

    Примеры:
      - max_ratio=1.0  -> все классы приводятся к размеру m (классический strict balance)
      - max_ratio=2.0  -> самый большой класс после балансировки не превышает 2*m

    Args:
        X: np.ndarray формы (N, ...) — массив окон.
        y: np.ndarray формы (N,) — метки классов.
        max_ratio: допустимое отношение max/min после балансировки (>= 1).
        exclude_labels: метка или набор меток, которые нужно полностью исключить из X и y
                        до балансировки.
        random_state: seed для воспроизводимости.
        verbose: печатать ли краткую статистику.

    Returns:
        balanced_X, balanced_y

    Raises:
        ValueError: если после исключений остаётся < 2 классов или max_ratio < 1.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError("X и y должны быть одинаковой длины")

    if max_ratio is None:
        max_ratio = 1.0
    max_ratio = float(max_ratio)
    if max_ratio < 1.0:
        raise ValueError("max_ratio должен быть >= 1.0")

    # --- исключение классов ---
    if exclude_labels is not None:
        if isinstance(exclude_labels, (str, bytes)) or not isinstance(exclude_labels, Iterable):
            exclude_set = {exclude_labels}
        else:
            exclude_set = set(exclude_labels)

        keep_mask = ~np.isin(y, list(exclude_set))
        X = X[keep_mask]
        y = y[keep_mask]

    if len(y) == 0:
        raise ValueError("После исключений не осталось данных.")

    rng = np.random.default_rng(random_state)

    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    if len(class_indices) < 2:
        raise ValueError(
            f"Для балансировки нужно минимум 2 класса. Сейчас: {list(class_indices.keys())}"
        )

    counts = {k: len(v) for k, v in class_indices.items()}
    min_samples = min(counts.values())
    cap = int(np.floor(max_ratio * min_samples))

    if cap <= 0:
        raise ValueError("cap <= 0 — проверьте данные и max_ratio.")

    if verbose:
        max_samples = max(counts.values())
        print(
            f"Балансировка: min={min_samples}, max={max_samples}, max_ratio={max_ratio} -> cap={cap}"
        )

    balanced_indices = []
    for label, indices in class_indices.items():
        k = min(len(indices), cap)
        # replace=False: только downsample
        chosen = rng.choice(indices, size=k, replace=False)
        balanced_indices.extend(chosen.tolist())

    rng.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]
