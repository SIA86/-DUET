import numpy as np
from collections import defaultdict

def balance_windows(X, y):
    """
    Балансирует набор временных окон по классам:
    - Оставляет одинаковое число окон для каждого класса
    - Классы определяются из y
    - Выборка происходит случайно

    Parameters:
        X: np.ndarray [T, D] или [T, C] — входные окна
        y: np.ndarray [T] — метки классов

    Returns:
        balanced_X: сбалансированные окна
        balanced_y: соответствующие метки
    """
    assert len(X) == len(y), "X и y должны быть одинаковой длины"

    class_indices = defaultdict(list)

    # Группируем индексы по классам
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    # Находим минимальное число примеров в классе
    min_samples = min(len(indices) for indices in class_indices.values())

    print(f"Балансируем до {min_samples} примеров на класс")

    # Выбираем случайные индексы из каждого класса
    balanced_indices = []
    for indices in class_indices.values():
        balanced_indices.extend(np.random.choice(indices, size=min_samples, replace=False))

    # Перемешиваем результат
    np.random.shuffle(balanced_indices)

    # Возвращаем сбалансированные данные
    return X[balanced_indices], y[balanced_indices]
