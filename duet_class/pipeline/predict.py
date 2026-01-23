import numpy as np
import torch
from typing import Any, Tuple, Optional


def _as_windows(x: np.ndarray, seq_len: int) -> np.ndarray:
    x_array = np.asarray(x)
    if x_array.ndim == 2:
        if x_array.shape[0] != seq_len:
            raise ValueError("x_window должно быть размером (seq_len, n_features)")
        return x_array[None, :, :]
    if x_array.ndim == 3:
        if x_array.shape[1] != seq_len:
            raise ValueError("windows должно быть размером (batch, seq_len, n_features)")
        return x_array
    raise ValueError("Ожидается np.ndarray размерности 2 или 3")


def predict_window(model, x_window: np.ndarray, config: Any, device: str = "cuda") -> np.ndarray:
    """
    Предикт для одного окна.

    Ожидается np.ndarray формы (seq_len, n_features).
    Возвращает вероятности классов формы (num_classes,).
    """
    model.eval()
    windows = _as_windows(x_window, config.seq_len)
    x_tensor = torch.tensor(windows, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(x_tensor)
        y_pred_tensor = torch.softmax(logits, dim=-1)

    return y_pred_tensor.cpu().numpy()[0]



def predict_dataset_batched(
    model,
    windows: np.ndarray,
    config: Any,
    batch_size: int = 512,
    device: str = "cuda",
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Делает batched предсказания модели по заранее нарезанным окнам и возвращает:
        - labels: np.ndarray (argmax), shape: (n_windows,)
        - probs: np.ndarray (вероятности), shape: (n_windows, n_classes)
    
    Если передан threshold, то классы с вероятностью < threshold помечаются как -1.
    """
    model.eval()
    # Используем windows напрямую, так как val_x уже нарезан на окна (N, seq_len, features)
    X = windows 

    preds = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        x_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x_tensor)
            y_batch = torch.softmax(logits, dim=-1).cpu().numpy()
        preds.append(y_batch)

    probs = np.concatenate(preds, axis=0) if preds else np.empty((0, config.num_classes))
    
    if probs.size > 0:
        labels = np.argmax(probs, axis=1)
        if threshold is not None:
            # Фильтруем предсказания по порогу уверенности
            max_probs = np.max(probs, axis=1)
            labels[max_probs < threshold] = -1
    else:
        labels = np.empty((0,), dtype=int)

    return labels, probs
