import torch
import numpy as np
import pandas as pd
from typing import Any, Tuple

from . import preprocess

   
def predict_window(model, x_window: pd.DataFrame, config: Any, device="cuda") -> np.ndarray:
    assert len(x_window) == config.seq_len , "X_window должно быть размером config.seq_len"

    model.eval()

    x_train = preprocess.prepare_windows_for_pred(x_window, config)
    x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred_tensor = model(x_tensor)

    y_pred = y_pred_tensor.cpu().numpy()

    return y_pred


def predict_dataset_batched(
    model,
    data: pd.DataFrame,
    config: Any,
    batch_size: int = 512,
    device: str = "cuda"
) -> Tuple[pd.Series, np.ndarray]:
    """
    Делает batched предсказания модели и возвращает:
        - labels_series: метки (argmax), pd.Series
        - probs: np.ndarray (вероятности), shape: (len(data), n_classes), заполнено NaN

    Эти объекты можно напрямую передать в plot_roc_auc(...).
    """
    model.eval()

    # Получаем входы
    X = preprocess.prepare_windows_for_pred(data, config)
    preds = []
    for i in range(0, len(X)+1, batch_size):
        batch = X[i:i + batch_size]
        x_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_batch = model(x_tensor).cpu().numpy()  # (batch_size, n_classes)
        preds.append(y_batch)

    y_pred_all = np.concatenate(preds, axis=0)  # shape: (n_preds, n_classes)
    n_preds, n_classes = y_pred_all.shape

    # Смещение
    if config.predict_type.upper() == "DETECT":
        start_idx = config.seq_len - 1
    elif config.predict_type.upper() == "NEXT":
        start_idx = config.seq_len
    else:
        raise ValueError("config.predict_type должен быть 'DETECT' или 'NEXT'")

    # Инициализируем Series и массив
    labels_series = pd.Series(np.nan, index=data.index)
    probs_array = np.full((len(data), n_classes), np.nan, dtype=np.float32)

    # Заполнение
    labels = np.argmax(y_pred_all, axis=1)
    labels_series.iloc[start_idx:start_idx + n_preds] = labels
    probs_array[start_idx:start_idx + n_preds, :] = y_pred_all

    return labels_series, probs_array
