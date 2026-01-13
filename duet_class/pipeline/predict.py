import torch
import numpy as np
import pandas as pd
from typing import Any, Tuple

from .wf_slicer import GlobalNormConfig, SplitConfig, WalkForwardWindowSlicerVec, WindowConfig

   
SCALER_MAP = {
    "STD": "standard",
    "MINMAX": "minmax",
    "QUANT": "quantile",
    "NONE": "none",
}

def _build_predict_slicer(config: Any) -> WalkForwardWindowSlicerVec:
    split = SplitConfig(
        n_folds=1,
        mode="expanding",
        ratios=(1.0, 0.0, 0.0),
        gap=0,
        step_size=None,
        sliding_train_size=None,
    )

    window = WindowConfig(
        x_window=config.seq_len,
        x_end_offset=0,
        y_window=0,
        y_end_offset=0,
        allow_left_context_for_x=False,
    )

    global_norm = GlobalNormConfig(
        scaler=SCALER_MAP.get(config.scaler.upper(), "none"),
    )

    return WalkForwardWindowSlicerVec(
        split=split,
        window=window,
        global_norm=global_norm,
        no_norm_cols=config.not_to_normalise,
        eps=1e-12,
        drop_incomplete_last_fold=True,
    )

def _predict_time_offset(config: Any) -> int:
    predict_type = config.predict_type.upper()
    if predict_type == "DETECT":
        return 0
    if predict_type == "NEXT":
        return 1
    raise ValueError("config.predict_type должен быть 'DETECT' или 'NEXT'")

def predict_window(model, x_window: pd.DataFrame, config: Any, device="cuda") -> np.ndarray:
    assert len(x_window) == config.seq_len , "X_window должно быть размером config.seq_len"

    model.eval()

    slicer = _build_predict_slicer(config)
    out = slicer.split_and_window(X=x_window[config.features])
    x_train = out["fold_0"]["train"]["X"]
    x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(x_tensor)
        y_pred_tensor = torch.softmax(logits, dim=-1)

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
    slicer = _build_predict_slicer(config)
    out = slicer.split_and_window(X=data[config.features])
    fold0 = out["fold_0"]["train"]
    X = fold0["X"]
    t0 = fold0["t0"]
    t_offset = _predict_time_offset(config)
    preds = []
    for i in range(0, len(X)+1, batch_size):
        batch = X[i:i + batch_size]
        x_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
            y_batch = torch.softmax(logits, dim=-1).cpu().numpy()  # (batch_size, n_classes)
        preds.append(y_batch)

    y_pred_all = np.concatenate(preds, axis=0)  # shape: (n_preds, n_classes)
    n_preds, n_classes = y_pred_all.shape

    # Инициализируем Series и массив
    labels_series = pd.Series(np.nan, index=data.index)
    probs_array = np.full((len(data), n_classes), np.nan, dtype=np.float32)

    # Заполнение
    labels = np.argmax(y_pred_all, axis=1)
    target_idx = t0 + t_offset
    valid = target_idx < len(data)
    target_idx = target_idx[valid]
    labels = labels[valid]
    y_pred_all = y_pred_all[valid]
    labels_series.iloc[target_idx] = labels
    probs_array[target_idx, :] = y_pred_all

    return labels_series, probs_array
