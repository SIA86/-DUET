
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from . import preprocess
from typing import Any, Tuple
import copy
import tqdm


def predict_window(model, x_window: pd.DataFrame, config: Any, device="cuda") -> np.ndarray:
    assert len(x_window) == config.seq_len , "X_window должно быть размером config.seq_len"
    
    model.eval()

    x_train, col_idx, scalers = preprocess.prepare_windows_for_pred(x_window, config)
    x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred_tensor = model(x_tensor)

    y_pred = y_pred_tensor.cpu().numpy()

    # Денормализация
    forecast_idxs = [col_idx[c] for c in config.forecast]
    y_denorm = preprocess.denormalize_target(y_pred, scalers, forecast_idxs)
    return y_denorm[0]  # (horizon, n_features)


def plot_random_windows(
    model,
    data: pd.DataFrame,
    config,
    n: int = 5,
    device: str = "cuda"
):
    """
    Выбирает n случайных окон из data, делает предикт и рисует графики:
    - входное окно
    - прогноз модели
    - реальный таргет

    Для каждой фичи из config.forecast — отдельный график.
    """

    seq_len = config.seq_len
    horizon = config.horizon
    forecast_cols = config.forecast
    forecast_indices = [data.columns.get_loc(col) for col in forecast_cols]

    max_start = len(data) - seq_len - horizon
    indices = np.random.choice(max_start, size=n, replace=False)

    for idx_example, i in enumerate(indices):
        window_df = data.iloc[i:i + seq_len + horizon]  # полное окно (вход + таргет)
        x_window_df = window_df.iloc[:seq_len]          # вход
        y_true_np = window_df.iloc[seq_len:].values     # таргет как numpy

        # Прогноз (модель + денормализация)
        y_pred_np = predict_window(model, x_window_df, config, device=device)

        # Временные координаты
        t_input = np.arange(seq_len)
        t_forecast = np.arange(seq_len, seq_len + horizon)

        # Графики по фичам
        fig, axes = plt.subplots(len(forecast_indices), 1, figsize=(10, 4 * len(forecast_indices)))
        if len(forecast_indices) == 1:
            axes = [axes]  # унификация

        for ax_idx, feat_idx in enumerate(forecast_indices):
            ax = axes[ax_idx]
            feat_name = data.columns[feat_idx]

            # Построение линий
            ax.plot(t_input, x_window_df.iloc[:, feat_idx], label="Input", color='blue')
            ax.plot(t_forecast, y_true_np[:, feat_idx], 'o-', label="Target", color='green')
            ax.plot(t_forecast, y_pred_np[:, ax_idx], 'x--', label="Prediction", color='red')

            ax.set_title(f"{feat_name}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)

        plt.suptitle(f"Example {idx_example + 1}: Input + Forecast", y=1.02)
        plt.tight_layout()
        plt.show()


def predict_dataset_batched(
    model,
    data: pd.DataFrame,
    config: Any,
    batch_size: int = 512,
    device: str = "cuda"
) -> np.ndarray:
    """
    Делает batched предсказания модели по всему датасету для задачи регрессии.
    
    Возвращает:
        preds_array: np.ndarray формы (len(data), horizon, n_features),
                     с NaN там, где предсказание невозможно.
    """
    model.eval()

    X, col_idx, scalers = preprocess.prepare_windows_for_pred(data, config)  # X.shape = (n_windows, seq_len, n_features)
    # print(f'Scalers {len(scalers)}')

    preds = []
    for i in range(0, len(X), batch_size):
        x_tensor = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            y_batch = model(x_tensor).cpu().numpy()  # (B, horizon, n_features)
        preds.append(y_batch)

    y_pred_all = np.concatenate(preds, axis=0)  # shape: (n_preds, horizon, n_features)
    # print(y_pred_all.shape)
    
    # Денормализация
    forecast_idxs = [col_idx[c] for c in config.forecast]
    y_denorm = preprocess.denormalize_target(y_pred_all, scalers, forecast_idxs)

    # Вставляем предсказания начиная с config.seq_len
    preds_array = np.full((len(data), config.horizon, len(config.forecast)), np.nan, dtype=np.float32)
    preds_array[config.seq_len - 1 : config.seq_len + len(y_pred_all)] = y_denorm

    return preds_array

def preds_array_to_dataframe(preds_array: np.ndarray, index: pd.Index, config) -> pd.DataFrame:
    """
    Преобразует (n_samples, horizon, n_features) в DataFrame с колонками:
    <feature>_t+0, <feature>_t+1, ..., <next_feature>_t+0, ...
    """
    n_samples, horizon, n_features = preds_array.shape

    # Переставить оси: (samples, features, horizon)
    arr = preds_array.transpose(0, 2, 1)  # (n_samples, n_features, horizon)

    # Разворачиваем: (n_samples, n_features * horizon)
    flat = arr.reshape(n_samples, -1)

    # Генерация имён колонок
    columns = [f"{feat}_t+{t+1}" for feat in config.forecast for t in range(horizon)]

    return pd.DataFrame(flat, index=index, columns=columns)
