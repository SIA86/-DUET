import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from . import preprocess
from . import predict

def evaluate_model(model, data_loader, device="cuda"):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    return y_true, y_pred

def compute_metrics(y_true, y_pred):
    """Вычисляет MAE, RMSE, MAPE, SMAPE"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "smape": smape
    }

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Вычисляет MAPE (в процентах).
    
    Аргументы:
    - y_true: массив истинных значений
    - y_pred: массив предсказанных значений
    
    Возвращает:
    - MAPE (%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    return mape

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Вычисляет SMAPE (в процентах).
    
    Аргументы:
    - y_true: массив истинных значений
    - y_pred: массив предсказанных значений
    
    Возвращает:
    - SMAPE (%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.abs(y_true - y_pred) / np.clip(denominator, 1e-8, None)) * 100
    return smape

def compute_feature_importance(model, x_sample: np.ndarray, config, device="cuda"):
    """
    Вычисляет важность признаков по их вкладу в предсказание.
    Требует model в режиме CI == True.

    x_sample: np.ndarray формы [seq_len, num_features] — одно окно
    Возвращает: Series с важностями по признакам
    """
    model.eval()
    with torch.no_grad():
        # Базовое предсказание
        x_train, col_idx, _ = preprocess.prepare_windows_for_pred(x_sample, config)
        x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            y_pred_tensor = model(x_tensor)

        y_pred = y_pred_tensor.cpu().numpy()

        contribs = []
        # print(x_train.shape)
        for i in range(x_train.shape[-1]):
            x_masked = x_train.copy()
            x_masked[:, i] = 0  # обнуляем один признак

            x_masked_tensor = torch.tensor(x_masked, dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_masked_tensor = model(x_masked_tensor)

            pred_masked = pred_masked_tensor.cpu().numpy()

            delta = np.abs(y_pred - pred_masked).mean()  # среднее отличие
            contribs.append(delta)

    imp = pd.Series(contribs, index=col_idx, name="importance").sort_values(ascending=False)
    _plot_feature_importance(imp)

def _plot_feature_importance(importance_series: pd.Series, top_n: int = None, title: str = "Feature Importance"):
    """
    Рисует горизонтальную гистограмму важности признаков.

    Аргументы:
        importance_series: Series с индексом = имя признака, значением = важность
        top_n: если задано — показать только top N признаков
        title: заголовок графика
    """
    if top_n:
        importance_series = importance_series.head(top_n)

    plt.figure(figsize=(10, 6))
    importance_series[::-1].plot(kind="barh", color="skyblue")
    plt.xlabel("Importance (Δ prediction)")
    plt.title(title)
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()