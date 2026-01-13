import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from typing import Tuple, List, Dict
from sklearn.base import BaseEstimator
from .timefeatures import time_features_from_index
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer


def check_data(df: pd.DataFrame, config):
    """
    Проверяет, что входной DataFrame корректный:
    - является pd.DataFrame
    - числовые колонки не содержат NaN / inf

    Не проверяет ненужные типы (например, str, datetime)
    """
    assert isinstance(df, pd.DataFrame), "Input is not a DataFrame"

    # Оставляем только числовые колонки для проверки
    numeric_df = df[config.features + config.forecast].select_dtypes(include=np.number)

    if numeric_df.isnull().values.any():
        raise ValueError("DataFrame contains NaN values in numeric columns")

    if np.isinf(numeric_df.values).any():
        raise ValueError("DataFrame contains inf values in numeric columns")


def prepare_time_series(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Подготавливает временной ряд:
    - Если индекс уже datetime → ничего не делает
    - Иначе:
        - Преобразует колонку `timestamp_col` из int UNIX в datetime
        - Делает её индексом
    - Проверяет наличие пропусков по времени
    - Восстанавливает регулярную частоту
    - Заполняет пропуски
    - Удаляет оставшиеся NaN

    Parameters:
        df (pd.DataFrame): исходный DataFrame
        config: объект с параметрами, должен содержать:
            - timestamp_col: имя колонки с UNIX-временем (int)

    Returns:
        pd.DataFrame: готовый временной ряд
    """
    # Проверяем наличие нужной колонки
    if not hasattr(config, 'timestamp_col'):
        raise ValueError("Config must contain 'timestamp_col'")
    timestamp_col = config.timestamp_col

    # 1. Если индекс уже datetime — ничего не делаем
    if not isinstance(df.index, pd.DatetimeIndex):        
        # 2. Иначе: конвертируем UNIX timestamp в datetime
        if timestamp_col not in df.columns:
            raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")

        print(f"Преобразуем {timestamp_col} в datetime")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')  # unit='s' для UNIX timestamp в секундах

        # 3. Делаем новую колонку индексом
        df = df.set_index(timestamp_col).sort_index()
        df = df.rename_axis('datetime')
    else:
        print(f"Индекс уже в формате datetime")
        df = df.rename_axis('datetime')

    # 4. Определяем частоту
    inferred_freq = pd.infer_freq(df.index)
    print(f"Инферированная частота: {inferred_freq}")

    # 5. Проверяем пропуски в индексе
    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=inferred_freq
    )

    missing_dates = ~full_idx.isin(df.index)
    if missing_dates.any():
        print(f"Обнаружены пропущенные даты: {full_idx[missing_dates].min()} — {full_idx[missing_dates].max()}")
        
        # Восстанавливаем полный временной индекс
        df = df.reindex(full_idx)

        # 6. Заполнение пропусков
        df = df.ffill().bfill()
        print("Пропуски заполнены (ffill + bfill)")
    else:
        print("Пропусков в индексе не обнаружено")

    # 7. Удаляем оставшиеся NaN
    before = len(df)
    df = df.dropna(subset=config.features + config.forecast)
    print(f"Окончательно удалено {before - len(df)} строк содеражащих NaN")

    return df


def split_dataframe(df: pd.DataFrame, train_ratio=0.8):
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()

    return df_train, df_val

def append_time_features(df: pd.DataFrame, timeenc=1):
    time_feats = time_features_from_index(df.index, timeenc=timeenc)
    time_features_cols = [f"time_feat_{i}" for i in range(time_feats.shape[1])]
    time_feats = pd.DataFrame(
        time_feats,
        index=df.index,
        columns=time_features_cols
    )
    return pd.concat([df, time_feats], axis=1), time_features_cols


def prepare_windows(df: pd.DataFrame, config):
    """
    Подготовка входов и таргетов с нормализацией и нарезкой на окна.
    Поддерживаются любые sklearn-преобразователи, имеющие fit/transform/inverse_transform.

    Аргументы:
        df (pd.DataFrame): исходный датафрейм
        config: объект с параметрами (seq_len, horizont, features, forecast, not_to_norm)

    Возвращает:
        X (np.ndarray): нормализованные входы
        y (np.ndarray): нормализованные таргеты
        column_index (dict): словарь {имя_колонки: индекс}
        scalers (list): список скейлеров, обученных на каждом окне
    """
    # print(df.shape)
    all_columns = pd.Index(config.features + config.forecast).drop_duplicates().tolist()
    df_with_time_features, time_features_cols = append_time_features(df[all_columns], config.timeenc)
    features_cols = config.features + time_features_cols
    # print(df_with_time_features.shape)

    column_index = {col: i for i, col in enumerate(df_with_time_features.columns)}
    data = df_with_time_features.values
    seq_len = config.seq_len
    horiz = config.horizon
    not_to_norm = set(config.not_to_normalise)

    feature_idxs = [column_index[col] for col in features_cols]
    forecast_idxs = [column_index[col] for col in config.forecast]
    not_to_norm_idxs = [column_index[col] for col in not_to_norm]
    norm_idxs = list(set(feature_idxs + forecast_idxs) - set(not_to_norm_idxs))

    X, y, scalers = [], [], []

    total_len = len(data)
    max_i = total_len - seq_len - horiz + 1

    if config.scaler.upper() == 'STD':
      scaler_template = StandardScaler()
    elif config.scaler.upper() == 'MINMAX':
      scaler_template = MinMaxScaler()
    elif config.scaler.upper() == 'QUANT':
      n_quantiles = min(100, config.seq_len) 
      scaler_template = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')

    for i in tqdm(range(max_i), desc="Windowing"):
        window = data[i:i + seq_len]
        target_window = data[i + seq_len:i + seq_len + horiz]

        # Копия скейлера на каждое окно
        scaler = copy.deepcopy(scaler_template)

        # Нормализация
        scaler.fit(window[:, norm_idxs])
        window_scaled = window.copy()
        target_scaled = target_window.copy()

        window_scaled[:, norm_idxs] = scaler.transform(window[:, norm_idxs])
        target_scaled[:, norm_idxs] = scaler.transform(target_window[:, norm_idxs])

        # Сохраняем
        X.append(window_scaled[:, feature_idxs])
        y.append(target_scaled[:, forecast_idxs])
        scalers.append((scaler, norm_idxs))

    return np.array(X), np.array(y), column_index, scalers


def prepare_windows_for_pred(df: pd.DataFrame, config):
    """
    Подготовка входов и таргетов с нормализацией и нарезкой на окна.
    Поддерживаются любые sklearn-преобразователи, имеющие fit/transform/inverse_transform.

    Аргументы:
        df (pd.DataFrame): исходный датафрейм
        config: объект с параметрами (seq_len, horizont, features, forecast, not_to_norm)

    Возвращает:
        X (np.ndarray): нормализованные входы
        y (np.ndarray): нормализованные таргеты
        column_index (dict): словарь {имя_колонки: индекс}
        scalers (list): список скейлеров, обученных на каждом окне
    """
    # print(df.shape)
    all_columns = pd.Index(config.features + config.forecast).drop_duplicates().tolist()
    df_with_time_features, time_features_cols = append_time_features(df[all_columns], config.timeenc)
    features_cols = config.features + time_features_cols
    # print(df_with_time_features.shape)

    column_index = {col: i for i, col in enumerate(df_with_time_features.columns)}
    data = df_with_time_features.values
    seq_len = config.seq_len
    horiz = config.horizon
    not_to_norm = set(config.not_to_normalise)

    feature_idxs = [column_index[col] for col in features_cols]
    forecast_idxs = [column_index[col] for col in config.forecast]
    not_to_norm_idxs = [column_index[col] for col in not_to_norm]
    norm_idxs = list(set(feature_idxs + forecast_idxs) - set(not_to_norm_idxs))

    X, scalers = [], []

    total_len = len(data)
    max_i = total_len - seq_len + 1

    if config.scaler.upper() == 'STD':
      scaler_template = StandardScaler()
    elif config.scaler.upper() == 'MINMAX':
      scaler_template = MinMaxScaler()
    elif config.scaler.upper() == 'QUANT':
      n_quantiles = min(100, config.seq_len) 
      scaler_template = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')

    for i in range(max_i):
        window = data[i:i + seq_len]

        # Копия скейлера на каждое окно
        scaler = copy.deepcopy(scaler_template)

        # Нормализация
        scaler.fit(window[:, norm_idxs])
        window_scaled = window.copy()
        window_scaled[:, norm_idxs] = scaler.transform(window[:, norm_idxs])

        # Сохраняем
        X.append(window_scaled[:, feature_idxs])
        scalers.append((scaler, norm_idxs))

    return np.array(X), column_index, scalers


def denormalize_target(normalized_targets, scalers, forecast_idxs):
    """
    Денормализация таргета для любых sklearn скейлеров.

    Аргументы:
        normalized_targets (np.ndarray): массив нормализованных таргетов (n_windows, horizon, n_features)
        scalers (list): список кортежей (скейлер, norm_idxs) по окнам
        forecast_idxs (list): индексы колонок в исходном df

    Возвращает:
        np.ndarray: денормализованный таргет
    """
    denorm = []

    for i, target in enumerate(normalized_targets):
        scaler, norm_idxs = scalers[i]
        target_denorm = target.copy()

        # Создаем "фиктивный" массив, где будут только нормализуемые колонки
        # остальные не трогаем
        restored = np.zeros((target.shape[0], len(norm_idxs)))

        for j, idx in enumerate(forecast_idxs):
            if idx in norm_idxs:
                col_pos = norm_idxs.index(idx)
                restored[:, col_pos] = target[:, j]

        # Инверсный трансформ только по нормализуемым колонкам
        restored = scaler.inverse_transform(restored)

        # Вставим обратно
        for j, idx in enumerate(forecast_idxs):
            if idx in norm_idxs:
                col_pos = norm_idxs.index(idx)
                target_denorm[:, j] = restored[:, col_pos]

        denorm.append(target_denorm)

    return np.array(denorm)