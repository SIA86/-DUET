import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from .timefeatures import time_features_from_index
from collections import defaultdict


def check_data(df: pd.DataFrame, config):
    """
    Проверяет, что входной DataFrame корректный:
    - является pd.DataFrame
    - числовые колонки не содержат NaN / inf

    Не проверяет ненужные типы (например, str, datetime)
    """
    assert isinstance(df, pd.DataFrame), "Input is not a DataFrame"

    # Оставляем только числовые колонки для проверки
    numeric_df = df[config.features + [config.forecast]].select_dtypes(include=np.number)

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
    df = df.dropna(subset=config.features + [config.forecast])
    print(f"Окончательно удалено {before - len(df)} строк содеражащих NaN")

    return df

def split_dataframe(df: pd.DataFrame, train_ratio=0.8):
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()

    return df_train, df_val

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
        df (pd.DataFrame):precision=0.73, recall=0.87 исходный датафрейм
        config: объект с параметрами (seq_len, horizont, features, forecast, not_to_norm)

    Возвращает:
        X (np.ndarray): нормализованные входы
        y (np.ndarray): нормализованные таргеты
    """
    # print(df.shape)
    all_columns = pd.Index(config.features + [config.forecast]).drop_duplicates().tolist()
    df_with_time_features, time_features_cols = append_time_features(df[all_columns], config.timeenc)
    features_cols = config.features + time_features_cols
    # print(df_with_time_features.shape)

    column_index = {col: i for i, col in enumerate(df_with_time_features.columns)}
    data = df_with_time_features.values
    seq_len = config.seq_len
    not_to_norm = set(config.not_to_normalise)

    feature_idxs = [column_index[col] for col in features_cols]
    forecast_idxs = [column_index[config.forecast]]
    not_to_norm_idxs = [column_index[col] for col in not_to_norm]
    norm_idxs = list(set(feature_idxs + forecast_idxs) - set(not_to_norm_idxs))

    X, y, scalers = [], [], []

    total_len = len(data)
    if config.predict_type.upper() == 'DETECT':
        max_i = total_len - seq_len + 1 
    elif config.predict_type.upper() == 'NEXT':
        max_i = max(1, total_len - seq_len)

    if config.scaler.upper() == 'STD':
      scaler_template = StandardScaler()
    elif config.scaler.upper() == 'MINMAX':
      scaler_template = MinMaxScaler()
    elif config.scaler.upper() == 'QUANT':
      n_quantiles = min(100, config.seq_len) 
      scaler_template = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')

    for i in range(max_i):
        window = data[i:i + seq_len]
        if config.predict_type.upper() == 'DETECT':
            target_window = data[i + seq_len - 1, forecast_idxs][None,...]
        elif config.predict_type.upper() == 'NEXT':
            target_window = data[i + seq_len, forecast_idxs][None,...]

        # Копия скейлера на каждое окно
        scaler = copy.deepcopy(scaler_template)

        # Нормализация
        scaler.fit(window[:, norm_idxs])
        window_scaled = window.copy()
        window_scaled[:, norm_idxs] = scaler.transform(window[:, norm_idxs])

        # Сохраняем
        X.append(window_scaled[:, feature_idxs])
        y.append(target_window.squeeze())

    return np.array(X), np.array(y)


def prepare_windows_for_pred(df: pd.DataFrame, config):
    """
    Подготовка входов и таргетов с нормализацией и нарезкой на окна.
    Поддерживаются любые sklearn-преобразователи, имеющие fit/transform/inverse_transform.

    Аргументы:
        df (pd.DataFrame):precision=0.73, recall=0.87 исходный датафрейм
        config: объект с параметрами (seq_len, horizont, features, forecast, not_to_norm)

    Возвращает:
        X (np.ndarray): нормализованные входы
    """
    # print(df.shape)
    all_columns = pd.Index(config.features + [config.forecast]).drop_duplicates().tolist()
    df_with_time_features, time_features_cols = append_time_features(df[all_columns], config.timeenc)
    features_cols = config.features + time_features_cols
    # print(df_with_time_features.shape)

    column_index = {col: i for i, col in enumerate(df_with_time_features.columns)}
    data = df_with_time_features.values
    seq_len = config.seq_len
    not_to_norm = set(config.not_to_normalise)

    feature_idxs = [column_index[col] for col in features_cols]
    forecast_idxs = [column_index[config.forecast]]
    not_to_norm_idxs = [column_index[col] for col in not_to_norm]
    norm_idxs = list(set(feature_idxs + forecast_idxs) - set(not_to_norm_idxs))

    X, scalers = [], []

    total_len = len(data)
    if config.predict_type.upper() == 'DETECT':
        max_i = total_len - seq_len + 1
    elif config.predict_type.upper() == 'NEXT':
        max_i = max(1, total_len - seq_len)

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

    return np.array(X)

