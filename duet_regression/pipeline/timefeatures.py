
import numpy as np
import pandas as pd

def time_features_from_index(index: pd.DatetimeIndex, timeenc: int = 1) -> np.ndarray:
    """
    Возвращает временные признаки из индекса времени:
    - если timeenc = 0: возвращает скалярные индексы [month, day, weekday, hour, minute]
    - если timeenc = 1: возвращает синусоиды по тем же величинам
    """
    if timeenc == 0:
        features = np.stack([
            index.month,
            index.day,
            index.weekday,
            index.hour if hasattr(index, "hour") else np.zeros(len(index)),
            index.minute if hasattr(index, "minute") else np.zeros(len(index))
        ], axis=1)
        return features

    elif timeenc == 1:
        # синусоиды
        month = index.month / 12.0 * 2 * np.pi
        day = index.day / 31.0 * 2 * np.pi
        weekday = index.weekday / 7.0 * 2 * np.pi
        hour = index.hour / 24.0 * 2 * np.pi if hasattr(index, "hour") else np.zeros(len(index))
        minute = index.minute / 60.0 * 2 * np.pi if hasattr(index, "minute") else np.zeros(len(index))

        features = np.stack([
            np.sin(month), np.cos(month),
            np.sin(day), np.cos(day),
            np.sin(weekday), np.cos(weekday),
            np.sin(hour), np.cos(hour),
            np.sin(minute), np.cos(minute)
        ], axis=1)
        return features.astype(np.float32)

    else:
        raise ValueError("timeenc должен быть 0 или 1")
