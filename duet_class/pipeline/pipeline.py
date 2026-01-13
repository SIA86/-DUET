from .config import DUETConfig
from . import preprocess
import pandas as pd
from ..duet.model import DUETModel
from torch.utils.data import DataLoader, TensorDataset
import torch
import joblib
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(df: pd.DataFrame, config):
  df = preprocess.prepare_time_series(df, config) # проверка пропусков, установка datetime индекса
  preprocess.check_data(df, config) # проверка на nan & inf
  df_train, df_val = preprocess.split_dataframe(df, train_ratio=0.75)

  # --- 2. Создание окон и меток ---
  # Предположим, что у вас есть отдельная колонка с метками классов: например, 'target'
  # Если нет, то вам нужно определить логику создания меток из временных окон

  # Создаем окна и метки классов
  x_train, y_train = preprocess.prepare_windows(df_train, config)
  x_val, y_val = preprocess.prepare_windows(df_val, config)


  # Балансируем (если не нужно, то закомментировать)
  x_train, y_train = preprocess.balance_windows(x_train, y_train)
  x_val, y_val = preprocess.balance_windows(x_val, y_val)

  # Проверяем балансировку
  unique, counts = np.unique(y_train, return_counts=True)
  print("Классы после балансировки:", dict(zip(unique, counts)))

  # --- 3. Подсчет весов классов ---
  class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
  class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
  print(f"Распределение весов между класcами {class_weights}")

  # --- 4. Создание DataLoader'ов ---
  train_dataset = TensorDataset(
      torch.tensor(x_train, dtype=torch.float32),
      torch.tensor(y_train, dtype=torch.long)
  )
  val_dataset = TensorDataset(
      torch.tensor(x_val, dtype=torch.float32),
      torch.tensor(y_val, dtype=torch.long)
  )

  train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

  # --- 6. Инициализация модели ---
  model = DUETModel(config).to(DEVICE)

  # --- 7. Обучение модели ---
  trained_model = train.train_model(
      model,
      config,
      train_loader,
      val_loader,
      device=DEVICE,
      class_weights = class_weights
  )

  return model