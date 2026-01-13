from .config import DUETConfig
from . import preprocess, train as train_module
from .prepare_and_check import FinancialTimeSeriesPreparer
from .wf_slicer import GlobalNormConfig, SplitConfig, WalkForwardWindowSlicerVec, WindowConfig
import pandas as pd
from duet.model import DUETModel
from torch.utils.data import DataLoader, TensorDataset
import torch
import joblib
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_MAP = {
  "STD": "standard",
  "MINMAX": "minmax",
  "QUANT": "quantile",
  "NONE": "none",
}

def _build_slicer(config: DUETConfig, train_ratio: float) -> WalkForwardWindowSlicerVec:
  split = SplitConfig(
    n_folds=1,
    mode="expanding",
    ratios=(train_ratio, 1 - train_ratio, 0.0),
    gap=0,
    step_size=None,
    sliding_train_size=None,
  )

  predict_type = config.predict_type.upper()
  if predict_type == "DETECT":
    y_end_offset = 0
  elif predict_type == "NEXT":
    y_end_offset = 1
  else:
    raise ValueError("config.predict_type должен быть 'DETECT' или 'NEXT'")

  window = WindowConfig(
    x_window=config.seq_len,
    x_end_offset=0,
    y_window=1,
    y_end_offset=y_end_offset,
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

def train(df: pd.DataFrame, config):
  preparer = FinancialTimeSeriesPreparer(
    tz="UTC",
    timestamp_col="timestamp",
    drop_warmup=True,
  )
  df, _ = preparer.prepare(df, ensure_ohlcv=True)
  slicer = _build_slicer(config, train_ratio=0.75)
  out = slicer.split_and_window(
    X=df[config.features],
    y=df[[config.forecast]],
  )
  fold0 = out["fold_0"]
  x_train = fold0["train"]["X"]
  y_train = fold0["train"]["y"][:, 0, 0].astype(int)
  x_val = fold0["val"]["X"]
  y_val = fold0["val"]["y"][:, 0, 0].astype(int)

  # Балансируем только train (val/test оставляем в исходном распределении)
  x_train, y_train = preprocess.balance_windows(x_train, y_train)

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
  trained_model = train_module.train_model(
      model,
      config,
      train_loader,
      val_loader,
      device=DEVICE,
      class_weights = class_weights
  )

  return trained_model
