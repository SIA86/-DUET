from .config import DUETConfig
from . import train as train_module
from .prepare_and_check import FinancialTimeSeriesPreparer
from .wf_slicer import GlobalNormConfig, SplitConfig, WalkForwardWindowSlicerVec, WindowConfig
import pandas as pd
from duet.model import DUETModel
from torch.utils.data import DataLoader, TensorDataset
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_MAP = {
  "STD": "standard",
  "MINMAX": "minmax",
  "QUANT": "quantile",
  "NONE": "none",
}

def _build_slicer(config: DUETConfig, train_ratio: float) -> WalkForwardWindowSlicerVec:
  test_ratio = 0.1
  val_ratio = 1 - train_ratio - test_ratio
  if val_ratio <= 0:
    raise ValueError("train_ratio too large for test_ratio=0.1")
  split = SplitConfig(
    n_folds=1,
    mode="expanding",
    ratios=(train_ratio, val_ratio, test_ratio),
    gap=0,
    step_size=None,
    sliding_train_size=None,
  )

  window = WindowConfig(
    x_window=config.seq_len,
    x_end_offset=0,
    y_window=config.horizon,
    y_end_offset=config.horizon,
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
    y=df[config.forecast],
  )
  fold0 = out["fold_0"]
  x_train = fold0["train"]["X"]
  y_train = fold0["train"]["y"]
  x_val = fold0["val"]["X"]
  y_val = fold0["val"]["y"]

  # --- 4. Создание DataLoader'ов ---
  train_dataset = TensorDataset(
      torch.tensor(x_train, dtype=torch.float32),
      torch.tensor(y_train, dtype=torch.float32)
  )
  val_dataset = TensorDataset(
      torch.tensor(x_val, dtype=torch.float32),
      torch.tensor(y_val, dtype=torch.float32)
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
      device=DEVICE
  )

  return trained_model
