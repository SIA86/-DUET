import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from duet.model import DUETModel
from pipeline import evaluate, preprocess, train
from pipeline.prepare_and_check import FinancialTimeSeriesPreparer
from pipeline.config import DUETConfig


def build_toy_dataframe(num_points: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=num_points, freq="h")
    unix_ts = (timestamps.view("int64") // 1_000_000_000).astype(int)

    signal = np.sin(np.linspace(0, 8 * np.pi, num_points)) + 0.1 * rng.standard_normal(num_points)
    feature_1 = signal + 0.05 * rng.standard_normal(num_points)
    feature_2 = np.cos(np.linspace(0, 4 * np.pi, num_points)) + 0.05 * rng.standard_normal(num_points)
    labels = np.digitize(signal, [-0.25, 0.25]).astype(int)

    return pd.DataFrame(
        {
            "timestamp": unix_ts,
            "feat1": feature_1,
            "feat2": feature_2,
            "target": labels,
        }
    )


def build_config(ci: bool, use_router: bool, predict_type: str) -> DUETConfig:
    return DUETConfig(
        timestamp_col="timestamp",
        features=["feat1", "feat2"],
        forecast="target",
        not_to_normalise=[],
        seq_len=16,
        num_classes=3,
        patch_len=4,
        stride=2,
        moving_avg=3,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
        dropout=0.1,
        fc_dropout=0.1,
        activation="gelu",
        num_experts=2,
        report_freq=0,
        CI=ci,
        use_router=use_router,
        timeenc=1,
        predict_type=predict_type,
        batch_size=8,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=3,
        checkpoint_best="",
        checkpoint_final="",
        seed=42,
        verbose=False,
    )


@pytest.mark.parametrize(
    "ci,use_router,predict_type",
    [
        (True, False, "detect"),
        (False, True, "detect"),
        (True, True, "next"),
        (False, False, "next"),
    ],
)
def test_train_run_with_toy_dataset(ci: bool, use_router: bool, predict_type: str) -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    config = build_config(ci=ci, use_router=use_router, predict_type=predict_type)
    df = build_toy_dataframe()

    preparer = FinancialTimeSeriesPreparer(
        tz="UTC",
        timestamp_col="timestamp",
        drop_warmup=True,
    )
    df, _ = preparer.prepare(df, ensure_ohlcv=True)
    df_train, df_val = preprocess.split_dataframe(df, train_ratio=0.8)

    x_train, y_train = preprocess.prepare_windows(df_train, config)
    x_val, y_val = preprocess.prepare_windows(df_val, config)

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = DUETModel(config)
    model = train.train_model(model, config, train_loader, val_loader, device="cpu")

    y_true, y_pred = evaluate.evaluate_model(model, val_loader, device="cpu")
    assert y_true.shape == y_pred.shape
    assert y_true.size > 0

    xb, _ = next(iter(val_loader))
    with torch.no_grad():
        logits = model(xb)
    assert logits.shape[1] == config.num_classes
