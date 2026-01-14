import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from duet.model import DUETModel
from pipeline import evaluate, train
from pipeline.prepare_and_check import FinancialTimeSeriesPreparer
from pipeline.config import DUETConfig
from pipeline.wf_slicer import GlobalNormConfig, SplitConfig, WalkForwardWindowSlicerVec, WindowConfig

SCALER_MAP = {
    "STD": "standard",
    "MINMAX": "minmax",
    "QUANT": "quantile",
    "NONE": "none",
}


def build_toy_dataframe(num_points: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=num_points, freq="h")
    unix_ts = (timestamps.view("int64") // 1_000_000_000).astype(int)

    signal = np.sin(np.linspace(0, 8 * np.pi, num_points)) + 0.1 * rng.standard_normal(num_points)
    feature_1 = signal + 0.05 * rng.standard_normal(num_points)
    feature_2 = np.cos(np.linspace(0, 4 * np.pi, num_points)) + 0.05 * rng.standard_normal(num_points)
    target = signal + 0.05 * rng.standard_normal(num_points)

    return pd.DataFrame(
        {
            "timestamp": unix_ts,
            "feat1": feature_1,
            "feat2": feature_2,
            "target": target,
        }
    )


def build_config(ci: bool, use_router: bool) -> DUETConfig:
    return DUETConfig(
        timestamp_col="timestamp",
        features=["feat1", "feat2"],
        forecast=["target"],
        not_to_normalise=[],
        seq_len=16,
        horizon=4,
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
        use_revin=False,
        timeenc=1,
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

def build_slicer(config: DUETConfig, train_ratio: float) -> WalkForwardWindowSlicerVec:
    val_ratio = (1 - train_ratio) * 0.5
    test_ratio = 1 - train_ratio - val_ratio
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


@pytest.mark.parametrize(
    "ci,use_router",
    [
        (True, False),
        (False, True),
        (True, True),
        (False, False),
    ],
)
def test_train_run_with_toy_dataset(ci: bool, use_router: bool) -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    config = build_config(ci=ci, use_router=use_router)
    df = build_toy_dataframe()

    preparer = FinancialTimeSeriesPreparer(
        tz="UTC",
        timestamp_col="timestamp",
        drop_warmup=True,
    )
    df, _ = preparer.prepare(df, ensure_ohlcv=True)
    slicer = build_slicer(config, train_ratio=0.6)
    out = slicer.split_and_window(
        X=df[config.features],
        y=df[config.forecast],
    )
    fold0 = out["fold_0"]
    x_train = fold0["train"]["X"]
    y_train = fold0["train"]["y"]
    x_val = fold0["val"]["X"]
    y_val = fold0["val"]["y"]

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
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
        preds = model(xb)
    assert preds.shape[1] == config.horizon
    assert preds.shape[2] == len(config.forecast)
