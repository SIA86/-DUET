import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline import pipeline
from pipeline.config import DUETConfig


def build_toy_dataframe(num_points: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=num_points, freq="h")
    unix_ts = (timestamps.view("int64") // 1_000_000_000).astype(int)

    signal = np.sin(np.linspace(0, 6 * np.pi, num_points)) + 0.05 * rng.standard_normal(num_points)
    feature_1 = signal + 0.03 * rng.standard_normal(num_points)
    feature_2 = np.cos(np.linspace(0, 4 * np.pi, num_points)) + 0.03 * rng.standard_normal(num_points)
    target = signal + 0.01 * rng.standard_normal(num_points)

    return pd.DataFrame(
        {
            "timestamp": unix_ts,
            "feat1": feature_1,
            "feat2": feature_2,
            "target": target,
        }
    )


def test_pipeline_train_end_to_end() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    config = DUETConfig(
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
        d_c=4,
        K_t=2,
        K_c=3,
        top_k=2,
        dropout=0.0,
        fc_dropout=0.0,
        report_freq=0,
        use_router=True,
        use_revin=False,
        batch_size=8,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=2,
        checkpoint_best="",
        checkpoint_final="",
        seed=42,
        verbose=False,
    )

    df = build_toy_dataframe()
    model = pipeline.train(df, config)

    assert model is not None
