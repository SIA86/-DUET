import numpy as np
import torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.config import DUETConfig
from pipeline.predict import predict_dataset_batched, predict_window


def build_config(seq_len: int, num_classes: int) -> DUETConfig:
    return DUETConfig(
        seq_len=seq_len,
        num_classes=num_classes,
        features=["feat1", "feat2"],
        forecast="target",
        not_to_normalise=[],
        scaler="none",
        predict_type="detect",
    )


def build_model(seq_len: int, n_features: int, num_classes: int) -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(seq_len * n_features, num_classes),
    )


def test_predict_window_single() -> None:
    seq_len = 4
    n_features = 2
    num_classes = 3
    config = build_config(seq_len, num_classes)
    model = build_model(seq_len, n_features, num_classes)

    x_window = np.random.RandomState(0).randn(seq_len, n_features).astype(np.float32)
    probs = predict_window(model, x_window, config, device="cpu")

    assert probs.shape == (num_classes,)
    assert np.isclose(probs.sum(), 1.0)


def test_predict_dataset_batched() -> None:
    seq_len = 5
    n_features = 3
    num_classes = 4
    config = build_config(seq_len, num_classes)
    model = build_model(seq_len, n_features, num_classes)

    windows = np.random.RandomState(1).randn(7, seq_len, n_features).astype(np.float32)
    labels, probs = predict_dataset_batched(model, windows, config, batch_size=3, device="cpu")

    assert probs.shape == (windows.shape[0], num_classes)
    assert labels.shape == (windows.shape[0],)
    assert np.all((labels >= 0) & (labels < num_classes))
