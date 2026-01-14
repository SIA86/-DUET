import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))

from wf_slicer import (
    GlobalNormConfig,
    SplitConfig,
    WalkForwardWindowSlicerVec,
    WindowConfig,
)


def make_series(length: int, features: int = 1, offset: float = 0.0) -> np.ndarray:
    data = np.arange(length * features, dtype=np.float64).reshape(length, features)
    return data + offset


def build_slicer(
    split: SplitConfig,
    window: WindowConfig,
    global_norm: GlobalNormConfig | None = None,
    local_norm_features: list[int] | None = None,
    no_norm_cols: list[int] | None = None,
    drop_nan_windows: bool = True,
    ) -> WalkForwardWindowSlicerVec:
    return WalkForwardWindowSlicerVec(
        split=split,
        window=window,
        global_norm=global_norm,
        local_norm_features=local_norm_features,
        no_norm_cols=no_norm_cols,
        drop_nan_windows=drop_nan_windows,
    )


def extract_windows(series: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    return np.stack([series[start : start + length] for start in starts], axis=0)


def test_build_folds_single_with_gap() -> None:
    split = SplitConfig(n_folds=1, ratios=(0.6, 0.2, 0.2), gap=1)
    window = WindowConfig(x_window=3)
    slicer = build_slicer(split, window)

    folds = slicer._build_folds(T=30)

    assert folds == [
        {"train": (0, 16), "val": (17, 22), "test": (23, 30)},
    ]


def test_build_folds_multi_expanding() -> None:
    split = SplitConfig(n_folds=2, ratios=(0.6, 0.2, 0.2), gap=1, mode="expanding")
    window = WindowConfig(x_window=3)
    slicer = build_slicer(split, window)

    folds = slicer._build_folds(T=50)

    assert folds == [
        {"train": (0, 15), "val": (16, 21), "test": (22, 27)},
        {"train": (0, 20), "val": (21, 26), "test": (27, 32)},
    ]


def test_build_folds_sliding_with_train_size() -> None:
    split = SplitConfig(
        n_folds=2,
        ratios=(0.6, 0.2, 0.2),
        gap=1,
        mode="sliding",
        sliding_train_size=10,
    )
    window = WindowConfig(x_window=3)
    slicer = build_slicer(split, window)

    folds = slicer._build_folds(T=60)

    assert folds == [
        {"train": (0, 10), "val": (11, 17), "test": (18, 24)},
        {"train": (6, 16), "val": (17, 23), "test": (24, 30)},
    ]


def test_build_folds_invalid_raises_with_incomplete_fold() -> None:
    split = SplitConfig(n_folds=2, ratios=(0.6, 0.2, 0.2), gap=2, mode="expanding")
    window = WindowConfig(x_window=3)
    slicer = build_slicer(split, window, drop_nan_windows=True)
    slicer.drop_incomplete_last_fold = False

    with pytest.raises(ValueError, match="Cannot build folds"):
        slicer._build_folds(T=5)


def test_windows_basic_offsets_no_y() -> None:
    split = SplitConfig(n_folds=1, ratios=(0.5, 0.3, 0.2))
    window = WindowConfig(x_window=3, x_end_offset=0, allow_left_context_for_x=False)
    slicer = build_slicer(split, window)

    X = make_series(10, 1)
    out = slicer.split_and_window(X)
    train = out["fold_0"]["train"]

    expected = np.array([[[0], [1], [2]], [[1], [2], [3]], [[2], [3], [4]]], dtype=np.float32)

    assert np.array_equal(train["X"], expected)
    assert np.array_equal(train["t0"], np.array([2, 3, 4]))
    assert train["y"] is None


def test_windows_allow_left_context() -> None:
    split = SplitConfig(n_folds=1, ratios=(0.3, 0.3, 0.4))
    window = WindowConfig(x_window=3, x_end_offset=0, allow_left_context_for_x=True)
    slicer = build_slicer(split, window)

    X = make_series(10, 1)
    out = slicer.split_and_window(X)
    val = out["fold_0"]["val"]

    expected = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]], dtype=np.float32)

    assert np.array_equal(val["X"], expected)
    assert np.array_equal(val["t0"], np.array([3, 4, 5]))


def test_windows_with_future_offsets_and_y() -> None:
    split = SplitConfig(n_folds=1, ratios=(0.8, 0.1, 0.1))
    window = WindowConfig(x_window=3, x_end_offset=1, y_window=2, y_end_offset=2)
    slicer = build_slicer(split, window)

    X = make_series(10, 1)
    y = make_series(10, 1, offset=100)
    out = slicer.split_and_window(X, y=y)
    train = out["fold_0"]["train"]

    assert np.array_equal(train["t0"], np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(train["X"][0], np.array([[0], [1], [2]], dtype=np.float32))
    assert np.array_equal(train["y"][0], np.array([[102], [103]], dtype=np.float32))
    assert np.array_equal(train["X"][-1], np.array([[4], [5], [6]], dtype=np.float32))
    assert np.array_equal(train["y"][-1], np.array([[106], [107]], dtype=np.float32))


def test_multi_target_y_and_nan_drop() -> None:
    split = SplitConfig(n_folds=1, ratios=(0.5, 0.25, 0.25))
    window = WindowConfig(x_window=3, y_window=1)
    slicer = build_slicer(split, window)

    X = make_series(12, 1)
    y1 = make_series(12, 1, offset=100)
    y2 = make_series(12, 1, offset=200)
    y2[3, 0] = np.nan
    out = slicer.split_and_window(X, y={"y1": y1, "y2": y2})
    train = out["fold_0"]["train"]

    assert set(train["y"].keys()) == {"y1", "y2"}
    assert train["y"]["y1"].shape[0] == train["y"]["y2"].shape[0]
    assert np.all(np.isfinite(train["y"]["y2"]))
    assert 3 not in train["t0"]
    assert train["y"]["y1"].shape[0] == 3


def test_aux_skip_and_drop_nan_windows() -> None:
    split = SplitConfig(n_folds=1, ratios=(0.75, 0.125, 0.125))
    window = WindowConfig(x_window=3)
    slicer = build_slicer(split, window)

    X = make_series(8, 1)
    X[0, 0] = np.nan
    aux = np.zeros((8, 1), dtype=np.float64)
    aux[1, 0] = 1.0

    out = slicer.split_and_window(X, aux=aux)
    train = out["fold_0"]["train"]

    assert np.array_equal(train["t0"], np.array([4, 5]))
    assert np.array_equal(train["X"][0, :, 0], np.array([2, 3, 4], dtype=np.float32))
    assert np.array_equal(train["X"][1, :, 0], np.array([3, 4, 5], dtype=np.float32))


def test_local_minmax_and_no_norm_cols() -> None:
    split = SplitConfig(n_folds=1, ratios=(0.7, 0.2, 0.1))
    window = WindowConfig(x_window=4)
    slicer = build_slicer(split, window, local_norm_features=[0], no_norm_cols=[1])

    X = make_series(10, 2)
    out = slicer.split_and_window(X)
    train = out["fold_0"]["train"]

    col0 = train["X"][:, :, 0]
    col1 = train["X"][:, :, 1]
    t0 = train["t0"]
    x_start = t0 - (window.x_window - 1)
    expected_col1 = extract_windows(X[:, 1], x_start, window.x_window).astype(np.float32)

    assert np.allclose(col0.min(axis=1), 0.0)
    assert np.allclose(col0.max(axis=1), 1.0)
    assert np.array_equal(col1, expected_col1)


@pytest.mark.parametrize("scaler", ["standard", "robust", "minmax", "quantile"])
def test_global_normalization_variants(scaler: str) -> None:
    split = SplitConfig(n_folds=1, ratios=(0.6, 0.2, 0.2))
    window = WindowConfig(x_window=3)
    global_norm = GlobalNormConfig(scaler=scaler, quantile_clip=(0.1, 0.9))
    slicer = build_slicer(split, window, global_norm=global_norm, no_norm_cols=[1])

    X = make_series(10, 2)
    out = slicer.split_and_window(X)
    train = out["fold_0"]["train"]

    normalized = train["X"][:, :, 0]
    untouched = train["X"][:, :, 1]
    t0 = train["t0"]
    x_start = t0 - (window.x_window - 1)
    expected_untouched = extract_windows(X[:, 1], x_start, window.x_window).astype(np.float32)

    assert np.array_equal(untouched, expected_untouched)

    if scaler == "standard":
        mean = np.mean(X[:6, 0])
        std = np.std(X[:6, 0])
        expected = (extract_windows(X[:, 0], x_start, window.x_window) - mean) / std
        assert np.allclose(normalized, expected.astype(np.float32))
    elif scaler == "robust":
        med = np.median(X[:6, 0])
        q1 = np.percentile(X[:6, 0], 25)
        q3 = np.percentile(X[:6, 0], 75)
        iqr = q3 - q1
        expected = (extract_windows(X[:, 0], x_start, window.x_window) - med) / iqr
        assert np.all(np.isfinite(normalized))
        assert np.allclose(normalized, expected.astype(np.float32))
    elif scaler == "minmax":
        mn = np.min(X[:6, 0])
        mx = np.max(X[:6, 0])
        denom = mx - mn
        expected = (extract_windows(X[:, 0], x_start, window.x_window) - mn) / denom
        assert np.allclose(normalized, expected.astype(np.float32))
    else:
        assert np.all(np.isfinite(normalized))
        assert normalized.min() >= 0.1 - 1e-6
        assert normalized.max() <= 0.9 + 1e-6


def test_y_window_required_when_target_provided() -> None:
    split = SplitConfig(n_folds=1, ratios=(0.7, 0.2, 0.1))
    window = WindowConfig(x_window=3, y_window=0)
    slicer = build_slicer(split, window)

    X = make_series(10, 1)
    y = make_series(10, 1)

    with pytest.raises(ValueError, match="y_window must be > 0"):
        slicer.split_and_window(X, y=y)
