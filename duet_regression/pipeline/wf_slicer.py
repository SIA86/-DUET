from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

ArrayLike = Union[np.ndarray, "pd.DataFrame", "pd.Series"]
Mode = Literal["expanding", "sliding"]
ScalerType = Literal["none", "robust", "standard", "minmax", "quantile"]


@dataclass(frozen=True)
class SplitConfig:
    n_folds: int = 1
    mode: Mode = "expanding"
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    step_size: Optional[int] = None
    gap: int = 0
    sliding_train_size: Optional[int] = None


@dataclass(frozen=True)
class WindowConfig:
    """
    Унифицированная логика "якоря" t и смещений.

    X-окно определяется как:
        x_end = t + x_end_offset
        X = [x_end - (x_window - 1), ..., x_end]   (длина x_window)

    Пример: x_window=10, x_end_offset=+3
        X = [t-6, t-5, t-4, t-3, t-2, t-1, t, t+1, t+2, t+3]

    y-окно (если y передан) аналогично:
        y_end = t + y_end_offset
        y = [y_end - (y_window - 1), ..., y_end]   (длина y_window)

    Типичные частные случаи:
    - "X слева от t0, без t0": x_window=lookback, x_end_offset=-1
    - "X слева и включая t0": x_window=lookback+1, x_end_offset=0
    - "y = t+1 (horizon=1)": y_window=1, y_end_offset=+1
    - "y = t0": y_window=1, y_end_offset=0
    """
    x_window: int
    x_end_offset: int = 0

    y_window: int = 0
    y_end_offset: int = 0

    # Если False: X-окно должно полностью лежать внутри части (start >= part_start)
    # Если True: X-окно может залезать левее part_start (но всё равно в пределах [0, T))
    allow_left_context_for_x: bool = False


@dataclass(frozen=True)
class GlobalNormConfig:
    scaler: ScalerType = "none"
    n_quantiles: int = 256
    quantile_clip: Tuple[float, float] = (0.0, 1.0)
    # robust clip: None или (lo, hi) в шкале робаст-скора
    robust_clip: Optional[Tuple[float, float]] = (-10.0, 10.0)


class WalkForwardWindowSlicerVec:
    """
    Векторный walk-forward slicer для временных рядов.

    Возвращаемые формы:
      X: (N, x_window, F)
      y: (N, y_window, Y) или None
      t: (N,) якорные индексы (абсолютные)

    aux (если передан) используется ТОЛЬКО для "skip" логики:
      окно выкидывается, если в соответствующем aux-окне есть хотя бы один 1 (или True).
    """

    def __init__(
        self,
        split: SplitConfig,
        window: WindowConfig,
        global_norm: Optional[GlobalNormConfig] = None,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        local_norm_features: Optional[List[Union[int, str]]] = None,
        no_norm_cols: Optional[List[Union[int, str]]] = None,
        eps: float = 1e-12,
        drop_incomplete_last_fold: bool = True,
        drop_nan_windows: bool = True,
    ):
        if sliding_window_view is None:
            raise ImportError("numpy.sliding_window_view is required (NumPy >= 1.20).")

        self.split = split
        self.window = window
        self.global_norm = global_norm or GlobalNormConfig("none")
        self.feature_names = feature_names
        self.target_names = target_names
        self.local_norm_features = local_norm_features or []
        self.no_norm_cols = no_norm_cols or []
        self.eps = float(eps)
        self.drop_incomplete_last_fold = drop_incomplete_last_fold

        self.drop_nan_windows = bool(drop_nan_windows)

        self._validate()

    def split_and_window(
        self,
        X: ArrayLike,
        y: Optional[Union[ArrayLike, Dict[str, ArrayLike]]] = None,
        aux: Optional[ArrayLike] = None,
    ) -> Dict[str, Any]:
        X_arr, X_cols = self._to_2d_array(X, is_target=False)

        # y can be either a single target array-like or a dict of targets {name: array-like}
        y_mode: Literal["none", "single", "multi"] = "none"
        y_arr, y_cols = (None, None)
        y_dict: Optional[Dict[str, Dict[str, Any]]] = None
        y_cols_dict: Optional[Dict[str, List[str]]] = None

        if y is not None:
            if isinstance(y, dict):
                y_mode = "multi"
                y_dict = {}
                y_cols_dict = {}
                for name, yv in y.items():
                    arr, cols = self._to_2d_array(yv, is_target=True)
                    if arr.shape[0] != X_arr.shape[0]:
                        raise ValueError(f"X and y[{name}] must have same length")
                    y_dict[name] = {"arr": arr, "cols": cols}
                    y_cols_dict[name] = cols
            else:
                y_mode = "single"
                y_arr, y_cols = self._to_2d_array(y, is_target=True)
                if y_arr.shape[0] != X_arr.shape[0]:
                    raise ValueError("X and y must have same length")

        aux_arr, aux_cols = (None, None)
        if aux is not None:
            aux_arr, aux_cols = self._to_2d_array(aux, is_target=False)
            if aux_arr.shape[0] != X_arr.shape[0]:
                raise ValueError("X and aux must have same length")

        T, F = X_arr.shape

        local_norm_idx = self._resolve_idx(self.local_norm_features, X_cols)
        no_norm_idx = self._resolve_idx(self.no_norm_cols, X_cols)
        global_norm_idx = [i for i in range(F) if i not in set(no_norm_idx)]

        folds = self._build_folds(T)

        # ---- Build X windows view ----
        Lx = int(self.window.x_window)
        if Lx <= 0:
            raise ValueError("window.x_window must be > 0")
        if T - Lx + 1 <= 0:
            raise ValueError("Not enough data for given x_window")

        # NumPy returns [T-L+1, F, L] for 2D input; swap to [T-L+1, L, F]
        X_view = sliding_window_view(X_arr, window_shape=(Lx,), axis=0)
        X_view = np.swapaxes(X_view, 1, 2)  # -> [T-Lx+1, Lx, F]

        # ---- Build aux windows view (if provided) ----
        aux_view = None
        if aux_arr is not None:
            if T - Lx + 1 <= 0:
                raise ValueError("Not enough data for aux windows")
            aux_view = sliding_window_view(aux_arr, window_shape=(Lx,), axis=0)
            aux_view = np.swapaxes(aux_view, 1, 2)  # -> [T-Lx+1, Lx, A]

        # ---- Build y windows view (if y provided) ----
        # y_view can be:
        #   - None (no target)
        #   - np.ndarray [T-Ly+1, Ly, Y] for single target
        #   - Dict[str, np.ndarray] for multi-target
        if y_mode != "none":
            Ly = int(self.window.y_window)
            if Ly <= 0:
                raise ValueError("window.y_window must be > 0 when y is provided")
            if T - Ly + 1 <= 0:
                raise ValueError("Not enough data for given y_window")

            if y_mode == "single":
                y_view = sliding_window_view(y_arr, window_shape=(Ly,), axis=0)
                y_view = np.swapaxes(y_view, 1, 2)  # -> [T-Ly+1, Ly, Y]
            else:
                y_view = {}
                assert y_dict is not None
                for name, yinfo in y_dict.items():
                    arr = yinfo["arr"]
                    yv = sliding_window_view(arr, window_shape=(Ly,), axis=0)
                    yv = np.swapaxes(yv, 1, 2)  # -> [T-Ly+1, Ly, Yk]
                    y_view[name] = yv
        else:
            Ly = 0
            y_view = None

        out: Dict[str, Any] = {}
        for k, fold in enumerate(folds):
            tr_s, tr_e = fold["train"]
            gparams = self._fit_global_scaler(X_arr, tr_s, tr_e, global_norm_idx)

            parts = {}
            for part_name in ("train", "val", "test"):
                ps, pe = fold[part_name]

                Xw, yw, t = self._windows_for_part_vectorized(
                    X_view=X_view,
                    y_view=y_view,
                    aux_view=aux_view,
                    part_start=ps,
                    part_end=pe,
                    T=T,
                    Lx=Lx,
                    Ly=Ly,
                    global_norm_idx=global_norm_idx,
                    gparams=gparams,
                    local_norm_idx=local_norm_idx,
                )
                parts[part_name] = {"X": Xw, "y": yw, "t0": t}

            parts["meta"] = {
                "fold_index": k,
                "bounds": fold,
                "feature_columns": X_cols,
                "target_columns": (y_cols_dict if y_mode == "multi" else y_cols),
                "aux_columns": aux_cols,
                "global_norm": {
                    "scaler": self.global_norm.scaler,
                    "no_norm_idx": no_norm_idx,
                    "params": self._summarize_gparams(gparams),
                },
                "configs": {"split": self.split, "window": self.window},
            }
            out[f"fold_{k}"] = parts

        return out

    # ---------------------- Vectorized windowing ----------------------
    def _windows_for_part_vectorized(
        self,
        X_view: np.ndarray,                  # [T-Lx+1, Lx, F]
        y_view: Optional[Union[np.ndarray, Dict[str, np.ndarray]]],        # [T-Ly+1, Ly, Y] or dict
        aux_view: Optional[np.ndarray],      # [T-Lx+1, Lx, A] или None
        part_start: int,
        part_end: int,
        T: int,
        Lx: int,
        Ly: int,
        global_norm_idx: List[int],
        gparams: Dict[str, Any],
        local_norm_idx: List[int],
    ) -> Tuple[np.ndarray, Optional[Union[np.ndarray, Dict[str, np.ndarray]]], np.ndarray]:
        """
        Якорь t принадлежит части: t ∈ [part_start, part_end)

        X окно:
          x_end = t + x_end_offset
          x_start = x_end - (Lx - 1)
          Xw = X_view[x_start]

        Ограничения:
          - x_end < part_end  (чтобы X не залезал в будущее за пределы части)
          - x_end < T
          - x_start >= 0
          - если allow_left_context_for_x=False: x_start >= part_start

        y окно (если y_view != None):
          y_end = t + y_end_offset
          y_start = y_end - (Ly - 1)
          y строго внутри части: y_start >= part_start и y_end < part_end
        """
        x_off = int(self.window.x_end_offset)
        y_off = int(self.window.y_end_offset)

        # аналитические границы для t, чтобы не строить лишние массивы
        t_min = part_start
        t_min = max(t_min, (Lx - 1) - x_off)  # x_start >= 0
        t_max = part_end - x_off              # x_end < part_end  => t < part_end - x_off
        t_max = min(t_max, T - x_off)         # x_end < T        => t < T - x_off

        if not self.window.allow_left_context_for_x:
            # x_start = t + x_off - (Lx-1) >= part_start  => t >= part_start - x_off + (Lx-1)
            t_min = max(t_min, part_start - x_off + (Lx - 1))

        if y_view is not None:
            # y_start >= 0  => t + y_off - (Ly-1) >= 0 => t >= (Ly-1) - y_off
            t_min = max(t_min, (Ly - 1) - y_off)

            # y_end < part_end => t < part_end - y_off
            t_max = min(t_max, part_end - y_off)

            # y_end < T => t < T - y_off
            t_max = min(t_max, T - y_off)

            # y_start >= part_start => t >= part_start - y_off + (Ly-1)
            t_min = max(t_min, part_start - y_off + (Ly - 1))

        if t_max <= t_min:
            X_empty = np.zeros((0, Lx, X_view.shape[2]), dtype=np.float32)
            y_empty = None if y_view is None else np.zeros((0, Ly, y_view.shape[2]), dtype=np.float32)
            return X_empty, y_empty, np.zeros((0,), dtype=np.int64)

        t = np.arange(t_min, t_max, dtype=np.int64)

        x_end = t + x_off
        x_start = x_end - (Lx - 1)

        # bounds для X_view: x_start ∈ [0, T-Lx]
        max_x_start = X_view.shape[0] - 1
        m = (x_start >= 0) & (x_start <= max_x_start) & (x_end >= 0) & (x_end < T) & (x_end < part_end)
        if not np.all(m):
            t = t[m]
            x_start = x_start[m]
            if t.size == 0:
                X_empty = np.zeros((0, Lx, X_view.shape[2]), dtype=np.float32)
                y_empty = None if y_view is None else np.zeros((0, Ly, y_view.shape[2]), dtype=np.float32)
                return X_empty, y_empty, np.zeros((0,), dtype=np.int64)

        Xw = X_view[x_start].astype(np.float32, copy=False)

        # ленивый global scaling: только на Xw
        if gparams.get("scaler", "none") != "none" and global_norm_idx:
            Xw = self._apply_global_scaler_to_windows(Xw, global_norm_idx, gparams)

        # --- SKIP по AUX: если aux_view есть, выбрасываем окна где где-либо aux==1/True
        if aux_view is not None:
            Aw = aux_view[x_start]  # [N, Lx, A]
            bad = np.any(Aw != 0, axis=(1, 2))
            if np.any(bad):
                keep = ~bad
                Xw = Xw[keep]
                t = t[keep]
                x_start = x_start[keep]
                if Xw.shape[0] == 0:
                    y_empty = None if y_view is None else np.zeros((0, Ly, y_view.shape[2]), dtype=np.float32)
                    return Xw, y_empty, np.zeros((0,), dtype=np.int64)

        # --- Drop NaN/Inf windows (X) ---
        if self.drop_nan_windows:
            bad_x = ~np.isfinite(Xw).all(axis=(1, 2))
            if np.any(bad_x):
                keep = ~bad_x
                Xw = Xw[keep]
                t = t[keep]
                x_start = x_start[keep]
                if Xw.shape[0] == 0:
                    y_empty = None if y_view is None else np.zeros((0, Ly, y_view.shape[2]), dtype=np.float32)
                    return Xw, y_empty, np.zeros((0,), dtype=np.int64)

# локальный minmax по окну
        if local_norm_idx:
            Xw = self._local_minmax_windows(Xw, local_norm_idx)

        # y windows: строго внутри части
        if y_view is None:
            yw = None

        elif isinstance(y_view, dict):
            # multi-target y: compute bounds mask once (same for all targets, only channel dims differ)
            any_name = next(iter(y_view.keys()))
            any_view = y_view[any_name]

            y_end = t + y_off
            y_start = y_end - (Ly - 1)

            max_y_start = any_view.shape[0] - 1
            m = (
                (y_start >= 0)
                & (y_start <= max_y_start)
                & (y_end >= 0)
                & (y_end < T)
                & (y_start >= part_start)
                & (y_end < part_end)
            )
            if not np.all(m):
                Xw = Xw[m]
                t = t[m]
                y_start = y_start[m]
                if Xw.shape[0] == 0:
                    y_empty = {k: np.zeros((0, Ly, v.shape[2]), dtype=np.float32) for k, v in y_view.items()}
                    return Xw, y_empty, np.zeros((0,), dtype=np.int64)

            yw = {}
            bad_any = None
            for name, v in y_view.items():
                yk = v[y_start].astype(np.float32, copy=False)
                yw[name] = yk
                if self.drop_nan_windows:
                    bad = ~np.isfinite(yk).all(axis=(1, 2))
                    bad_any = bad if bad_any is None else (bad_any | bad)

            if self.drop_nan_windows and bad_any is not None and np.any(bad_any):
                keep = ~bad_any
                Xw = Xw[keep]
                t = t[keep]
                yw = {name: yk[keep] for name, yk in yw.items()}

        else:
            # single target y
            y_end = t + y_off
            y_start = y_end - (Ly - 1)

            max_y_start = y_view.shape[0] - 1
            m = (
                (y_start >= 0)
                & (y_start <= max_y_start)
                & (y_end >= 0)
                & (y_end < T)
                & (y_start >= part_start)
                & (y_end < part_end)
            )
            if not np.all(m):
                Xw = Xw[m]
                t = t[m]
                y_start = y_start[m]
                if Xw.shape[0] == 0:
                    return Xw, np.zeros((0, Ly, y_view.shape[2]), dtype=np.float32), np.zeros((0,), dtype=np.int64)

            yw = y_view[y_start].astype(np.float32, copy=False)

            if self.drop_nan_windows:
                bad_y = ~np.isfinite(yw).all(axis=(1, 2))
                if np.any(bad_y):
                    keep = ~bad_y
                    Xw = Xw[keep]
                    yw = yw[keep]
                    t = t[keep]

        return Xw, yw, t

    # ---------------------- Global scaler: fit/apply ----------------------

    def _fit_global_scaler(self, X: np.ndarray, start: int, end: int, cols: List[int]) -> Dict[str, Any]:
        scaler = self.global_norm.scaler
        if scaler == "none" or not cols:
            return {"scaler": "none"}

        data = X[start:end, :][:, cols].astype(np.float64, copy=False)
        if data.shape[0] == 0:
            raise ValueError("Empty train slice for scaler fit")

        if scaler == "standard":
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            std = np.where(std < self.eps, 1.0, std)
            return {"scaler": "standard", "mean": mean, "std": std}

        if scaler == "robust":
            med = np.nanmedian(data, axis=0)
            q1 = np.nanpercentile(data, 25, axis=0)
            q3 = np.nanpercentile(data, 75, axis=0)
            iqr = q3 - q1
            iqr = np.where(np.abs(iqr) < self.eps, 1.0, iqr)
            return {"scaler": "robust", "median": med, "iqr": iqr, "clip": self.global_norm.robust_clip}

        if scaler == "minmax":
            mn = np.nanmin(data, axis=0)
            mx = np.nanmax(data, axis=0)
            denom = mx - mn
            denom = np.where(np.abs(denom) < self.eps, 1.0, denom)
            return {"scaler": "minmax", "min": mn, "denom": denom}

        if scaler == "quantile":
            nq = int(self.global_norm.n_quantiles)
            nq = max(8, min(nq, 4096))
            probs = np.linspace(0.0, 1.0, nq)
            qvals = np.nanquantile(data, probs, axis=0)  # [nq, K]
            return {"scaler": "quantile", "probs": probs, "qvals": qvals, "clip": self.global_norm.quantile_clip}

        raise ValueError(f"Unknown scaler: {scaler}")

    def _apply_global_scaler_to_windows(self, Xw: np.ndarray, cols: List[int], p: Dict[str, Any]) -> np.ndarray:
        # Xw: [N, Lx, F], преобразуем только Xw[..., cols]
        scaler = p["scaler"]
        sub = Xw[..., cols].astype(np.float64, copy=False)

        if scaler == "standard":
            sub = (sub - p["mean"]) / p["std"]

        elif scaler == "robust":
            sub = (sub - p["median"]) / p["iqr"]
            clip = p.get("clip", None)
            if clip is not None:
                lo, hi = clip
                sub = np.clip(sub, lo, hi)

        elif scaler == "minmax":
            sub = (sub - p["min"]) / p["denom"]

        elif scaler == "quantile":
            lo, hi = p["clip"]
            sub = self._quantile_transform_windows(sub, p["probs"], p["qvals"], lo=lo, hi=hi)

        else:
            return Xw

        Xw = Xw.copy()
        Xw[..., cols] = sub.astype(np.float32, copy=False)
        return Xw

    def _quantile_transform_windows(
        self,
        sub: np.ndarray,          # [N, L, K]
        probs: np.ndarray,        # [nq]
        qvals: np.ndarray,        # [nq, K]
        lo: float,
        hi: float,
    ) -> np.ndarray:
        N, L, K = sub.shape
        x = sub.reshape(-1, K)  # [M, K], M=N*L
        out = np.empty_like(x, dtype=np.float64)

        for j in range(K):
            v = x[:, j]
            q = qvals[:, j]
            mask = np.isfinite(v)
            if not np.any(mask):
                out[:, j] = v
                continue
            q_mono = np.maximum.accumulate(q)

            idx = np.searchsorted(q_mono, v[mask], side="right")
            idx = np.clip(idx, 1, len(q_mono) - 1)

            q0 = q_mono[idx - 1]
            q1 = q_mono[idx]
            p0 = probs[idx - 1]
            p1 = probs[idx]

            denom = (q1 - q0)
            denom = np.where(np.abs(denom) < self.eps, 1.0, denom)
            u = p0 + (v[mask] - q0) * (p1 - p0) / denom
            u = np.clip(u, lo, hi)

            col = np.full_like(v, np.nan, dtype=np.float64)
            col[mask] = u
            out[:, j] = col

        return out.reshape(N, L, K)

    # ---------------------- Local minmax windows ----------------------

    def _local_minmax_windows(self, Xw: np.ndarray, cols: List[int]) -> np.ndarray:
        """Локальная min-max нормализация внутри каждого окна по оси времени.

        Предполагается, что окна уже очищены от NaN/Inf (если включен drop_nan_windows).
        Но для устойчивости используем nanmin/nanmax.
        """
        Xw = Xw.copy()
        sub = Xw[..., cols]  # [N, L, K]

        mn = np.nanmin(sub, axis=1)  # [N, K]
        mx = np.nanmax(sub, axis=1)  # [N, K]

        # если окно полностью NaN по какой-то фиче, nanmin/nanmax вернут NaN => позже отфильтруем
        denom = mx - mn
        denom = np.where(np.abs(denom) < self.eps, 1.0, denom)

        Xw[..., cols] = (sub - mn[:, None, :]) / denom[:, None, :]
        return Xw

    # ---------------------- Helpers ----------------------

    def _validate(self) -> None:
        tr, va, te = self.split.ratios
        if any(r < 0 for r in (tr, va, te)) or not np.isclose(tr + va + te, 1.0):
            raise ValueError("ratios must be non-negative and sum to 1.0")
        if self.window.x_window <= 0:
            raise ValueError("window.x_window must be > 0")
        if self.split.n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if self.global_norm.scaler not in ("none", "robust", "standard", "minmax", "quantile"):
            raise ValueError("Unsupported scaler")

    def _to_2d_array(self, obj: ArrayLike, is_target: bool) -> Tuple[np.ndarray, List[str]]:
        if pd is not None and isinstance(obj, (pd.Series, pd.DataFrame)):
            if isinstance(obj, pd.Series):
                arr = obj.to_numpy().reshape(-1, 1)
                cols = [obj.name or ("target" if is_target else "feat")]
            else:
                arr = obj.to_numpy()
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                cols = list(obj.columns.astype(str))
            return np.asarray(arr), cols

        arr = np.asarray(obj)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("Input must be 1D or 2D")
        cols = (self.target_names if is_target else self.feature_names) or [
            ("y" if is_target else "x") + str(i) for i in range(arr.shape[1])
        ]
        return arr, cols

    def _resolve_idx(self, feats: List[Union[int, str]], columns: List[str]) -> List[int]:
        if not feats:
            return []
        out: List[int] = []
        for f in feats:
            if isinstance(f, int):
                if f < 0 or f >= len(columns):
                    raise ValueError(f"Index out of range: {f}")
                out.append(f)
            elif isinstance(f, str):
                if f not in columns:
                    raise ValueError(f"Column not found: {f}")
                out.append(columns.index(f))
            else:
                raise TypeError("Columns must be int or str")
        # unique preserve order
        seen = set()
        uniq = []
        for i in out:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        return uniq

    def _summarize_gparams(self, p: Dict[str, Any]) -> Dict[str, Any]:
        s = p.get("scaler", "none")
        if s == "none":
            return {"scaler": "none"}
        if s == "quantile":
            return {"scaler": "quantile", "n_quantiles": int(len(p["probs"]))}
        if s == "robust":
            return {"scaler": "robust", "clip": p.get("clip", None)}
        return {"scaler": s}

    def _build_folds(self, T: int) -> List[Dict[str, Tuple[int, int]]]:
        """Build walk-forward folds on index range [0, T).

        Special-case n_folds==1: split the whole series into train/val/test with gaps.
        """
        n_folds = int(self.split.n_folds)
        tr_r, va_r, te_r = self.split.ratios
        gap = int(self.split.gap)

        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        if not np.isclose(tr_r + va_r + te_r, 1.0):
            raise ValueError("ratios must sum to 1.0")
        if n_folds == 1:
            if va_r == 0 and te_r == 0:
                return [{"train": (0, T), "val": (T, T), "test": (T, T)}]

            usable = T - 2 * gap
            if usable <= 0:
                raise ValueError(f"Not enough data for gap={gap}: need T > 2*gap, got T={T}")

            train_len = int(np.floor(usable * tr_r))
            val_len = int(np.floor(usable * va_r))
            test_len = usable - train_len - val_len

            def ensure_min(len_val: int, ratio: float) -> int:
                return max(1, len_val) if ratio > 0 else 0

            train_len = ensure_min(train_len, tr_r)
            val_len = ensure_min(val_len, va_r)
            test_len = ensure_min(test_len, te_r)

            total = train_len + val_len + test_len
            if total > usable:
                overflow = total - usable
                take = min(overflow, max(0, train_len - (1 if tr_r > 0 else 0)))
                train_len -= take
                overflow -= take
                if overflow > 0:
                    take = min(overflow, max(0, val_len - (1 if va_r > 0 else 0)))
                    val_len -= take
                    overflow -= take
                if overflow > 0:
                    test_len = max(1 if te_r > 0 else 0, test_len - overflow)

            train_start = 0
            train_end = train_start + train_len

            val_start = train_end + gap
            val_end = val_start + val_len

            test_start = val_end + gap
            test_end = min(T, test_start + test_len)

            return [{"train": (train_start, train_end), "val": (val_start, val_end), "test": (test_start, test_end)}]

        if te_r <= 0:
            raise ValueError("test ratio must be > 0")

        # Multi-fold heuristic
        test_len = max(1, int(np.floor((T * te_r) / n_folds)))
        step = test_len if self.split.step_size is None else int(self.split.step_size)
        if step <= 0:
            raise ValueError("step_size must be > 0")

        val_len = max(1, int(np.round(test_len * (va_r / te_r)))) if va_r > 0 else 0
        train_len_from_ratio = max(1, int(np.round(test_len * (tr_r / te_r)))) if tr_r > 0 else 0

        if self.split.mode == "sliding" and self.split.sliding_train_size is not None:
            train_len = int(self.split.sliding_train_size)
            if train_len <= 0:
                raise ValueError("sliding_train_size must be > 0")
        else:
            train_len = train_len_from_ratio

        block_len = train_len + gap + val_len + gap + test_len

        folds: List[Dict[str, Tuple[int, int]]] = []
        test_end = block_len
        for _ in range(n_folds):
            test_start = test_end - test_len
            val_end = test_start - gap
            val_start = val_end - val_len
            train_end = val_start - gap
            train_start = 0 if self.split.mode == "expanding" else train_end - train_len

            fold = {"train": (train_start, train_end), "val": (val_start, val_end), "test": (test_start, test_end)}
            ok = (train_start >= 0 and train_end <= val_start and val_end <= test_start and test_end <= T)
            if not ok:
                if self.drop_incomplete_last_fold:
                    break
                raise ValueError("Cannot build folds with given params and T")
            folds.append(fold)
            test_end += step

        if not folds:
            raise ValueError("No folds could be constructed")
        return folds
