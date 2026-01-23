# @title back
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FutureEffectConfig:
    horizons: Sequence[int] = (5, 15, 30, 60)   # баров удержания
    price_col: str = "Close"                    # цена входа/выхода
    drop_noise_cluster: bool = True             # исключать cluster_id = -1
    min_samples_per_cluster: int = 10           # фильтр для отчёта


class ClusterFutureEffect:
    """
    Оценка будущего эффекта по кластерам:
    - long/short forward return на горизонтах
    - max drawdown (MAE) внутри окна удержания на горизонтах

    Входы:
      df_prices: DataFrame с колонкой price_col (обычно Close)
      event_indices: индексы событий (позиции int или значения df.index)
      cluster_ids: cluster_id для каждого события (len == len(event_indices))
    """

    def __init__(self, cfg: FutureEffectConfig = FutureEffectConfig()):
        self.cfg = cfg

    def _to_positions(self, df: pd.DataFrame, event_indices: Sequence[Union[int, pd.Timestamp]]) -> np.ndarray:
        if isinstance(event_indices, (np.ndarray, list, tuple)) and len(event_indices) == 0:
            return np.array([], dtype=np.int64)

        idx = df.index
        pos = []
        for e in event_indices:
            if isinstance(e, (int, np.integer)):
                p = int(e)
                if p < 0 or p >= len(df):
                    raise IndexError(f"Event position {p} out of range [0, {len(df)-1}]")
                pos.append(p)
            else:
                # timestamp / label
                try:
                    p = idx.get_loc(e)
                except KeyError:
                    raise KeyError(f"Event index {e} not found in df.index")
                if isinstance(p, slice):
                    raise ValueError(f"Event index {e} is not unique in df.index")
                pos.append(int(p))
        return np.asarray(pos, dtype=np.int64)

    @staticmethod
    def _safe_take(a: np.ndarray, start: int, end: int) -> np.ndarray:
        # returns a[start:end] with bounds safety
        start = max(0, start)
        end = min(len(a), end)
        return a[start:end]

    def evaluate(
        self,
        df_prices: pd.DataFrame,
        event_indices: Sequence[Union[int, pd.Timestamp]],
        cluster_ids: Sequence[int],
        open_price_shift: int = 0
    ) -> Dict[str, pd.DataFrame]:
        cfg = self.cfg

        if cfg.price_col not in df_prices.columns:
            raise ValueError(f"df_prices must contain column '{cfg.price_col}'")

        if len(event_indices) != len(cluster_ids):
            raise ValueError("event_indices and cluster_ids length mismatch")

        prices = np.asarray(df_prices[cfg.price_col], dtype=np.float64)
        pos = self._to_positions(df_prices, event_indices)
        clusters = np.asarray(cluster_ids, dtype=np.int64)

        # optionally drop noise cluster -1
        if cfg.drop_noise_cluster:
            keep = clusters != -1
            pos = pos[keep]
            clusters = clusters[keep]

        if len(pos) == 0:
            empty = pd.DataFrame()
            return {"per_event": empty, "summary": empty}

        rows: List[dict] = []
        for p, c in zip(pos, clusters):
            entry = prices[p + open_price_shift]
            if not np.isfinite(entry) or entry <= 0:
                continue

            for h in cfg.horizons:
                exit_pos = p + h
                if exit_pos >= len(prices):
                    continue

                exit_price = prices[exit_pos]
                if not np.isfinite(exit_price) or exit_price <= 0:
                    continue

                window = self._safe_take(prices, p + 1, p + h + 1)  # future path (exclude entry)
                if window.size == 0:
                    continue

                p_min = float(np.min(window))
                p_max = float(np.max(window))

                # LONG
                ret_long = (exit_price / entry) - 1.0
                dd_long = (p_min / entry) - 1.0  # <= 0 ; worst adverse move while holding long

                # SHORT (PnL is inverse of price move)
                ret_short = (entry / exit_price) - 1.0
                dd_short = (entry / p_max) - 1.0  # <= 0 ; worst adverse move while holding short (price spikes up)

                rows.append({
                    "cluster": int(c),
                    "pos": int(p),
                    "timestamp": df_prices.index[p],
                    "h": int(h),

                    "entry": float(entry),
                    "exit": float(exit_price),

                    "ret_long": float(ret_long),
                    "dd_long": float(dd_long),

                    "ret_short": float(ret_short),
                    "dd_short": float(dd_short),
                })

        per_event = pd.DataFrame(rows)
        if per_event.empty:
            empty = pd.DataFrame()
            return {"per_event": empty, "summary": empty}

        # Summary per cluster & horizon
        def _agg(col: str):
            return {
                (col, "mean"): "mean",
                (col, "median"): "median",
                (col, "std"): "std",
                (col, "p25"): lambda s: float(np.nanpercentile(s, 25)),
                (col, "p75"): lambda s: float(np.nanpercentile(s, 75)),
                (col, "winrate"): lambda s: float(np.mean(np.asarray(s) > 0)),
            }

        # build aggregation dict
        agg = {}
        for col in ["ret_long", "dd_long", "ret_short", "dd_short"]:
            agg.update(_agg(col))
        # count
        agg[("meta", "count")] = ("ret_long", "size")

        # pandas requires mapping col->list of funcs, easier: groupby then apply
        grp = per_event.groupby(["cluster", "h"], sort=True)

        summary = grp.agg(
            ret_long_mean=("ret_long", "mean"),
            ret_long_median=("ret_long", "median"),
            ret_long_std=("ret_long", "std"),
            ret_long_p25=("ret_long", lambda s: float(np.nanpercentile(s, 25))),
            ret_long_p75=("ret_long", lambda s: float(np.nanpercentile(s, 75))),
            ret_long_winrate=("ret_long", lambda s: float(np.mean(np.asarray(s) > 0))),

            dd_long_mean=("dd_long", "mean"),
            dd_long_median=("dd_long", "median"),
            dd_long_std=("dd_long", "std"),
            dd_long_p25=("dd_long", lambda s: float(np.nanpercentile(s, 25))),
            dd_long_p75=("dd_long", lambda s: float(np.nanpercentile(s, 75))),

            ret_short_mean=("ret_short", "mean"),
            ret_short_median=("ret_short", "median"),
            ret_short_std=("ret_short", "std"),
            ret_short_p25=("ret_short", lambda s: float(np.nanpercentile(s, 25))),
            ret_short_p75=("ret_short", lambda s: float(np.nanpercentile(s, 75))),
            ret_short_winrate=("ret_short", lambda s: float(np.mean(np.asarray(s) > 0))),

            dd_short_mean=("dd_short", "mean"),
            dd_short_median=("dd_short", "median"),
            dd_short_std=("dd_short", "std"),
            dd_short_p25=("dd_short", lambda s: float(np.nanpercentile(s, 25))),
            dd_short_p75=("dd_short", lambda s: float(np.nanpercentile(s, 75))),

            count=("ret_long", "size"),
        ).reset_index()

        # optional: filter small clusters
        if cfg.min_samples_per_cluster > 1:
            summary = summary[summary["count"] >= cfg.min_samples_per_cluster].reset_index(drop=True)

        return {"per_event": per_event, "summary": summary}


class ClusterFutureEffectViz:
    """
    Визуализация результатов ClusterFutureEffect (summary DataFrame).
    summary ожидается в формате, который возвращает evaluate():
      columns: cluster, h, ret_long_mean, ret_long_winrate, dd_long_mean,
               ret_short_mean, ret_short_winrate, dd_short_mean, count, ...
    """

    @staticmethod
    def _pivot(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
        # rows=cluster, cols=h
        p = summary.pivot_table(index="cluster", columns="h", values=value_col, aggfunc="mean")
        # сортировка кластеров по "среднему по горизонтам" (для читабельности)
        p = p.loc[p.mean(axis=1).sort_values(ascending=False).index]
        return p

    @staticmethod
    def _heatmap(pivot_df: pd.DataFrame, title: str, fmt: str = ".3f"):
        data = pivot_df.to_numpy()
        fig, ax = plt.subplots(figsize=(1.2 * (data.shape[1] + 2), 0.35 * (data.shape[0] + 6)))
        im = ax.imshow(data, aspect="auto")

        ax.set_title(title)
        ax.set_xlabel("horizon (bars)")
        ax.set_ylabel("cluster")

        ax.set_xticks(np.arange(pivot_df.shape[1]))
        ax.set_xticklabels([str(c) for c in pivot_df.columns.tolist()])

        ax.set_yticks(np.arange(pivot_df.shape[0]))
        ax.set_yticklabels([str(i) for i in pivot_df.index.tolist()])

        # подписи значений (если кластеров не слишком много)
        if data.shape[0] <= 40 and data.shape[1] <= 8:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    v = data[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, format(float(v), fmt), ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    def plot_heatmaps(self, summary: pd.DataFrame):
        """
        4 heatmap'а:
        - mean return long
        - mean drawdown long
        - mean return short
        - mean drawdown short
        """
        for col, title in [
            ("ret_long_mean",  "Mean forward return (LONG)"),
            ("dd_long_mean",   "Mean max drawdown (LONG)  (<=0)"),
            ("ret_short_mean", "Mean forward return (SHORT)"),
            ("dd_short_mean",  "Mean max drawdown (SHORT) (<=0)"),
        ]:
            p = self._pivot(summary, col)
            self._heatmap(p, title, fmt=".3f")


    @staticmethod
    def plot_winrate_bars(
        summary: pd.DataFrame,
        horizons: list[int] | tuple[int, ...],
        *,
        side: str = "long",          # "long" | "short"
        min_count: int = 30,
        top_k: int = 20,             # сколько кластеров показывать
        sort_by: str = "mean",       # "mean" | "last_horizon"
    ):
        """
        Группированный bar chart winrate по кластерам для выбранных горизонтов.

        summary: DataFrame из ClusterFutureEffect.evaluate()["summary"]
        Требует колонки:
          cluster, h, count,
          ret_long_winrate / ret_short_winrate
        """
        if side not in ("long", "short"):
            raise ValueError("side must be 'long' or 'short'")

        win_col = f"ret_{side}_winrate"

        df = summary.copy()
        df = df[df["h"].isin(horizons) & (df["count"] >= min_count)]
        if df.empty:
            raise ValueError("Нет данных после фильтрации (проверь horizons/min_count).")

        pivot = df.pivot_table(index="cluster", columns="h", values=win_col, aggfunc="mean")

        # Сортировка кластеров
        if sort_by == "mean":
            order = pivot.mean(axis=1).sort_values(ascending=False).index
        elif sort_by == "last_horizon":
            last_h = max(horizons)
            if last_h not in pivot.columns:
                raise ValueError("last_horizon not present in pivot columns after filtering")
            order = pivot[last_h].sort_values(ascending=False).index
        else:
            raise ValueError("sort_by must be 'mean' or 'last_horizon'")

        pivot = pivot.loc[order].head(top_k)

        clusters = pivot.index.to_list()
        hs = list(pivot.columns.to_list())
        data = pivot.to_numpy(dtype=float)  # (top_k, len(hs))

        n_clusters = len(clusters)
        n_h = len(hs)

        x = np.arange(n_clusters)
        group_width = 0.8
        bar_w = group_width / max(1, n_h)

        fig, ax = plt.subplots(figsize=(max(12, 0.6 * n_clusters + 4), 5))

        for j, h in enumerate(hs):
            ax.bar(
                x - group_width/2 + (j + 0.5) * bar_w,
                data[:, j],
                width=bar_w,
                label=f"h={h}",
            )

        ax.set_title(f"Winrate by cluster | {side.upper()} | horizons={hs} | min_count={min_count}")
        ax.set_xlabel("cluster")
        ax.set_ylabel("winrate")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in clusters], rotation=0)
        ax.axhline(0.5, linewidth=1)  # ориентир 50%
        ax.set_ylim(0.0, 1.0)
        ax.legend(ncol=min(4, n_h))

        plt.tight_layout()
        plt.show()


