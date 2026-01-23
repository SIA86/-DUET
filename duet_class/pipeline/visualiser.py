import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import pandas as pd

import mplfinance as mpf
from IPython.display import clear_output


def _slice_around_index(
    df: pd.DataFrame,
    center_idx: int | pd.Timestamp,
    window: int,
) -> tuple[pd.DataFrame, pd.Timestamp, int]:
    """Return (df_slice, center_timestamp, center_pos_in_df) for ±window around center_idx."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    # center_idx -> integer position + timestamp
    if isinstance(center_idx, int):
        pos = int(center_idx)
        if pos < 0 or pos >= len(df):
            raise IndexError(f"center_idx out of range: {pos}")
        ts = df.index[pos]
    else:
        try:
            pos = df.index.get_loc(center_idx)
            if isinstance(pos, slice):
                raise ValueError("Timestamp is not unique in index")
            pos = int(pos)
            ts = pd.Timestamp(center_idx)
        except KeyError as e:
            raise ValueError("center_idx timestamp not found in index") from e

    left = max(0, pos - int(window))
    right = min(len(df), pos + int(window) + 1)

    df_slice = df.iloc[left:right]
    if df_slice.empty:
        raise ValueError("Selected slice is empty")

    return df_slice, ts, pos


def _apply_legends(axes) -> None:
    """Add legends to axes that have labeled artists."""
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="upper left", fontsize=8)


def plot_candles_around_index(
    df: pd.DataFrame,
    center_idx: int | pd.Timestamp,
    window: int,
    *,
    title: str | None = None,
    style: str = "yahoo",
    volume: bool = True,
    figsize: tuple[int, int] = (12, 6),
    vline_color: str = "blue",
    vline_width: float = 1.2,
    returnfig: bool = False,
):
    """Рисует свечи вокруг заданного индекса ±window и подсвечивает центральную свечу."""

    df_slice, ts, _ = _slice_around_index(df, center_idx, window)

    vlines = dict(
        vlines=[ts],
        colors=vline_color,
        linewidths=vline_width,
        alpha=0.9,
    )

    use_volume = bool(volume and ("Volume" in df_slice.columns))

    # Важно: не используем title= в mpf.plot(), чтобы заголовок не налезал на свечи.
    fig, axes = mpf.plot(
        df_slice,
        type="candle",
        style=style,
        volume=use_volume,
        figsize=figsize,
        show_nontrading=True,
        vlines=vlines,
        returnfig=True,
    )

    fig.suptitle(title or f"Candles around {ts}", fontsize=12, y=0.98)

    if returnfig:
        return fig, axes
    return None


def plot_candles_with_add_around_index(
    df: pd.DataFrame,
    center_idx: int | pd.Timestamp,
    window: int,
    *,
    add: dict[int, list[str]] | None = None,
    title: str | None = None,
    style: str = "yahoo",
    volume: bool = True,
    figsize: tuple[int, int] = (12, 7),
    vline_color: str = "blue",
    vline_width: float = 1.2,
    addplot_type: str = "line",
    addplot_width: float = 1.0,
    addplot_alpha: float = 1.0,
    addplot_secondary_y: bool = False,
    legend: bool = True,
    returnfig: bool = False,
):
    """Свечи вокруг индекса ±window + дополнительные графики по словарю add.

    add = {
      0: ['BBL_20_2.0','BBU_20_2.0','BBM_20_2.0'],  # overlay на свечах
      1: ['RSI_14'],                                 # отдельная панель
      2: ['x','y','z'],
    }

    Правила:
    - панель 0: линии рисуются поверх свечей
    - панели 1..N: отдельные панели
    - label берётся из названия колонки (из add) и показывается через legend
    - заголовок рисуется через fig.suptitle() (не через title=)

    Примечание по volume:
    - если volume=True и есть колонка Volume, объём уходит в последнюю панель,
      не ломая вашу нумерацию панелей из add.
    """

    df_slice, ts, _ = _slice_around_index(df, center_idx, window)

    vlines = dict(
        vlines=[ts],
        colors=vline_color,
        linewidths=vline_width,
        alpha=0.9,
    )

    add = add or {}

    # Проверка колонок
    missing = [c for cols in add.values() for c in cols if c not in df_slice.columns]
    if missing:
        raise ValueError(f"Columns not found in df: {sorted(set(missing))}")

    addplots = []
    for panel, cols in add.items():
        for col in cols:
            addplots.append(
                mpf.make_addplot(
                    df_slice[col],
                    panel=int(panel),
                    type=addplot_type,
                    width=addplot_width,  # в mplfinance это width, не linewidth
                    alpha=addplot_alpha,
                    secondary_y=addplot_secondary_y,
                    label=str(col),
                )
            )

    # volume -> последняя панель
    use_volume = bool(volume and ("Volume" in df_slice.columns))
    max_user_panel = max(add.keys(), default=0)
    volume_panel = (max_user_panel + 1) if use_volume else None

    # panel ratios
    panel_ratios = [3]  # panel 0
    if max_user_panel >= 1:
        panel_ratios += [1] * max_user_panel
    if use_volume:
        panel_ratios += [1]

    plot_kwargs = dict(
        type="candle",
        style=style,
        volume=use_volume,
        figsize=figsize,
        tight_layout=False,
        show_nontrading=True,
        vlines=vlines,
        addplot=addplots if addplots else None,
        panel_ratios=panel_ratios,
        returnfig=True,
        warn_too_much_data=10000,
    )
    if use_volume:
        plot_kwargs["volume_panel"] = volume_panel

    fig, axes = mpf.plot(df_slice, **plot_kwargs)

    fig.suptitle(title or f"Candles around {ts}", fontsize=12, y=0.98)

    if legend:
        _apply_legends(axes)

    if returnfig:
        return fig, axes
    return None

def _valid_mask(y_true, y_pred_proba):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if y_pred_proba.ndim == 1:
        return (~np.isnan(y_true)) & (~np.isnan(y_pred_proba))
    elif y_pred_proba.ndim == 2:
        return (~np.isnan(y_true)) & (~np.isnan(y_pred_proba).any(axis=1))
    else:
        raise ValueError(f"y_pred_proba must be 1D or 2D, got ndim={y_pred_proba.ndim}")

def _infer_classes_and_k(y_true, y_pred_proba, n_classes=None):
    """
    Возвращает:
      classes_for_binarize: массив меток классов для label_binarize
      k: сколько классов реально будем строить (чтобы не упасть)
      col_to_label: список подписей для легенды (по колонкам)
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    n_cols = y_pred_proba.shape[1]
    observed = np.unique(y_true)

    # Частый случай: классы кодированы 0..K-1, а proba[:,i] соответствует классу i
    is_int_like = np.issubdtype(observed.dtype, np.integer) or np.all(np.equal(observed, observed.astype(int)))
    if is_int_like:
        y_true_int = observed.astype(int)
        if y_true_int.min() >= 0 and y_true_int.max() < n_cols:
            classes_for_binarize = np.arange(n_cols)
        else:
            # fallback: бинариуем только наблюдаемые классы
            classes_for_binarize = np.sort(observed)
    else:
        classes_for_binarize = np.sort(observed)

    # k — сколько кривых построим
    k = n_cols

    if n_classes is not None:
        try:
            n_classes = int(n_classes)
            if n_classes <= 0:
                n_classes = None
        except Exception:
            n_classes = None

    if n_classes is not None:
        # не даём выйти за границы матрицы вероятностей
        k = min(k, n_classes)

    # Также не даём выйти за границы бинариованной матрицы (если она короче)
    # Важно: label_binarize вернёт столбцов столько, сколько classes_for_binarize
    k = min(k, len(classes_for_binarize))

    # подписи
    col_to_label = [f"class {c}" for c in classes_for_binarize[:k]]

    # если есть несоответствия — просто предупреждаем, но не падаем
    if n_classes is not None and n_classes != n_cols:
        print(f"[warn] n_classes={n_classes} but y_pred_proba has {n_cols} columns. Using k={k}.")
    if len(classes_for_binarize) != n_cols:
        # это нормальная ситуация: например, класс отсутствует в y_true, но модель его предсказывает
        print(f"[warn] y_true has {len(np.unique(y_true))} observed classes, "
              f"binarize uses {len(classes_for_binarize)} classes, proba has {n_cols} columns. Using k={k}.")

    return classes_for_binarize, k, col_to_label

def plot_pr_auc(y_true, y_pred_proba, n_classes=None, title="Precision-Recall Curve"):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    mask = _valid_mask(y_true, y_pred_proba)
    y_true = y_true[mask]
    y_pred_proba = y_pred_proba[mask]

    if y_true.size == 0:
        print("[warn] No valid rows after NaN filtering.")
        return

    # Бинарная классификация (1D score)
    if y_pred_proba.ndim == 1:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        return

    # Мультикласс: One-vs-Rest
    classes_for_binarize, k, labels = _infer_classes_and_k(y_true, y_pred_proba, n_classes=n_classes)
    y_true_bin = label_binarize(y_true, classes=classes_for_binarize)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab10')  # хватает на 10, дальше будет циклично

    for i in range(k):
        y_i = y_true_bin[:, i]
        # нет ни одного positive — PR не имеет смысла
        if np.sum(y_i) == 0:
            print(f"[warn] Skipping PR for {labels[i]}: no positive samples in y_true.")
            continue
        # Иногда столбец может быть весь нули (класс отсутствует в y_true) — AP тогда определён, но кривая тривиальна.
        precision_i, recall_i, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        ap_i = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])

        plt.plot(recall_i, precision_i, lw=2, color=cmap(i % 10),
                 label=f'PR {labels[i]} (AP = {ap_i:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc_auc(y_true, y_pred_proba, n_classes=None, title="ROC Curve"):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    mask = _valid_mask(y_true, y_pred_proba)
    y_true = y_true[mask]
    y_pred_proba = y_pred_proba[mask]

    if y_true.size == 0:
        print("[warn] No valid rows after NaN filtering.")
        return

    # Бинарная классификация (1D score)
    if y_pred_proba.ndim == 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        return

    # Мультикласс: One-vs-Rest
    classes_for_binarize, k, labels = _infer_classes_and_k(y_true, y_pred_proba, n_classes=n_classes)
    y_true_bin = label_binarize(y_true, classes=classes_for_binarize)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab10')

    for i in range(k):
        # Если класс отсутствует в y_true, roc_curve может ругаться (нужны оба класса 0/1).
        # Тогда просто пропускаем без падения.
        y_i = y_true_bin[:, i]
        if np.all(y_i == 0) or np.all(y_i == 1):
            print(f"[warn] Skipping ROC for {labels[i]}: only one class present in y_true_bin.")
            continue

        fpr_i, tpr_i, _ = roc_curve(y_i, y_pred_proba[:, i])
        roc_auc_i = auc(fpr_i, tpr_i)
        plt.plot(fpr_i, tpr_i, lw=2, color=cmap(i % 10),
                 label=f'ROC {labels[i]} (AUC = {roc_auc_i:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
