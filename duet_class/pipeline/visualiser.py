import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap


def plot_classification_forecast(
    df: pd.DataFrame,
    price_col="Close",
    target_col="target",
    forecast_col="forecast",
    class_labels: list = None,
    start: int = 0,
    length: int = 200,
    title: str = "DUET Classification Forecast vs Target"
):
    """
    Визуализирует результаты модели для задачи классификации:
    - Верхний график: цена (например, Close)
    - Средний график: реальные классы (target)
    - Нижний график: предсказанные классы (forecast)

    Аргументы:
        df: DataFrame с данными
        price_col: имя колонки с ценой
        target_col: имя колонки с истинными классами
        forecast_col: имя колонки с предсказанными классами
        class_labels: список названий классов (например, ['Down', 'Neutral', 'Up'])
        start: начальный индекс
        length: количество точек
        title: заголовок графика
    """
    end = start + length
    df_plot = df.iloc[start:end]
    x = df_plot.index

    if class_labels is None:
        unique_classes = sorted(df_plot[target_col].dropna().unique())
        class_labels = [f"Class {int(c)}" for c in unique_classes]

    num_classes = len(class_labels)
    cmap = ListedColormap(plt.cm.get_cmap('Set3').colors[:num_classes])

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1.5, 1, 1]})

    # --- График цены ---
    axs[0].plot(df_plot[price_col], color='black', label=price_col)
    axs[0].set_ylabel("Price")
    axs[0].legend()
    axs[0].grid(True)

    # --- График целевых классов (target) ---
    y_target = df_plot[target_col].fillna(-1).astype(int)
    axs[1].fill_between(x, 0, 1, where=(y_target >= 0),
                        color='lightgray', alpha=0.3, label="Valid Data")

    for cls in sorted(y_target.unique()):
        if cls == -1:
            continue
        axs[1].fill_between(x, 0, 1, where=(y_target == cls),
                            color=cmap(cls % num_classes), alpha=0.7, label=f"{class_labels[cls]} (True)")

    axs[1].set_yticks([])
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("Target")
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.0, 1))
    axs[1].grid(True)

    # --- График прогнозов (forecast) ---
    y_pred = df_plot[forecast_col].fillna(-1).astype(int)
    axs[2].fill_between(x, 0, 1, where=(y_pred >= 0),
                        color='lightgray', alpha=0.3, label="Predicted")

    for cls in sorted(y_pred.unique()):
        if cls == -1:
            continue
        axs[2].fill_between(x, 0, 1, where=(y_pred == cls),
                            color=cmap(cls % num_classes), alpha=0.7, label=f"{class_labels[cls]} (Pred)")

    axs[2].set_yticks([])
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel("Forecast")
    axs[2].legend(loc='upper left', bbox_to_anchor=(1.0, 1))
    axs[2].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, right=0.85)
    plt.show()

def plot_pr_auc(y_true, y_pred_proba, n_classes=None, title="Precision-Recall Curve"):
    # Очистка от NaN
    valid_mask = (~np.isnan(y_true)) & (~np.isnan(y_pred_proba).any(axis=1))
    y_true = y_true[valid_mask]
    y_pred_proba = y_pred_proba[valid_mask]
    
    if len(y_pred_proba.shape) == 1:
        # Бинарная классификация
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        # Мультикласс: One-vs-Rest
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        precision = dict()
        recall = dict()
        ap = dict()

        colors = ['blue', 'red', 'green', 'orange', 'purple']
        plt.figure(figsize=(8, 6))

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            ap[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(recall[i], precision[i], color=colors[i],
                     lw=2, label=f'PR curve of class {i} (AP = {ap[i]:.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_roc_auc(y_true, y_pred_proba, n_classes=None, title="ROC Curve"):
    # Проверка на NaN
    valid_mask = (~np.isnan(y_true)) & (~np.isnan(y_pred_proba).any(axis=1))
    y_true = y_true[valid_mask]
    y_pred_proba = y_pred_proba[valid_mask]

    if len(y_pred_proba.shape) == 1:
        # Бинарная классификация
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    else:
        # Мультикласс: One-vs-Rest
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()