import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

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
