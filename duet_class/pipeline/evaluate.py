import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, data_loader, device="cuda", return_proba: bool = False):
    model.eval()
    preds = []
    trues = []
    probs = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            pred_logits = model(xb).cpu()  # [B, num_classes]
            pred_labels = torch.argmax(pred_logits, dim=1)  # [B]

            preds.append(pred_labels.numpy())
            trues.append(yb.numpy())
            if return_proba:
                probs.append(torch.softmax(pred_logits, dim=1).numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    if return_proba:
        y_pred_proba = np.concatenate(probs, axis=0) if probs else np.array([])
        return y_true, y_pred, y_pred_proba

    return y_true, y_pred


def compute_pr_auc(y_true, y_pred_proba, num_classes: int):
    if y_pred_proba is None or y_pred_proba.size == 0:
        return np.nan

    if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 1:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return average_precision_score(y_true, y_pred_proba)

    class_count = y_pred_proba.shape[1]
    class_count = num_classes if num_classes is not None else class_count
    y_true_bin = label_binarize(y_true, classes=np.arange(class_count))
    ap_scores = []
    for idx in range(class_count):
        class_true = y_true_bin[:, idx]
        if class_true.sum() == 0 or class_true.sum() == class_true.shape[0]:
            ap_scores.append(np.nan)
        else:
            ap_scores.append(average_precision_score(class_true, y_pred_proba[:, idx]))
    return np.nanmean(ap_scores)


def compute_metrics(y_true, y_pred, y_pred_proba=None, num_classes: int = None):
    """
    Считает метрики: accuracy, precision, recall, f1-score
    """
    labels = list(range(num_classes)) if num_classes is not None else None
    report = classification_report(
        y_true,
        y_pred,
        output_dict=False,
        zero_division=0,
        labels=labels,
    )
    print(report)

    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
        labels=labels,
    )
    macro_precision = report_dict['macro avg']['precision']
    macro_recall = report_dict['macro avg']['recall']
    macro_f1 = report_dict['macro avg']['f1-score']
    accuracy = (y_pred == y_true).mean()
    pr_auc = compute_pr_auc(y_true, y_pred_proba, num_classes)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "pr_auc_macro": pr_auc,
    }


def plot_confusion_matrix(y_true, y_pred, class_names=None, num_classes: int = None):
    """
    Рисует матрицу ошибок (confusion matrix)
    """
    labels = list(range(num_classes)) if num_classes is not None else None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names if class_names else None)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_classification_examples(y_true, y_pred, n: int = 5):
    """
    Выводит случайные примеры с истинными и предсказанными метками
    """
    idx = np.random.choice(len(y_true), size=n, replace=False)

    print("\nExamples of predictions:")
    for i in idx:
        true_label = y_true[i]
        pred_label = y_pred[i]
        print(f"Sample {i}: True={true_label}, Pred={pred_label} {'✅' if true_label == pred_label else '❌'}")
