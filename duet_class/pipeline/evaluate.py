import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(model, data_loader, device="cuda"):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            pred_logits = model(xb).cpu()  # [B, num_classes]
            pred_labels = torch.argmax(pred_logits, dim=1)  # [B]

            preds.append(pred_labels.numpy())
            trues.append(yb.numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    """
    Считает метрики: accuracy, precision, recall, f1-score
    """
    report = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
    print(report)

    macro_precision = classification_report(y_true, y_pred, output_dict=True)['macro avg']['precision']
    macro_recall = classification_report(y_true, y_pred, output_dict=True)['macro avg']['recall']
    macro_f1 = classification_report(y_true, y_pred, output_dict=True)['macro avg']['f1-score']
    accuracy = (y_pred == y_true).mean()

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Рисует матрицу ошибок (confusion matrix)
    """
    cm = confusion_matrix(y_true, y_pred)
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