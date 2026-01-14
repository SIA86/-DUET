import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from duet.model import DUETModel  # должен возвращать [B, num_classes]
from pipeline.config import DUETConfig
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# --- Loss functions ---
class WeightedCrossEntropyLoss(nn.Module):
    """Взвешенная CrossEntropy для несбалансированных классов"""
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.weights is not None:
            ce = nn.CrossEntropyLoss(weight=self.weights)
        else:
            ce = nn.CrossEntropyLoss()
        return ce(input, target.long())


# --- Optimizer ---
def compile_optimizer(model, config):
    return optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

# --- Accuracy metric ---
def calculate_accuracy(preds, targets):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)

# --- Training loop ---
# --- Training loop ---
def train_model(model, config: DUETConfig, train_loader, val_loader, device="cuda", class_weights=None):
    """
    Обучает модель для задачи классификации.
    Раз в config.report_freq эпох выводит confusion matrix.
    """
    model = model.to(device)

    loss_fn = WeightedCrossEntropyLoss(weights=class_weights)
    optimizer = compile_optimizer(model, config)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        train_accs = []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            acc = calculate_accuracy(pred, yb)
            train_accs.append(acc)

        model.eval()
        val_losses = []
        val_accs = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())
                acc = calculate_accuracy(pred, yb)
                val_accs.append(acc)

                # Сохраняем предсказания и таргеты
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)

        if config.verbose:
            print(f"[{epoch+1}/{config.epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        # --- Вывод confusion matrix раз в report_freq эпох ---
        if config.report_freq and (epoch + 1) % config.report_freq == 0:

            cm = confusion_matrix(all_targets, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(all_targets))
            disp.plot(cmap='Blues')
            plt.title(f"Confusion Matrix — Epoch {epoch+1}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            if config.checkpoint_best:
                print(f'Saving best val_acc model weights to {config.checkpoint_best}')
                torch.save(best_state, config.checkpoint_best)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if config.verbose:
                    print("🔁 Early stopping triggered")
                break

    if config.checkpoint_final:
        print(f'Saving final model weights to {config.checkpoint_final}')
        torch.save(model.state_dict(), config.checkpoint_final)
        
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    return model
