
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from duet.model import DUETModel
from pipeline.config import DUETConfig
import numpy as np
from tqdm import tqdm


def get_loss_fn(name: str, gamma=None):
    """Возвращает функцию потерь по названию"""
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    else:
        raise ValueError(f"Неизвестная функция потерь: {name}")


def compile_optimizer(model, config):
    """Настраивает оптимизатор"""
    return optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )


def train_model(model, config: DUETConfig, train_loader, val_loader, device="cuda"):
    """Обучает модель DUET"""
    model = model.to(device)
    loss_fn = get_loss_fn(config.loss)
    optimizer = compile_optimizer(model, config)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_losses = []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss = loss_fn(pred, yb)
                val_losses.append(val_loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        if config.verbose:
            print(f"[{epoch+1}/{config.epochs}] Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
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

    return model
