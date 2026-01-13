import torch
import torch.nn as nn

class DUETHead(nn.Module):
    """
    Финальный head модели DUET для задачи классификации:
    - принимает выходы encoder'а
    - возвращает логиты/вероятности классов
    """

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.0, activation: str = "softmax"):
        super().__init__()

        # Активация
        self.activation = nn.Softmax(dim=1)
        
        # Head для классификации
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),  # ⬅️ Выход по числу классов
        )

    def forward(self, x):
        # x: [B, N, d_model]
        x = x.mean(dim=1)           # [B, d_model], усреднение по последовательности
        logits = self.head(x)       # [B, num_classes]
        return self.activation(logits)