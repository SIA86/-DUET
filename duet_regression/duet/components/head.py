
import torch
import torch.nn as nn

import torch.nn as nn

class DUETHead(nn.Module):
    """
    Финальный head модели DUET:
    - проецирует выходы encoder'а на прогноз только c_out
    - учитывает длину прогноза (horizon)
    """

    def __init__(self, d_model: int, horizon: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.horizon = horizon
        self.c_out = c_out

        # Head без проекции на c_out
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.horizon * self.c_out), 
        )

    def forward(self, x):
        # x: [B, N, d_model]
        x = x.mean(dim=1)  # [B, d_model]
        out = self.head(x)  # [B, horizon * c_out]
        out = out.view(-1, self.horizon, self.c_out)  # [B, horizon, c_out]

        return out