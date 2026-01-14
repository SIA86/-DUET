
import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    """
    Простая распределительная логика (soft routing):
    принимает вход x, считает агрегацию, передаёт через MLP,
    возвращает веса маршрутизации по экспертам.
    """
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.routing_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: [B, N, d_model]
        # агрегируем по времени
        summary = x.mean(dim=1)  # [B, d_model]
        routing_weights = self.routing_mlp(summary)  # [B, num_experts]
        return routing_weights
