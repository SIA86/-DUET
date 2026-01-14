import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Instance Normalization по временной оси для каждого канала.
    """
    def __init__(self, num_channels: int | None, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.affine = affine
        self.eps = eps
        self.weight = None
        self.bias = None

        if self.affine and self.num_channels is not None:
            self.weight = nn.Parameter(torch.ones(1, 1, self.num_channels))
            self.bias = nn.Parameter(torch.zeros(1, 1, self.num_channels))

    def _init_affine(self, num_channels: int, device: torch.device):
        if self.affine and self.weight is None:
            self.num_channels = num_channels
            self.weight = nn.Parameter(torch.ones(1, 1, num_channels, device=device))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_channels, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        x_norm = (x - mean) / std

        if self.affine:
            self._init_affine(x.shape[-1], x.device)
            x_norm = x_norm * self.weight + self.bias

        return x_norm
