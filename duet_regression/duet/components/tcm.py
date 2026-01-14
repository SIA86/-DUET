
import torch
import torch.nn as nn

class SeriesDecomposition(nn.Module):
    """
    Простая скользящая фильтрация для выделения тренда и сезонности
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)

    def forward(self, x):
        # x: [B, T, C]
        x_t = x.permute(0, 2, 1)                          # → [B, C, T]
        trend = self.avg_pool(x_t)                        # сглаженный тренд
        trend = trend.permute(0, 2, 1)                    # → [B, T, C]
        seasonal = x - trend
        return seasonal, trend

class LinearPatternExtractor(nn.Module):
    """
    Линейная проекция патча (в духе Transformer patch embedding)
    """
    def __init__(self, patch_len, d_model, dropout=0.0):
        super().__init__()
        self.projection = nn.Linear(patch_len * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seasonal, trend):
        # seasonal/trend: [B, N, C, patch_len]
        x = torch.cat([seasonal, trend], dim=-1)   # [B, N, C, 2*patch_len]
        out = self.projection(x)
        return self.dropout(out)

class TCM(nn.Module):
    """
    Temporal Clustering Module
    - Seasonal/trend декомпозиция
    - Patching (по времени)
    - Линейная проекция патчей
    """
    def __init__(self, config):
        super().__init__()
        self.decomp = SeriesDecomposition(config.moving_avg)
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.num_clusters = config.K_t
        self.use_router = config.use_router
        self.extractors = nn.ModuleList([
            LinearPatternExtractor(
                patch_len=config.patch_len,
                d_model=config.d_model,
                dropout=config.dropout
            )
            for _ in range(config.K_t)
        ])
        self.router = nn.Sequential(
            nn.Linear(config.patch_len, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.K_t)
        )

    def forward(self, x):
        # x: [B, T, C]
        seasonal, trend = self.decomp(x)
        B, T, C = x.shape

        seasonal = seasonal.transpose(1, 2).unfold(dimension=2, size=self.patch_len, step=self.stride)
        trend = trend.transpose(1, 2).unfold(dimension=2, size=self.patch_len, step=self.stride)
        seasonal = seasonal.permute(0, 2, 1, 3)  # [B, N, C, patch_len]
        trend = trend.permute(0, 2, 1, 3)        # [B, N, C, patch_len]

        patch_summary = (seasonal + trend).mean(dim=2)  # [B, N, patch_len]
        if self.use_router:
            weights_router = torch.softmax(self.router(patch_summary), dim=-1)
        else:
            weights_router = torch.full(
                (B, patch_summary.size(1), self.num_clusters),
                1.0 / self.num_clusters,
                device=x.device,
                dtype=x.dtype,
            )

        expert_outputs = [extractor(seasonal, trend) for extractor in self.extractors]
        stacked = torch.stack(expert_outputs, dim=3)  # [B, N, C, K_t, d_model]
        weights = weights_router.unsqueeze(2).unsqueeze(-1)  # [B, N, 1, K_t, 1]
        out = (stacked * weights).sum(dim=3)                 # [B, N, C, d_model]
        return out, weights_router
