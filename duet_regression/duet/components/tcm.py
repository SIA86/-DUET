
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, patch_len, d_model, ci=True, dropout=0.0):
        super().__init__()
        self.ci = ci
        self.projection = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.patch_len = patch_len

        if self.ci:
            # Отдельная проекция для каждого канала
            for _ in range(1):  # Проекция на 1 признаковое измерение (для seasonal + trend)
                self.projection.append(nn.Linear(patch_len * 2, d_model))
        else:
            self.shared_proj = nn.Linear(patch_len * 2, d_model)

    def forward(self, seasonal, trend):
        # x: [B, T, C] и [B, T, C]
        x = torch.cat([seasonal, trend], dim=-1)   # [B, T, 2*C]
        if self.ci:
            B, T, _ = x.shape
            C = 1
            x = x.view(B, T, C, -1)                 # [B, T, 1, 2]
            out = self.projection[0](x.squeeze(2))  # [B, T, d_model]
        else:
            out = self.shared_proj(x)              # [B, T, d_model]

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
        self.num_experts = config.num_experts
        self.extractor = LinearPatternExtractor(
            patch_len=config.patch_len,
            d_model=config.d_model,
            ci=config.CI,
            dropout=config.dropout
        )
        self.use_router = config.use_router
        if self.use_router:
            # Distributional Router (обучаемый softmax)
            self.router = nn.Sequential(
                nn.Linear(config.d_model, config.num_experts),
                nn.Softmax(dim=-1)
            )
            # K независимых экспертов
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.d_model, config.d_model),
                    nn.ReLU(),
                    nn.Dropout(config.dropout)
                )
                for _ in range(config.num_experts)
            ])

        self.dist_weights = None  # сохраняем для анализа

    def forward(self, x):
        # x: [B, T, C]
        seasonal, trend = self.decomp(x)
        B, T, C = x.shape

        # Patching по временной оси
        n_patches = (T - self.patch_len) // self.stride + 1
        patches_seasonal = []
        patches_trend = []

        for i in range(n_patches):
            s_patch = seasonal[:, i*self.stride : i*self.stride+self.patch_len, :]
            t_patch = trend[:, i*self.stride : i*self.stride+self.patch_len, :]
            patches_seasonal.append(s_patch)
            patches_trend.append(t_patch)

        seasonal = torch.stack(patches_seasonal, dim=1)  # [B, N, patch_len, C]
        trend = torch.stack(patches_trend, dim=1)        # [B, N, patch_len, C]

        # Объединяем патчи обратно: [B, N, patch_len * C]
        seasonal = seasonal.permute(0, 1, 3, 2).reshape(B, -1, self.patch_len)
        trend = trend.permute(0, 1, 3, 2).reshape(B, -1, self.patch_len)

        x_proj = self.extractor(seasonal, trend)  # [B, N, d_model]

        if self.use_router:
            # Router → веса принадлежности патча к каждому кластеру (эксперту)
            dist_weights = self.router(x_proj)        # [B, N, K]
            self.dist_weights = dist_weights          # сохранить для анализа

            # Вызов всех экспертов
            expert_outs = []
            for expert in self.experts:
                expert_out = expert(x_proj)           # [B, N, d_model]
                expert_outs.append(expert_out)

            # Агрегация: взвешенная сумма
            stacked = torch.stack(expert_outs, dim=-1)   # [B, N, d_model, K]
            dist_weights = dist_weights.unsqueeze(2)     # [B, N, 1, K]
            out = torch.sum(stacked * dist_weights, dim=-1)  # [B, N, d_model]

            return out
        else:
            return x_proj
