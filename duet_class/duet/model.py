
import torch
import torch.nn as nn
from duet.components import TCM, CCM, Fusion, RevIN, DUETHead

class DUETModel(nn.Module):
    """
    Основной класс модели DUET.
    Состоит из:
    - RevIN (опционально)
    - Temporal Clustering Module (TCM)
    - Channel Clustering Module (CCM)
    - Fusion Module
    - DUETHead
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        num_channels = None
        if config.features:
            time_features = 0
            has_time_features = any(
                isinstance(name, str) and name.startswith("timeenc_")
                for name in config.features
            )
            if not has_time_features:
                if getattr(config, "timeenc", None) == 1:
                    time_features = 10
                elif getattr(config, "timeenc", None) == 0:
                    time_features = 5
            num_channels = len(config.features) + time_features
        self.use_revin = config.use_revin
        self.revin = RevIN(num_channels=num_channels, affine=config.revin_affine, eps=config.revin_eps)
        self.tcm = TCM(config)
        self.ccm = CCM(config)
        self.fusion = Fusion()
        self.last_W_t = None
        self.last_M = None

        self.head = DUETHead(
            d_model=config.d_model,
            num_classes=config.num_classes,
            dropout=config.fc_dropout
        )

    def forward(self, x):
        # x: [B, T, C]
        if self.use_revin:
            x = self.revin(x)

        z_t, w_t = self.tcm(x)       # [B, N, C, d_model], [B, N, K_t]
        mask = self.ccm(x)           # [B, C, C]
        z = self.fusion(z_t, mask)   # [B, N, d_model]

        self.last_W_t = w_t
        self.last_M = mask

        out = self.head(z)    # [B, num_classes]

        return out
