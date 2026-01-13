
import torch
import torch.nn as nn
from duet.components import TCM, CCM, DUETHead
from pipeline import preprocess

class DUETModel(nn.Module):
    """
    Основной класс модели DUET.
    Состоит из:
    - RevIN (опционально)
    - Temporal Clustering Module (TCM)
    - Channel Clustering Module (CCM)
    - Router (опционально)
    - DUETHead
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tcm = TCM(config)
        self.ccm = CCM(config)
        self.channel_proj = None

        self.head = DUETHead(
            d_model=config.d_model,
            horizon=config.horizon,
            c_out=len(config.forecast),
            dropout=config.fc_dropout
        )

    def forward(self, x):
        # x: [B, T, C]
        if self.config.CI:
            outs = []
            for i in range(x.size(-1)):
                xi = x[:, :, i].unsqueeze(-1)  # [B, L, 1]
                outi = self.tcm(xi)            # [B, N, d_model]
                outs.append(outi)
            x = torch.cat(outs, dim=-1)        # [B, N, d_model * D]
            if self.channel_proj is None:
                D = x.size(-1) // self.config.d_model
                self.channel_proj = nn.Linear(self.config.d_model * D, self.config.d_model).to(x.device)
            x = self.channel_proj(x)
        else:
            x = self.tcm(x)  # [B, N, d_model]
            
        x = self.ccm(x)       # [B, N, d_model]
        out = self.head(x)    # [B, horizon]

        return out