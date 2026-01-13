import torch
import torch.nn as nn


class Fusion(nn.Module):
    """
    Masked channel mixing с разреженной маской.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z_t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # z_t: [B, N, C, d_model]
        # mask: [B, C, C]
        row_sum = mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mask_norm = mask / row_sum
        mixed = torch.einsum("bcj,bnjd->bncd", mask_norm, z_t)
        return mixed.mean(dim=2)
