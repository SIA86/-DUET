import numpy as np
import pytest
import torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from duet.components import TCM, CCM, Fusion, DUETHead
from pipeline.config import DUETConfig


def build_config(use_router: bool = True) -> DUETConfig:
    return DUETConfig(
        features=["feat1", "feat2"],
        forecast=["target"],
        seq_len=16,
        horizon=3,
        patch_len=4,
        stride=2,
        moving_avg=3,
        d_model=8,
        d_c=4,
        K_t=2,
        K_c=3,
        top_k=2,
        dropout=0.0,
        fc_dropout=0.0,
        report_freq=0,
        use_router=use_router,
        use_revin=True,
        batch_size=4,
        epochs=1,
        verbose=False,
    )


def test_tcm_shapes_and_router_weights() -> None:
    config = build_config(use_router=True)
    tcm = TCM(config)

    x = torch.randn(2, config.seq_len, 3)
    z_t, weights = tcm(x)

    assert z_t.shape == (2, (config.seq_len - config.patch_len) // config.stride + 1, 3, config.d_model)
    assert weights.shape == (2, z_t.shape[1], config.K_t)
    ones = torch.ones_like(weights.sum(dim=-1))
    assert torch.allclose(weights.sum(dim=-1), ones, atol=1e-5)


def test_ccm_mask_shape_and_symmetry() -> None:
    config = build_config()
    ccm = CCM(config)

    x = torch.randn(2, config.seq_len, 3)
    mask = ccm(x)

    assert mask.shape == (2, 3, 3)
    assert torch.allclose(mask, mask.transpose(1, 2), atol=1e-6)


def test_fusion_matches_mean_when_mask_uniform() -> None:
    fusion = Fusion()
    z_t = torch.randn(2, 5, 4, 6)
    mask = torch.ones(2, 4, 4)

    fused = fusion(z_t, mask)
    expected = z_t.mean(dim=2)

    assert fused.shape == expected.shape
    assert torch.allclose(fused, expected, atol=1e-6)


def test_head_outputs_logits() -> None:
    head = DUETHead(d_model=8, horizon=3, c_out=2, dropout=0.0)
    x = torch.randn(2, 5, 8)
    out = head(x)

    assert out.shape == (2, 3, 2)
