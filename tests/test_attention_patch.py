import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tide_core.config import TIDEConfig
from tide_core.patches import TIDEAttentionPatch, TIDEAttentionOverride


def test_attention_patch_adds_text_bias_and_scales_pe():
    cfg = TIDEConfig(width=2048, height=2048, base_width=1024, base_height=1024)
    patch = TIDEAttentionPatch(cfg)
    text_tokens = 4
    image_tokens = 16
    total = text_tokens + image_tokens
    q = torch.randn(1, 2, total, 8)
    k = torch.randn(1, 2, total, 8)
    v = torch.randn(1, 2, total, 8)
    pe = torch.ones(1, 1, total, sum(cfg.axes_dim) // 2, 2, 2)

    out = patch(q, k, v, pe=pe, attn_mask=None, extra_options={"img_slice": [text_tokens, total], "block_type": "double", "tide": {"timestep": 1.0}})

    mask = out["attn_mask"]
    assert mask.shape == (1, 1, 1, total)
    assert torch.allclose(mask[..., :text_tokens], torch.full((1, 1, 1, text_tokens), math.log(4.0)))
    assert torch.allclose(mask[..., text_tokens:], torch.zeros(1, 1, 1, image_tokens))
    assert out["pe"].shape == pe.shape
    assert torch.max(out["pe"]) > 1.0


def test_attention_patch_respects_block_toggles():
    cfg = TIDEConfig(width=2048, height=2048, apply_to_double_blocks=False, apply_to_single_blocks=True)
    patch = TIDEAttentionPatch(cfg)
    q = torch.randn(1, 1, 8, 4)
    k = torch.randn(1, 1, 8, 4)
    v = torch.randn(1, 1, 8, 4)
    out = patch(q, k, v, pe=None, attn_mask=None, extra_options={"img_slice": [2, 8], "block_type": "double"})
    assert out["attn_mask"] is None


def test_attention_override_runs_masked_sdpa_without_comfy_imports():
    cfg = TIDEConfig(width=2048, height=2048, force_pytorch_attention_with_mask=True)
    override = TIDEAttentionOverride(cfg)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    mask = torch.zeros(1, 1, 1, 5)

    def should_not_run(*args, **kwargs):
        raise AssertionError("delegate should not run when mask is present")

    out = override(should_not_run, q, k, v, 2, mask=mask, skip_reshape=True)
    assert out.shape == (1, 5, 16)
