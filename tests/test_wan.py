import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tide_core.config import TIDEConfig
from tide_core.patches import TIDEAttentionPatch
from tide_core.wan import _scale_wan_freqs


class _FakeRopeEmbedder:
    axes_dim = (8, 12, 12)


class _FakeWanInner:
    rope_embedder = _FakeRopeEmbedder()


def test_wan_rope_temperature_scale_preserves_shape_and_relaxes_late():
    cfg = TIDEConfig(width=1280, height=640, base_width=640, base_height=640)
    freqs = torch.ones(1, 32, 1, sum(_FakeRopeEmbedder.axes_dim) // 2, 2, 2)

    early = _scale_wan_freqs(cfg, _FakeWanInner(), freqs, timestep=1.0)
    late = _scale_wan_freqs(cfg, _FakeWanInner(), freqs, timestep=0.0)

    assert early.shape == freqs.shape
    assert late.shape == freqs.shape
    assert torch.max(early) > 1.0
    assert torch.allclose(late, freqs, atol=1e-6)


def test_flux_attn_patch_tolerates_wan_post_attention_patch_payload():
    cfg = TIDEConfig(width=1280, height=720, base_width=640, base_height=640)
    patch = TIDEAttentionPatch(cfg)
    x = torch.randn(1, 8, 16)
    payload = {"x": x, "q": torch.randn(1, 8, 2, 8), "k": torch.randn(1, 8, 2, 8), "transformer_options": {}}

    assert patch(payload) is x


def test_flux_attn_patch_only_treats_full_wan_payload_as_passthrough():
    cfg = TIDEConfig(width=1280, height=720, base_width=640, base_height=640)
    patch = TIDEAttentionPatch(cfg)
    payload = {"x": torch.randn(1, 8, 16)}

    assert patch(payload)["q"] is payload
