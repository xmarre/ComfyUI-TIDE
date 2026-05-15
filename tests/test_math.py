import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tide_core.config import TIDEConfig
from tide_core.math import adaptive_text_bias, aspect_adaptive_base_resolution, get_default_temperature, rope_temperature_scale


def test_adaptive_text_bias_matches_paper_and_official_flux_script():
    cfg = TIDEConfig(width=2048, height=2048, base_width=1024, base_height=1024)
    assert math.isclose(adaptive_text_bias(cfg), math.log(4.0), rel_tol=1e-6)


def test_native_resolution_bias_is_zero_by_default():
    cfg = TIDEConfig(width=1024, height=1024, base_width=1024, base_height=1024)
    assert adaptive_text_bias(cfg) == 0.0


def test_yarn_temperature_formula():
    expected_mscale = 0.1 * math.log(4.0) + 1.0
    assert math.isclose(get_default_temperature(4.0), 1.0 / (expected_mscale * expected_mscale), rel_tol=1e-6)


def test_rope_temperature_scale_shape_and_progression():
    cfg = TIDEConfig(width=4096, height=2048, base_width=1024, base_height=1024)
    early = rope_temperature_scale(cfg, timestep=1.0, device=torch.device("cpu"))
    late = rope_temperature_scale(cfg, timestep=0.0, device=torch.device("cpu"))
    assert early.shape == (sum(cfg.axes_dim) // 2,)
    assert late.shape == (sum(cfg.axes_dim) // 2,)
    assert torch.all(torch.isfinite(early))
    assert torch.all(torch.isfinite(late))
    # At t=0, Eq. 21 reaches tau_max=1, so the multiplier is 1.
    assert torch.allclose(late, torch.ones_like(late), atol=1e-6)
    assert torch.max(early) > 1.0


def test_temperature_strength_zero_disables_scaling():
    cfg = TIDEConfig(width=4096, height=4096, temperature_strength=0.0)
    scale = rope_temperature_scale(cfg, timestep=1.0, device=torch.device("cpu"))
    assert torch.allclose(scale, torch.ones_like(scale), atol=1e-6)


def test_aspect_adaptive_base_resolution_preserves_720p_area_budget_for_square_targets():
    assert aspect_adaptive_base_resolution(960, 960, 1280, 720) == (960, 960)


def test_aspect_adaptive_base_resolution_preserves_native_landscape_and_portrait():
    assert aspect_adaptive_base_resolution(1280, 720, 1280, 720) == (1280, 720)
    assert aspect_adaptive_base_resolution(720, 1280, 1280, 720) == (720, 1280)


def test_aspect_adaptive_base_resolution_uses_target_aspect_for_nonstandard_i2v_size():
    assert aspect_adaptive_base_resolution(896, 656, 1280, 720) == (1120, 816)


def test_aspect_adaptive_base_resolution_keeps_snapped_result_within_native_budget():
    base_width, base_height = aspect_adaptive_base_resolution(1000, 777, 1280, 720)
    assert base_width * base_height <= 1280 * 720


def test_aspect_adaptive_base_resolution_preserves_wan_480p_budget():
    assert aspect_adaptive_base_resolution(832, 480, 832, 480) == (832, 480)
    base_width, base_height = aspect_adaptive_base_resolution(480, 832, 832, 480)
    assert base_width * base_height <= 832 * 480


def test_aspect_adaptive_base_resolution_rejects_unsnappable_budget():
    with pytest.raises(ValueError, match="No grid-aligned base resolution"):
        aspect_adaptive_base_resolution(100_000_000, 1, 1280, 720)
