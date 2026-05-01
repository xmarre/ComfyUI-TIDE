import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tide_core.config import TIDEConfig
from tide_core.math import adaptive_text_bias, get_default_temperature, rope_temperature_scale


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
