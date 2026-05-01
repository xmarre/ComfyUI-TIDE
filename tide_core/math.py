from __future__ import annotations

import math
from typing import Iterable

import torch

from .config import TIDEConfig


def get_mscale(scale: float) -> float:
    """YaRN mscale used by the official TIDE implementation.

    Official code computes 0.1 * log(scale) + 1 for scale > 1 and 1 otherwise.
    The corresponding attention temperature is 1 / mscale**2.
    """

    scale = float(scale)
    if scale <= 1.0:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def get_default_temperature(scale: float) -> float:
    mscale = get_mscale(scale)
    return 1.0 / (mscale * mscale)


def adaptive_text_bias(config: TIDEConfig) -> float:
    """Paper Eq. 17/18: beta = log(lambda), with lambda = pixel ratio.

    The official FLUX script computes log(width / 1024) + log(height / 1024),
    which is identical to log((width * height) / 1024**2). Strength scales the
    paper value linearly; values <= base resolution return zero by default.
    """

    if not config.should_apply():
        return 0.0
    beta = math.log(config.target_pixel_ratio)
    if beta < 0.0 and not config.apply_to_native_or_smaller:
        beta = 0.0
    return float(config.text_anchor_strength) * beta


def _axis_frequencies(axis_dim: int, theta: float, device: torch.device) -> torch.Tensor:
    if axis_dim <= 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    if axis_dim % 2 != 0:
        raise ValueError(f"axis_dim must be even, got {axis_dim}")
    return 1.0 / (float(theta) ** (torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim))


def _frequency_parameter(freqs: torch.Tensor, mode: str) -> torch.Tensor:
    if freqs.numel() == 0:
        return freqs
    if mode == "official_raw":
        # Matches the released code: alpha = alpha_low + (alpha_high-alpha_low) * freqs.
        return freqs
    if mode == "paper_normalized":
        # The paper describes a normalized frequency f, but the released FLUX code uses raw RoPE frequencies.
        # This option provides the closest literal normalized interpretation for comparison.
        lo = freqs.min()
        hi = freqs.max()
        if torch.isclose(hi, lo):
            return torch.zeros_like(freqs)
        return (freqs - lo) / (hi - lo)
    raise ValueError(f"Unsupported frequency_mode: {mode!r}")


def axis_temperature_scale(
    *,
    axis_dim: int,
    axis_scale: float,
    timestep: float,
    alpha_low: float,
    alpha_high: float,
    tau_max: float,
    theta: float,
    frequency_mode: str,
    strength: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return per-RoPE-pair multiplier equivalent to 1 / sqrt(tau(t, f))."""

    freqs = _axis_frequencies(axis_dim, theta, device)
    if freqs.numel() == 0:
        return freqs.to(dtype=dtype)

    low_temperature = get_default_temperature(axis_scale)
    t = float(max(0.0, min(1.0, timestep)))
    tau_max = float(tau_max)
    if tau_max <= 0.0:
        raise ValueError(f"tau_max must be > 0, got {tau_max}")

    if axis_scale <= 1.0 or strength <= 0.0:
        target = torch.ones_like(freqs)
    else:
        f = _frequency_parameter(freqs, frequency_mode)
        alphas = float(alpha_low) + (float(alpha_high) - float(alpha_low)) * f
        # Paper Eq. 21. Official implementation names this dyheating.
        temps = tau_max - ((tau_max - float(low_temperature)) * torch.pow(torch.tensor(t, device=device), alphas))
        temps = temps.clamp_min(1.0e-6)
        target = torch.rsqrt(temps)

    if strength < 1.0:
        target = 1.0 + float(strength) * (target - 1.0)
    elif strength > 1.0:
        target = 1.0 + float(strength) * (target - 1.0)
    return target.to(dtype=dtype)


def rope_temperature_scale(
    config: TIDEConfig,
    *,
    timestep: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a concatenated scale vector for ComfyUI's complex RoPE matrix.

    ComfyUI's FLUX EmbedND concatenates RoPE axes in the order (t, y, x), and
    each RoPE pair is represented by one 2x2 complex-rotation matrix. The output
    length is therefore sum(axes_dim) // 2, matching pe.shape[-3].
    """

    axes_dim = tuple(int(v) for v in config.axes_dim)
    if len(axes_dim) != 3:
        raise ValueError(f"Expected three axes dims (t, y, x), got {axes_dim!r}")

    parts = [
        axis_temperature_scale(
            axis_dim=axes_dim[0],
            axis_scale=1.0,
            timestep=timestep,
            alpha_low=config.alpha_low,
            alpha_high=config.alpha_high,
            tau_max=config.tau_max,
            theta=config.theta,
            frequency_mode=config.frequency_mode,
            strength=config.temperature_strength,
            device=device,
            dtype=dtype,
        ),
        axis_temperature_scale(
            axis_dim=axes_dim[1],
            axis_scale=config.scale_y,
            timestep=timestep,
            alpha_low=config.alpha_low,
            alpha_high=config.alpha_high,
            tau_max=config.tau_max,
            theta=config.theta,
            frequency_mode=config.frequency_mode,
            strength=config.temperature_strength,
            device=device,
            dtype=dtype,
        ),
        axis_temperature_scale(
            axis_dim=axes_dim[2],
            axis_scale=config.scale_x,
            timestep=timestep,
            alpha_low=config.alpha_low,
            alpha_high=config.alpha_high,
            tau_max=config.tau_max,
            theta=config.theta,
            frequency_mode=config.frequency_mode,
            strength=config.temperature_strength,
            device=device,
            dtype=dtype,
        ),
    ]
    return torch.cat(parts, dim=0)
