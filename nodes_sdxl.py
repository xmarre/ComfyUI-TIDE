import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


@dataclass(frozen=True)
class TIDESDXLConfig:
    width: int
    height: int
    base_width: int
    base_height: int
    temperature_strength: float
    alpha: float
    tau_max: float
    apply_to: str

    @property
    def pixel_ratio(self) -> float:
        base_pixels = max(1.0, float(self.base_width) * float(self.base_height))
        target_pixels = max(1.0, float(self.width) * float(self.height))
        return max(1.0, target_pixels / base_pixels)

    @property
    def extrapolation_scale(self) -> float:
        return math.sqrt(self.pixel_ratio)

    @property
    def inv_tau_min(self) -> float:
        # TIDE inherits YaRN's attention-temperature convention:
        # sqrt(1/tau) = 0.1 * ln(scale) + 1.
        scale = self.extrapolation_scale
        if scale <= 1.0:
            return 1.0
        yarn_mscale = 0.1 * math.log(scale) + 1.0
        return yarn_mscale * yarn_mscale


def _token_count(x: torch.Tensor, skip_reshape: bool) -> int:
    if skip_reshape and x.ndim >= 4:
        return int(x.shape[-2])
    return int(x.shape[1])


def _normalised_sigma_t(transformer_options: dict[str, Any]) -> float:
    """Return t in [0, 1], where 1 is the noisy/start side and 0 is the clean/end side."""
    cur = transformer_options.get("sigmas", None)
    all_sigmas = transformer_options.get("sample_sigmas", None)
    if cur is None or all_sigmas is None:
        return 1.0

    try:
        cur_f = float(cur.detach().float().mean().item()) if torch.is_tensor(cur) else float(cur)
        if torch.is_tensor(all_sigmas):
            sigmas = all_sigmas.detach().float().flatten()
            sigmas = sigmas[torch.isfinite(sigmas)]
            if sigmas.numel() == 0:
                return 1.0
            sigma_max = float(sigmas.max().item())
            positive = sigmas[sigmas > 0]
            sigma_min = float(positive.min().item()) if positive.numel() else float(sigmas.min().item())
        else:
            vals = [float(v) for v in all_sigmas if math.isfinite(float(v))]
            if not vals:
                return 1.0
            sigma_max = max(vals)
            positives = [v for v in vals if v > 0]
            sigma_min = min(positives) if positives else min(vals)

        denom = sigma_max - sigma_min
        if denom <= 1e-12:
            return 1.0
        return max(0.0, min(1.0, (cur_f - sigma_min) / denom))
    except Exception:
        return 1.0


def _temperature_q_scale(config: TIDESDXLConfig, transformer_options: dict[str, Any]) -> float:
    if config.temperature_strength <= 0.0:
        return 1.0

    inv_tau_min = config.inv_tau_min
    if inv_tau_min <= 1.0:
        return 1.0

    tau_min = 1.0 / inv_tau_min
    tau_max = max(tau_min, float(config.tau_max))
    t = _normalised_sigma_t(transformer_options)
    alpha = max(1e-6, float(config.alpha))
    tau = tau_max - (tau_max - tau_min) * (t ** alpha)
    inv_tau = 1.0 / max(tau, 1e-6)

    # Strength blends from no-op to the full TIDE/YaRN temperature.
    return 1.0 + (inv_tau - 1.0) * max(0.0, min(1.0, float(config.temperature_strength)))


def _looks_like_unet_spatial_transformer(transformer_options: dict[str, Any]) -> bool:
    # SDXL/SD1 UNet SpatialTransformer sets activations_shape before calling
    # BasicTransformerBlock. FLUX-style DiT paths instead use block_type/img_slice.
    if "activations_shape" not in transformer_options:
        return False
    if "block_type" in transformer_options or "img_slice" in transformer_options:
        return False
    return True


def _attention_kind(q_tokens: int, k_tokens: int) -> str:
    # In SDXL UNet blocks, self-attention has image-token K/V (q_len == k_len),
    # while cross-attention has text-token K/V (usually 77 tokens).
    if q_tokens == k_tokens:
        return "self"
    return "cross"


def _enabled_for_kind(apply_to: str, kind: str) -> bool:
    return apply_to == "both" or apply_to == kind


def build_sdxl_attention_override(
    config: TIDESDXLConfig,
    previous_override: Optional[Callable[..., torch.Tensor]] = None,
) -> Callable[..., torch.Tensor]:
    def tide_sdxl_attention_override(func: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) < 4:
            if previous_override is not None:
                return previous_override(func, *args, **kwargs)
            return func(*args, **kwargs)

        q, k, v, heads = args[:4]
        rest = args[4:]
        transformer_options = kwargs.get("transformer_options", {}) or {}

        if (
            not torch.is_tensor(q)
            or not torch.is_tensor(k)
            or not _looks_like_unet_spatial_transformer(transformer_options)
        ):
            if previous_override is not None:
                return previous_override(func, *args, **kwargs)
            return func(*args, **kwargs)

        skip_reshape = bool(kwargs.get("skip_reshape", False))
        q_tokens = _token_count(q, skip_reshape)
        k_tokens = _token_count(k, skip_reshape)
        kind = _attention_kind(q_tokens, k_tokens)

        if _enabled_for_kind(config.apply_to, kind):
            q_scale = _temperature_q_scale(config, transformer_options)
            if q_scale != 1.0:
                q = q * q_scale
                args = (q, k, v, heads, *rest)

        if previous_override is not None:
            return previous_override(func, *args, **kwargs)
        return func(*args, **kwargs)

    return tide_sdxl_attention_override


class TIDESDXLHighRes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", {"default": 1536, "min": 64, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1536, "min": 64, "max": 16384, "step": 8}),
                "temperature_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "base_width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "base_height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "alpha": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 4.0, "step": 0.05}),
                "tau_max": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.05}),
                "apply_to": (["cross", "self", "both"], {"default": "both"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/TIDE"

    def apply(
        self,
        model,
        width: int,
        height: int,
        temperature_strength: float,
        base_width: int,
        base_height: int,
        alpha: float,
        tau_max: float,
        apply_to: str,
    ):
        patched = model.clone()
        config = TIDESDXLConfig(
            width=int(width),
            height=int(height),
            base_width=int(base_width),
            base_height=int(base_height),
            temperature_strength=float(temperature_strength),
            alpha=float(alpha),
            tau_max=float(tau_max),
            apply_to=str(apply_to),
        )

        transformer_options = patched.model_options.get("transformer_options", {}).copy()
        previous_override = transformer_options.get("optimized_attention_override", None)
        transformer_options["optimized_attention_override"] = build_sdxl_attention_override(config, previous_override)
        transformer_options["tide_sdxl"] = {
            "width": config.width,
            "height": config.height,
            "base_width": config.base_width,
            "base_height": config.base_height,
            "temperature_strength": config.temperature_strength,
            "alpha": config.alpha,
            "tau_max": config.tau_max,
            "apply_to": config.apply_to,
        }
        patched.model_options["transformer_options"] = transformer_options
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "TIDESDXLHighRes": TIDESDXLHighRes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TIDESDXLHighRes": "TIDE SDXL High-Resolution Extrapolation",
}
