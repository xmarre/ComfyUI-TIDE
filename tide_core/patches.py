from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from .config import TIDEConfig
from .math import adaptive_text_bias, rope_temperature_scale

_LOG = logging.getLogger("ComfyUI-TIDE")


def _safe_timestep01(value: Any) -> float:
    if value is None:
        return 1.0
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return 1.0
            v = value.detach().float().max().cpu().item()
        else:
            v = float(value)
    except Exception:
        return 1.0
    # FLUX-family Comfy models normally receive flow timesteps/sigmas in [0, 1].
    # Clamp rather than normalize by an unknown scheduler-specific maximum.
    return max(0.0, min(1.0, float(v)))


def _block_enabled(config: TIDEConfig, extra_options: dict[str, Any]) -> bool:
    block_type = extra_options.get("block_type")
    if block_type == "double":
        return config.apply_to_double_blocks
    if block_type == "single":
        return config.apply_to_single_blocks
    # Unknown Flux-like DiT patch site. Apply conservatively if either stream type is enabled.
    return config.apply_to_double_blocks or config.apply_to_single_blocks


def _add_text_bias_mask(
    attn_mask: Optional[torch.Tensor],
    *,
    text_tokens: int,
    key_tokens: int,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    bias = torch.zeros((1, 1, 1, key_tokens), device=device, dtype=dtype)
    if text_tokens > 0 and beta != 0.0:
        bias[..., :text_tokens] = float(beta)

    if attn_mask is None:
        return bias

    if not torch.is_floating_point(attn_mask):
        # Boolean masks represent validity, not additive logits. Preserve them rather than
        # accidentally converting padding semantics into a dense additive bias. Current Flux
        # T2I path normally has no boolean mask; Qwen's text mask is handled upstream.
        _LOG.warning("TIDE skipped text anchoring because the existing attention mask is boolean/non-floating.")
        return attn_mask

    # Comfy attention accepts additive masks of shape [B, H|1, Q|1, K].
    # Rely on PyTorch broadcasting and do not materialize full QxK masks.
    return attn_mask.to(device=device, dtype=dtype) + bias


class TIDEAttentionPatch:
    """ComfyUI attn1_patch implementing TIDE text anchoring and RoPE temperature scaling."""

    def __init__(self, config: TIDEConfig):
        self.config = config

    def to(self, device: torch.device | str):  # Comfy calls .to on patches during model moves.
        return self

    def __call__(self, q, k=None, v=None, pe=None, attn_mask=None, extra_options=None):
        # ComfyUI WAN currently calls attn1_patch after self-attention with a
        # single dict payload: {"x", "q", "k", "transformer_options"}.
        # The Flux TIDE patch cannot modify that already-computed attention, so
        # leave it as a no-op instead of failing when the generic node is used
        # on a WAN model. WAN support is installed through tide_core.wan.
        if k is None and v is None and isinstance(q, dict) and {"x", "q", "k", "transformer_options"}.issubset(q):
            return q.get("x", q)

        extra_options = extra_options or {}
        if not _block_enabled(self.config, extra_options):
            return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

        out_mask = attn_mask
        beta = adaptive_text_bias(self.config)
        if beta != 0.0:
            img_slice = extra_options.get("img_slice")
            if img_slice and len(img_slice) == 2:
                try:
                    text_tokens = int(img_slice[0])
                    total_tokens = int(k.shape[2])
                except (TypeError, ValueError, IndexError, AttributeError):
                    text_tokens = 0
                    total_tokens = 0

                if text_tokens > 0 and total_tokens > text_tokens:
                    out_mask = _add_text_bias_mask(
                        attn_mask,
                        text_tokens=text_tokens,
                        key_tokens=total_tokens,
                        beta=beta,
                        device=k.device,
                        dtype=q.dtype if torch.is_floating_point(q) else torch.float32,
                    )
                elif self.config.debug:
                    _LOG.warning("TIDE skipped text anchoring because Flux text/image token slices were unavailable.")
            elif self.config.debug:
                _LOG.warning("TIDE skipped text anchoring because Flux img_slice metadata was unavailable.")

        out_pe = pe
        if pe is not None and self.config.temperature_strength != 0.0 and self.config.should_apply_temperature():
            tide_opts = extra_options.get("tide", {})
            timestep = _safe_timestep01(tide_opts.get("timestep", extra_options.get("timestep")))
            try:
                scale = rope_temperature_scale(
                    self.config,
                    timestep=timestep,
                    device=pe.device,
                    dtype=pe.dtype if torch.is_floating_point(pe) else torch.float32,
                )
                # Comfy FLUX pe shape is [B, 1, N, Dpair, 2, 2] or broadcast-compatible.
                if pe.shape[-3] == scale.numel():
                    view_shape = (1,) * (pe.ndim - 3) + (scale.numel(), 1, 1)
                    out_pe = pe * scale.reshape(view_shape)
                elif self.config.debug:
                    _LOG.warning(
                        "TIDE skipped dynamic temperature: pe axis dimension %s != scale length %s",
                        pe.shape[-3], scale.numel(),
                    )
            except Exception as exc:
                if self.config.debug:
                    _LOG.exception("TIDE dynamic temperature failed and was skipped: %s", exc)

        return {"q": q, "k": k, "v": v, "pe": out_pe, "attn_mask": out_mask}


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: Optional[torch.Tensor] = None,
    attn_precision=None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Small unwrapped PyTorch SDPA path for additive TIDE masks.

    This avoids xFormers/Sage/Flash backends that may reject additive masks or
    materialize dense high-resolution masks. It intentionally mirrors ComfyUI's
    attention_pytorch tensor layout without invoking the wrapped function again.
    """

    if skip_reshape:
        b, _, _, dim_head = q.shape
        q_s, k_s, v_s = q, k, v
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q_s, k_s, v_s = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=q_s.device, dtype=q_s.dtype if torch.is_floating_point(mask) else mask.dtype)

    out = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=mask, dropout_p=0.0, is_causal=False)

    if skip_reshape:
        if not skip_output_reshape:
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    else:
        if skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


class TIDEAttentionOverride:
    """Optional optimized_attention_override used only when an additive mask is present."""

    def __init__(self, config: TIDEConfig, old_override: Optional[Callable] = None):
        self.config = config
        self.old_override = old_override

    def to(self, device: torch.device | str):
        return self

    def __call__(self, original_func: Callable, *args, **kwargs):
        mask = kwargs.get("mask", None)
        if mask is None:
            mask = kwargs.get("attn_mask", None)
        if self.config.force_pytorch_attention_with_mask and mask is not None:
            sdpa_kwargs = dict(kwargs)
            if "mask" not in sdpa_kwargs and "attn_mask" in sdpa_kwargs:
                sdpa_kwargs["mask"] = sdpa_kwargs["attn_mask"]
            return _sdpa_attention(*args, **sdpa_kwargs)
        if self.old_override is not None:
            return self.old_override(original_func, *args, **kwargs)
        return original_func(*args, **kwargs)


class TIDEModelWrapper:
    """Inject current denoising timestep into transformer_options for DTC."""

    def __init__(self, config: TIDEConfig, old_wrapper: Optional[Callable] = None):
        self.config = config
        self.old_wrapper = old_wrapper

    def to(self, device: torch.device | str):
        return self

    def __call__(self, apply_model: Callable, args: dict[str, Any]) -> torch.Tensor:
        c = args.get("c", {}).copy()
        transformer_options = c.get("transformer_options", {}).copy()
        tide_opts = transformer_options.get("tide", {}).copy()
        tide_opts["timestep"] = _safe_timestep01(args.get("timestep"))
        tide_opts["width"] = self.config.width
        tide_opts["height"] = self.config.height
        transformer_options["tide"] = tide_opts
        c["transformer_options"] = transformer_options

        if self.config.preserve_existing_wrapper and self.old_wrapper is not None:
            return self.old_wrapper(apply_model, args | {"c": c})
        return apply_model(args["input"], args["timestep"], **c)
