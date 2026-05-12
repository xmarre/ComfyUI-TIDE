from __future__ import annotations

import logging
import sys
from dataclasses import asdict, replace
from typing import Any, Optional

import torch

from .config import TIDEConfig
from .math import rope_temperature_scale
from .patches import _safe_timestep01

_LOG = logging.getLogger("ComfyUI-TIDE")

_CONFIG_KEY = "tide_wan_config"
_ENABLED_KEY = "tide_wan_enabled"
_WRAPPER_KEY = "tide_wan_rope_temperature"
_SCALED_FREQS_ID_KEY = "_tide_wan_scaled_freqs_id"
_DIFFUSION_MODEL_WRAPPER_TYPE = "diffusion_model"
_DEBUG_SCALE_LOG_LIMIT = 12


def _debug_print(config: Optional[TIDEConfig], message: str) -> None:
    if config is not None and config.debug:
        print(message, file=sys.stderr, flush=True)

try:  # pragma: no cover - ComfyUI is not importable in standalone tests.
    import comfy.patcher_extension as _comfy_patcher_extension
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name and exc.name.startswith("comfy"):
        _comfy_patcher_extension = None
    else:
        raise


def _diffusion_wrapper_type() -> str:
    if _comfy_patcher_extension is None:
        return _DIFFUSION_MODEL_WRAPPER_TYPE
    return _comfy_patcher_extension.WrappersMP.DIFFUSION_MODEL


def _ensure_transformer_options(model: Any) -> dict[str, Any]:
    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    transformer_options = model.model_options.get("transformer_options")
    if not isinstance(transformer_options, dict):
        transformer_options = {}
    model.model_options["transformer_options"] = transformer_options
    return transformer_options


def _config_dict(config: TIDEConfig) -> dict[str, Any]:
    data = asdict(config)
    data["axes_dim"] = tuple(config.axes_dim)
    data["backend"] = "wan"
    data["text_anchoring"] = "not_applicable_cross_attention_only"
    return data


def _looks_like_wan_inner(inner: Any) -> bool:
    # ComfyUI WAN 2.1/2.2 models expose this narrow runtime contract; avoid
    # class-name checks so compatible WAN subclasses can use the same path.
    return (
        inner is not None
        and callable(getattr(inner, "rope_encode", None))
        and callable(getattr(inner, "forward_orig", None))
        and hasattr(inner, "rope_embedder")
    )


def _read_wan_axes_dim(inner: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    rope_embedder = getattr(inner, "rope_embedder", None)
    axes_dim = getattr(rope_embedder, "axes_dim", None)
    if axes_dim is None:
        return tuple(int(v) for v in fallback)
    try:
        axes = tuple(int(v) for v in axes_dim)
    except Exception:
        return tuple(int(v) for v in fallback)
    if len(axes) != 3 or any(v <= 0 for v in axes):
        return tuple(int(v) for v in fallback)
    return axes


def _scale_wan_freqs(
    config: TIDEConfig,
    inner: Any,
    freqs: torch.Tensor,
    *,
    timestep: float,
) -> torch.Tensor:
    """Apply TIDE Dynamic Temperature Control to WAN RoPE matrices.

    ComfyUI WAN computes RoPE as a broadcastable matrix with shape compatible
    with [B, tokens, 1, rope_pairs, 2, 2].  The paper's Text Anchoring term is
    not applied here: WAN uses separate self-attention and cross-attention, so
    text and image keys do not compete in one joint softmax.
    """

    if not torch.is_tensor(freqs):
        return freqs
    if config.temperature_strength == 0.0:
        if config.debug:
            _debug_print(config, "[ComfyUI-TIDE] WAN DTC disabled: temperature_strength=0")
        return freqs
    if not config.should_apply_temperature():
        if config.debug:
            _debug_print(
                config,
                "[ComfyUI-TIDE] WAN DTC inactive: "
                f"width={config.width} height={config.height} "
                f"base={config.base_width}x{config.base_height} "
                f"scale_x={config.scale_x:.4f} scale_y={config.scale_y:.4f} "
                "and apply_to_native_or_smaller=false",
            )
        return freqs

    axes_dim = _read_wan_axes_dim(inner, tuple(config.axes_dim))
    local_config = replace(config, axes_dim=axes_dim)

    try:
        scale = rope_temperature_scale(
            local_config,
            timestep=timestep,
            device=freqs.device,
            dtype=freqs.dtype if torch.is_floating_point(freqs) else torch.float32,
        )
    except Exception as exc:
        if config.debug:
            _LOG.exception("TIDE WAN dynamic temperature scale failed and was skipped: %s", exc)
        return freqs

    if freqs.ndim < 3 or freqs.shape[-3] != scale.numel():
        if config.debug:
            _LOG.warning(
                "TIDE WAN skipped dynamic temperature: freqs axis dimension %s != scale length %s",
                freqs.shape[-3] if freqs.ndim >= 3 else None,
                scale.numel(),
            )
            _debug_print(
                config,
                "[ComfyUI-TIDE] WAN DTC skipped: "
                f"freqs_shape={tuple(freqs.shape)} scale_len={int(scale.numel())}",
            )
        return freqs

    view_shape = (1,) * (freqs.ndim - 3) + (scale.numel(), 1, 1)
    out = freqs * scale.reshape(view_shape)

    if config.debug:
        count = int(getattr(inner, "_tide_wan_scale_log_count", 0))
        if count < _DEBUG_SCALE_LOG_LIMIT:
            setattr(inner, "_tide_wan_scale_log_count", count + 1)
            _debug_print(
                config,
                "[ComfyUI-TIDE] WAN DTC applied: "
                f"step_log={count + 1}/{_DEBUG_SCALE_LOG_LIMIT} "
                f"timestep={float(timestep):.6f} "
                f"shape={tuple(freqs.shape)} axes_dim={axes_dim} "
                f"scale_x={config.scale_x:.4f} scale_y={config.scale_y:.4f} "
                f"scale_min={float(scale.min().detach().cpu()):.6f} "
                f"scale_max={float(scale.max().detach().cpu()):.6f}",
            )
    return out


def _mark_scaled_freqs(transformer_options: Optional[dict[str, Any]], freqs: Any) -> None:
    if isinstance(transformer_options, dict) and torch.is_tensor(freqs):
        transformer_options[_SCALED_FREQS_ID_KEY] = id(freqs)


def _freqs_already_scaled(transformer_options: Optional[dict[str, Any]], freqs: Any) -> bool:
    return isinstance(transformer_options, dict) and torch.is_tensor(freqs) and transformer_options.get(_SCALED_FREQS_ID_KEY) == id(freqs)


def _resolve_config(transformer_options: Optional[dict[str, Any]], fallback: Optional[TIDEConfig]) -> Optional[TIDEConfig]:
    cfg = transformer_options.get(_CONFIG_KEY) if isinstance(transformer_options, dict) else None
    if isinstance(cfg, TIDEConfig):
        return cfg
    return fallback


def _resolve_timestep(transformer_options: Optional[dict[str, Any]], timestep: Any = None) -> float:
    if isinstance(transformer_options, dict):
        tide_opts = transformer_options.get("tide", {})
        if isinstance(tide_opts, dict):
            value = tide_opts.get("timestep", None)
            if value is not None:
                return _safe_timestep01(value)
        value = transformer_options.get("timestep", None)
        if value is not None:
            return _safe_timestep01(value)
    return _safe_timestep01(timestep)


def _ensure_wan_rope_encode_wrapped(inner: Any, fallback_config: Optional[TIDEConfig]) -> bool:
    if not _looks_like_wan_inner(inner):
        return False
    if getattr(inner, "_tide_wan_rope_encode_wrapped", False):
        return True

    original_rope_encode = inner.rope_encode

    def tide_wan_rope_encode(*args: Any, **kwargs: Any):
        transformer_options = kwargs.get("transformer_options", None)
        freqs = original_rope_encode(*args, **kwargs)
        config = _resolve_config(transformer_options, fallback_config)
        if config is None:
            return freqs
        timestep = _resolve_timestep(transformer_options)
        out = _scale_wan_freqs(config, inner, freqs, timestep=timestep)
        _mark_scaled_freqs(transformer_options, out)
        return out

    inner._tide_wan_original_rope_encode = original_rope_encode
    inner.rope_encode = tide_wan_rope_encode
    inner._tide_wan_rope_encode_wrapped = True
    _debug_print(fallback_config, f"[ComfyUI-TIDE] WAN wrapped rope_encode on {type(inner).__name__} id={id(inner)}")
    return True


def _ensure_wan_forward_orig_wrapped(inner: Any, fallback_config: Optional[TIDEConfig]) -> bool:
    if not _looks_like_wan_inner(inner):
        return False
    if getattr(inner, "_tide_wan_forward_orig_wrapped", False):
        return True

    original_forward_orig = inner.forward_orig

    def tide_wan_forward_orig(
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options=None,
        **kwargs,
    ):
        config = _resolve_config(transformer_options, fallback_config)
        if config is not None and torch.is_tensor(freqs) and not _freqs_already_scaled(transformer_options, freqs):
            timestep = _resolve_timestep(transformer_options, t)
            freqs = _scale_wan_freqs(config, inner, freqs, timestep=timestep)
            _mark_scaled_freqs(transformer_options, freqs)

        if transformer_options is None:
            return original_forward_orig(x, t, context, clip_fea=clip_fea, freqs=freqs, **kwargs)
        return original_forward_orig(
            x,
            t,
            context,
            clip_fea=clip_fea,
            freqs=freqs,
            transformer_options=transformer_options,
            **kwargs,
        )

    inner._tide_wan_original_forward_orig = original_forward_orig
    inner.forward_orig = tide_wan_forward_orig
    inner._tide_wan_forward_orig_wrapped = True
    _debug_print(fallback_config, f"[ComfyUI-TIDE] WAN wrapped forward_orig on {type(inner).__name__} id={id(inner)}")
    return True


class TIDEWanDiffusionWrapper:
    """ComfyUI diffusion_model wrapper that enables WAN RoPE DTC hooks lazily."""

    def __init__(self, config: TIDEConfig):
        self.config = config

    def to(self, device: torch.device | str):
        return self

    def __call__(
        self,
        executor,
        x,
        timestep,
        context,
        clip_fea=None,
        time_dim_concat=None,
        transformer_options=None,
        **kwargs,
    ):
        if transformer_options is None:
            transformer_options = {}
        transformer_options[_CONFIG_KEY] = self.config
        transformer_options[_ENABLED_KEY] = self.config.should_apply_temperature() and self.config.temperature_strength != 0.0

        tide_opts = transformer_options.get("tide", {})
        if not isinstance(tide_opts, dict):
            tide_opts = {}
        tide_opts["timestep"] = _safe_timestep01(timestep)
        tide_opts["width"] = self.config.width
        tide_opts["height"] = self.config.height
        transformer_options["tide"] = tide_opts

        inner = getattr(executor, "class_obj", None)
        wrapped_rope = _ensure_wan_rope_encode_wrapped(inner, self.config)
        wrapped_forward_orig = _ensure_wan_forward_orig_wrapped(inner, self.config)
        if self.config.debug:
            if not getattr(self, "_logged_runtime", False):
                _debug_print(
                    self.config,
                    "[ComfyUI-TIDE] WAN diffusion wrapper active: "
                    f"inner={type(inner).__name__ if inner is not None else None} "
                    f"wrapped_rope={bool(wrapped_rope)} wrapped_forward_orig={bool(wrapped_forward_orig)} "
                    f"enabled={bool(transformer_options[_ENABLED_KEY])}",
                )
                self._logged_runtime = True
            if not (wrapped_rope or wrapped_forward_orig):
                _LOG.warning("TIDE WAN wrapper did not find a WAN-like rope_encode/forward_orig target on %s", type(inner).__name__ if inner is not None else None)
                _debug_print(self.config, f"[ComfyUI-TIDE] WAN wrapper did not find WAN target: inner={type(inner).__name__ if inner is not None else None}")

        return executor(
            x,
            timestep,
            context,
            clip_fea,
            time_dim_concat,
            transformer_options,
            **kwargs,
        )


def install_tide_wan_patch(model: Any, config: TIDEConfig) -> Any:
    """Install the WAN-specific TIDE DTC path on a cloned ComfyUI MODEL."""

    transformer_options = _ensure_transformer_options(model)
    transformer_options[_CONFIG_KEY] = config
    transformer_options[_ENABLED_KEY] = config.should_apply_temperature() and config.temperature_strength != 0.0
    transformer_options["tide_wan"] = _config_dict(config)

    wrapper = TIDEWanDiffusionWrapper(config)
    wrapper_type = _diffusion_wrapper_type()

    installed_on_patcher = False
    if callable(getattr(model, "remove_wrappers_with_key", None)) and callable(getattr(model, "add_wrapper_with_key", None)):
        model.remove_wrappers_with_key(wrapper_type, _WRAPPER_KEY)
        model.add_wrapper_with_key(wrapper_type, _WRAPPER_KEY, wrapper)
        installed_on_patcher = True

    if not installed_on_patcher:
        wrappers = transformer_options.setdefault("wrappers", {})
        wrappers_for_type = wrappers.setdefault(wrapper_type, {})
        wrappers_for_type[_WRAPPER_KEY] = [wrapper]

    _debug_print(
        config,
        "[ComfyUI-TIDE] WAN patch installed: "
        f"wrapper_type={wrapper_type} model_patcher_wrapper={installed_on_patcher} "
        f"enabled={bool(transformer_options[_ENABLED_KEY])} "
        f"width={config.width} height={config.height} base={config.base_width}x{config.base_height} "
        f"scale_x={config.scale_x:.4f} scale_y={config.scale_y:.4f} "
        f"temperature_strength={config.temperature_strength:.4f}",
    )

    # Best-effort eager wrapping for already-materialized WAN models. The runtime
    # diffusion wrapper repeats this lazily because ComfyUI can replace/live-wrap
    # the diffusion module during dynamic loading.
    outer = getattr(model, "model", None)
    inner = getattr(outer, "diffusion_model", None) if outer is not None else getattr(model, "diffusion_model", None)
    _ensure_wan_rope_encode_wrapped(inner, config)
    _ensure_wan_forward_orig_wrapped(inner, config)
    return model


__all__ = [
    "TIDEWanDiffusionWrapper",
    "install_tide_wan_patch",
    "_scale_wan_freqs",
]
