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
_CONFIG_SOURCE_KEY = "_tide_wan_config_source"
_INNER_KEY = "_tide_wan_inner"
_BLOCK_PATCH_KEY = "_tide_wan_block_dtc_patch"
_BLOCK_PATCH_CONFIG_KEY = "_tide_wan_block_dtc_config"
_BLOCK_PATCH_INNER_KEY = "_tide_wan_block_dtc_inner"
_BLOCK_PATCH_SOURCE_KEY = "_tide_wan_block_dtc_source"
_BLOCK_SCALED_CACHE_KEY = "_tide_wan_block_scaled_freqs_cache"
_TRACE_LOG_COUNT_KEY = "_tide_wan_trace_log_count"
_DIFFUSION_MODEL_WRAPPER_TYPE = "diffusion_model"
_DEBUG_SCALE_LOG_LIMIT = 12
_DEBUG_TRACE_LOG_LIMIT = 24


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


def inject_tide_wan_options(
    transformer_options: Optional[dict[str, Any]],
    config: TIDEConfig,
    *,
    timestep: Any = None,
    source: str = "unknown",
) -> Optional[dict[str, Any]]:
    """Inject the WAN TIDE runtime contract into the live transformer_options.

    The WAN RoPE/forward hooks run much lower than the Comfy model_function_wrapper.
    They must therefore be able to resolve the current TIDE config from the exact
    transformer_options dict that reaches WanModel.rope_encode/forward_orig.  This
    helper intentionally mutates that dict in place.
    """

    if not isinstance(transformer_options, dict):
        return transformer_options

    transformer_options[_CONFIG_KEY] = config
    transformer_options[_ENABLED_KEY] = config.should_apply_temperature() and config.temperature_strength != 0.0
    transformer_options[_CONFIG_SOURCE_KEY] = source

    tide_opts = transformer_options.get("tide", {})
    if not isinstance(tide_opts, dict):
        tide_opts = {}
    if timestep is not None:
        tide_opts["timestep"] = _safe_timestep01(timestep)
    tide_opts["width"] = config.width
    tide_opts["height"] = config.height
    transformer_options["tide"] = tide_opts
    return transformer_options


def has_tide_wan_options(transformer_options: Optional[dict[str, Any]]) -> bool:
    return isinstance(transformer_options, dict) and (
        "tide_wan" in transformer_options or isinstance(transformer_options.get(_CONFIG_KEY), TIDEConfig)
    )


def _trace_inner(config: Optional[TIDEConfig], inner: Any, message: str) -> None:
    if config is None or not config.debug:
        return
    count = int(getattr(inner, _TRACE_LOG_COUNT_KEY, 0)) if inner is not None else 0
    if count >= _DEBUG_TRACE_LOG_LIMIT:
        return
    if inner is not None:
        setattr(inner, _TRACE_LOG_COUNT_KEY, count + 1)
    _debug_print(config, message)


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
        count = int(getattr(inner, "_tide_wan_scale_log_count", 0)) if inner is not None else 0
        if count < _DEBUG_SCALE_LOG_LIMIT:
            if inner is not None:
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
        inner._tide_wan_fallback_config = fallback_config
        return True

    original_rope_encode = inner.rope_encode

    def tide_wan_rope_encode(*args: Any, **kwargs: Any):
        transformer_options = kwargs.get("transformer_options", None)
        freqs = original_rope_encode(*args, **kwargs)
        config = _resolve_config(transformer_options, getattr(inner, "_tide_wan_fallback_config", fallback_config))
        _trace_inner(
            config,
            inner,
            "[ComfyUI-TIDE] WAN rope_encode hook entered: "
            f"freqs_tensor={torch.is_tensor(freqs)} "
            f"freqs_shape={tuple(freqs.shape) if torch.is_tensor(freqs) else None} "
            f"config_source={transformer_options.get(_CONFIG_SOURCE_KEY) if isinstance(transformer_options, dict) else None}",
        )
        if config is None:
            return freqs
        timestep = _resolve_timestep(transformer_options)
        out = _scale_wan_freqs(config, inner, freqs, timestep=timestep)
        _mark_scaled_freqs(transformer_options, out)
        return out

    inner._tide_wan_original_rope_encode = original_rope_encode
    inner._tide_wan_fallback_config = fallback_config
    inner.rope_encode = tide_wan_rope_encode
    inner._tide_wan_rope_encode_wrapped = True
    _debug_print(fallback_config, f"[ComfyUI-TIDE] WAN wrapped rope_encode on {type(inner).__name__} id={id(inner)}")
    return True


def _ensure_wan_forward_orig_wrapped(inner: Any, fallback_config: Optional[TIDEConfig]) -> bool:
    if not _looks_like_wan_inner(inner):
        return False
    if getattr(inner, "_tide_wan_forward_orig_wrapped", False):
        inner._tide_wan_fallback_config = fallback_config
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
        config = _resolve_config(transformer_options, getattr(inner, "_tide_wan_fallback_config", fallback_config))
        already_scaled = _freqs_already_scaled(transformer_options, freqs)
        _trace_inner(
            config,
            inner,
            "[ComfyUI-TIDE] WAN forward_orig hook entered: "
            f"freqs_tensor={torch.is_tensor(freqs)} "
            f"freqs_shape={tuple(freqs.shape) if torch.is_tensor(freqs) else None} "
            f"already_scaled={bool(already_scaled)} "
            f"config_source={transformer_options.get(_CONFIG_SOURCE_KEY) if isinstance(transformer_options, dict) else None}",
        )
        if config is not None and torch.is_tensor(freqs) and not already_scaled:
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
    inner._tide_wan_fallback_config = fallback_config
    inner.forward_orig = tide_wan_forward_orig
    inner._tide_wan_forward_orig_wrapped = True
    _debug_print(fallback_config, f"[ComfyUI-TIDE] WAN wrapped forward_orig on {type(inner).__name__} id={id(inner)}")
    return True


def _wan_candidates_from_outer(outer: Any) -> list[Any]:
    if outer is None:
        return []

    candidates = [getattr(outer, "diffusion_model", None)]
    model = getattr(outer, "model", None)
    if model is not None:
        candidates.extend(
            [
                getattr(model, "diffusion_model", None),
                getattr(getattr(model, "model", None), "diffusion_model", None),
            ]
        )
    return candidates


def _resolve_apply_model_outer(apply_model: Any) -> Any:
    outer = getattr(apply_model, "__self__", None)
    if outer is not None:
        return outer

    # Spectrum WAN replaces BaseModel.apply_model with a per-instance closure.
    # Functions assigned to an instance are not descriptors, so __self__ is lost.
    # Recover the captured BaseModel/outer object from the closure instead of
    # silently skipping the WAN prepare path.
    closure = getattr(apply_model, "__closure__", None)
    if not closure:
        return None

    matches: list[tuple[Any, Any]] = []
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        bound_self = getattr(value, "__self__", None)
        if bound_self is not None:
            value = bound_self
        for candidate in _wan_candidates_from_outer(value):
            if _looks_like_wan_inner(candidate) and not any(candidate is seen for _, seen in matches):
                matches.append((value, candidate))
    if len(matches) == 1:
        return matches[0][0]
    return None


def _resolve_apply_model_wan_inner(outer: Any) -> Any:
    for candidate in _wan_candidates_from_outer(outer):
        if _looks_like_wan_inner(candidate):
            return candidate
    return None



def _get_block_scaled_freqs(
    transformer_options: Optional[dict[str, Any]],
    config: TIDEConfig,
    inner: Any,
    freqs: Any,
) -> Any:
    if not torch.is_tensor(freqs):
        return freqs
    if _freqs_already_scaled(transformer_options, freqs):
        return freqs

    cache = None
    if isinstance(transformer_options, dict):
        cache = transformer_options.setdefault(_BLOCK_SCALED_CACHE_KEY, {})
        if isinstance(cache, dict):
            cached = cache.get(id(freqs))
            if torch.is_tensor(cached):
                return cached

    timestep = _resolve_timestep(transformer_options)
    out = _scale_wan_freqs(config, inner, freqs, timestep=timestep)
    _mark_scaled_freqs(transformer_options, out)
    if isinstance(cache, dict) and torch.is_tensor(out):
        cache[id(freqs)] = out
    return out


def _install_wan_block_dtc_patches(
    transformer_options: Optional[dict[str, Any]],
    config: TIDEConfig,
    inner: Any,
    *,
    source: str,
) -> bool:
    """Install a block-level WAN PE scaling fallback.

    Spectrum WAN 2.1 can run actual forwards through an APPLY_MODEL driver where
    neither the native WanModel.rope_encode hook nor the WanModel.forward_orig
    hook is reached by TIDE, even though transformer block execution still goes
    through Comfy's patches_replace path.  Wrapping the block replacement hooks
    lets TIDE scale the shared RoPE matrix at the last stable point before each
    attention block while preserving existing DiffAid/Spectrum patches.
    """

    if not isinstance(transformer_options, dict) or not _looks_like_wan_inner(inner):
        return False

    patches_replace = transformer_options.get("patches_replace")
    if not isinstance(patches_replace, dict):
        patches_replace = {}
    dit_replace = patches_replace.get("dit")
    if not isinstance(dit_replace, dict):
        dit_replace = {}

    block_count = len(getattr(inner, "blocks", ()) or ())
    if block_count <= 0:
        return False

    def make_tide_wan_block_patch(previous_patch: Any, block_index: int):
        def tide_wan_block_patch(args, extra_options):
            local_config = getattr(tide_wan_block_patch, _BLOCK_PATCH_CONFIG_KEY, config)
            local_inner = getattr(tide_wan_block_patch, _BLOCK_PATCH_INNER_KEY, inner)
            block_transformer_options = None
            if isinstance(args, dict):
                block_transformer_options = args.get("transformer_options")
            if not isinstance(block_transformer_options, dict):
                block_transformer_options = transformer_options

            resolved_config = _resolve_config(block_transformer_options, local_config)
            if resolved_config is not None and isinstance(args, dict) and "pe" in args:
                if isinstance(block_transformer_options, dict):
                    block_transformer_options[_INNER_KEY] = local_inner
                pe = args.get("pe")
                scaled_pe = _get_block_scaled_freqs(block_transformer_options, resolved_config, local_inner, pe)
                if scaled_pe is not pe:
                    args = dict(args)
                    args["pe"] = scaled_pe
                    if resolved_config.debug and not getattr(tide_wan_block_patch, "_tide_wan_block_scaled_logged", False):
                        setattr(tide_wan_block_patch, "_tide_wan_block_scaled_logged", True)
                        _debug_print(
                            resolved_config,
                            "[ComfyUI-TIDE] WAN block DTC fallback scaled PE: "
                            f"block={block_index} source={getattr(tide_wan_block_patch, _BLOCK_PATCH_SOURCE_KEY, source)} "
                            f"pe_shape={tuple(pe.shape) if torch.is_tensor(pe) else None}",
                        )

            if callable(previous_patch):
                return previous_patch(args, extra_options)
            original_block = extra_options.get("original_block") if isinstance(extra_options, dict) else None
            if not callable(original_block):
                raise RuntimeError("TIDE WAN block DTC patch missing original_block")
            return original_block(args)

        return tide_wan_block_patch

    installed = 0
    for block_index in range(block_count):
        block_key = ("double_block", block_index)
        previous_patch = dit_replace.get(block_key)

        if callable(previous_patch) and getattr(previous_patch, _BLOCK_PATCH_KEY, False):
            setattr(previous_patch, _BLOCK_PATCH_CONFIG_KEY, config)
            setattr(previous_patch, _BLOCK_PATCH_INNER_KEY, inner)
            setattr(previous_patch, _BLOCK_PATCH_SOURCE_KEY, source)
            installed += 1
            continue

        tide_wan_block_patch = make_tide_wan_block_patch(previous_patch, block_index)
        setattr(tide_wan_block_patch, _BLOCK_PATCH_KEY, True)
        setattr(tide_wan_block_patch, _BLOCK_PATCH_CONFIG_KEY, config)
        setattr(tide_wan_block_patch, _BLOCK_PATCH_INNER_KEY, inner)
        setattr(tide_wan_block_patch, _BLOCK_PATCH_SOURCE_KEY, source)
        dit_replace[block_key] = tide_wan_block_patch
        installed += 1

    patches_replace["dit"] = dit_replace
    transformer_options["patches_replace"] = patches_replace

    if config.debug:
        marker = "_tide_wan_block_dtc_logged"
        if not getattr(inner, marker, False):
            setattr(inner, marker, True)
            _debug_print(
                config,
                "[ComfyUI-TIDE] WAN block DTC fallback installed: "
                f"blocks={installed}/{block_count} source={source}",
            )
    return installed > 0


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
        inject_tide_wan_options(transformer_options, self.config, timestep=timestep, source="diffusion_wrapper")

        inner = getattr(executor, "class_obj", None)
        if isinstance(transformer_options, dict):
            transformer_options[_INNER_KEY] = inner
        wrapped_rope = _ensure_wan_rope_encode_wrapped(inner, self.config)
        wrapped_forward_orig = _ensure_wan_forward_orig_wrapped(inner, self.config)
        installed_block_dtc = _install_wan_block_dtc_patches(transformer_options, self.config, inner, source="diffusion_wrapper")
        if self.config.debug:
            if not getattr(self, "_logged_runtime", False):
                _debug_print(
                    self.config,
                    "[ComfyUI-TIDE] WAN diffusion wrapper active: "
                    f"inner={type(inner).__name__ if inner is not None else None} "
                    f"wrapped_rope={bool(wrapped_rope)} wrapped_forward_orig={bool(wrapped_forward_orig)} "
                    f"block_dtc={bool(installed_block_dtc)} "
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


def prepare_tide_wan_apply_model(
    apply_model: Any,
    config: TIDEConfig,
    transformer_options: Optional[dict[str, Any]],
) -> None:
    """Prepare WAN DTC before Comfy's APPLY_MODEL wrappers run.

    Spectrum WAN 2.1 can drive the WAN model from an APPLY_MODEL wrapper and
    return without entering Comfy's native DIFFUSION_MODEL wrapper chain.  The
    normal TIDE WAN diffusion wrapper therefore cannot be the only lazy binding
    point.  This hook runs from TIDE's model_function_wrapper, mutates the same
    transformer_options dict the sampler will pass into apply_model, and wraps
    the live WanModel before Spectrum's APPLY_MODEL wrapper can bypass the lower
    wrapper chain.
    """

    if transformer_options is None:
        return

    outer = _resolve_apply_model_outer(apply_model)
    inner = _resolve_apply_model_wan_inner(outer)
    if not _looks_like_wan_inner(inner):
        return

    inject_tide_wan_options(transformer_options, config, source="model_function_wrapper")
    transformer_options[_INNER_KEY] = inner

    wrapped_rope = _ensure_wan_rope_encode_wrapped(inner, config)
    wrapped_forward_orig = _ensure_wan_forward_orig_wrapped(inner, config)
    installed_block_dtc = _install_wan_block_dtc_patches(transformer_options, config, inner, source="model_function_wrapper")

    if config.debug:
        # Log once per live inner. This is intentionally separate from
        # TIDEWanDiffusionWrapper's runtime log because Spectrum WAN 2.1 may
        # bypass that path completely.
        marker = "_tide_wan_apply_model_prepare_logged"
        if not getattr(inner, marker, False):
            setattr(inner, marker, True)
            _debug_print(
                config,
                "[ComfyUI-TIDE] WAN apply_model prepare active: "
                f"outer={type(outer).__name__ if outer is not None else None} "
                f"inner={type(inner).__name__ if inner is not None else None} "
                f"wrapped_rope={bool(wrapped_rope)} wrapped_forward_orig={bool(wrapped_forward_orig)} "
                f"block_dtc={bool(installed_block_dtc)} "
                f"enabled={bool(transformer_options[_ENABLED_KEY])}",
            )


def install_tide_wan_patch(model: Any, config: TIDEConfig) -> Any:
    """Install the WAN-specific TIDE DTC path on a cloned ComfyUI MODEL."""

    transformer_options = _ensure_transformer_options(model)
    inject_tide_wan_options(transformer_options, config, source="node_install")
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
    "has_tide_wan_options",
    "install_tide_wan_patch",
    "prepare_tide_wan_apply_model",
    "inject_tide_wan_options",
    "_scale_wan_freqs",
]
