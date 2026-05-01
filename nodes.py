from __future__ import annotations

from typing import Any

try:
    from .tide_core import TIDEConfig, TIDEAttentionOverride, TIDEAttentionPatch, TIDEModelWrapper
except ImportError:
    from tide_core import TIDEConfig, TIDEAttentionOverride, TIDEAttentionPatch, TIDEModelWrapper


class TIDEHighResolutionExtrapolation:
    """Patch a Flux-style DiT model with TIDE text anchoring and dynamic temperature."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", {"default": 2048, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 2048, "min": 16, "max": 16384, "step": 16}),
                "text_anchor_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Multiplier on paper beta=log(target_pixels/base_pixels). 1.0 matches the paper/official code."},
                ),
                "temperature_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "0 disables Dynamic Temperature Control; 1.0 matches the official dyheating curve."},
                ),
            },
            "optional": {
                "base_width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
                "base_height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
                "alpha_low": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 8.0, "step": 0.05}),
                "alpha_high": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 8.0, "step": 0.05}),
                "tau_max": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01}),
                "frequency_mode": (["official_raw", "paper_normalized"], {"default": "official_raw"}),
                "apply_to_double_blocks": ("BOOLEAN", {"default": True}),
                "apply_to_single_blocks": ("BOOLEAN", {"default": True}),
                "apply_to_native_or_smaller": ("BOOLEAN", {"default": False}),
                "force_pytorch_attention_with_mask": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Use PyTorch SDPA for masked TIDE attention to avoid backends that reject or densify additive masks."},
                ),
                "preserve_existing_wrapper": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/TIDE"

    def patch(
        self,
        model,
        width: int,
        height: int,
        text_anchor_strength: float,
        temperature_strength: float,
        base_width: int = 1024,
        base_height: int = 1024,
        alpha_low: float = 0.6,
        alpha_high: float = 0.2,
        tau_max: float = 1.0,
        frequency_mode: str = "official_raw",
        apply_to_double_blocks: bool = True,
        apply_to_single_blocks: bool = True,
        apply_to_native_or_smaller: bool = False,
        force_pytorch_attention_with_mask: bool = True,
        preserve_existing_wrapper: bool = True,
        debug: bool = False,
    ):
        config = TIDEConfig(
            width=int(width),
            height=int(height),
            base_width=int(base_width),
            base_height=int(base_height),
            text_anchor_strength=float(text_anchor_strength),
            temperature_strength=float(temperature_strength),
            alpha_low=float(alpha_low),
            alpha_high=float(alpha_high),
            tau_max=float(tau_max),
            frequency_mode=str(frequency_mode),
            apply_to_double_blocks=bool(apply_to_double_blocks),
            apply_to_single_blocks=bool(apply_to_single_blocks),
            apply_to_native_or_smaller=bool(apply_to_native_or_smaller),
            force_pytorch_attention_with_mask=bool(force_pytorch_attention_with_mask),
            preserve_existing_wrapper=bool(preserve_existing_wrapper),
            debug=bool(debug),
        )

        patched = model.clone()

        old_wrapper = patched.model_options.get("model_function_wrapper")
        patched.set_model_unet_function_wrapper(TIDEModelWrapper(config, old_wrapper=old_wrapper))
        patched.set_model_attn1_patch(TIDEAttentionPatch(config))

        transformer_options = patched.model_options.get("transformer_options", {}).copy()
        old_override = transformer_options.get("optimized_attention_override")
        transformer_options["optimized_attention_override"] = TIDEAttentionOverride(config, old_override=old_override)
        patched.model_options["transformer_options"] = transformer_options

        return (patched,)


NODE_CLASS_MAPPINGS = {
    "TIDEHighResolutionExtrapolation": TIDEHighResolutionExtrapolation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TIDEHighResolutionExtrapolation": "TIDE High-Resolution Extrapolation",
}
