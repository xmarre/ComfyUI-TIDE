from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TIDEConfig:
    """Runtime configuration for the ComfyUI TIDE patch.

    The defaults intentionally match the paper/official-code path where feasible:
    base 1024x1024 FLUX training resolution, alpha_low=0.6, alpha_high=0.2,
    tau_max=1.0, and Flux-style latent patch granularity of 16 image pixels per
    transformer token.
    """

    width: int
    height: int
    base_width: int = 1024
    base_height: int = 1024
    token_px: int = 16
    text_anchor_strength: float = 1.0
    temperature_strength: float = 1.0
    alpha_low: float = 0.6
    alpha_high: float = 0.2
    tau_max: float = 1.0
    theta: float = 10000.0
    axes_dim: tuple[int, int, int] = (16, 56, 56)
    frequency_mode: str = "official_raw"
    apply_to_double_blocks: bool = True
    apply_to_single_blocks: bool = True
    apply_to_native_or_smaller: bool = False
    force_pytorch_attention_with_mask: bool = True
    preserve_existing_wrapper: bool = True
    debug: bool = False

    @property
    def target_image_tokens(self) -> int:
        return max(1, (int(self.width) // self.token_px) * (int(self.height) // self.token_px))

    @property
    def base_image_tokens(self) -> int:
        return max(1, (int(self.base_width) // self.token_px) * (int(self.base_height) // self.token_px))

    @property
    def target_pixel_ratio(self) -> float:
        return max(1.0e-12, (float(self.width) * float(self.height)) / (float(self.base_width) * float(self.base_height)))

    @property
    def scale_x(self) -> float:
        return max(1.0e-12, float(self.width) / float(self.base_width))

    @property
    def scale_y(self) -> float:
        return max(1.0e-12, float(self.height) / float(self.base_height))

    @property
    def is_extrapolating(self) -> bool:
        return self.target_image_tokens > self.base_image_tokens

    def should_apply(self) -> bool:
        return self.apply_to_native_or_smaller or self.is_extrapolating
