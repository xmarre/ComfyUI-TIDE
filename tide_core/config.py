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

    def __post_init__(self) -> None:
        for name in ("width", "height", "base_width", "base_height", "token_px"):
            value = getattr(self, name)
            try:
                numeric_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a positive integer-like value, got {value!r}") from exc
            if numeric_value <= 0:
                raise ValueError(f"{name} must be > 0, got {value!r}")

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

    @property
    def is_axis_extrapolating(self) -> bool:
        return self.scale_x > 1.0 or self.scale_y > 1.0

    def should_apply_text_anchor(self) -> bool:
        # Text anchoring is derived from total text/image token dilution, so keep
        # the area-based gate unless the user explicitly forces native/smaller use.
        return self.apply_to_native_or_smaller or self.is_extrapolating

    def should_apply_temperature(self) -> bool:
        # Dynamic Temperature Control is RoPE-axis based. Wide or tall WAN/Flux
        # generations can extrapolate on one spatial axis while total pixel count
        # is <= the square base area, e.g. 832x480 vs 640x640. In that case the
        # width axis still needs the DTC path.
        return self.apply_to_native_or_smaller or self.is_axis_extrapolating

    def should_apply(self) -> bool:
        return self.should_apply_text_anchor() or self.should_apply_temperature()
