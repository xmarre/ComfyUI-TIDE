# ComfyUI-TIDE

ComfyUI-TIDE is a ComfyUI custom node implementation of **TIDE: Text-Informed Dynamic Extrapolation with Step-Aware Temperature Control for Diffusion Transformers**.

The node is intended for FLUX-family DiT models in ComfyUI, including FLUX.2-style model paths that use the same joint text/image attention structure as ComfyUI's FLUX implementation. It patches the `MODEL` object and does not add sampling steps.

## What is implemented

TIDE has two core inference-time mechanisms:

1. **Text Anchoring**
   - Adds a positive additive bias to attention logits whose keys are text tokens.
   - The default adaptive bias is:

     ```text
     beta = log((target_width * target_height) / (base_width * base_height))
     ```

   - With the default base resolution of 1024x1024, this matches the official FLUX script's `log(width / 1024) + log(height / 1024)`.

2. **Dynamic Temperature Control**
   - Applies the paper/official-code YaRN temperature concept as a per-frequency multiplier on ComfyUI's RoPE matrix.
   - The default curve follows the released implementation's `dyheating` form:

     ```text
     tau(t, f) = tau_max - (tau_max - tau_min) * t ** alpha(f)
     alpha(f) = alpha_low + (alpha_high - alpha_low) * f
     ```

   - Defaults: `tau_max=1.0`, `alpha_low=0.6`, `alpha_high=0.2`.
   - `frequency_mode=official_raw` uses raw RoPE frequencies, matching the released code. `paper_normalized` is available for comparison because the paper text describes a normalized frequency variable.

## Implementation plan

1. Patch ComfyUI attention rather than cloning a full model implementation.
2. Use ComfyUI's existing `attn1_patch` hook to modify joint DiT attention inputs.
3. Use a model function wrapper to expose the current denoising timestep to the attention patch.
4. Keep TIDE state local to the cloned `MODEL` object.
5. Avoid global monkey-patching and avoid changing the scheduler or sampler.
6. Force a small PyTorch SDPA attention path only when an additive TIDE mask is present, preventing backends from rejecting or materializing a dense high-resolution mask.

## Paper / official-code / local-module mapping

| Paper component | Official implementation component | This repository |
|---|---|---|
| Text-token influence decay analysis | `run.py::get_attn_mask()` creates an additive text-token mask | `tide_core.math.adaptive_text_bias`, `tide_core.patches.TIDEAttentionPatch` |
| Text Anchoring, beta = log(lambda) | `log(width / 1024) + log(height / 1024)` for the first 512 text tokens | Adaptive beta computed from node `width`, `height`, `base_width`, `base_height`; text-token count is inferred from ComfyUI `img_slice` instead of hard-coded to 512 |
| YaRN temperature baseline | `get_mscale`, `get_default_temperature` | `tide_core.math.get_mscale`, `get_default_temperature` |
| Dynamic Temperature Control | `dyheating()` and `tuning_temperature()` multiply RoPE cos/sin by `1/sqrt(tau)` | `tide_core.math.rope_temperature_scale`, applied to ComfyUI's RoPE matrix in `TIDEAttentionPatch` |
| Denoising-step-aware RoPE | Official `FluxPosEmbed.update_timestep(timestep.item())` | `TIDEModelWrapper` injects normalized current timestep into `transformer_options["tide"]` |
| FLUX attention integration | Official fork modifies Diffusers `FluxTransformer2DModel` / processor | ComfyUI `attn1_patch` and `optimized_attention_override`; no fork of ComfyUI core |
| Logarithmic FLUX scheduler shift | Official `FluxPipeline(..., shift_mode="log")` | Not implemented as a node-level sampler change; see limitations |
| DyPE / NTK-by-parts / YaRN positional interpolation | Official fork implements custom RoPE interpolation | Not fully implemented; this node implements TIDE's attention-side mechanisms and optional frequency interpretation only |

## Installation

Clone or copy this repository into ComfyUI's custom node directory:

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-TIDE
```

Restart ComfyUI.

No extra Python dependency is required beyond the PyTorch/ComfyUI environment. `requirements.txt` lists `torch` only for standalone unit tests.

## Usage

1. Add **TIDE High-Resolution Extrapolation** after your model loader.
2. Connect its `model` output to the sampler model input.
3. Set `width` and `height` to the final generation size used by your latent node.
4. Use a FLUX-family DiT model.

Recommended starting values:

| Setting | Value |
|---|---:|
| `width` / `height` | final latent/image size |
| `base_width` / `base_height` | `1024` / `1024` for FLUX-family models |
| `text_anchor_strength` | `1.0` |
| `temperature_strength` | `1.0` |
| `alpha_low` | `0.6` |
| `alpha_high` | `0.2` |
| `frequency_mode` | `official_raw` |
| `force_pytorch_attention_with_mask` | `True` |

For ablations:

- Text Anchoring only: `temperature_strength=0.0`, `text_anchor_strength=1.0`
- Dynamic Temperature only: `text_anchor_strength=0.0`, `temperature_strength=1.0`
- Disable the node behavior without removing it: set both strengths to `0.0`

## Node inputs

### Required

- `model`: ComfyUI MODEL to patch.
- `width`, `height`: final target generation dimensions in pixels.
- `text_anchor_strength`: multiplier on beta. `1.0` follows paper/official behavior.
- `temperature_strength`: multiplier on the RoPE temperature scale. `1.0` follows official behavior; `0.0` disables DTC.

### Optional

- `base_width`, `base_height`: training/native resolution used for adaptive scaling. Default is 1024x1024.
- `alpha_low`, `alpha_high`: Dynamic Temperature Control exponents.
- `tau_max`: maximum temperature. Default 1.0.
- `frequency_mode`:
  - `official_raw`: matches the released code's raw RoPE-frequency use.
  - `paper_normalized`: normalizes frequency to [0, 1] as a literal reading of the paper notation.
- `apply_to_double_blocks`, `apply_to_single_blocks`: choose which FLUX block types receive the patch.
- `apply_to_native_or_smaller`: default false. Prevents non-positive or native-resolution anchoring from changing normal-resolution behavior.
- `force_pytorch_attention_with_mask`: default true. Uses an internal PyTorch SDPA path only when the TIDE additive mask is active.
- `preserve_existing_wrapper`: default true. Delegates to an existing model wrapper after injecting TIDE timestep metadata.
- `debug`: logs skipped dynamic-temperature shape mismatches and exceptions.

## Repository structure

```text
ComfyUI-TIDE/
├── __init__.py
├── nodes.py
├── requirements.txt
├── README.md
├── examples/
│   └── README.md
├── tide_core/
│   ├── __init__.py
│   ├── config.py
│   ├── math.py
│   └── patches.py
└── tests/
    ├── test_attention_patch.py
    └── test_math.py
```

## Tests

Standalone math/patch tests:

```bash
cd ComfyUI-TIDE
python -m pip install pytest torch
python -m pytest -q
```

These tests validate:

- adaptive beta computation,
- YaRN temperature formula,
- RoPE scale shape and timestep progression,
- attention-mask creation,
- masked SDPA override behavior.

They do not validate visual quality or live ComfyUI model execution.

## Paper vs official code mismatches and resolutions

### 1. Frequency variable in Dynamic Temperature Control

- Paper: describes `f` as frequency normalized into a range used by `alpha(f)`.
- Official code: multiplies `(alpha_high - alpha_low)` by raw RoPE frequencies.
- Resolution: default `frequency_mode=official_raw` to match official behavior; `paper_normalized` is exposed for controlled comparison.

### 2. Dynamic Temperature implementation site

- Paper: formulates the final attention with a temperature term in the softmax denominator.
- Official FLUX code: implements temperature by multiplying RoPE cos/sin by `1/sqrt(tau)` inside YaRN RoPE generation.
- Resolution: ComfyUI exposes RoPE matrices through `pe`; this repo applies the official-code equivalent multiplier to `pe`.

### 3. Text token count

- Paper: text-token length is abstract `L_T`.
- Official FLUX script: hard-codes 512 text tokens for FLUX.1.
- Resolution: this repo infers text-token count from ComfyUI's `img_slice`, avoiding hard-coding 512 and making FLUX-family variants more likely to work.

### 4. Scheduler time shifting

- Paper appendix and official pipeline use a logarithmic FLUX time-shift schedule for high resolutions.
- This custom node receives an already-built sampler schedule and should not silently alter the user's sampler.
- Resolution: no sampler schedule rewrite is performed. This is a deliberate deviation. Use a ComfyUI sampler/scheduler setup that does not over-shift high-resolution FLUX timesteps.

### 5. Positional interpolation / DyPE / YaRN

- Paper experiments combine TIDE with YaRN/DyPE-style positional handling.
- Official code includes a Diffusers FLUX fork with NTK, NTK-by-parts, YaRN, and DyPE RoPE logic.
- Resolution: this repo implements TIDE's attention-side contribution in ComfyUI and does not clone the full official Diffusers transformer. This avoids replacing ComfyUI internals but means it is not a complete official YaRN/DyPE port.

### 6. Qwen/general DiT support

- Official code includes Qwen-Image support.
- ComfyUI Qwen's current patch path does not propagate a patch-returned additive attention mask to the final attention call in the same way as FLUX.
- Resolution: this repo is implemented for Flux-style ComfyUI joint attention first. Other DiTs may work only if their ComfyUI implementation exposes compatible `attn1_patch`, `img_slice`, and additive-mask propagation.

## Assumptions

- The model uses ComfyUI's Flux-style joint attention with text tokens before image tokens.
- ComfyUI provides `extra_options["img_slice"]` in attention patches.
- `width` and `height` passed to this node match the actual generated latent/image dimensions.
- The timestep seen by the wrapper is already normalized or sigma-like in [0, 1]. Values outside the interval are clamped.
- FLUX-family token granularity is 16 image pixels per transformer token.

## Limitations

- Not a full official repository clone.
- Does not implement official Diffusers pipeline scripts, benchmark code, metric evaluation, datasets, or training code.
- Does not modify the sampler's high-resolution time-shift schedule.
- Does not implement full NTK-by-parts, YaRN positional interpolation, or DyPE positional interpolation.
- Visual quality is unverified in this static repository export.
- Very large resolutions still require enough VRAM for the chosen model, sampler, attention backend, and VAE path.

## Unresolved uncertainties

- Exact FLUX.2 internal token layout may differ from FLUX.1 depending on the ComfyUI model implementation. If it still uses Flux-style `img_slice` with text tokens first, this patch should apply.
- Whether ComfyUI's current default scheduler for FLUX.2 already avoids the extreme high-resolution time-shift problem described in the paper needs live workflow verification.
- The paper notation and official code differ on frequency normalization; defaulting to official code is the safest reproducibility choice, but it is still a documented mismatch.

## License

This repository is an implementation scaffold for ComfyUI. ComfyUI itself is GPL-licensed. Review license compatibility before redistributing as part of a larger package.
