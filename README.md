# ComfyUI-TIDE

ComfyUI-TIDE is a ComfyUI custom node for applying the core inference-time mechanisms from **TIDE: Text-Informed Dynamic Extrapolation with Step-Aware Temperature Control for Diffusion Transformers** to Flux-style DiT attention in ComfyUI.

The node patches a cloned ComfyUI `MODEL` object. It does **not** add extra sampling steps, replace the sampler, replace the scheduler, or fork ComfyUI core.

## Credits and attribution

The algorithmic method implemented here is based on the TIDE paper:

**TIDE: Text-Informed Dynamic Extrapolation with Step-Aware Temperature Control for Diffusion Transformers**  
Yihua Liu, Fanjiang Ye, Bowen Lin, Rongyu Fang, Chengming Zhang  
arXiv:2603.08928, 2026  
<https://arxiv.org/abs/2603.08928>

Credit for the TIDE method, including **Text Anchoring**, **Dynamic Temperature Control**, and the paper's analysis of attention dilution in high-resolution Diffusion Transformer generation, belongs to the paper authors.

This repository is an independent ComfyUI custom-node implementation. It is not the official TIDE repository, not affiliated with the paper authors, and should not be cited as the original method. If this node is useful in your work, cite the TIDE paper.

```bibtex
@misc{liu2026tide,
  title         = {TIDE: Text-Informed Dynamic Extrapolation with Step-Aware Temperature Control for Diffusion Transformers},
  author        = {Yihua Liu and Fanjiang Ye and Bowen Lin and Rongyu Fang and Chengming Zhang},
  year          = {2026},
  eprint        = {2603.08928},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2603.08928}
}
```

## Scope

This repository implements the practical ComfyUI attention-side parts of TIDE for Flux-style joint text/image attention:

- text-token additive bias for Text Anchoring;
- step-aware RoPE temperature scaling for Dynamic Temperature Control;
- a lightweight ComfyUI model wrapper to pass the current denoising timestep into the attention patch;
- a small PyTorch SDPA fallback used only when an additive TIDE attention mask is active.

It does **not** implement the full official Diffusers pipeline, benchmark harness, datasets, metric evaluation scripts, Qwen pipeline, or complete YaRN/DyPE/NTK positional interpolation stack.

## Supported model path

The target path is FLUX-family DiT models in ComfyUI, including FLUX.2-style paths if they use the same practical structure as ComfyUI's Flux implementation:

- joint text/image attention;
- text tokens before image tokens;
- `attn1_patch` support;
- `extra_options["img_slice"]` available at the attention patch site;
- RoPE matrix passed as `pe`.

Other DiT models may require model-specific patch paths. They should not be assumed to work unless their ComfyUI implementation exposes the same attention-patch contract.

## What is implemented

### 1. Text Anchoring

TIDE identifies text-token influence decay as a core failure mode at high resolution: image token count grows with resolution while text token count stays fixed. Text Anchoring counteracts this by adding a positive bias to attention logits whose keys are text tokens.

This node computes the default adaptive bias as:

```text
beta = log((target_width * target_height) / (base_width * base_height))
```

With the default `base_width=1024` and `base_height=1024`, this is equivalent to:

```text
log(width / 1024) + log(height / 1024)
```

The final applied value is:

```text
applied_beta = text_anchor_strength * beta
```

By default, the node applies no Text Anchoring at native-or-smaller token counts unless `apply_to_native_or_smaller=True`.

### 2. Dynamic Temperature Control

TIDE uses a step-aware temperature curve so attention sharpening is stronger in the early/global part of denoising and relaxes toward the late/detail part of denoising. This repository applies that idea as a RoPE temperature multiplier, matching the reference implementation strategy rather than inserting a new attention kernel for every backend.

Default curve:

```text
tau(t, f) = tau_max - (tau_max - tau_min) * t ** alpha(f)
alpha(f) = alpha_low + (alpha_high - alpha_low) * f
```

Defaults:

| Parameter | Default |
|---|---:|
| `tau_max` | `1.0` |
| `alpha_low` | `0.6` |
| `alpha_high` | `0.2` |
| `temperature_strength` | `1.0` |
| `frequency_mode` | `official_raw` |

`frequency_mode=official_raw` uses raw RoPE frequencies, matching the released implementation behavior this port was written against. `paper_normalized` is exposed for comparison because the paper notation describes a normalized frequency variable.

## Implementation notes

1. The node clones and patches the incoming ComfyUI `MODEL`.
2. TIDE state is stored on the cloned model through ComfyUI model options.
3. `TIDEModelWrapper` injects current timestep metadata into `transformer_options["tide"]`.
4. `TIDEAttentionPatch` applies Text Anchoring and Dynamic Temperature Control through ComfyUI's `attn1_patch` hook.
5. `TIDEAttentionOverride` forces a compact PyTorch SDPA path only when an additive TIDE mask is active. This avoids attention backends that reject additive masks or try to materialize a dense high-resolution mask.
6. No global monkey-patching is used.
7. No sampler or scheduler rewrite is performed.

## Paper / reference-code / local-module mapping

| Paper component | Reference implementation behavior | This repository |
|---|---|---|
| Text-token influence decay | Additive text-token attention mask | `tide_core.math.adaptive_text_bias`, `tide_core.patches.TIDEAttentionPatch` |
| Text Anchoring, `beta = log(lambda)` | `log(width / 1024) + log(height / 1024)` for FLUX-sized text prefix | Adaptive beta from node `width`, `height`, `base_width`, `base_height`; text-token count inferred from ComfyUI `img_slice` |
| YaRN temperature baseline | `get_mscale`, default temperature from extrapolation scale | `tide_core.math.get_mscale`, `get_default_temperature` |
| Dynamic Temperature Control | `dyheating()` / temperature-aware RoPE scaling | `tide_core.math.rope_temperature_scale`, applied to ComfyUI `pe` |
| Denoising-step-aware behavior | Update position embedding state from current timestep | `TIDEModelWrapper` injects normalized timestep into `transformer_options` |
| FLUX attention integration | Modified Diffusers FLUX transformer/processor | ComfyUI `attn1_patch` plus optional `optimized_attention_override` |
| Logarithmic FLUX scheduler shift | Scheduler/pipeline-level change | Not implemented by this node |
| DyPE / NTK-by-parts / YaRN positional interpolation | Custom positional interpolation stack | Not fully implemented; this node implements the TIDE attention-side mechanisms only |

## Installation

Clone this repository into ComfyUI's custom node directory:

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-TIDE
```

Restart ComfyUI.

No extra runtime dependency is required beyond the PyTorch/ComfyUI environment. `requirements.txt` lists `torch` for standalone tests.

## Usage

1. Load a FLUX-family model as usual.
2. Add **TIDE High-Resolution Extrapolation** after the model loader.
3. Connect the patched `model` output to your sampler.
4. Set `width` and `height` to the final generated image dimensions used by your latent node.
5. Keep `base_width=1024` and `base_height=1024` for FLUX-family models unless you know the model's native training target differs.

Recommended starting values:

| Setting | Value |
|---|---:|
| `width` / `height` | final image dimensions |
| `base_width` / `base_height` | `1024` / `1024` |
| `text_anchor_strength` | `1.0` |
| `temperature_strength` | `1.0` |
| `alpha_low` | `0.6` |
| `alpha_high` | `0.2` |
| `tau_max` | `1.0` |
| `frequency_mode` | `official_raw` |
| `force_pytorch_attention_with_mask` | `True` |

Ablation settings:

| Test | Settings |
|---|---|
| Text Anchoring only | `text_anchor_strength=1.0`, `temperature_strength=0.0` |
| Dynamic Temperature only | `text_anchor_strength=0.0`, `temperature_strength=1.0` |
| Disabled | `text_anchor_strength=0.0`, `temperature_strength=0.0` |

## Node inputs

### Required

| Input | Description |
|---|---|
| `model` | ComfyUI `MODEL` object to patch. |
| `width`, `height` | Final target generation dimensions in pixels. Must match the latent/image size used by the workflow. |
| `text_anchor_strength` | Multiplier on adaptive beta. `1.0` follows the paper/reference behavior. `0.0` disables Text Anchoring. |
| `temperature_strength` | Multiplier on Dynamic Temperature Control. `1.0` follows the reference curve. `0.0` disables DTC. |

### Optional

| Input | Default | Description |
|---|---:|---|
| `base_width`, `base_height` | `1024`, `1024` | Native/training resolution used for adaptive scaling. |
| `alpha_low`, `alpha_high` | `0.6`, `0.2` | DTC exponents for low/high RoPE frequency behavior. |
| `tau_max` | `1.0` | Maximum temperature reached near the end of denoising. |
| `frequency_mode` | `official_raw` | `official_raw` or `paper_normalized`. |
| `apply_to_double_blocks` | `True` | Apply patch to FLUX double-stream blocks. |
| `apply_to_single_blocks` | `True` | Apply patch to FLUX single-stream blocks. |
| `apply_to_native_or_smaller` | `False` | Allow patching even when target token count is not above base token count. |
| `force_pytorch_attention_with_mask` | `True` | Use internal PyTorch SDPA only when TIDE's additive mask is active. |
| `preserve_existing_wrapper` | `True` | Delegate to an existing ComfyUI model wrapper after injecting TIDE metadata. |
| `debug` | `False` | Log skipped DTC shape mismatches and exceptions. |

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

Run the standalone tests from the repository root:

```bash
python -m pip install pytest torch
python -m pytest -q
```

The tests cover:

- adaptive Text Anchoring beta computation;
- YaRN/default temperature formula;
- RoPE temperature scale shape and timestep progression;
- additive attention-mask creation;
- masked SDPA override behavior.

The tests do not validate visual quality or live ComfyUI execution.

## Paper vs implementation differences

### Scheduler time shifting

The paper appendix describes a logarithmic FLUX time-shift schedule for high resolutions. This node receives an already-built ComfyUI sampler schedule and does not silently rewrite it. Use a scheduler setup that does not over-shift high-resolution FLUX timesteps.

### Positional interpolation

The paper evaluates TIDE in combination with positional extrapolation/interpolation methods such as YaRN/DyPE-style handling. This repository does not port the full positional interpolation stack. It applies the TIDE attention-side mechanisms through ComfyUI's patch system.

### Frequency variable

The paper notation describes a normalized frequency variable for `alpha(f)`. The reference implementation behavior used by this port applies the curve using raw RoPE frequencies. This mismatch is exposed as `frequency_mode` so the default can follow the reference behavior while still allowing controlled comparison.

### Text-token count

The paper writes the text-token length abstractly as `L_T`. Some FLUX implementations use a fixed text-token prefix length. This repository infers the text prefix from ComfyUI's `img_slice` rather than hard-coding a token count.

### General DiT support

The method is architecture-relevant to DiTs, but this implementation is tied to ComfyUI's Flux-style patch interface. Non-Flux DiTs need compatible attention hooks and token-layout metadata.

## Assumptions

- The model uses Flux-style joint attention with text tokens before image tokens.
- `extra_options["img_slice"]` identifies the split between text and image tokens.
- The ComfyUI attention patch receives a RoPE matrix through `pe`.
- The node `width` and `height` match the actual generated dimensions.
- The timestep passed through the model wrapper is normalized or sigma-like in `[0, 1]`; values outside the interval are clamped.
- FLUX-family image token granularity is 16 pixels per transformer token.

## Limitations

- Not an official TIDE release.
- Not a full reproduction of the paper's complete experimental pipeline.
- Does not include paper benchmark scripts, datasets, generated result images, or metric evaluation.
- Does not modify the sampler's high-resolution time-shift schedule.
- Does not fully implement NTK-by-parts, YaRN positional interpolation, or DyPE positional interpolation.
- Visual quality still requires live workflow testing.
- Very large resolutions still require sufficient VRAM for the selected model, sampler, attention path, latent size, and VAE path.

## License

This repository is released under the MIT License; see [`LICENSE`](LICENSE).

ComfyUI is GPL-3.0 licensed. This custom node is distributed as a separate plugin, but it imports and runs inside ComfyUI. Review license compatibility before redistributing this node as part of a larger bundled package.
