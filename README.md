# ComfyUI-TIDE

ComfyUI-TIDE is a ComfyUI custom node implementation of inference-time mechanisms from **TIDE: Text-Informed Dynamic Extrapolation with Step-Aware Temperature Control for Diffusion Transformers**.

The primary implementation targets Flux-style DiT attention in ComfyUI. The repository also includes an experimental SDXL/UNet adaptation that applies the usable attention-temperature part of the method to SDXL-style `SpatialTransformer` attention.

The nodes patch a cloned ComfyUI `MODEL` object. They do **not** add extra sampling steps, replace the sampler, replace the scheduler, or fork ComfyUI core.

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
````

## Scope

This repository implements practical ComfyUI attention-side mechanisms inspired by TIDE.

For Flux-style joint text/image attention, it implements:

* text-token additive bias for Text Anchoring;
* step-aware RoPE temperature scaling for Dynamic Temperature Control;
* a lightweight ComfyUI model wrapper to pass the current denoising timestep into the attention patch;
* a small PyTorch SDPA fallback used only when an additive TIDE attention mask is active.

For SDXL-style UNet attention, it implements:

* step-aware attention-temperature scaling through ComfyUI's `optimized_attention_override`;
* optional application to SDXL cross-attention, self-attention, or both;
* model-local patching that chains with a pre-existing attention override when present.

It does **not** implement the full official Diffusers pipeline, benchmark harness, datasets, metric evaluation scripts, Qwen pipeline, or complete YaRN/DyPE/NTK positional interpolation stack.

## Supported model paths

### Flux / Flux.2-style DiT path

The main target path is FLUX-family DiT models in ComfyUI, including FLUX.2-style paths if they use the same practical structure as ComfyUI's Flux implementation:

* joint text/image attention;
* text tokens before image tokens;
* `attn1_patch` support;
* `extra_options["img_slice"]` available at the attention patch site;
* RoPE matrix passed as `pe`.

Other DiT models may require model-specific patch paths. They should not be assumed to work unless their ComfyUI implementation exposes the same attention-patch contract.

### SDXL / UNet path

The SDXL node targets ComfyUI's UNet `SpatialTransformer` attention path:

* SDXL-style cross-attention and self-attention;
* `optimized_attention_override` support;
* `transformer_options["activations_shape"]` present at the attention site.

The SDXL path is intentionally separate from the Flux path.

Important: SDXL support is **not** a full paper-faithful TIDE implementation. TIDE Text Anchoring is defined for joint text/image attention, where text keys and image keys compete inside one softmax. In SDXL UNet cross-attention, the keys/values are text-only. Adding the same positive bias to every text key would be cancelled by softmax shift invariance and would not change the output. SDXL self-attention has image tokens but no text keys. Therefore the SDXL node implements the usable part: step-aware attention-temperature control.

## Nodes

### TIDE High-Resolution Extrapolation

Use this node for FLUX-family DiT models.

Implemented mechanisms:

* Text Anchoring;
* Dynamic Temperature Control;
* optional PyTorch SDPA fallback when the additive text-anchor mask is active.

### TIDE SDXL High-Resolution Extrapolation

Use this node for SDXL-style UNet models.

Implemented mechanism:

* step-aware attention-temperature scaling.

Not implemented for SDXL:

* Text Anchoring;
* Flux-style RoPE temperature scaling;
* MM-DiT text/image token balancing.

## What is implemented

### 1. Text Anchoring for Flux-style joint attention

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

### 2. Dynamic Temperature Control for Flux-style RoPE attention

TIDE uses a step-aware temperature curve so attention sharpening is stronger in the early/global part of denoising and relaxes toward the late/detail part of denoising. For Flux-style models, this repository applies that idea as a RoPE temperature multiplier, matching the reference implementation strategy rather than inserting a new attention kernel for every backend.

Default curve:

```text
tau(t, f) = tau_max - (tau_max - tau_min) * t ** alpha(f)
alpha(f) = alpha_low + (alpha_high - alpha_low) * f
```

Defaults:

| Parameter              |        Default |
| ---------------------- | -------------: |
| `tau_max`              |          `1.0` |
| `alpha_low`            |          `0.6` |
| `alpha_high`           |          `0.2` |
| `temperature_strength` |          `1.0` |
| `frequency_mode`       | `official_raw` |

`frequency_mode=official_raw` uses raw RoPE frequencies, matching the released implementation behavior this port was written against. `paper_normalized` is exposed for comparison because the paper notation describes a normalized frequency variable.

### 3. Dynamic attention temperature for SDXL/UNet attention

The SDXL node applies the attention-temperature part of the method by scaling the attention query tensor before ComfyUI's optimized attention function:

```text
attention_logits = (Q * inv_tau) K^T / sqrt(d)
```

This is equivalent to applying the temperature factor to the attention logits:

```text
attention_logits = Q K^T / (tau * sqrt(d))
```

The SDXL curve uses a single exponent:

```text
tau(t) = tau_max - (tau_max - tau_min) * t ** alpha
```

The minimum temperature is derived from the YaRN-style extrapolation scale:

```text
scale = sqrt((width * height) / (base_width * base_height))
sqrt(1 / tau_min) = 0.1 * log(scale) + 1
```

The node blends from no-op to full temperature control with:

```text
applied_inv_tau = 1 + (inv_tau - 1) * temperature_strength
```

SDXL defaults:

| Parameter                    |         Default |
| ---------------------------- | --------------: |
| `base_width` / `base_height` | `1024` / `1024` |
| `temperature_strength`       |           `1.0` |
| `alpha`                      |           `0.6` |
| `tau_max`                    |           `1.0` |
| `apply_to`                   |          `both` |

## Implementation notes

### Flux path

1. The node clones and patches the incoming ComfyUI `MODEL`.
2. TIDE state is stored on the cloned model through ComfyUI model options.
3. `TIDEModelWrapper` injects current timestep metadata into `transformer_options["tide"]`.
4. `TIDEAttentionPatch` applies Text Anchoring and Dynamic Temperature Control through ComfyUI's `attn1_patch` hook.
5. `TIDEAttentionOverride` forces a compact PyTorch SDPA path only when an additive TIDE mask is active. This avoids attention backends that reject additive masks or try to materialize a dense high-resolution mask.
6. No global monkey-patching is used.
7. No sampler or scheduler rewrite is performed.

### SDXL path

1. The SDXL node clones and patches the incoming ComfyUI `MODEL`.
2. It installs an `optimized_attention_override` on the cloned model only.
3. The override detects SDXL/UNet `SpatialTransformer` attention by checking for `transformer_options["activations_shape"]`.
4. It avoids Flux-style paths by skipping attention calls that expose Flux-specific `block_type` or `img_slice` metadata.
5. It classifies attention as:

   * `self` when query-token count equals key-token count;
   * `cross` when query-token count differs from key-token count.
6. It applies query scaling only to the selected attention kind: `cross`, `self`, or `both`.
7. If another attention override already exists, the SDXL node delegates to it after applying its query scaling.

## Paper / reference-code / local-module mapping

| Paper component                                     | Reference implementation behavior                                   | This repository                                                                                                            |
| --------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Text-token influence decay                          | Additive text-token attention mask                                  | `tide_core.math.adaptive_text_bias`, `tide_core.patches.TIDEAttentionPatch`                                                |
| Text Anchoring, `beta = log(lambda)`                | `log(width / 1024) + log(height / 1024)` for FLUX-sized text prefix | Adaptive beta from node `width`, `height`, `base_width`, `base_height`; text-token count inferred from ComfyUI `img_slice` |
| YaRN temperature baseline                           | `get_mscale`, default temperature from extrapolation scale          | `tide_core.math.get_mscale`, `get_default_temperature`                                                                     |
| Dynamic Temperature Control                         | `dyheating()` / temperature-aware RoPE scaling                      | `tide_core.math.rope_temperature_scale`, applied to ComfyUI `pe`                                                           |
| Denoising-step-aware behavior                       | Update position embedding state from current timestep               | `TIDEModelWrapper` injects normalized timestep into `transformer_options`                                                  |
| FLUX attention integration                          | Modified Diffusers FLUX transformer/processor                       | ComfyUI `attn1_patch` plus optional `optimized_attention_override`                                                         |
| SDXL attention integration                          | Not part of the paper's main Flux/MM-DiT implementation             | `nodes_sdxl.py`, ComfyUI `optimized_attention_override`, query scaling for UNet attention                                  |
| Logarithmic FLUX scheduler shift                    | Scheduler/pipeline-level change                                     | Not implemented by this node                                                                                               |
| DyPE / NTK-by-parts / YaRN positional interpolation | Custom positional interpolation stack                               | Not fully implemented; this node implements the TIDE attention-side mechanisms only                                        |

## Installation

Clone this repository into ComfyUI's custom node directory:

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-TIDE
```

Restart ComfyUI.

No extra runtime dependency is required beyond the PyTorch/ComfyUI environment. `requirements.txt` lists `torch` for standalone tests.

## Usage

### Flux / Flux.2 usage

1. Load a FLUX-family model as usual.
2. Add **TIDE High-Resolution Extrapolation** after the model loader.
3. Connect the patched `model` output to your sampler.
4. Set `width` and `height` to the final generated image dimensions used by your latent node.
5. Keep `base_width=1024` and `base_height=1024` for FLUX-family models unless you know the model's native training target differs.

Recommended starting values:

| Setting                             |                  Value |
| ----------------------------------- | ---------------------: |
| `width` / `height`                  | final image dimensions |
| `base_width` / `base_height`        |        `1024` / `1024` |
| `text_anchor_strength`              |                  `1.0` |
| `temperature_strength`              |                  `1.0` |
| `alpha_low`                         |                  `0.6` |
| `alpha_high`                        |                  `0.2` |
| `tau_max`                           |                  `1.0` |
| `frequency_mode`                    |         `official_raw` |
| `force_pytorch_attention_with_mask` |                 `True` |

Ablation settings:

| Test                     | Settings                                               |
| ------------------------ | ------------------------------------------------------ |
| Text Anchoring only      | `text_anchor_strength=1.0`, `temperature_strength=0.0` |
| Dynamic Temperature only | `text_anchor_strength=0.0`, `temperature_strength=1.0` |
| Disabled                 | `text_anchor_strength=0.0`, `temperature_strength=0.0` |

### SDXL usage

1. Load an SDXL checkpoint as usual.
2. Add **TIDE SDXL High-Resolution Extrapolation** after the model loader.
3. Connect the patched `model` output to your sampler.
4. Set `width` and `height` to the final generated image dimensions used by your latent node.
5. Keep `base_width=1024` and `base_height=1024` for SDXL unless you intentionally want a different native-resolution reference.
6. Start with `apply_to=both`. If results are unstable, test `cross` and `self` separately.

Recommended starting values:

| Setting                      |                  Value |
| ---------------------------- | ---------------------: |
| `width` / `height`           | final image dimensions |
| `base_width` / `base_height` |        `1024` / `1024` |
| `temperature_strength`       |                  `1.0` |
| `alpha`                      |                  `0.6` |
| `tau_max`                    |                  `1.0` |
| `apply_to`                   |                 `both` |

SDXL ablation settings:

| Test                 | Settings                                     |
| -------------------- | -------------------------------------------- |
| Cross-attention only | `apply_to=cross`, `temperature_strength=1.0` |
| Self-attention only  | `apply_to=self`, `temperature_strength=1.0`  |
| Both                 | `apply_to=both`, `temperature_strength=1.0`  |
| Disabled             | `temperature_strength=0.0`                   |

## Node inputs

### TIDE High-Resolution Extrapolation

#### Required

| Input                  | Description                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `model`                | ComfyUI `MODEL` object to patch.                                                                        |
| `width`, `height`      | Final target generation dimensions in pixels. Must match the latent/image size used by the workflow.    |
| `text_anchor_strength` | Multiplier on adaptive beta. `1.0` follows the paper/reference behavior. `0.0` disables Text Anchoring. |
| `temperature_strength` | Multiplier on Dynamic Temperature Control. `1.0` follows the reference curve. `0.0` disables DTC.       |

#### Optional

| Input                               |        Default | Description                                                                  |
| ----------------------------------- | -------------: | ---------------------------------------------------------------------------- |
| `base_width`, `base_height`         | `1024`, `1024` | Native/training resolution used for adaptive scaling.                        |
| `alpha_low`, `alpha_high`           |   `0.6`, `0.2` | DTC exponents for low/high RoPE frequency behavior.                          |
| `tau_max`                           |          `1.0` | Maximum temperature reached near the end of denoising.                       |
| `frequency_mode`                    | `official_raw` | `official_raw` or `paper_normalized`.                                        |
| `apply_to_double_blocks`            |         `True` | Apply patch to FLUX double-stream blocks.                                    |
| `apply_to_single_blocks`            |         `True` | Apply patch to FLUX single-stream blocks.                                    |
| `apply_to_native_or_smaller`        |        `False` | Allow patching even when target token count is not above base token count.   |
| `force_pytorch_attention_with_mask` |         `True` | Use internal PyTorch SDPA only when TIDE's additive mask is active.          |
| `preserve_existing_wrapper`         |         `True` | Delegate to an existing ComfyUI model wrapper after injecting TIDE metadata. |
| `debug`                             |        `False` | Log skipped DTC shape mismatches and exceptions.                             |

### TIDE SDXL High-Resolution Extrapolation

| Input                       |        Default | Description                                                                                          |
| --------------------------- | -------------: | ---------------------------------------------------------------------------------------------------- |
| `model`                     |       required | ComfyUI `MODEL` object to patch.                                                                     |
| `width`, `height`           | `1536`, `1536` | Final target generation dimensions in pixels. Must match the latent/image size used by the workflow. |
| `temperature_strength`      |          `1.0` | Strength of SDXL attention-temperature scaling. `0.0` disables the patch.                            |
| `base_width`, `base_height` | `1024`, `1024` | Native/training resolution used for adaptive scaling.                                                |
| `alpha`                     |          `0.6` | Single exponent for the SDXL step-aware temperature curve.                                           |
| `tau_max`                   |          `1.0` | Maximum temperature reached near the end of denoising.                                               |
| `apply_to`                  |         `both` | `cross`, `self`, or `both`. Controls which SDXL attention calls are patched.                         |

## Repository structure

```text
ComfyUI-TIDE/
├── __init__.py
├── nodes.py
├── nodes_sdxl.py
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

* adaptive Text Anchoring beta computation;
* YaRN/default temperature formula;
* RoPE temperature scale shape and timestep progression;
* additive attention-mask creation;
* masked SDPA override behavior.

The tests do not validate visual quality or live ComfyUI execution.

For the SDXL module, a basic syntax check can be run with:

```bash
python -m py_compile nodes_sdxl.py
```

## Paper vs implementation differences

### SDXL support is an adaptation, not full TIDE

The full TIDE method is designed around DiT/MM-DiT attention where text tokens and image tokens are present in the same attention sequence. That makes Text Anchoring meaningful because a positive bias on text-key logits changes the balance between text keys and image keys.

SDXL uses a UNet architecture with separate attention patterns. In cross-attention, the keys and values are text-only, so adding the same bias to all text logits would not change the softmax result. In self-attention, the keys are image tokens and there are no text keys to anchor. For that reason, the SDXL node only applies Dynamic Temperature Control as attention-logit sharpening.

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

### Flux path

* The model uses Flux-style joint attention with text tokens before image tokens.
* `extra_options["img_slice"]` identifies the split between text and image tokens.
* The ComfyUI attention patch receives a RoPE matrix through `pe`.
* The node `width` and `height` match the actual generated dimensions.
* The timestep passed through the model wrapper is normalized or sigma-like in `[0, 1]`; values outside the interval are clamped.
* FLUX-family image token granularity is 16 pixels per transformer token.

### SDXL path

* The model uses ComfyUI's UNet `SpatialTransformer` attention path.
* `optimized_attention_override` is honored by the active ComfyUI attention backend.
* `transformer_options["activations_shape"]` is present for the attention calls that should be patched.
* The node `width` and `height` match the actual generated dimensions.
* SDXL's native-resolution reference is treated as `1024x1024` unless overridden.
* Sigma metadata is available through `transformer_options`; if not, the SDXL patch falls back to the noisy/start side of the curve.

## Limitations

* Not an official TIDE release.
* Not a full reproduction of the paper's complete experimental pipeline.
* Does not include paper benchmark scripts, datasets, generated result images, or metric evaluation.
* Does not modify the sampler's high-resolution time-shift schedule.
* Does not fully implement NTK-by-parts, YaRN positional interpolation, or DyPE positional interpolation.
* Flux support is tied to ComfyUI's Flux-style attention patch contract.
* SDXL support is an experimental attention-temperature adaptation, not full Text Anchoring.
* SDXL visual quality needs live workflow testing across checkpoints, resolutions, samplers, and attention backends.
* Very large resolutions still require sufficient VRAM for the selected model, sampler, attention path, latent size, and VAE path.

## License

This repository is released under the MIT License; see [`LICENSE`](LICENSE).

ComfyUI is GPL-3.0 licensed. This custom node is distributed as a separate plugin, but it imports and runs inside ComfyUI. Review license compatibility before redistributing this node as part of a larger bundled package.
