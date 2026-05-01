# Example workflow wiring

Minimal ComfyUI graph placement:

1. Load a FLUX-family model as usual.
2. Insert **TIDE High-Resolution Extrapolation** immediately after the model loader and before the sampler.
3. Set `width` and `height` to the same final latent/image dimensions used by your empty latent node.
4. Use a high-resolution latent such as 2048x2048, 4096x2048, or another multiple of 16.

The node modifies only the MODEL object. It does not create latents, change the sampler, or decode images.
