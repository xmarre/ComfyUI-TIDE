try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ModuleNotFoundError as exc:
    if exc.name not in {f"{__package__}.nodes", "nodes"}:
        raise
    # Allows standalone pytest execution from a directory whose name is not a
    # valid Python package identifier. ComfyUI imports this file as a package,
    # so the relative import path above remains the normal runtime path.
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as exc:
    if "attempted relative import with no known parent package" not in str(exc):
        raise
    # Allows standalone pytest execution from a directory whose name is not a
    # valid Python package identifier. ComfyUI imports this file as a package,
    # so the relative import path above remains the normal runtime path.
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# SDXL/UNet support is intentionally kept in a separate module because the
# faithful TIDE text-anchor path is MM-DiT-specific. This registers the SDXL
# adaptation without changing existing FLUX nodes.
try:
    from .nodes_sdxl import NODE_CLASS_MAPPINGS as _SDXL_NODE_CLASS_MAPPINGS
    from .nodes_sdxl import NODE_DISPLAY_NAME_MAPPINGS as _SDXL_NODE_DISPLAY_NAME_MAPPINGS
except ModuleNotFoundError as exc:
    if exc.name not in {f"{__package__}.nodes_sdxl", "nodes_sdxl"}:
        raise
    from nodes_sdxl import NODE_CLASS_MAPPINGS as _SDXL_NODE_CLASS_MAPPINGS
    from nodes_sdxl import NODE_DISPLAY_NAME_MAPPINGS as _SDXL_NODE_DISPLAY_NAME_MAPPINGS
except ImportError as exc:
    if "attempted relative import with no known parent package" not in str(exc):
        raise
    from nodes_sdxl import NODE_CLASS_MAPPINGS as _SDXL_NODE_CLASS_MAPPINGS
    from nodes_sdxl import NODE_DISPLAY_NAME_MAPPINGS as _SDXL_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS.update(_SDXL_NODE_CLASS_MAPPINGS)
try:
    NODE_DISPLAY_NAME_MAPPINGS
except NameError:
    NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(_SDXL_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
