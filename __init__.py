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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
