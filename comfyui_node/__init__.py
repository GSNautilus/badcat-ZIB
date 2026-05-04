"""Z-Image Style Adapter — ComfyUI custom nodes.

Install: copy this folder to ComfyUI/custom_nodes/zimage_style_adapter/
Place the adapter weights at ComfyUI/models/style_adapters/phase2b_ssl.pt
(or pass an absolute path in the loader node).
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
