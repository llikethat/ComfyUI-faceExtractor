"""
DFL Face Extractor for ComfyUI
Automatically extracts faces from video/images matching a reference face.
Designed for DeepFaceLab de-aging and face swap workflows.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
