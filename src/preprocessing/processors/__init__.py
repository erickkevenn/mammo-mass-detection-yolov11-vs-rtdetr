"""Pacote processors (plano v2)."""

from .clahe import apply_clahe
from .crop import crop_breast_region
from .normalize import normalize_bit_depth
from .resize import resize_and_scale_annotations, save_processed_image

__all__ = [
    "normalize_bit_depth",
    "crop_breast_region",
    "apply_clahe",
    "resize_and_scale_annotations",
    "save_processed_image",
]
