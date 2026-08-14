"""
bbox_ops.py — Geometria de bounding boxes (sem dependência de MammographyCase).

Usado por audit_datasets, mammo_loaders e annotations para evitar import circular.
"""

from __future__ import annotations

from typing import Tuple


def compute_size_mm(bbox: Tuple[int, int, int, int], pixel_spacing_mm: float) -> float:
    """Maior lado em px × pixel spacing (mm/px), conforme plano v3."""
    x1, y1, x2, y2 = bbox
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return max(w, h) * pixel_spacing_mm


def clip_bbox(
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """Clipa a bounding box nos limites da imagem (x2/y2 tratados como limites superiores)."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return (x1, y1, x2, y2)


def is_valid_bbox(bbox: Tuple[int, int, int, int], min_side_px: int = 2) -> bool:
    """Extensão mínima em x e y (plano v3: > 1 px por lado → default min_side_px=2)."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False
    return (x2 - x1) >= min_side_px and (y2 - y1) >= min_side_px
