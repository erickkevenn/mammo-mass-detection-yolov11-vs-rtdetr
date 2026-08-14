"""
annotations.py — Normalização de anotações e cálculo de tamanho real.

Responsabilidades:
  1. Calcular size_mm para cada NoduleAnnotation a partir de pixel_spacing
  2. Validar e clipar bboxes nos limites da imagem
  3. Normalizar bboxes para coordenadas absolutas inteiras

Deve ser chamado ANTES do crop (as bboxes serão recalculadas após crop).
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.preprocessing.bbox_ops import clip_bbox, compute_size_mm, is_valid_bbox
from src.preprocessing.loaders import MammographyCase, NoduleAnnotation
from src.utils.logger import get_logger

log = get_logger("annotations")


def normalize_bit_depth(image: np.ndarray, target_bits: int = 8) -> np.ndarray:
    """
    Normaliza a profundidade de bits de uma imagem para uint8.

    Essencial para uniformizar INbreast (12-14 bits), CBIS-DDSM (variável)
    e MIAS (8 bits) antes do CLAHE e demais etapas de processamento.

    Args:
        image:       Array numpy (qualquer dtype inteiro/float).
        target_bits: Bits do output. Apenas 8 é suportado atualmente.

    Returns:
        Array uint8 com pixels em [0, 255].
    """
    if target_bits != 8:
        raise ValueError("Apenas target_bits=8 é suportado.")

    if image.dtype == np.uint8:
        return image

    max_val = image.max()
    if max_val == 0:
        return np.zeros_like(image, dtype=np.uint8)

    return ((image.astype(np.float32) / float(max_val)) * 255.0).astype(np.uint8)


def normalize_to_uint8_percentile(
    image: np.ndarray,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """
    Escalona intensidades por percentis para uint8 (plano v3 — mamografia multi-bit).
    """
    if image.size == 0:
        return image.astype(np.uint8)
    arr = image.astype(np.float64)
    lo, hi = np.percentile(arr, [p_lo, p_hi])
    if hi <= lo + 1e-9:
        return normalize_bit_depth(image.astype(np.uint16) if arr.max() > 255 else image, target_bits=8)
    out = (arr - lo) / (hi - lo + 1e-9)
    out = np.clip(out, 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def compute_annotations(
    case: MammographyCase,
    img_w: int,
    img_h: int,
) -> MammographyCase:
    """
    Calcula size_mm e valida bboxes para todas as anotações de um caso.

    Args:
        case:  MammographyCase com pixel_spacing_mm preenchido.
        img_w: Largura da imagem original em pixels.
        img_h: Altura da imagem original em pixels.

    Returns:
        MammographyCase com anotações atualizadas (size_mm preenchido,
        bboxes clipadas, anotações inválidas removidas).
    """
    valid_annotations: List[NoduleAnnotation] = []

    for ann in case.annotations:
        bbox = clip_bbox(ann.bbox_abs, img_w, img_h)

        if not is_valid_bbox(bbox):
            log.warning(
                f"[annotations] {case.case_id}: bbox inválida após clip "
                f"{ann.bbox_abs} → {bbox}. Removida."
            )
            continue

        size_mm = compute_size_mm(bbox, case.pixel_spacing_mm)
        ann.bbox_abs = bbox
        ann.size_mm = size_mm
        valid_annotations.append(ann)

    if len(valid_annotations) < len(case.annotations):
        removed = len(case.annotations) - len(valid_annotations)
        log.warning(
            f"[annotations] {case.case_id}: {removed} anotação(ões) removida(s) "
            f"por bbox inválida."
        )

    case.annotations = valid_annotations
    return case
