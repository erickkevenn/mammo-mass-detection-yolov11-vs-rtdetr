"""
crop.py — Crop da região mamária por limiarização.

Pipeline desta etapa (etapa 2 do pipeline geral):
  1. Limiarização binária para isolar silhueta mamária
  2. Encontrar maior contorno (ignora artefatos externos)
  3. Expandir por margin_px configurável
  4. Recalcular todas as bboxes para novo sistema de coordenadas
  5. Descartar anotações fora do crop
  6. Salvar exemplos visuais before/after

Deve ser chamado APÓS normalize_bit_depth e ANTES de apply_clahe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.preprocessing.loaders import MammographyCase, NoduleAnnotation
from src.utils.logger import get_logger

log = get_logger("crop")


@dataclass
class CropInfo:
    """Informação sobre o crop realizado — para rastreabilidade."""
    x0: int
    y0: int
    x1: int
    y1: int
    orig_w: int
    orig_h: int
    crop_w: int
    crop_h: int


def _find_breast_bbox(
    image: np.ndarray,
    threshold: int,
    margin_px: int,
) -> Tuple[int, int, int, int]:
    """
    Encontra a bounding box da silhueta mamária via limiarização.

    Args:
        image:      Imagem uint8 em escala de cinza.
        threshold:  Limiar de pixel (0-255) para binarização.
        margin_px:  Margem adicionada ao redor da bbox.

    Returns:
        (x0, y0, x1, y1) em pixels absolutos (clampado nos limites).
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Morfologia leve para fechar buracos pequenos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _bbox_from_otsu(image, margin_px)

    largest = max(contours, key=cv2.contourArea)
    img_h, img_w = image.shape[:2]
    area_ratio = cv2.contourArea(largest) / float(max(1, img_w * img_h))
    if area_ratio < 0.30:
        log.warning(
            f"[crop] Maior contorno cobre apenas {area_ratio:.1%} da imagem — usando Otsu."
        )
        return _bbox_from_otsu(image, margin_px)

    x, y, w, hc = cv2.boundingRect(largest)

    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(img_w, x + w + margin_px)
    y1 = min(img_h, y + hc + margin_px)

    return (x0, y0, x1, y1)


def _bbox_from_otsu(image: np.ndarray, margin_px: int) -> Tuple[int, int, int, int]:
    """Fallback: binarização Otsu + maior contorno."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = image.shape[:2]
    if not contours:
        return (0, 0, img_w, img_h)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, hc = cv2.boundingRect(largest)
    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(img_w, x + w + margin_px)
    y1 = min(img_h, y + hc + margin_px)
    return (x0, y0, x1, y1)


def _recalc_annotations(
    annotations: List[NoduleAnnotation],
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
) -> Tuple[List[NoduleAnnotation], List[NoduleAnnotation]]:
    """
    Recalcula bboxes para o sistema de coordenadas pós-crop.

    Returns:
        (anotações válidas, anotações descartadas)
    """
    valid, discarded = [], []
    for ann in annotations:
        ax1, ay1, ax2, ay2 = ann.bbox_abs
        # Subtrair offset do crop
        nx1 = ax1 - x0
        ny1 = ay1 - y0
        nx2 = ax2 - x0
        ny2 = ay2 - y0
        # Clipar nos limites do crop
        nx1 = max(0, min(nx1, crop_w))
        ny1 = max(0, min(ny1, crop_h))
        nx2 = max(0, min(nx2, crop_w))
        ny2 = max(0, min(ny2, crop_h))
        # Verificar se ainda tem área válida
        if nx2 > nx1 and ny2 > ny1:
            ann.bbox_abs = (nx1, ny1, nx2, ny2)
            valid.append(ann)
        else:
            discarded.append(ann)
    return valid, discarded


def _save_visual(
    before: np.ndarray,
    after: np.ndarray,
    case_id: str,
    save_dir: Path,
    annotations_before: Optional[List[NoduleAnnotation]] = None,
    annotations_after: Optional[List[NoduleAnnotation]] = None,
) -> None:
    """Salva imagem comparativa before/after com bboxes desenhadas."""
    save_dir.mkdir(parents=True, exist_ok=True)

    def draw_boxes(img_gray, anns):
        rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        for ann in (anns or []):
            x1, y1, x2, y2 = ann.bbox_abs
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return rgb

    before_rgb = draw_boxes(before, annotations_before)
    after_rgb  = draw_boxes(after, annotations_after)

    # Redimensionar before para mesma altura que after para concatenar
    h_after = after_rgb.shape[0]
    ratio = h_after / before_rgb.shape[0]
    w_before_resized = int(before_rgb.shape[1] * ratio)
    before_resized = cv2.resize(before_rgb, (w_before_resized, h_after))

    combined = np.hstack([before_resized, after_rgb])

    # Adicionar labels
    cv2.putText(combined, "BEFORE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(combined, "AFTER CROP", (w_before_resized + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out_path = save_dir / f"{case_id}_crop.jpg"
    cv2.imwrite(str(out_path), combined)


def crop_breast_region(
    image: np.ndarray,
    case: MammographyCase,
    threshold: int = 10,
    margin_px: int = 10,
    save_dir: Optional[Path] = None,
    save_visual: bool = True,
) -> Tuple[np.ndarray, MammographyCase, CropInfo]:
    """
    Recorta a região mamária e recalcula as anotações.

    Args:
        image:       Imagem uint8 em escala de cinza (pós normalize_bit_depth).
        case:        MammographyCase com anotações em coordenadas originais.
        threshold:   Limiar de pixel para binarização da silhoueta.
        margin_px:   Margem em pixels ao redor da bbox da mama.
        save_dir:    Diretório para salvar exemplos visuais (None = não salvar).
        save_visual: Se True e save_dir fornecido, salva comparativo before/after.

    Returns:
        (imagem_cropada, case_atualizado, crop_info)
    """
    orig_h, orig_w = image.shape[:2]
    x0, y0, x1, y1 = _find_breast_bbox(image, threshold, margin_px)

    cropped = image[y0:y1, x0:x1]
    crop_w = x1 - x0
    crop_h = y1 - y0

    annotations_before = list(case.annotations)  # cópia para o visual
    valid_anns, discarded = _recalc_annotations(
        case.annotations, x0, y0, crop_w, crop_h
    )

    if discarded:
        log.warning(
            f"[crop] {case.case_id}: {len(discarded)} anotação(ões) descartada(s) "
            f"por ficarem fora do crop. Bboxes: "
            f"{[a.bbox_abs for a in discarded]}"
        )

    case.annotations = valid_anns

    crop_info = CropInfo(
        x0=x0, y0=y0, x1=x1, y1=y1,
        orig_w=orig_w, orig_h=orig_h,
        crop_w=crop_w, crop_h=crop_h,
    )

    if save_visual and save_dir is not None:
        try:
            _save_visual(
                before=image,
                after=cropped,
                case_id=case.case_id,
                save_dir=save_dir,
                annotations_before=annotations_before,
                annotations_after=valid_anns,
            )
        except Exception as e:
            log.warning(f"[crop] Erro ao salvar visual para {case.case_id}: {e}")

    log.debug(
        f"[crop] {case.case_id}: ({orig_w}x{orig_h}) → "
        f"({crop_w}x{crop_h}) | offset=({x0},{y0}) | "
        f"anns válidas={len(valid_anns)}"
    )

    return cropped, case, crop_info
