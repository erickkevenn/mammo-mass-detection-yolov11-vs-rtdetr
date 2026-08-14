"""
resize_norm.py — Redimensionamento para 640x640 e escala proporcional de bboxes.

Ordem no pipeline: etapa 4 (após CLAHE, antes da exportação).

Salva a imagem processada como PNG (para YOLO) e registra metadados
por imagem em JSON para rastreabilidade.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.preprocessing.loaders import MammographyCase, NoduleAnnotation
from src.utils.logger import get_logger

log = get_logger("resize_norm")


def resize_and_scale_annotations(
    image: np.ndarray,
    case: MammographyCase,
    target_size: int = 640,
    pad_value: int = 114,
) -> Tuple[np.ndarray, MammographyCase]:
    """
    Letterbox: escala uniforme para caber em target×target e padding cinza (plano v3).

    Preserva aspect ratio das lesões; bboxes transformadas com o mesmo fator + offset.
    """
    orig_h, orig_w = image.shape[:2]
    if orig_w <= 0 or orig_h <= 0:
        return image, case

    r = min(target_size / orig_h, target_size / orig_w)
    nw = max(1, int(round(orig_w * r)))
    nh = max(1, int(round(orig_h * r)))

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = target_size - nw
    pad_h = target_size - nh
    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )

    scaled_annotations: List[NoduleAnnotation] = []
    for ann in case.annotations:
        x1, y1, x2, y2 = ann.bbox_abs
        nx1 = int(round(x1 * r + left))
        ny1 = int(round(y1 * r + top))
        nx2 = int(round(x2 * r + left))
        ny2 = int(round(y2 * r + top))
        nx1 = max(0, min(nx1, target_size - 1))
        ny1 = max(0, min(ny1, target_size - 1))
        nx2 = max(0, min(nx2, target_size))
        ny2 = max(0, min(ny2, target_size))
        if nx2 > nx1 and ny2 > ny1:
            ann.bbox_abs = (nx1, ny1, nx2, ny2)
            scaled_annotations.append(ann)
        else:
            log.warning(
                f"[resize] {case.case_id}: bbox degenerou após letterbox: "
                f"({x1},{y1},{x2},{y2}) → ({nx1},{ny1},{nx2},{ny2}). Descartada."
            )

    case.annotations = scaled_annotations
    log.debug(
        f"[resize] {case.case_id}: ({orig_w}x{orig_h}) → letterbox {target_size} "
        f"| r={r:.4f} pad=({left},{top}) | anns={len(scaled_annotations)}"
    )
    return padded, case


def save_processed_image(
    image: np.ndarray,
    case: MammographyCase,
    out_dir: Path,
    target_size: int = 640,
) -> Path:
    """
    Salva imagem processada como PNG e metadados como JSON.

    Args:
        image:       Imagem uint8 640×640.
        case:        MammographyCase com bboxes finais.
        out_dir:     Diretório de saída.
        target_size: Tamanho da imagem (para registro nos metadados).

    Returns:
        Caminho do PNG salvo.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{case.case_id}.png"
    cv2.imwrite(str(png_path), image)

    meta = {
        "case_id":         case.case_id,
        "patient_id":      case.patient_id,
        "dataset":         case.dataset,
        "view":            case.view,
        "laterality":      case.laterality,
        "birads":          case.birads,
        "pixel_spacing_mm": case.pixel_spacing_mm,
        "img_size":        target_size,
        "annotations": [
            {
                "bbox_abs":  list(ann.bbox_abs),
                "size_mm":   round(ann.size_mm, 3),
                "size_bin":  ann.size_bin,
                "malignant": ann.malignant,
                "birads":    ann.birads,
            }
            for ann in case.annotations
        ],
    }

    json_path = out_dir / f"{case.case_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return png_path
