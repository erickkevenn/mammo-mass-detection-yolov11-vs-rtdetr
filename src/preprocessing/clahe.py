"""
clahe.py — Aplicação de CLAHE padronizada e idêntica nos três datasets.

Ordem no pipeline: etapa 3 (após normalize_bit_depth + crop, antes de resize).

CLAHE (Contrast Limited Adaptive Histogram Equalization) é aplicado
identicamente nos três datasets para garantir comparabilidade.
Parâmetros lidos de config.yaml e registrados em log por imagem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("clahe")


def apply_clahe(
    image: np.ndarray,
    tile_grid_size: Tuple[int, int] = (8, 8),
    clip_limit: float = 2.0,
    case_id: str = "",
    save_dir: Optional[Path] = None,
    save_visual: bool = True,
) -> np.ndarray:
    """
    Aplica CLAHE a uma imagem uint8 em escala de cinza.

    Args:
        image:          Array uint8 (H, W) — deve estar em escala de cinza.
        tile_grid_size: Tamanho do tile para o histograma local.
        clip_limit:     Limite de contraste para evitar amplificação de ruído.
        case_id:        Identificador para log e nome do arquivo de saída.
        save_dir:       Diretório para salvar exemplo visual (None = não salvar).
        save_visual:    Se True e save_dir fornecido, salva before/after.

    Returns:
        Imagem uint8 com CLAHE aplicado.

    Raises:
        ValueError: Se a imagem não for uint8 ou tiver mais de 1 canal.
    """
    if image.dtype != np.uint8:
        raise ValueError(
            f"apply_clahe espera imagem uint8, recebeu {image.dtype}. "
            "Chame normalize_bit_depth antes."
        )
    if image.ndim != 2:
        raise ValueError(
            f"apply_clahe espera imagem 2D (grayscale), recebeu shape {image.shape}."
        )

    clahe_obj = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tuple(tile_grid_size),
    )
    result = clahe_obj.apply(image)

    log.debug(
        f"[clahe] {case_id}: tile_grid={tile_grid_size}, "
        f"clip_limit={clip_limit}"
    )

    if save_visual and save_dir is not None and case_id:
        _save_clahe_visual(image, result, case_id, save_dir)

    return result


def _save_clahe_visual(
    before: np.ndarray,
    after: np.ndarray,
    case_id: str,
    save_dir: Path,
) -> None:
    """Salva comparativo before/after do CLAHE como JPEG."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    before_bgr = cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
    after_bgr  = cv2.cvtColor(after,  cv2.COLOR_GRAY2BGR)

    cv2.putText(before_bgr, "BEFORE CLAHE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(after_bgr,  "AFTER CLAHE",  (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    combined = np.hstack([before_bgr, after_bgr])
    out_path = save_dir / f"{case_id}_clahe.jpg"

    try:
        cv2.imwrite(str(out_path), combined)
    except Exception as e:
        log.warning(f"[clahe] Erro ao salvar visual {out_path}: {e}")
