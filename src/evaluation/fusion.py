"""
fusion.py — Late fusion de detecções CC + MLO por caso clínico.

Análise secundária do artigo: cada modelo detecta independentemente
nas vistas CC e MLO. Os resultados são fundidos por caso clínico.

Critério de fusão (documentado no artigo):
  - Detectado em CC e MLO com IoU projetado > iou_threshold → CONFIRMADO
  - Detectado em apenas uma vista:
      - Se BIRADS >= birads_threshold → possível lesão unilateral (mantém)
      - Caso contrário → falso positivo candidato (descartado)
  - MIAS (vista='unknown') → excluído da análise de fusão

Métricas calculadas em nível de CASO (patient_id + lateralidade),
não de imagem individual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger("fusion")


@dataclass
class Detection:
    """Detecção de um modelo em uma imagem."""
    case_id:    str
    patient_id: str
    laterality: str       # 'LEFT' | 'RIGHT'
    view:       str       # 'CC' | 'MLO'
    bbox:       Tuple[int, int, int, int]   # (x1, y1, x2, y2) em 640x640
    confidence: float
    birads:     int = 0


@dataclass
class CaseDetections:
    """Detecções agrupadas por caso clínico."""
    patient_id:  str
    laterality:  str
    cc_dets:     List[Detection] = field(default_factory=list)
    mlo_dets:    List[Detection] = field(default_factory=list)
    birads:      int = 0


def compute_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """Calcula IoU entre dois bounding boxes (x1, y1, x2, y2)."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def group_by_case(detections: List[Detection]) -> Dict[Tuple[str, str], CaseDetections]:
    """
    Agrupa detecções por (patient_id, laterality).

    Detecções de casos MIAS (view='unknown') são excluídas.
    """
    groups: Dict[Tuple[str, str], CaseDetections] = {}

    for det in detections:
        if det.view == "unknown":
            continue   # MIAS excluído da análise de fusão

        key = (det.patient_id, det.laterality)
        if key not in groups:
            groups[key] = CaseDetections(
                patient_id=det.patient_id,
                laterality=det.laterality,
                birads=det.birads,
            )

        if det.view == "CC":
            groups[key].cc_dets.append(det)
        elif det.view == "MLO":
            groups[key].mlo_dets.append(det)

        groups[key].birads = max(groups[key].birads, det.birads)

    return groups


def fuse_case(
    case: CaseDetections,
    iou_threshold: float = 0.3,
    birads_threshold: int = 4,
) -> Dict:
    """
    Aplica a lógica de late fusion para um caso clínico.

    Returns:
        Dicionário com resultado da fusão:
          - confirmed: bool — ao menos uma detecção confirmada
          - n_confirmed: int — número de lesões confirmadas
          - n_unilateral: int — lesões detectadas em apenas uma vista
          - n_fp_candidate: int — falsos positivos candidatos descartados
          - fusion_details: list — detalhes por par de detecções
    """
    cc_dets  = case.cc_dets
    mlo_dets = case.mlo_dets
    confirmed, unilateral, fp_candidates = 0, 0, 0
    details = []

    if not cc_dets and not mlo_dets:
        return {
            "confirmed": False, "n_confirmed": 0,
            "n_unilateral": 0, "n_fp_candidate": 0,
            "fusion_details": [],
        }

    # Caso com apenas uma vista com detecção
    if not cc_dets or not mlo_dets:
        single_dets = cc_dets or mlo_dets
        view_present = "CC" if cc_dets else "MLO"
        for det in single_dets:
            if case.birads >= birads_threshold:
                unilateral += 1
                details.append({
                    "type": "unilateral",
                    "view": view_present,
                    "confidence": det.confidence,
                    "birads": case.birads,
                })
            else:
                fp_candidates += 1
                details.append({
                    "type": "fp_candidate",
                    "view": view_present,
                    "confidence": det.confidence,
                    "birads": case.birads,
                })
        return {
            "confirmed": unilateral > 0,
            "n_confirmed": 0, "n_unilateral": unilateral,
            "n_fp_candidate": fp_candidates,
            "fusion_details": details,
        }

    # Ambas as vistas com detecções — calcular IoU projetado
    matched_cc  = set()
    matched_mlo = set()

    for i, cc_det in enumerate(cc_dets):
        best_iou   = 0.0
        best_j     = -1
        for j, mlo_det in enumerate(mlo_dets):
            if j in matched_mlo:
                continue
            iou = compute_iou(cc_det.bbox, mlo_det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            confirmed += 1
            matched_cc.add(i)
            matched_mlo.add(best_j)
            details.append({
                "type": "confirmed",
                "iou": round(best_iou, 4),
                "conf_cc": cc_dets[i].confidence,
                "conf_mlo": mlo_dets[best_j].confidence,
            })

    # Detecções não pareadas
    for i, cc_det in enumerate(cc_dets):
        if i not in matched_cc:
            if case.birads >= birads_threshold:
                unilateral += 1
                details.append({"type": "unilateral", "view": "CC",
                                 "confidence": cc_det.confidence})
            else:
                fp_candidates += 1

    for j, mlo_det in enumerate(mlo_dets):
        if j not in matched_mlo:
            if case.birads >= birads_threshold:
                unilateral += 1
                details.append({"type": "unilateral", "view": "MLO",
                                 "confidence": mlo_det.confidence})
            else:
                fp_candidates += 1

    return {
        "confirmed": confirmed > 0 or unilateral > 0,
        "n_confirmed": confirmed, "n_unilateral": unilateral,
        "n_fp_candidate": fp_candidates,
        "fusion_details": details,
    }


def late_fusion(
    detections: List[Detection],
    iou_threshold: float = 0.3,
    birads_threshold: int = 4,
) -> Dict[Tuple[str, str], Dict]:
    """
    Aplica late fusion em todos os casos de uma lista de detecções.

    Args:
        detections:       Lista de Detection (todos os casos).
        iou_threshold:    IoU mínimo para confirmar detecção entre CC e MLO.
        birads_threshold: BI-RADS mínimo para tratar detecção unilateral
                          como lesão (não falso positivo).

    Returns:
        Dicionário {(patient_id, laterality): resultado_fusao}.
    """
    groups = group_by_case(detections)
    results: Dict[Tuple[str, str], Dict] = {}

    mias_excluded = sum(1 for d in detections if d.view == "unknown")
    if mias_excluded > 0:
        log.info(
            f"[fusion] {mias_excluded} detecções MIAS (view=unknown) "
            "excluídas da análise de fusão."
        )

    for key, case in groups.items():
        results[key] = fuse_case(case, iou_threshold, birads_threshold)

    total_confirmed = sum(r["n_confirmed"] for r in results.values())
    total_unilateral = sum(r["n_unilateral"] for r in results.values())
    log.info(
        f"[fusion] {len(results)} casos avaliados | "
        f"confirmados={total_confirmed} | "
        f"unilaterais={total_unilateral}"
    )

    return results
