"""CDD-CESM — apenas imagens LE + CSV (massas)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image

from src.preprocessing.bbox_ops import clip_bbox, compute_size_mm, is_valid_bbox
from src.preprocessing.mammo_loaders import MammographyCase, NoduleAnnotation, log

CDD_PIXEL_SPACING_FALLBACK = 0.094


def _polygon_to_bbox(xs: List[float], ys: List[float]) -> Optional[Tuple[int, int, int, int]]:
    if not xs or not ys:
        return None
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _extract_patient_id(fname: str, row: pd.Series) -> str:
    for col in ("patient_id", "Patient ID", "patient", "PatientID"):
        if col in row.index and pd.notna(row.get(col)) and str(row.get(col)).strip():
            return str(row.get(col)).strip()
    m = re.match(r"(P\d+)", fname, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return Path(fname).stem


def _pixel_spacing(row: pd.Series) -> float:
    for col in (
        "pixel_spacing_mm",
        "PixelSpacing",
        "pixel_spacing",
        "spacing_mm",
        "res_mm",
        "resolution_mm",
    ):
        if col in row.index and pd.notna(row.get(col)):
            try:
                v = float(row.get(col))
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    for key in ("file_attributes", "region_attributes"):
        if key not in row.index:
            continue
        raw = row.get(key)
        if pd.isna(raw):
            continue
        try:
            d = json.loads(str(raw)) if isinstance(raw, str) else dict(raw)
            for k in ("pixel_spacing_mm", "spacing", "PixelSpacing"):
                if k in d:
                    return float(str(d[k]).strip("[]").split(",")[0])
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            pass
    return CDD_PIXEL_SPACING_FALLBACK


def _parse_birads(row: pd.Series) -> int:
    for col in ("birads", "BI-RADS", "BI_RADS", "BIRADS", "bi_rads"):
        if col in row.index and pd.notna(row.get(col)):
            s = str(row.get(col)).strip()
            for d in ("1", "2", "3", "4", "5"):
                if d in s:
                    return int(d)
    raw = row.get("region_attributes", "{}")
    try:
        d = json.loads(str(raw))
        for k in ("birads", "BI-RADS", "assessment"):
            if k in d:
                s = str(d[k])
                for digit in ("1", "2", "3", "4", "5"):
                    if digit in s:
                        return int(digit)
    except (json.JSONDecodeError, TypeError):
        pass
    return 0


def load_cdd_cesm(raw_dir: str | Path) -> List[MammographyCase]:
    """
    Carrega casos CDD-CESM a partir do CSV VIA e da pasta LE exclusivamente.

    Um :class:`MammographyCase` por ficheiro de imagem; vários polígonos na mesma
    imagem viram várias :class:`NoduleAnnotation` (após clip e filtros ]0,150] mm).
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / "Radiology_hand_drawn_segmentations_v2.csv"
    img_dir = raw_dir / "Low energy images of CDD-CESM"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV CDD-CESM não encontrado: {csv_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Pasta LE não encontrada: {img_dir}")

    le_files = {p.name.lower(): p for p in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))}

    raw_df = pd.read_csv(csv_path, header=0)
    raw_df.columns = [c.lstrip("#").strip() for c in raw_df.columns]
    if "filename" not in raw_df.columns:
        raise ValueError(f"Coluna filename ausente: {raw_df.columns.tolist()}")

    cases: List[MammographyCase] = []
    by_file: dict[str, list] = defaultdict(list)
    for _, row in raw_df.iterrows():
        fname = str(row.get("filename", "")).strip()
        if not fname or fname.lower() not in le_files:
            continue
        by_file[fname].append(row)

    for fname, row_list in by_file.items():
        img_path = le_files[fname.lower()]
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            log.warning(f"[CDD-CESM] Falha ao ler {fname}: {e}")
            continue

        anns: List[NoduleAnnotation] = []
        for row in row_list:
            if not isinstance(row, pd.Series):
                row = pd.Series(row)
            try:
                shape = json.loads(str(row.get("region_shape_attributes", "{}")))
            except json.JSONDecodeError:
                continue
            xs, ys = shape.get("all_points_x", []), shape.get("all_points_y", [])
            bbox_raw = _polygon_to_bbox(xs, ys)
            if bbox_raw is None:
                continue
            bbox = clip_bbox(bbox_raw, w, h)
            if not is_valid_bbox(bbox):
                continue
            ps = _pixel_spacing(row)
            size_mm = compute_size_mm(bbox, ps)
            if not (0 < size_mm <= 150):
                continue
            br = _parse_birads(row)
            anns.append(
                NoduleAnnotation(
                    bbox_abs=bbox,
                    size_mm=size_mm,
                    malignant=br >= 4,
                    birads=br,
                )
            )

        if not anns:
            continue

        first = row_list[0]
        if not isinstance(first, pd.Series):
            first = pd.Series(first)
        patient_id = _extract_patient_id(fname, first)
        case_birads = max((a.birads for a in anns if a.birads), default=0)

        cases.append(
            MammographyCase(
                case_id=f"cdd_{Path(fname).stem}",
                patient_id=patient_id,
                dataset="cdd_cesm",
                view="unknown",
                image_path=img_path,
                pixel_spacing_mm=_pixel_spacing(first),
                annotations=anns,
                birads=case_birads,
                laterality="",
            )
        )

    log.info(f"[CDD-CESM] {len(cases)} imagens com ≥1 massa válida (LE).")
    return cases
