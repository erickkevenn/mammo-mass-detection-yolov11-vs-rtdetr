"""
audit_datasets.py — Etapa 0: auditoria e relatório de faixas (0-B).

Execução (Windows / plano v2):
    py -3.11 -m src.preprocessing.audit_datasets
    py -3.11 -m src.preprocessing.audit_datasets --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import yaml
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from src.preprocessing.bbox_ops import clip_bbox, compute_size_mm, is_valid_bbox
from src.preprocessing.mammo_loaders import _parse_inbreast_xml
from src.utils.logger import get_logger

log = get_logger("audit_datasets")

CDD_PIXEL_SPACING_FALLBACK = 0.094  # mm/px
VINDR_DEFAULT_PS = 0.085
INB_DEFAULT_PS = 0.070


@dataclass
class QualityReport:
    """Falhas bloqueantes vs notas informativas (excluídos por regra ≠ falha de pipeline)."""

    blocking: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.blocking.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def has_fail(self) -> bool:
        return len(self.blocking) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers gerais
# ─────────────────────────────────────────────────────────────────────────────


def _polygon_to_bbox(xs: List[float], ys: List[float]) -> Optional[Tuple[int, int, int, int]]:
    if not xs or not ys:
        return None
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _dicom_pixel_spacing(dcm_path: Path) -> float:
    try:
        dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        for tag in ("PixelSpacing", "ImagerPixelSpacing"):
            val = getattr(dcm, tag, None)
            if val is not None:
                return float(val[0])
    except Exception:
        pass
    return -1.0


def _dicom_rows_cols(dcm_path: Path) -> Optional[Tuple[int, int]]:
    try:
        dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        return int(dcm.Rows), int(dcm.Columns)
    except Exception:
        return None


def _bbox_out_of_bounds(bbox: Tuple[int, int, int, int], w: int, h: int) -> bool:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return True
    return x1 < 0 or y1 < 0 or x2 > w or y2 > h


def _parse_int_birads(val) -> int:
    if pd.isna(val):
        return 0
    s = str(val).strip()
    for d in ("1", "2", "3", "4", "5"):
        if re.search(rf"\b{d}\b", s) or d in s.replace(" ", ""):
            try:
                return int(d)
            except ValueError:
                pass
    try:
        v = int(float(s))
        return v if 1 <= v <= 5 else 0
    except (ValueError, TypeError):
        return 0


def _cdd_extract_patient_id(fname: str, row: pd.Series) -> str:
    """Prioridade: coluna patient_id no CSV; senão prefixo P{n} no filename (não usar filename completo)."""
    for col in ("patient_id", "Patient ID", "patient", "PatientID"):
        if col in row.index and pd.notna(row.get(col)) and str(row.get(col)).strip():
            return str(row.get(col)).strip()
    m = re.match(r"(P\d+)", fname, re.IGNORECASE)
    if m:
        return m.group(1).upper() if m.group(1).upper().startswith("P") else m.group(1)
    return Path(fname).stem


def _cdd_pixel_spacing(row: pd.Series) -> float:
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
    # JSON em file_attributes / region_attributes
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


def _cdd_parse_birads_region(row: pd.Series) -> int:
    """BI-RADS em colunas diretas ou dentro de region_attributes (JSON)."""
    for col in ("birads", "BI-RADS", "BI_RADS", "BIRADS", "bi_rads"):
        if col in row.index and pd.notna(row.get(col)):
            v = _parse_int_birads(row.get(col))
            if v:
                return v
    raw = row.get("region_attributes", "{}")
    try:
        d = json.loads(str(raw))
        for k in ("birads", "BI-RADS", "assessment"):
            if k in d:
                v = _parse_int_birads(d[k])
                if v:
                    return v
    except (json.JSONDecodeError, TypeError):
        pass
    return 0


def scan_cdd_non_le_folders(raw_dir: Path, q: QualityReport) -> None:
    """Sinaliza pastas/arquivos suspeitos de não-LE no root do CDD (recombined / HE)."""
    bad_names = []
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        low = str(p).lower()
        if any(
            tok in low
            for tok in (
                "recombined",
                "re-combined",
                "subtraction",
                "high energy",
                "high_energy",
                "he image",
                "cesm_he",
            )
        ):
            bad_names.append(str(p.relative_to(raw_dir)))
    le_dir = raw_dir / "Low energy images of CDD-CESM"
    if bad_names:
        q.fail(
            f"❌ [CDD-CESM] Arquivos/pastas possivelmente não-LE encontrados ({len(bad_names)}): "
            f"ex.: {bad_names[:5]}"
        )
    if le_dir.exists():
        for p in le_dir.iterdir():
            pl = p.name.lower()
            if "recombined" in pl or ("sub" in pl and "subtract" in pl):
                q.fail(f"❌ [CDD-CESM] Ficheiro suspeito na pasta LE: {p.name}")


def simulate_group_splits(groups: np.ndarray, n_samples: int, seed: int = 42):
    """70% / 15% / 15% por grupo (patient), sem overlap."""
    idx = np.arange(n_samples)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_val_rel, te_rel = next(gss1.split(idx, groups=groups))

    sub_groups = groups[tr_val_rel]
    sub_idx = np.arange(len(tr_val_rel))
    val_fraction = 0.15 / 0.85
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_sub, va_sub = next(gss2.split(sub_idx, groups=sub_groups))

    train_idx = tr_val_rel[tr_sub]
    val_idx = tr_val_rel[va_sub]
    test_idx = te_rel.astype(np.int64)

    def _pids(ix):
        return set(groups[ix])

    ok_disjoint = (
        _pids(train_idx).isdisjoint(_pids(val_idx))
        and _pids(train_idx).isdisjoint(_pids(test_idx))
        and _pids(val_idx).isdisjoint(_pids(test_idx))
    )
    return train_idx, val_idx, test_idx, ok_disjoint


def print_histogram_5mm(df_valid: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    edges = list(range(0, 155, 5))
    for ds in sorted(df_valid["dataset"].unique()):
        sub = df_valid[df_valid["dataset"] == ds]["size_mm"]
        hist, _ = np.histogram(sub, bins=edges)
        print(f"  [{ds}] intervalos de 5 mm (limite inferior inclusivo, exclusivo superior):")
        for i in range(len(hist)):
            lo, hi = edges[i], edges[i + 1]
            if hist[i]:
                print(f"    [{lo:3d},{hi:3d}) mm: {int(hist[i])}")
    sub_all = df_valid["size_mm"]
    hist_all, _ = np.histogram(sub_all, bins=edges)
    print("  [TOTAL agregado]")
    for i in range(len(hist_all)):
        lo, hi = edges[i], edges[i + 1]
        if hist_all[i]:
            print(f"    [{lo:3d},{hi:3d}) mm: {int(hist_all[i])}")


# ─────────────────────────────────────────────────────────────────────────────
# Auditores
# ─────────────────────────────────────────────────────────────────────────────


def audit_cdd_cesm(raw_dir: Path, q: QualityReport) -> Tuple[pd.DataFrame, Dict[str, int]]:
    csv_path = raw_dir / "Radiology_hand_drawn_segmentations_v2.csv"
    img_dir = raw_dir / "Low energy images of CDD-CESM"
    stats = {
        "rows_csv": 0,
        "skipped_not_le_folder": 0,
        "null_polygon": 0,
        "loaded": 0,
        "bbox_clamped": 0,
        "excluded_size_mm": 0,
        "excluded_degenerate": 0,
    }

    scan_cdd_non_le_folders(raw_dir, q)

    if not csv_path.exists():
        log.error(f"[CDD-CESM] CSV não encontrado: {csv_path}")
        q.fail(f"❌ [CDD-CESM] CSV obrigatório em falta: {csv_path}")
        return pd.DataFrame(), stats

    if not img_dir.exists():
        q.fail(f"❌ [CDD-CESM] Pasta LE não encontrada: {img_dir}")
        return pd.DataFrame(), stats

    le_files = {p.name.lower(): p for p in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))}

    raw_df = pd.read_csv(csv_path, header=0)
    raw_df.columns = [c.lstrip("#").strip() for c in raw_df.columns]
    has_patient_col = any(c in raw_df.columns for c in ("patient_id", "Patient ID", "patient", "PatientID"))
    if not has_patient_col:
        log.warning(
            "[CDD-CESM] Coluna patient_id ausente no CSV VIA — usando identificador P{{n}} "
            "extraído do filename (não o filename completo)."
        )

    if "filename" not in raw_df.columns:
        q.fail(f"❌ [CDD-CESM] Coluna filename ausente: {raw_df.columns.tolist()}")
        return pd.DataFrame(), stats

    rows = []
    stats["rows_csv"] = len(raw_df)

    for _, row in raw_df.iterrows():
        fname = str(row.get("filename", "")).strip()
        if not fname:
            continue

        low_name = fname.lower()
        if low_name not in le_files:
            stats["skipped_not_le_folder"] += 1
            continue

        img_path = le_files[low_name]
        patient_id = _cdd_extract_patient_id(fname, row)
        ps = _cdd_pixel_spacing(row)
        birads = _cdd_parse_birads_region(row)

        shape_str = str(row.get("region_shape_attributes", "{}"))
        try:
            shape = json.loads(shape_str)
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
        except json.JSONDecodeError:
            xs, ys = [], []

        bbox_raw = _polygon_to_bbox(xs, ys)
        if bbox_raw is None:
            stats["null_polygon"] += 1
            continue

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            q.fail(f"❌ [CDD-CESM] Falha ao ler imagem {fname}: {e}")
            continue

        bbox = clip_bbox(bbox_raw, w, h)
        clamped = _bbox_out_of_bounds(bbox_raw, w, h)
        if clamped:
            stats["bbox_clamped"] += 1

        size_mm = compute_size_mm(bbox, ps)

        if not is_valid_bbox(bbox):
            stats["excluded_degenerate"] += 1
            valid = False
        elif not (0 < size_mm <= 150):
            stats["excluded_size_mm"] += 1
            valid = False
        else:
            valid = True

        stats["loaded"] += 1

        rows.append(
            {
                "dataset": "cdd_cesm",
                "patient_id": patient_id,
                "split_group_key": f"cdd_cesm|{patient_id}",
                "case_id": fname,
                "view": "",
                "laterality": "",
                "size_mm": size_mm,
                "birads": birads if birads else np.nan,
                "image_w": w,
                "image_h": h,
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "bbox_oob": clamped,
                "valid": valid,
                "pixel_spacing_mm": ps,
            }
        )

    if stats["skipped_not_le_folder"]:
        q.note(
            f"⚠️  [CDD-CESM] {stats['skipped_not_le_folder']} linhas do CSV sem ficheiro correspondente "
            f"na pasta LE (não carregadas — possíveis recombined/outro estudo)."
        )
    if stats["bbox_clamped"]:
        q.note(f"ℹ️  [CDD-CESM] Bboxes ajustadas (clamp) aos limites da imagem: {stats['bbox_clamped']}")
    if stats["excluded_size_mm"]:
        q.note(
            f"ℹ️  [CDD-CESM] Lesões excluídas (size_mm ∉ ]0,150] após clip): {stats['excluded_size_mm']}"
        )
    if stats["excluded_degenerate"]:
        q.note(
            f"ℹ️  [CDD-CESM] Lesões excluídas (bbox degenerada após clip): {stats['excluded_degenerate']}"
        )

    df = pd.DataFrame(rows)
    log.info(f"[CDD-CESM] anotações carregadas do LE: {len(df)} | válidas: {df['valid'].sum() if len(df) else 0}")
    return df, stats


def audit_inbreast(raw_dir: Path, q: QualityReport) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "xml_missing": 0,
        "mass_rois": 0,
        "parse_failed": 0,
        "bbox_clamped": 0,
        "excluded_size_mm": 0,
        "excluded_degenerate": 0,
        "no_dicom_shape": 0,
    }
    dicom_dir = raw_dir / "AllDICOMs"
    xml_dir = raw_dir / "AllXML"
    csv_path = raw_dir / "INbreast.csv"

    if not csv_path.exists():
        q.fail(f"❌ [INbreast] INbreast.csv não encontrado: {csv_path}")
        return pd.DataFrame(), stats

    meta = pd.read_csv(csv_path, sep=";", encoding="latin-1")
    meta.columns = [c.strip() for c in meta.columns]

    rows = []
    for _, row in meta.iterrows():
        file_name = str(row.get("File Name", "")).strip()
        if not file_name:
            continue

        patient_id_file = file_name.split("_")[0] if "_" in file_name else Path(file_name).stem

        dcm_candidates = list(dicom_dir.glob(f"{file_name}*.dcm")) if dicom_dir.exists() else []
        ps = INB_DEFAULT_PS
        rc: Optional[Tuple[int, int]] = None
        if dcm_candidates:
            ps_f = _dicom_pixel_spacing(dcm_candidates[0])
            if ps_f > 0:
                ps = ps_f
            rc = _dicom_rows_cols(dcm_candidates[0])

        patient_dicom = ""
        if dcm_candidates:
            try:
                dcm = pydicom.dcmread(str(dcm_candidates[0]), stop_before_pixels=True)
                patient_dicom = str(getattr(dcm, "PatientID", "") or "").strip()
            except Exception:
                pass
        if patient_dicom and patient_dicom != patient_id_file:
            log.debug(f"[INbreast] PatientID DICOM ({patient_dicom}) ≠ prefixo filename ({patient_id_file})")

        xml_candidates = list(xml_dir.glob(f"{file_name}*.xml")) if xml_dir.exists() else []
        if not xml_candidates:
            stats["xml_missing"] += 1
            continue

        anns = _parse_inbreast_xml(xml_candidates[0])
        if not anns:
            stats["parse_failed"] += 1
            continue

        stats["mass_rois"] += len(anns)

        birads_csv = _parse_int_birads(row.get("Bi-Rads", 0))
        birads = float(birads_csv) if 1 <= birads_csv <= 5 else np.nan

        view = str(row.get("View", "")).strip().upper()
        lat = str(row.get("Laterality", "")).strip().upper()

        for ann in anns:
            bbox_raw = ann.bbox_abs

            if rc is None:
                stats["no_dicom_shape"] += 1
                rows.append(
                    {
                        "dataset": "inbreast",
                        "patient_id": patient_id_file,
                        "split_group_key": f"inbreast|{patient_id_file}",
                        "case_id": file_name,
                        "view": view if view in ("CC", "MLO") else "unknown",
                        "laterality": lat,
                        "size_mm": np.nan,
                        "birads": birads,
                        "image_w": np.nan,
                        "image_h": np.nan,
                        "bbox_x1": bbox_raw[0],
                        "bbox_y1": bbox_raw[1],
                        "bbox_x2": bbox_raw[2],
                        "bbox_y2": bbox_raw[3],
                        "bbox_oob": False,
                        "valid": False,
                        "pixel_spacing_mm": ps,
                    }
                )
                continue

            h, w = int(rc[0]), int(rc[1])
            bbox = clip_bbox(bbox_raw, w, h)
            clamped = _bbox_out_of_bounds(bbox_raw, w, h)
            if clamped:
                stats["bbox_clamped"] += 1

            size_mm = compute_size_mm(bbox, ps)

            if not is_valid_bbox(bbox):
                stats["excluded_degenerate"] += 1
                valid = False
            elif not (0 < size_mm <= 150):
                stats["excluded_size_mm"] += 1
                valid = False
            else:
                valid = True

            rows.append(
                {
                    "dataset": "inbreast",
                    "patient_id": patient_id_file,
                    "split_group_key": f"inbreast|{patient_id_file}",
                    "case_id": file_name,
                    "view": view if view in ("CC", "MLO") else "unknown",
                    "laterality": lat,
                    "size_mm": size_mm,
                    "birads": birads,
                    "image_w": w,
                    "image_h": h,
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "bbox_oob": clamped,
                    "valid": valid,
                    "pixel_spacing_mm": ps,
                }
            )

    if stats["bbox_clamped"]:
        q.note(f"ℹ️  [INbreast] Bboxes ajustadas (clamp): {stats['bbox_clamped']}")
    if stats["excluded_size_mm"]:
        q.note(f"ℹ️  [INbreast] Lesões excluídas (size_mm ∉ ]0,150] após clip): {stats['excluded_size_mm']}")
    if stats["excluded_degenerate"]:
        q.note(f"ℹ️  [INbreast] Lesões excluídas (bbox degenerada após clip): {stats['excluded_degenerate']}")
    if stats["no_dicom_shape"]:
        q.note(f"⚠️  [INbreast] ROIs sem Rows/Columns no DICOM (excluídas): {stats['no_dicom_shape']}")

    df = pd.DataFrame(rows)
    log.info(f"[INbreast] massas (ROI): {len(df)} | válidas: {df['valid'].sum() if len(df) else 0}")
    return df, stats


def audit_vindr(raw_dir: Path, q: QualityReport) -> Tuple[pd.DataFrame, Dict[str, int]]:
    finding_csv = raw_dir / "finding_annotations.csv"
    metadata_csv = raw_dir / "metadata.csv"
    stats = {
        "rows_mass": 0,
        "skipped_nan_bbox": 0,
        "bbox_clamped": 0,
        "excluded_size_mm": 0,
        "excluded_degenerate": 0,
        "invalid_dims": 0,
    }

    if not finding_csv.exists():
        q.fail("❌ [VinDr] finding_annotations.csv não encontrado")
        return pd.DataFrame(), stats

    df_f = pd.read_csv(finding_csv)
    meta_by_uid: Dict[str, pd.Series] = {}
    if metadata_csv.exists():
        df_m = pd.read_csv(metadata_csv)
        for _, r in df_m.iterrows():
            uid = str(r.get("SOP Instance UID", "")).strip()
            if uid:
                meta_by_uid[uid] = r

    rows = []
    for _, row in df_f.iterrows():
        finding_cat = str(row.get("finding_categories", ""))
        if "Mass" not in finding_cat:
            continue

        stats["rows_mass"] += 1
        image_id = str(row.get("image_id", "")).strip()
        study_id = str(row.get("study_id", "")).strip()

        xmin, ymin, xmax, ymax = row.get("xmin"), row.get("ymin"), row.get("xmax"), row.get("ymax")
        if pd.isna(xmin) or pd.isna(ymin) or pd.isna(xmax) or pd.isna(ymax):
            stats["skipped_nan_bbox"] += 1
            continue

        bbox_raw = (
            int(round(float(xmin))),
            int(round(float(ymin))),
            int(round(float(xmax))),
            int(round(float(ymax))),
        )

        meta = meta_by_uid.get(image_id)
        ps = VINDR_DEFAULT_PS
        w = int(row.get("width", 0)) if pd.notna(row.get("width")) else 0
        h = int(row.get("height", 0)) if pd.notna(row.get("height")) else 0
        if meta is not None:
            ws = meta.get("Imager Pixel Spacing", meta.get("Pixel Spacing", ""))
            try:
                ps = float(str(ws).strip("[]").split(",")[0])
                if ps <= 0:
                    ps = VINDR_DEFAULT_PS
            except (ValueError, TypeError):
                ps = VINDR_DEFAULT_PS
            try:
                h = int(meta["Rows"]) if pd.notna(meta.get("Rows")) else h
                w = int(meta["Columns"]) if pd.notna(meta.get("Columns")) else w
            except (KeyError, TypeError, ValueError):
                pass

        birads = 0
        if "assessment" in row.index and pd.notna(row.get("assessment")):
            birads = _parse_int_birads(row.get("assessment"))
        if not birads:
            birads = _parse_int_birads(row.get("finding_birads", ""))
        if not birads:
            birads = _parse_int_birads(row.get("breast_birads", ""))

        view = str(row.get("view_position", "")).strip().upper()
        lat = str(row.get("laterality", "")).strip().upper()

        if w <= 0 or h <= 0:
            stats["invalid_dims"] += 1
            rows.append(
                {
                    "dataset": "vindr",
                    "patient_id": study_id,
                    "split_group_key": f"vindr|{study_id}",
                    "case_id": image_id,
                    "view": view if view in ("CC", "MLO") else "unknown",
                    "laterality": lat,
                    "size_mm": np.nan,
                    "birads": birads if birads else np.nan,
                    "image_w": w,
                    "image_h": h,
                    "bbox_x1": bbox_raw[0],
                    "bbox_y1": bbox_raw[1],
                    "bbox_x2": bbox_raw[2],
                    "bbox_y2": bbox_raw[3],
                    "bbox_oob": False,
                    "valid": False,
                    "pixel_spacing_mm": ps,
                }
            )
            continue

        bbox = clip_bbox(bbox_raw, w, h)
        clamped = _bbox_out_of_bounds(bbox_raw, w, h)
        if clamped:
            stats["bbox_clamped"] += 1

        size_mm = compute_size_mm(bbox, ps)

        if not is_valid_bbox(bbox):
            stats["excluded_degenerate"] += 1
            valid = False
        elif not (0 < size_mm <= 150):
            stats["excluded_size_mm"] += 1
            valid = False
        else:
            valid = True

        rows.append(
            {
                "dataset": "vindr",
                "patient_id": study_id,
                "split_group_key": f"vindr|{study_id}",
                "case_id": image_id,
                "view": view if view in ("CC", "MLO") else "unknown",
                "laterality": lat,
                "size_mm": size_mm,
                "birads": birads if birads else np.nan,
                "image_w": w,
                "image_h": h,
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "bbox_oob": clamped,
                "valid": valid,
                "pixel_spacing_mm": ps,
            }
        )

    if stats["skipped_nan_bbox"]:
        q.note(f"⚠️  [VinDr] {stats['skipped_nan_bbox']} anotações Mass com bbox nula (excluídas).")
    if stats["bbox_clamped"]:
        q.note(f"ℹ️  [VinDr] Bboxes ajustadas (clamp): {stats['bbox_clamped']}")
    if stats["excluded_size_mm"]:
        q.note(f"ℹ️  [VinDr] Lesões excluídas (size_mm ∉ ]0,150] após clip): {stats['excluded_size_mm']}")
    if stats["excluded_degenerate"]:
        q.note(f"ℹ️  [VinDr] Lesões excluídas (bbox degenerada após clip): {stats['excluded_degenerate']}")
    if stats["invalid_dims"]:
        q.note(f"⚠️  [VinDr] Anotações sem dimensões de imagem válidas: {stats['invalid_dims']}")

    df = pd.DataFrame(rows)
    log.info(f"[VinDr] Mass findings: {len(df)} linhas | válidas: {df['valid'].sum() if len(df) else 0}")
    return df, stats


# ─────────────────────────────────────────────────────────────────────────────
# Relatório 0-B
# ─────────────────────────────────────────────────────────────────────────────


def print_final_bins_report(df_all: pd.DataFrame, q: QualityReport, min_test_n: int = 30) -> None:
    df_v = df_all[df_all["valid"]].copy()
    N = len(df_v)

    print("\n" + "=" * 60)
    print("RELATÓRIO DE FAIXAS — AGUARDANDO CONFIRMAÇÃO DO USUÁRIO")
    print("=" * 60)

    if N == 0:
        print("\n❌ Nenhuma anotação válida após filtros de qualidade.\n")
        print("=" * 60)
        return

    sizes = df_v["size_mm"].values
    p10, p25, p33, p50, p67, p75, p90 = np.percentile(sizes, [10, 25, 33, 50, 67, 75, 90])

    print("\nDISTRIBUIÇÃO AGREGADA (todos os datasets, apenas massas válidas):")
    print(f"  Total de amostras: {N}")
    print(
        f"  P10={p10:.1f} mm | P25={p25:.1f} mm | P33={p33:.1f} mm | P50={p50:.1f} mm | "
        f"P67={p67:.1f} mm | P75={p75:.1f} mm | P90={p90:.1f} mm"
    )

    # Tercis P33 / P67 — opção principal (plano v3)
    n_b_small = int(np.sum(sizes < p33))
    n_b_med = int(np.sum((sizes >= p33) & (sizes < p67)))
    n_b_large = int(np.sum((sizes >= p67) & (sizes <= 150)))
    nB = [n_b_small, n_b_med, n_b_large]
    ok_flags_b = [(n * 0.15) >= min_test_n for n in nB]
    viable_b = all(ok_flags_b)

    print("\nOPÇÃO PRINCIPAL — Tercis empíricos (P33 / P67) [RECOMENDADA]:")
    tier_b = [
        ("small ", f" ( 0 – {p33:.1f} mm)", n_b_small),
        ("medium", f" ({p33:.1f} – {p67:.1f} mm)", n_b_med),
        ("large ", f" (>{p67:.1f} mm)", n_b_large),
    ]
    for label, suf, n in tier_b:
        est = n * 0.15
        ok = est >= min_test_n
        flag = "✅" if ok else "❌"
        print(f"  {label}{suf}: {n:4d} amostras | teste estimado: {est:.0f} | {flag}")
    print(
        f"  Viável para publicação (inferência formal nas 3 faixas): "
        f"{'SIM' if viable_b else 'NÃO'}"
    )

    # TNM 0 / 10 / 20 mm — opção alternativa (descritiva se n baixo)
    ok_flags_a = []
    print("\nOPÇÃO ALTERNATIVA — TNM (0 / 10 / 20 mm) [apenas se viável]:")
    n_tnm_small = int(np.sum(sizes < 10))
    n_tnm_med = int(np.sum((sizes >= 10) & (sizes <= 20)))
    n_tnm_large = int(np.sum(sizes > 20))
    fmt_a = (
        ("small ", " ( 0–10 mm):", n_tnm_small),
        ("medium", " (10–20 mm):", n_tnm_med),
        ("large ", " (>20 mm):   ", n_tnm_large),
    )
    for label, suf, n in fmt_a:
        est = n * 0.15
        ok = est >= min_test_n
        ok_flags_a.append(ok)
        flag = "✅" if ok else f"❌ ABAIXO DE {min_test_n} — só exploratório"
        print(f"  {label}{suf}  {n:4d} amostras | teste estimado: {est:.0f} | {flag}")
    viable_a = all(ok_flags_a)
    print(f"  Viável para inferência formal nas 3 faixas: {'SIM' if viable_a else 'NÃO'}")

    n_above_30 = int(np.sum(sizes > 30))
    pct_above = 100.0 * n_above_30 / N if N else 0.0
    print(f"\nLESÕES > 30 mm: {n_above_30} amostras ({pct_above:.1f}% do total)")
    if pct_above > 10:
        print("  → Se > 10%: teto em 30 mm seria indefensável — tercis são obrigatórios.")
    elif pct_above < 5:
        print("  → Se < 5%: teto em 30 mm seria aceitável mas ainda desnecessário com tercis.")
    else:
        print("  → Entre 5% e 10%: tercis continuam preferíveis para poder nos estratos.")

    print("\nDISTRIBUIÇÃO POR DATASET (Opção Principal):")
    for ds in sorted(df_v["dataset"].unique()):
        sub = df_v[df_v["dataset"] == ds]["size_mm"].values
        c0 = int(np.sum(sub < p33))
        c1 = int(np.sum((sub >= p33) & (sub < p67)))
        c2 = int(np.sum(sub >= p67))
        print(f"  {ds:12s}  small={c0:4d}  medium={c1:4d}  large={c2:4d}")

    print("\n" + "=" * 60)
    print(
        "⏸️  PAUSA — O pipeline NÃO prossegue até o usuário confirmar:\n"
        "    1. Qual opção de faixas (ex.: \"confirmo tercis P33/P67\")\n"
        "    2. Se os ❌ de qualidade foram resolvidos"
    )
    print("=" * 60)
    if q.has_fail():
        print(
            "\n❌ NOTA — Auditoria de qualidade: existem falhas ❌ em ETAPA 0-A; "
            "não avançar para Etapa 1 até corrigir.\n"
        )
    else:
        print("\n✅ NOTA — Nenhuma falha ❌ obrigatória registada na auditoria de qualidade.\n")


def print_etapa_0a_summary(
    df_all: pd.DataFrame,
    q: QualityReport,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    ok_disjoint: bool,
    min_test_n: int,
) -> None:
    """Checks obrigatórios antes do relatório final."""
    print("\n" + "=" * 60)
    print("ETAPA 0-A — VERIFICAÇÕES")
    print("=" * 60)

    df_v = df_all[df_all["valid"]].copy()
    groups = df_all["split_group_key"].astype(str).values

    print("\n1) Patient ID / agrupamento para split:")
    for ds in sorted(df_all["dataset"].unique()):
        sub = df_all[df_all["dataset"] == ds]
        print(
            f"  · {ds}: {sub['patient_id'].nunique()} IDs paciente/exame | "
            f"{sub['split_group_key'].nunique()} chaves de grupo | {len(sub)} anotações"
        )

    print("\n2) Bounding boxes e anotações nulas:")
    n_tot = len(df_all)
    n_valid = df_all["valid"].sum()
    n_oob = df_all["bbox_oob"].sum()
    n_bad_size = ((df_all["size_mm"] <= 0) | (df_all["size_mm"] > 150)).sum()
    print(f"  · Total de linhas de massa carregadas: {n_tot}")
    print(f"  · Válidas (mm in (0,150] com bbox dentro da imagem): {int(n_valid)}")
    print(f"  · Anotações com bbox original fora da imagem (corrigidas por clamp): {int(n_oob)}")
    print(f"  · size_mm <= 0 ou > 150 mm: {int(n_bad_size)}")

    print("\n3) Percentis e histograma (apenas válidas) — ver secção seguinte.")

    print_histogram_5mm(df_v, "Contagem por intervalo de 5 mm (por dataset e total):")

    print("\n4) Simulação GroupShuffleSplit (70% / 15% / 15%, seed=42):")
    print(
        f"  · Sobreposição de pacientes/grupo entre treino, val e teste: "
        f"{'❌ DETETADA' if not ok_disjoint else 'ausente ✅'}"
    )
    tr_g = set(groups[train_idx])
    va_g = set(groups[val_idx])
    te_g = set(groups[test_idx])
    print(f"  · Grupos treino={len(tr_g)} | val={len(va_g)} | teste={len(te_g)}")

    def bin_counts(ix: np.ndarray, edges: Tuple[float, float, float]) -> np.ndarray:
        e1, e2 = edges[1], edges[2]
        s = df_all.iloc[ix]
        s = s[s["valid"]]["size_mm"].values
        c0 = int(np.sum((s >= 0) & (s < e1)))
        c1 = int(np.sum((s >= e1) & (s < e2)))
        c2 = int(np.sum(s >= e2))
        return np.array([c0, c1, c2])

    if len(df_v):
        p33, p67 = np.percentile(df_v["size_mm"], [33, 67])
        edges_b = (0.0, p33, p67)

        s_te = df_all.iloc[test_idx]
        s_te = s_te[s_te["valid"]]["size_mm"].values
        tc_a = np.array(
            [
                int(np.sum(s_te < 10)),
                int(np.sum((s_te >= 10) & (s_te <= 20))),
                int(np.sum(s_te > 20)),
            ]
        )
        tc_b = bin_counts(test_idx, edges_b)

        print("  · Contagem no TESTE por faixa — TNM (0 / 10 / 20 mm):")
        for name, c in zip(("small", "medium", "large"), tc_a):
            flag = "✅" if c >= min_test_n else f"❌ ABAIXO DE {min_test_n}"
            print(f"      {name}: {c} amostras | {flag}")

        print(f"  · Contagem no TESTE por faixa — Tercis empíricos (P33={p33:.1f} / P67={p67:.1f} mm):")
        for name, c in zip(("small", "medium", "large"), tc_b):
            flag = "✅" if c >= min_test_n else f"❌ ABAIXO DE {min_test_n}"
            print(f"      {name}: {c} amostras | {flag}")

        print("\n  · TESTE por dataset × faixa (TNM 0/10/20 mm):")
        sub_te = df_all.iloc[test_idx]
        sub_te = sub_te[sub_te["valid"]]
        for ds in sorted(sub_te["dataset"].unique()):
            s = sub_te[sub_te["dataset"] == ds]["size_mm"].values
            c0 = int(np.sum(s < 10))
            c1 = int(np.sum((s >= 10) & (s <= 20)))
            c2 = int(np.sum(s > 20))
            print(f"      {ds}: small={c0} medium={c1} large={c2}")

    print("\n5) BI-RADS por dataset (valores devem estar em 1-5 quando presentes):")
    for ds in sorted(df_all["dataset"].unique()):
        sub = df_all[df_all["dataset"] == ds]
        b = pd.to_numeric(sub["birads"], errors="coerce").dropna()
        if b.empty:
            print(f"  · {ds}: sem BI-RADS numérico nas anotações analisadas")
            continue
        b_int = b.astype(int)
        invalid = b_int[~b_int.isin([1, 2, 3, 4, 5])]
        dist = b_int[b_int.isin([1, 2, 3, 4, 5])].value_counts().sort_index()
        txt = " | ".join(f"{k}:{int(dist[k])}" for k in sorted(dist.index))
        if len(invalid):
            q.fail(f"❌ [{ds}] BI-RADS inválidos (fora de 1–5): {len(invalid)}")
            print(f"  · {ds}: {txt} | ❌ inválidos:{len(invalid)}")
        else:
            print(f"  · {ds}: {txt}")

    print("\n6) CDD-CESM — apenas Low-Energy:")
    print(
        "  · Imagens devem residir apenas na pasta LE e corresponder ao CSV VIA.\n"
        "  · Referências a recombined/HE no relatório de qualidade indicam problema ❌."
    )

    if q.blocking or q.notes:
        print("\n--- Registo de qualidade ---")
        for m in q.blocking:
            print(m)
        for m in q.notes:
            print(m)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Etapa 0 — Auditoria dos datasets")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="results/audit/audit_report.csv")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    q = QualityReport()
    dfs: List[pd.DataFrame] = []

    cdd_path = Path(paths.get("cdd_cesm_raw", "ds/CDD-CESM"))
    if cdd_path.exists():
        df_c, st_c = audit_cdd_cesm(cdd_path, q)
        log.info(f"[CDD-CESM] stats: {st_c}")
        if not df_c.empty:
            dfs.append(df_c)
    else:
        log.warning(f"CDD-CESM não encontrado: {cdd_path}")

    inb_path = Path(paths.get("inbreast_raw", "ds/INBREAST/INbreast Release 1.0"))
    if inb_path.exists():
        df_i, st_i = audit_inbreast(inb_path, q)
        log.info(f"[INbreast] stats: {st_i}")
        if not df_i.empty:
            dfs.append(df_i)
    else:
        log.warning(f"INbreast não encontrado: {inb_path}")

    vindr_path = Path(paths.get("vindr_raw", ""))
    if not vindr_path.exists():
        cand = list(Path("ds").glob("vindr*"))
        if cand:
            vindr_path = cand[0]
            log.info(f"VinDr auto-detectado: {vindr_path}")
    if vindr_path.exists():
        df_v, st_v = audit_vindr(vindr_path, q)
        log.info(f"[VinDr] stats: {st_v}")
        if not df_v.empty:
            dfs.append(df_v)
    else:
        log.warning(f"VinDr não encontrado: {vindr_path}")

    if not dfs:
        log.error("Nenhum dataset carregado.")
        print("\n❌ Nenhum dataset encontrado em paths.* Verifique config.yaml.\n")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    groups = df_all["split_group_key"].astype(str).values
    train_idx, val_idx, test_idx, ok_disjoint = simulate_group_splits(groups, len(df_all))
    if not ok_disjoint:
        q.fail("❌ GroupShuffleSplit produziu overlap de pacientes entre splits (inesperado).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)
    log.info(f"Tabela unificada da auditoria salva em {out_path}")

    print_etapa_0a_summary(df_all, q, train_idx, val_idx, test_idx, ok_disjoint, min_test_n=30)
    print_final_bins_report(df_all, q, min_test_n=30)


if __name__ == "__main__":
    main()
