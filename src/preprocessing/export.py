"""
export.py — Exportação nos formatos YOLO (.txt) e COCO JSON.

Funções públicas:
  export_yolo_label(case, img_path, label_dir)  → .txt
  export_coco_json(cases_by_split, out_dir)     → annotations_{split}.json
  export_yolo_dataset_yaml(size_bin, hub_path)  → data_{size_bin}.yaml

Formato YOLO (normalizado, classe única 0 = nódulo):
  <class> <cx_norm> <cy_norm> <w_norm> <h_norm>

Formato COCO JSON (com atributos extras para análise por faixa):
  Inclui size_bin, dataset, birads, size_mm e patient_id em cada anotação.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from src.preprocessing.loaders import MammographyCase
from src.utils.logger import get_logger

log = get_logger("export")

IMG_SIZE = 640   # tamanho fixo após resize — usado para normalização YOLO


def export_annotations_csv(
    cases: List[MammographyCase],
    split_map: Dict[str, str],
    out_path: Path,
) -> Path:
    """
    Tabela unificada (uma linha por anotação) para o hub — plano Etapa 4.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for case in cases:
        sp = split_map.get(case.case_id, "")
        for ann in case.annotations:
            rows.append(
                {
                    "case_id": case.case_id,
                    "patient_id": case.patient_id,
                    "dataset": case.dataset,
                    "split": sp,
                    "view": case.view,
                    "laterality": case.laterality,
                    "size_mm": round(ann.size_mm, 4),
                    "size_bin": ann.size_bin,
                    "birads": ann.birads,
                    "bbox_x1": ann.bbox_abs[0],
                    "bbox_y1": ann.bbox_abs[1],
                    "bbox_x2": ann.bbox_abs[2],
                    "bbox_y2": ann.bbox_abs[3],
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info(f"[export] annotations.csv → {out_path} ({len(rows)} linhas)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# YOLO
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_to_yolo(
    bbox: Tuple[int, int, int, int],
    img_size: int = IMG_SIZE,
) -> Tuple[float, float, float, float]:
    """Converte (x1,y1,x2,y2) absoluto para (cx,cy,w,h) normalizado."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / img_size
    cy = ((y1 + y2) / 2.0) / img_size
    w  = (x2 - x1) / img_size
    h  = (y2 - y1) / img_size
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def export_yolo_label(
    case: MammographyCase,
    label_dir: Path,
    img_size: int = IMG_SIZE,
) -> Path:
    """
    Exporta as anotações de um caso no formato YOLO (.txt).

    Args:
        case:      MammographyCase com bboxes em coordenadas 640×640.
        label_dir: Diretório onde o .txt será salvo.
        img_size:  Tamanho da imagem (default 640).

    Returns:
        Caminho do .txt salvo.
    """
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    txt_path = label_dir / f"{case.case_id}.txt"

    lines = []
    for ann in case.annotations:
        cx, cy, w, h = _bbox_to_yolo(ann.bbox_abs, img_size)
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return txt_path


def copy_image_to_hub(
    processed_png: Path,
    img_dir: Path,
) -> Path:
    """Copia o PNG processado para o diretório de imagens do hub."""
    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    dst = img_dir / processed_png.name
    if not dst.exists():
        shutil.copy2(str(processed_png), str(dst))
    return dst


def export_yolo_dataset_yaml(
    size_bin: str,
    hub_yolo_path: Path,
) -> Path:
    """
    Gera o arquivo data_{size_bin}.yaml necessário para o Ultralytics.

    Args:
        size_bin:      Nome da faixa (ex: 'small').
        hub_yolo_path: Raiz do hub YOLO (ex: 'data/hub/yolo').

    Returns:
        Caminho do YAML gerado.
    """
    hub_yolo_path = Path(hub_yolo_path).resolve()
    content = {
        "path":   str(hub_yolo_path),
        "train":  f"{size_bin}/train/images",
        "val":    f"{size_bin}/val/images",
        "test":   f"{size_bin}/test/images",
        "nc":     1,
        "names":  ["nodule"],
        "size_bin": size_bin,
    }
    yaml_path = hub_yolo_path / f"data_{size_bin}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
    log.info(f"[export] YAML gerado: {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# COCO JSON
# ─────────────────────────────────────────────────────────────────────────────

def export_coco_json(
    cases_by_split: Dict[str, List[Tuple[MammographyCase, Path]]],
    out_dir: Path,
    img_size: int = IMG_SIZE,
) -> Dict[str, Path]:
    """
    Exporta anotações no formato COCO JSON para cada split.

    Args:
        cases_by_split: Dict {'train': [(case, png_path), ...], 'val': ..., 'test': ...}
        out_dir:        Diretório de saída para os JSONs.
        img_size:       Tamanho das imagens (default 640).

    Returns:
        Dict {split: caminho_do_json}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths: Dict[str, Path] = {}

    for split_name, case_list in cases_by_split.items():
        images_list    = []
        annotations_list = []
        ann_id = 1

        for img_id, (case, png_path) in enumerate(case_list, start=1):
            images_list.append({
                "id":        img_id,
                "file_name": png_path.name,
                "width":     img_size,
                "height":    img_size,
                "dataset":   case.dataset,
                "patient_id": case.patient_id,
                "view":      case.view,
                "laterality": case.laterality,
                "birads":    case.birads,
            })

            for ann in case.annotations:
                x1, y1, x2, y2 = ann.bbox_abs
                w = x2 - x1
                h = y2 - y1
                annotations_list.append({
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": 1,
                    "bbox":        [x1, y1, w, h],   # COCO: [x, y, w, h]
                    "area":        float(w * h),
                    "iscrowd":     0,
                    "attributes": {
                        "size_bin":  ann.size_bin,
                        "size_mm":   round(ann.size_mm, 3),
                        "dataset":   case.dataset,
                        "birads":    ann.birads,
                        "malignant": ann.malignant,
                        "patient_id": case.patient_id,
                        "split":     split_name,
                    },
                })
                ann_id += 1

        coco_dict = {
            "info": {
                "description": f"Mammography Nodule Detection — {split_name}",
                "version": "1.0",
            },
            "categories": [{"id": 1, "name": "nodule", "supercategory": "lesion"}],
            "images":      images_list,
            "annotations": annotations_list,
        }

        json_path = out_dir / f"annotations_{split_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, indent=2, ensure_ascii=False)

        log.info(
            f"[export] COCO JSON '{split_name}': "
            f"{len(images_list)} imagens, {len(annotations_list)} anotações → {json_path}"
        )
        output_paths[split_name] = json_path

    return output_paths
