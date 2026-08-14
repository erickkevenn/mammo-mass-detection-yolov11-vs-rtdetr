"""
metrics.py — Coleta e organização de métricas por faixa de tamanho.

Extrai do diretório de resultados do Ultralytics:
  - mAP@0.5
  - mAP@0.5:0.95
  - Precision, Recall, F1
  - AR@100 (Average Recall com 100 detecções)
  - AP_small, AP_medium, AP_large (áreas COCO, via val quando disponível)

Uso:
    python -m src.evaluation.metrics --config config.yaml
    python -m src.evaluation.metrics --config config.yaml --coco-val
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from src.evaluation.cross_dataset import find_best_weights
from src.preprocessing.stratify import EXPORT_BINS
from src.utils.logger import get_logger

log = get_logger("metrics")

MODELS = ["yolo", "detr"]
MODEL_NAMES = {"yolo": "YOLOv11m", "detr": "RT-DETR-L"}


def load_ultralytics_results(run_dir: Path) -> Optional[Dict]:
    """
    Carrega métricas de um diretório de run do Ultralytics.

    Tenta ler results.csv (gerado por model.train()) e
    extrai os valores da última época (ou melhor época).

    Args:
        run_dir: Caminho para o diretório do run (ex: results/yolo/yolo11m_small).

    Returns:
        Dicionário com métricas ou None se não encontrado.
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        log.warning(f"[metrics] results.csv não encontrado em {run_dir}")
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Usar linha com melhor mAP@0.5 em val
    map50_col = next(
        (c for c in df.columns if "mAP50" in c and "95" not in c), None
    )
    if map50_col is None:
        log.warning(f"[metrics] Coluna mAP50 não encontrada em {csv_path}")
        return None

    best_row = df.loc[df[map50_col].idxmax()]

    map50    = best_row.get(map50_col, 0.0)
    map5095  = best_row.get(
        next((c for c in df.columns if "mAP50-95" in c), ""), 0.0
    )
    precision = best_row.get(
        next((c for c in df.columns if "precision" in c.lower()), ""), 0.0
    )
    recall = best_row.get(
        next((c for c in df.columns if "recall" in c.lower()), ""), 0.0
    )

    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "mAP@0.5":      round(float(map50),    4),
        "mAP@0.5:0.95": round(float(map5095),  4),
        "Precision":    round(float(precision), 4),
        "Recall":       round(float(recall),    4),
        "F1":           round(f1,               4),
        "AR@100":       None,
        "AP_small":     None,
        "AP_medium":    None,
        "AP_large":     None,
        "epoch_best":   int(df[map50_col].idxmax()) + 1,
    }


def _compute_iou(box_a, box_b):
    """Calcula IoU entre duas bboxes [x1,y1,x2,y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_froc(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.3,
    fp_rates: List[float] = None,
    n_images: int = 0,
) -> Dict:
    """
    Calcula curva FROC (Free-Response ROC) para detecção mamográfica.

    FROC mede sensibilidade (True Positive Rate) em função do número
    de falsos positivos por imagem (FP/img). É o padrão em avaliação
    de CAD mamográfico (VinDr-Mammo benchmark, DREAM challenge).

    Args:
        predictions:   Lista de dicts com keys: image_id, bbox [x1,y1,x2,y2], score.
        ground_truths: Lista de dicts com keys: image_id, bbox [x1,y1,x2,y2].
        iou_threshold: Limiar de IoU para considerar match (default 0.3 — padrão mamografia).
        fp_rates:      Taxas de FP/img para reportar sensibilidade.
                       Default: [0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
        n_images:      Número total de imagens no conjunto de teste.

    Returns:
        Dict com:
          - 'froc_points': lista de (fp_per_img, sensitivity)
          - 'mean_sensitivity': média da sensibilidade nos fp_rates especificados
          - 'sensitivity_at_fp': {fp_rate: sensitivity}
    """
    if fp_rates is None:
        fp_rates = [0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]

    if n_images == 0:
        all_ids = set(d["image_id"] for d in ground_truths)
        all_ids.update(d["image_id"] for d in predictions)
        n_images = len(all_ids)

    if n_images == 0:
        return {"froc_points": [], "mean_sensitivity": 0.0, "sensitivity_at_fp": {}}

    # Agrupar GTs por imagem
    gt_by_image: Dict[str, List] = {}
    for gt in ground_truths:
        gt_by_image.setdefault(gt["image_id"], []).append(gt["bbox"])

    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0:
        return {"froc_points": [], "mean_sensitivity": 0.0, "sensitivity_at_fp": {}}

    # Ordenar predições por score decrescente
    sorted_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)

    # Rastrear GTs já matcheados
    gt_matched: Dict[str, List[bool]] = {
        img_id: [False] * len(bboxes)
        for img_id, bboxes in gt_by_image.items()
    }

    tp_list = []
    fp_list = []

    for pred in sorted_preds:
        img_id = pred["image_id"]
        pred_bbox = pred["bbox"]

        gt_bboxes = gt_by_image.get(img_id, [])
        matched = gt_matched.get(img_id, [])

        best_iou = 0.0
        best_idx = -1
        for idx, gt_bbox in enumerate(gt_bboxes):
            if matched[idx]:
                continue
            iou = _compute_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[img_id][best_idx] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

    # Acumular
    import numpy as np
    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    sensitivity = tp_cum / total_gt
    fp_per_image = fp_cum / n_images

    # Interpolar sensibilidade nos fp_rates solicitados
    sensitivity_at_fp = {}
    for fp_rate in fp_rates:
        # Encontrar sensibilidade no ponto fp_per_image <= fp_rate
        idxs = np.where(fp_per_image <= fp_rate)[0]
        if len(idxs) > 0:
            sensitivity_at_fp[fp_rate] = float(sensitivity[idxs[-1]])
        else:
            sensitivity_at_fp[fp_rate] = 0.0

    mean_sens = float(np.mean(list(sensitivity_at_fp.values())))

    # Subamostrar pontos para curva
    n_points = min(200, len(sensitivity))
    step = max(1, len(sensitivity) // n_points)
    froc_points = [
        (float(fp_per_image[i]), float(sensitivity[i]))
        for i in range(0, len(sensitivity), step)
    ]

    log.info(
        f"[FROC] mean_sensitivity={mean_sens:.4f} | "
        f"sens@0.5FP={sensitivity_at_fp.get(0.5, 0):.4f} | "
        f"sens@1.0FP={sensitivity_at_fp.get(1.0, 0):.4f} | "
        f"sens@2.0FP={sensitivity_at_fp.get(2.0, 0):.4f}"
    )

    return {
        "froc_points": froc_points,
        "mean_sensitivity": round(mean_sens, 4),
        "sensitivity_at_fp": {k: round(v, 4) for k, v in sensitivity_at_fp.items()},
    }


def _pick_first(metrics: Dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        if k not in metrics:
            continue
        try:
            return float(metrics[k])
        except (TypeError, ValueError):
            continue
    return None


def _coco_area_ap(metrics: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    mAP por faixa de área COCO (S/M/L). Tenta chaves Ultralytics comuns
    e faz fallback por regex em `results_dict`.
    """
    small = _pick_first(
        metrics,
        [
            "metrics/mAP50-95(S)",
            "metrics/mAP50(S)",
            "metrics/AP50(S)",
            "metrics/AP(S)",
        ],
    )
    medium = _pick_first(
        metrics,
        [
            "metrics/mAP50-95(M)",
            "metrics/mAP50(M)",
            "metrics/AP50(M)",
            "metrics/AP(M)",
        ],
    )
    large = _pick_first(
        metrics,
        [
            "metrics/mAP50-95(L)",
            "metrics/mAP50(L)",
            "metrics/AP50(L)",
            "metrics/AP(L)",
        ],
    )

    best_s: Dict[str, float] = {}
    best_m: Dict[str, float] = {}
    best_l: Dict[str, float] = {}

    for k, raw in metrics.items():
        if not isinstance(raw, (int, float)):
            continue
        v = float(raw)
        ku = k.upper()
        if re.search(r"\(S\)", ku) and "MAP" in ku:
            prio = 2 if "95" in ku or "0.95" in ku else 1
            cur = best_s.get("p", 0)
            if prio >= cur:
                best_s = {"p": prio, "v": v}
        if re.search(r"\(M\)", ku) and "MAP" in ku:
            prio = 2 if "95" in ku else 1
            cur = best_m.get("p", 0)
            if prio >= cur:
                best_m = {"p": prio, "v": v}
        if re.search(r"\(L\)", ku) and "MAP" in ku:
            prio = 2 if "95" in ku else 1
            cur = best_l.get("p", 0)
            if prio >= cur:
                best_l = {"p": prio, "v": v}

    if small is None and best_s:
        small = best_s["v"]
    if medium is None and best_m:
        medium = best_m["v"]
    if large is None and best_l:
        large = best_l["v"]

    return small, medium, large


def _ar100_from_metrics(metrics: Dict, box) -> Optional[float]:
    keys = [
        "metrics/AR@100(B)",
        "metrics/AR100(B)",
        "metrics/AR@100",
        "metrics/AR100",
    ]
    v = _pick_first(metrics, keys)
    if v is not None:
        return v
    ar = getattr(box, "ar", None)
    if ar is not None:
        try:
            return float(ar)
        except (TypeError, ValueError):
            pass
    return None


def evaluate_coco_metrics(
    model_path: Path,
    data_yaml: Path,
    device: str = "cuda:0",
    verbose: bool = False,
) -> Dict:
    """
    Avalia o modelo no conjunto de teste e coleta métricas COCO completas.

    Inclui AR@100 e AP por área (small / medium / large) quando expostos
    pelo Ultralytics em `results_dict`.
    """
    model_name = model_path.name.lower()

    if "rtdetr" in model_name or "detr" in model_name:
        from ultralytics import RTDETR

        model = RTDETR(str(model_path))
    else:
        from ultralytics import YOLO

        model = YOLO(str(model_path))

    results = model.val(
        data=str(data_yaml),
        split="test",
        device=device,
        save_json=True,
        plots=False,
        verbose=verbose,
    )

    metrics = results.results_dict or {}
    box = results.box

    small, medium, large = _coco_area_ap(metrics)
    ar100 = _ar100_from_metrics(metrics, box)

    def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
        if x is None:
            return None
        return round(float(x), nd)

    return {
        "mAP@0.5":      round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP@0.5:0.95": round(float(metrics.get("metrics/mAP50-95(B)", 0.0)), 4),
        "Precision":    round(float(metrics.get("metrics/precision(B)", 0.0)), 4),
        "Recall":       round(float(metrics.get("metrics/recall(B)", 0.0)), 4),
        "F1":           round(float(metrics.get("metrics/F1(B)", 0.0)), 4),
        "AR@100":       _r(ar100),
        "AP_small":     _r(small),
        "AP_medium":    _r(medium),
        "AP_large":     _r(large),
    }


def collect_all_metrics(
    results_root: Path,
    cfg: Optional[dict] = None,
    coco_val: bool = False,
) -> pd.DataFrame:
    """
    Coleta métricas de todos os modelos e faixas e retorna DataFrame.

    Com `coco_val=True` e `cfg` definido, re-executa `model.val()` no teste
    com `best.pt` para preencher AR@100 e AP por área (S/M/L), quando disponíveis.
    """
    rows: List[Dict] = []

    for model_key in MODELS:
        model_dir = results_root / model_key
        if not model_dir.exists():
            continue

        for bin_name in EXPORT_BINS:
            candidates = list(model_dir.glob(f"*{bin_name}*"))
            if not candidates:
                log.warning(f"[metrics] Sem resultados: {model_key}/{bin_name}")
                continue

            run_dir = sorted(candidates)[-1]
            m = load_ultralytics_results(run_dir)
            if m is None:
                continue

            row: Dict = {
                "model": MODEL_NAMES[model_key],
                "bin": bin_name,
                **m,
            }

            if coco_val and cfg is not None:
                hub = Path(cfg["paths"]["hub_yolo"])
                data_yaml = hub / f"data_{bin_name}.yaml"
                weights = find_best_weights(results_root, model_key, bin_name)
                if weights is not None and data_yaml.exists():
                    dev = str(cfg["hardware"].get("device", "cpu"))
                    try:
                        extra = evaluate_coco_metrics(
                            weights, data_yaml, device=dev, verbose=False
                        )
                        for fld in (
                            "AR@100",
                            "AP_small",
                            "AP_medium",
                            "AP_large",
                        ):
                            if fld in extra and extra[fld] is not None:
                                row[fld] = extra[fld]
                    except Exception as e:
                        log.warning(
                            f"[metrics] val teste falhou {model_key}/{bin_name}: {e}"
                        )

            rows.append(row)
            log.info(
                f"[metrics] {MODEL_NAMES[model_key]} | {bin_name} | "
                f"mAP@0.5={row.get('mAP@0.5')} | Recall={row.get('Recall')}"
            )

    return pd.DataFrame(rows)


def save_metrics_table(df: pd.DataFrame, out_path: Path) -> None:
    """Salva tabela de métricas em CSV e JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    df.to_json(out_path.with_suffix(".json"), orient="records", indent=2)
    log.info(f"[metrics] Tabela salva em {out_path.with_suffix('.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta métricas de todos os runs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="results/tables/metrics_all.csv")
    parser.add_argument(
        "--coco-val",
        action="store_true",
        help="Re-roda val() no split teste com best.pt (AR@100, AP S/M/L).",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_root = Path(cfg["paths"].get("results", "results"))
    df = collect_all_metrics(
        results_root, cfg=cfg, coco_val=args.coco_val
    )
    if df.empty:
        log.warning("[metrics] Nenhuma métrica encontrada. Execute os treinamentos primeiro.")
        return
    save_metrics_table(df, Path(args.out))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
