"""
evaluate_test.py — Avaliação formal no split de teste (plano v3).

Carrega best.pt (YOLOv11m / RT-DETR-L × small/medium/large), infere no teste
de cada faixa e grava métricas por imagem e resumo agregado.

Uso:
    py -3.11 -m src.evaluation.evaluate_test
    py -3.11 -m src.evaluation.evaluate_test --config config.yaml --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.evaluation.cross_dataset import find_best_weights
from src.evaluation.per_image_scores import (
    _average_precision_at_iou,
    _greedy_match_counts,
    _list_test_images,
    _load_gt_xyxy,
    _load_model,
    _mean_ap_50_95,
    _parse_split_dirs,
    _predict_batch,
)
from src.preprocessing.stratify import BIN_ORDER
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

log = get_logger("evaluate_test")

MODEL_KEYS = ("yolo", "detr")
MODEL_LABELS = {"yolo": "YOLOv11m", "detr": "RT-DETR-L"}
IOU_AP = 0.5
CONF_THR = 0.25


def _precision_recall_f1_at_conf(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float,
    conf_thr: float,
) -> Tuple[float, float, float]:
    if gt_boxes.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    m = pred_scores >= conf_thr
    pb, ps = pred_boxes[m], pred_scores[m]
    tp, fp, matched = _greedy_match_counts(pb, ps, gt_boxes, iou_thr)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = matched / gt_boxes.shape[0]
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def resolve_weights(
    model_key: str,
    bin_name: str,
    train_root: Path,
    results_root: Optional[Path],
) -> Optional[Path]:
    """
    Pesos: runs/train/{modelo}_{faixa}/weights/best.pt (prompt) com fallback
    em paths.results (runs/detect/results/...).
    """
    primary = train_root / f"{model_key}_{bin_name}" / "weights" / "best.pt"
    if primary.is_file():
        return primary
    if results_root is not None:
        w = find_best_weights(results_root, model_key, bin_name)
        if w is not None and w.is_file():
            log.info(
                f"[evaluate_test] {model_key}/{bin_name}: usando {w} "
                f"(runs/train ausente)."
            )
            return w
    log.error(f"[evaluate_test] best.pt não encontrado para {model_key}/{bin_name}")
    return None


def evaluate_model_bin(
    cfg: dict,
    model_key: str,
    bin_name: str,
    weights: Path,
    device: str,
    batch: int,
) -> List[dict]:
    hub = Path(cfg["paths"]["hub_yolo"])
    data_yaml = hub / f"data_{bin_name}.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"YAML ausente: {data_yaml}")

    ev = cfg.get("evaluation", {})
    imgsz = int(ev.get("per_image_imgsz", cfg.get("image", {}).get("target_size", 640)))
    img_dir, lab_dir = _parse_split_dirs(data_yaml, "test")
    img_paths = _list_test_images(img_dir)
    if not img_paths:
        raise RuntimeError(f"Nenhuma imagem de teste em {img_dir}")

    use_half = bool(ev.get("per_image_half", True)) and device.lower().startswith("cuda")
    gts = [_load_gt_xyxy(lab_dir / f"{p.stem}.txt", img_size=imgsz) for p in img_paths]

    log.info(
        f"[evaluate_test] {MODEL_LABELS[model_key]} | {bin_name} | "
        f"n={len(img_paths)} | {weights.name}"
    )
    model = _load_model(model_key, weights)
    pred_xy, pred_cf = _predict_batch(
        model, img_paths, device, batch=batch, imgsz=imgsz, half=use_half
    )
    del model

    rows: List[dict] = []
    for path, gt, pxy, pcf in zip(img_paths, gts, pred_xy, pred_cf):
        if gt.shape[0] == 0:
            ap50 = ap5095 = prec = rec = f1 = float("nan")
        else:
            ap50 = _average_precision_at_iou(pxy, pcf, gt, IOU_AP)
            ap5095 = _mean_ap_50_95(pxy, pcf, gt)
            prec, rec, f1 = _precision_recall_f1_at_conf(
                pxy, pcf, gt, IOU_AP, CONF_THR
            )
        rows.append(
            {
                "image_id": path.stem,
                "model": MODEL_LABELS[model_key],
                "bin": bin_name,
                "ap50": ap50,
                "ap5095": ap5095,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )
    return rows


def run_evaluation(
    cfg: dict,
    device: str = "cuda:0",
    batch: int = 1,
    train_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_global_seed(int(cfg.get("hardware", {}).get("seed", cfg.get("training", {}).get("seed", 42))))

    train_root = train_root or Path("runs/train")
    results_root = Path(cfg["paths"].get("results", "runs/detect/results"))
    out_dir = out_dir or Path("results/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_rows: List[dict] = []
    for bin_name in BIN_ORDER:
        for model_key in MODEL_KEYS:
            weights = resolve_weights(model_key, bin_name, train_root, results_root)
            if weights is None:
                continue
            per_image_rows.extend(
                evaluate_model_bin(cfg, model_key, bin_name, weights, device, batch)
            )

    if not per_image_rows:
        raise RuntimeError("Nenhuma métrica calculada — verifique pesos e hub.")

    df_img = pd.DataFrame(per_image_rows)
    per_image_path = out_dir / "test_per_image.csv"
    df_img.to_csv(per_image_path, index=False, float_format="%.6f")
    log.info(f"[evaluate_test] Salvo {per_image_path} ({len(df_img)} linhas)")

    counts = df_img.groupby(["model", "bin"]).size().reset_index(name="n_images")
    df_sum = (
        df_img.groupby(["model", "bin"], as_index=False)[
            ["ap50", "ap5095", "precision", "recall", "f1"]
        ]
        .mean(numeric_only=True)
        .merge(counts, on=["model", "bin"])
    )

    summary_path = out_dir / "test_summary.csv"
    col_order = [
        "model",
        "bin",
        "n_images",
        "ap50",
        "ap5095",
        "precision",
        "recall",
        "f1",
    ]
    df_sum = df_sum[col_order]
    df_sum.to_csv(summary_path, index=False, float_format="%.6f")
    log.info(f"[evaluate_test] Salvo {summary_path}")

    return df_img, df_sum


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avaliação formal no split de teste (métricas por imagem)."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--device",
        default=None,
        help="cuda:0 | cpu (default: hardware.device do config)",
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--train-root",
        default="runs/train",
        help="runs/train/{yolo|detr}_{faixa}/weights/best.pt",
    )
    parser.add_argument("--out-dir", default="results/evaluation")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = (
        args.device
        or cfg.get("hardware", {}).get("device", "cuda:0")
    )
    ev = cfg.get("evaluation", {})
    batch = (
        args.batch
        if args.batch is not None
        else int(ev.get("per_image_inference_batch", 1))
    )

    _, df_sum = run_evaluation(
        cfg,
        device=str(device),
        batch=int(batch),
        train_root=Path(args.train_root),
        out_dir=Path(args.out_dir),
    )
    print("\n=== test_summary.csv ===\n")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
