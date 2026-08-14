"""
froc.py — Curvas FROC por faixa (YOLOv11m vs RT-DETR-L).

Inferencia no teste com conf=0.001, match IoU>=0.3, apenas imagens com GT>0.

Uso:
    py -3.11 -m src.evaluation.froc
    py -3.11 -m src.evaluation.froc --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluation.evaluate_test import MODEL_LABELS, resolve_weights
from src.evaluation.metrics import _compute_iou
from src.evaluation.per_image_scores import (
    _list_test_images,
    _load_gt_xyxy,
    _load_model,
    _parse_split_dirs,
    _predict_batch,
)
from src.preprocessing.stratify import BIN_ORDER
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

log = get_logger("froc")

IOU_MATCH = 0.3
CONF_PRED = 0.001
OPERATIONAL_FP = [0.125, 0.25, 0.5, 1.0, 2.0]
PRINT_FP = [0.5, 1.0]
COLORS = {"yolo": "#1f77b4", "detr": "#d62728"}


def _gather_preds_gts(
    img_paths: List[Path],
    gts_xyxy: List[np.ndarray],
    pred_xy: List[np.ndarray],
    pred_cf: List[np.ndarray],
) -> Tuple[List[dict], List[dict], int]:
    """Apenas imagens com GT>0 (estudo sem negativos)."""
    predictions: List[dict] = []
    ground_truths: List[dict] = []
    n_pos_images = 0
    for path, gt, pxy, pcf in zip(img_paths, gts_xyxy, pred_xy, pred_cf):
        if gt.shape[0] == 0:
            continue
        n_pos_images += 1
        iid = path.stem
        for row in gt:
            ground_truths.append({"image_id": iid, "bbox": row.tolist()})
        for b, s in zip(pxy, pcf):
            predictions.append(
                {"image_id": iid, "bbox": b.tolist(), "score": float(s)}
            )
    return predictions, ground_truths, n_pos_images


def _full_froc_points(
    predictions: List[dict],
    ground_truths: List[dict],
    n_images: int,
    iou_threshold: float = IOU_MATCH,
) -> List[Tuple[float, float]]:
    """Todos os pontos (fp/img, sensibilidade) ao varrer conf decrescente."""
    gt_by_image: Dict[str, List] = {}
    for gt in ground_truths:
        gt_by_image.setdefault(gt["image_id"], []).append(gt["bbox"])
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0 or n_images == 0:
        return []

    sorted_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)
    gt_matched: Dict[str, List[bool]] = {
        img_id: [False] * len(bboxes)
        for img_id, bboxes in gt_by_image.items()
    }
    tp_list: List[int] = []
    fp_list: List[int] = []
    for pred in sorted_preds:
        img_id = pred["image_id"]
        pred_bbox = pred["bbox"]
        gt_bboxes = gt_by_image.get(img_id, [])
        matched = gt_matched.get(img_id, [])
        best_iou, best_idx = 0.0, -1
        for idx, gt_bbox in enumerate(gt_bboxes):
            if matched[idx]:
                continue
            iou = _compute_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_iou >= iou_threshold and best_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[img_id][best_idx] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    sensitivity = tp_cum / total_gt
    fp_per_image = fp_cum / n_images

    points = [(0.0, 0.0)]
    for fp, sens in zip(fp_per_image, sensitivity):
        points.append((float(fp), float(sens)))
    # Remover duplicatas de fp mantendo max sensibilidade
    by_fp: Dict[float, float] = {}
    for fp, sens in points:
        by_fp[fp] = max(by_fp.get(fp, 0.0), sens)
    return sorted(by_fp.items(), key=lambda x: x[0])


def _interpolate_sensitivity(
    curve: List[Tuple[float, float]], fp_target: float
) -> float:
    if not curve:
        return 0.0
    fps = np.array([p[0] for p in curve])
    sens = np.array([p[1] for p in curve])
    idxs = np.where(fps <= fp_target)[0]
    if len(idxs) == 0:
        return 0.0
    return float(sens[idxs[-1]])


def run_froc_for_bin(
    cfg: dict,
    model_key: str,
    bin_name: str,
    weights: Path,
    device: str,
    batch: int,
) -> Tuple[List[Tuple[float, float]], Dict[float, float], int]:
    hub = Path(cfg["paths"]["hub_yolo"])
    data_yaml = hub / f"data_{bin_name}.yaml"
    ev = cfg.get("evaluation", {})
    imgsz = int(ev.get("per_image_imgsz", cfg.get("image", {}).get("target_size", 640)))
    img_dir, lab_dir = _parse_split_dirs(data_yaml, "test")
    img_paths = _list_test_images(img_dir)
    gts = [_load_gt_xyxy(lab_dir / f"{p.stem}.txt", img_size=imgsz) for p in img_paths]
    use_half = bool(ev.get("per_image_half", True)) and str(device).lower().startswith("cuda")

    log.info(
        f"[froc] {MODEL_LABELS[model_key]} | {bin_name} | "
        f"n_test={len(img_paths)} | inferencia..."
    )
    model = _load_model(model_key, weights)
    pred_xy, pred_cf = _predict_batch(
        model,
        img_paths,
        device,
        batch=batch,
        imgsz=imgsz,
        half=use_half,
    )
    del model

    preds, gts_list, n_pos = _gather_preds_gts(img_paths, gts, pred_xy, pred_cf)
    curve = _full_froc_points(preds, gts_list, n_pos, IOU_MATCH)
    sens_at = {fp: _interpolate_sensitivity(curve, fp) for fp in OPERATIONAL_FP}
    log.info(
        f"[froc] {MODEL_LABELS[model_key]}/{bin_name} | n_pos={n_pos} | "
        f"GT={len(gts_list)} | sens@0.5={sens_at[0.5]:.4f} | sens@1.0={sens_at[1.0]:.4f}"
    )
    return curve, sens_at, n_pos


def _plot_froc_curves(
    curves: Dict[str, Dict[str, List[Tuple[float, float]]]],
    n_images: Dict[str, int],
    out_png: Path,
    out_pdf: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, bin_name in zip(axes, BIN_ORDER):
        for model_key, label, color in (
            ("yolo", "YOLOv11m", COLORS["yolo"]),
            ("detr", "RT-DETR-L", COLORS["detr"]),
        ):
            pts = curves[bin_name][model_key]
            if not pts:
                continue
            fps = [p[0] for p in pts if p[0] <= 4.0]
            sens = [p[1] for p in pts if p[0] <= 4.0]
            ax.plot(fps, sens, color=color, label=label, linewidth=2)
        for xv in (0.5, 1.0):
            ax.axvline(xv, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        n = n_images.get(bin_name, 0)
        ax.set_title(f"{bin_name.capitalize()} (n={n} imagens positivas)")
        ax.set_xlim(0, 4.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("FP por imagem")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Sensibilidade")
    fig.suptitle("Curvas FROC — split de teste (IoU >= 0.3)", fontsize=12, y=1.02)
    fig.text(
        0.5,
        -0.02,
        "FROC calculado sobre imagens positivas do teste — ausencia de imagens "
        "negativas (ver limitacoes)",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def run_all(
    cfg: dict,
    device: str = "cuda:0",
    batch: int = 1,
    train_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    train_root = train_root or Path("runs/train")
    results_root = Path(cfg["paths"].get("results", "runs/detect/results"))
    out_dir = out_dir or Path("results/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: List[dict] = []
    op_rows: List[dict] = []
    curves: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
        b: {} for b in BIN_ORDER
    }
    n_pos_by_bin: Dict[str, int] = {}

    for bin_name in BIN_ORDER:
        n_pos_by_bin[bin_name] = 0
        for model_key in ("yolo", "detr"):
            weights = resolve_weights(model_key, bin_name, train_root, results_root)
            if weights is None:
                raise FileNotFoundError(f"Pesos ausentes: {model_key}/{bin_name}")
            curve, sens_at, n_pos = run_froc_for_bin(
                cfg, model_key, bin_name, weights, device, batch
            )
            curves[bin_name][model_key] = curve
            n_pos_by_bin[bin_name] = n_pos
            model_label = MODEL_LABELS[model_key]
            for fp, sens in curve:
                csv_rows.append(
                    {
                        "model": model_label,
                        "bin": bin_name,
                        "fp_per_img": fp,
                        "sensitivity": sens,
                    }
                )
            for fp in OPERATIONAL_FP:
                op_rows.append(
                    {
                        "model": model_label,
                        "bin": bin_name,
                        "fp_per_img": fp,
                        "sensitivity": sens_at[fp],
                    }
                )

    df_curve = pd.DataFrame(csv_rows)
    df_op = pd.DataFrame(op_rows)
    curve_path = out_dir / "froc_values.csv"
    df_curve.to_csv(curve_path, index=False, float_format="%.6f")
    log.info(f"[froc] Salvo {curve_path} ({len(df_curve)} linhas)")

    op_path = out_dir / "froc_operational_points.csv"
    df_op.to_csv(op_path, index=False, float_format="%.6f")

    png_path = out_dir / "froc_curves.png"
    pdf_path = out_dir / "froc_curves.pdf"
    _plot_froc_curves(curves, n_pos_by_bin, png_path, pdf_path)
    log.info(f"[froc] Salvo {png_path} e {pdf_path}")

    print("\n" + "=" * 72)
    print("Sensibilidade nos pontos operacionais (FP/img)")
    print("=" * 72)
    for fp_show in PRINT_FP:
        print(f"\n--- FP/img = {fp_show} ---")
        sub = df_op[df_op["fp_per_img"] == fp_show][
            ["model", "bin", "sensitivity"]
        ].copy()
        sub["sensitivity"] = sub["sensitivity"].map(lambda x: f"{x:.4f}")
        print(sub.to_string(index=False))

    print("\n--- Pontos operacionais completos (0.125 a 2.0) ---")
    pivot = df_op.pivot_table(
        index=["model", "bin"],
        columns="fp_per_img",
        values="sensitivity",
    )
    print(pivot.round(4).to_string())

    print(f"\nArquivos: {curve_path}")
    print(f"          {png_path}")
    print(f"          {pdf_path}")
    return df_op


def main() -> None:
    parser = argparse.ArgumentParser(description="Curvas FROC por faixa")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--train-root", default="runs/train")
    parser.add_argument("--out-dir", default="results/evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_global_seed(args.seed)
    run_all(
        cfg,
        device=str(args.device),
        batch=int(args.batch),
        train_root=Path(args.train_root),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
