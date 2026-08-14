"""
sahi_inference.py — SAHI na faixa small (YOLOv11m vs RT-DETR-L).

Compara inferencia com e sem SAHI no split de teste small; metricas
alinhadas a evaluate_test.py + McNemar IoU>=0.3 (YOLO+SAHI vs RT-DETR+SAHI).

Uso:
    py -3.11 -m src.evaluation.sahi_inference
    py -3.11 -m src.evaluation.sahi_inference --device cuda:0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.evaluation.evaluate_test import (
    CONF_THR,
    IOU_AP,
    MODEL_LABELS,
    _precision_recall_f1_at_conf,
    resolve_weights,
)
from src.evaluation.per_image_scores import (
    _average_precision_at_iou,
    _list_test_images,
    _load_gt_xyxy,
    _load_model,
    _mean_ap_50_95,
    _parse_split_dirs,
    _predict_batch,
)
from src.evaluation.statistical_tests import (
    IOU_HIT,
    MODEL_DETR,
    MODEL_YOLO,
    _contingency_counts,
    _image_detection_hit,
    _mcnemar_from_contingency,
)
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

log = get_logger("sahi_inference")

BIN_NAME = "small"
SLICE_SIZE = 320
OVERLAP_RATIO = 0.2
NMS_IOU = 0.5
ALPHA_MCNEMAR = 0.05
METRIC_COLS = ["ap50", "ap5095", "precision", "recall", "f1", "detected_iou03"]


def ensure_sahi() -> None:
    try:
        import sahi  # noqa: F401
        log.info(f"[sahi] sahi {sahi.__version__} disponivel")
    except ImportError:
        log.info("[sahi] Instalando sahi...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sahi"])


def _create_sahi_detection_model(weights: Path, model_key: str, device: str):
    from sahi import AutoDetectionModel

    model_type = "yolo11" if model_key == "yolo" else "rtdetr"
    return AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=str(weights),
        confidence_threshold=0.001,
        device=device,
        image_size=640,
    )


def _predictions_from_sahi_result(result) -> Tuple[np.ndarray, np.ndarray]:
    boxes: List[List[float]] = []
    scores: List[float] = []
    for op in result.object_prediction_list:
        x1, y1, x2, y2 = op.bbox.to_xyxy()
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        scores.append(float(op.score.value))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.array(boxes, dtype=np.float64), np.array(scores, dtype=np.float64)


def _predict_sahi_image(
    detection_model,
    image_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=NMS_IOU,
        verbose=0,
    )
    return _predictions_from_sahi_result(result)


def _metrics_row(
    image_id: str,
    model_label: str,
    use_sahi: bool,
    gt: np.ndarray,
    pxy: np.ndarray,
    pcf: np.ndarray,
) -> dict:
    if gt.shape[0] == 0:
        ap50 = ap5095 = prec = rec = f1 = float("nan")
        det03 = 0
    else:
        ap50 = _average_precision_at_iou(pxy, pcf, gt, IOU_AP)
        ap5095 = _mean_ap_50_95(pxy, pcf, gt)
        prec, rec, f1 = _precision_recall_f1_at_conf(pxy, pcf, gt, IOU_AP, CONF_THR)
        det03 = _image_detection_hit(pxy, pcf, gt, IOU_HIT, CONF_THR)
    return {
        "image_id": image_id,
        "model": model_label,
        "sahi": use_sahi,
        "ap50": ap50,
        "ap5095": ap5095,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "detected_iou03": det03,
    }


def _summary_label(model_label: str, use_sahi: bool) -> str:
    tag = "com SAHI" if use_sahi else "sem SAHI"
    return f"{model_label} {tag}"


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    order = [
        (MODEL_YOLO, False),
        (MODEL_YOLO, True),
        (MODEL_DETR, False),
        (MODEL_DETR, True),
    ]
    sub = df[df["ap50"].notna() | (df["detected_iou03"].notna())]
    for model_label, use_sahi in order:
        part = sub[(sub["model"] == model_label) & (sub["sahi"] == use_sahi)]
        if part.empty:
            continue
        row = {
            "config": _summary_label(model_label, use_sahi),
            "model": model_label,
            "sahi": use_sahi,
            "n_images": len(part),
        }
        for c in METRIC_COLS:
            if c == "detected_iou03":
                row["detection_rate_iou03"] = float(part[c].mean())
            else:
                row[c] = float(part[c].mean(numeric_only=True))
        rows.append(row)
    return pd.DataFrame(rows)


def _mcnemar_yolo_sahi_vs_detr_sahi(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    y = df[(df["model"] == MODEL_YOLO) & (df["sahi"] == True)].set_index("image_id")  # noqa: E712
    d = df[(df["model"] == MODEL_DETR) & (df["sahi"] == True)].set_index("image_id")  # noqa: E712
    common = sorted(set(y.index) & set(d.index))
    hy = y.loc[common, "detected_iou03"].to_numpy(dtype=int)
    hd = d.loc[common, "detected_iou03"].to_numpy(dtype=int)
    n11, n12, n21, n22 = _contingency_counts(hd, hy)
    p_val, n_disc, chi2 = _mcnemar_from_contingency(n12, n21)
    sig = p_val is not None and p_val < ALPHA_MCNEMAR
    rate_y = float(np.mean(hy))
    rate_d = float(np.mean(hd))
    winner = MODEL_DETR if sig and rate_d > rate_y else (
        MODEL_YOLO if sig and rate_y > rate_d else "n.s."
    )
    tbl = pd.DataFrame(
        [
            {
                "comparison": "YOLO+SAHI vs RT-DETR+SAHI",
                "bin": BIN_NAME,
                "n_both_pos": n11,
                "n_detr_only": n12,
                "n_yolo_only": n21,
                "n_both_neg": n22,
                "n_discord": n_disc,
                "chi2": chi2,
                "p_value": p_val,
                "significant": sig,
                "winner": winner,
            }
        ]
    )
    matrix = (
        f"\n--- McNemar IoU>=0.3 | YOLO+SAHI vs RT-DETR+SAHI | n={len(common)} ---\n"
        f"                 YOLO+    YOLO-\n"
        f"  DETR+          {n11:5d}    {n12:5d}\n"
        f"  DETR-          {n21:5d}    {n22:5d}\n"
        f"  Taxa deteccao: YOLO+SAHI={rate_y:.4f}  RT-DETR+SAHI={rate_d:.4f}  "
        f"p={p_val:.6f}  sig={sig}  winner={winner}"
    )
    return tbl, matrix


def _conclusion(summary: pd.DataFrame, mcnemar_p: Optional[float]) -> str:
    def _row(model: str, sahi: bool) -> Optional[pd.Series]:
        m = summary[(summary["model"] == model) & (summary["sahi"] == sahi)]
        return m.iloc[0] if len(m) else None

    y0, y1 = _row(MODEL_YOLO, False), _row(MODEL_YOLO, True)
    d0, d1 = _row(MODEL_DETR, False), _row(MODEL_DETR, True)
    if y0 is None or y1 is None or d0 is None or d1 is None:
        return "Conclusao: dados insuficientes para comparar SAHI."

    gap_rec_no = float(d0["recall"]) - float(y0["recall"])
    gap_rec_sahi = float(d1["recall"]) - float(y1["recall"])
    gap_f1_no = float(d0["f1"]) - float(y0["f1"])
    gap_f1_sahi = float(d1["f1"]) - float(y1["f1"])

    y_rec_gain = float(y1["recall"]) - float(y0["recall"])
    y_f1_gain = float(y1["f1"]) - float(y0["f1"])

    sig = mcnemar_p is not None and mcnemar_p < ALPHA_MCNEMAR
    gap_closed = (not sig) and abs(gap_rec_sahi) < abs(gap_rec_no) * 0.5
    gap_reduced = abs(gap_rec_sahi) < abs(gap_rec_no) or abs(gap_f1_sahi) < abs(gap_f1_no)

    if gap_closed:
        verdict = "gap fechou"
    elif gap_reduced and not sig:
        verdict = "gap reduziu (McNemar n.s. apos SAHI)"
    elif gap_reduced:
        verdict = "gap reduziu (McNemar ainda significativo)"
    else:
        verdict = "gap permanece"

    return (
        f"Conclusao: {verdict}.\n"
        f"  Recall: gap RT-DETR-YOLO sem SAHI={gap_rec_no:+.4f} | com SAHI={gap_rec_sahi:+.4f} "
        f"| ganho YOLO com SAHI={y_rec_gain:+.4f}.\n"
        f"  F1: gap sem SAHI={gap_f1_no:+.4f} | com SAHI={gap_f1_sahi:+.4f} "
        f"| ganho YOLO com SAHI={y_f1_gain:+.4f}.\n"
        f"  McNemar YOLO+SAHI vs RT-DETR+SAHI: p={mcnemar_p:.6f} "
        f"({'significativo' if sig else 'n.s.'} em alpha=0.05)."
    )


def run_sahi_evaluation(
    cfg: dict,
    device: str = "cuda:0",
    batch: int = 1,
    train_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_sahi()

    train_root = train_root or Path("runs/train")
    results_root = Path(cfg["paths"].get("results", "runs/detect/results"))
    out_dir = out_dir or Path("results/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    hub = Path(cfg["paths"]["hub_yolo"])
    data_yaml = hub / f"data_{BIN_NAME}.yaml"
    ev = cfg.get("evaluation", {})
    imgsz = int(ev.get("per_image_imgsz", cfg.get("image", {}).get("target_size", 640)))
    img_dir, lab_dir = _parse_split_dirs(data_yaml, "test")
    img_paths = _list_test_images(img_dir)
    if not img_paths:
        raise RuntimeError(f"Sem imagens em {img_dir}")

    log.info(f"[sahi] Teste small: {len(img_paths)} imagens em {img_dir}")
    gts = [_load_gt_xyxy(lab_dir / f"{p.stem}.txt", img_size=imgsz) for p in img_paths]
    use_half = bool(ev.get("per_image_half", True)) and str(device).lower().startswith("cuda")

    all_rows: List[dict] = []

    for model_key in ("yolo", "detr"):
        weights = resolve_weights(model_key, BIN_NAME, train_root, results_root)
        if weights is None:
            raise FileNotFoundError(f"Pesos ausentes: {model_key}/{BIN_NAME}")
        model_label = MODEL_LABELS[model_key]

        log.info(f"[sahi] {model_label} sem SAHI (batch)...")
        umodel = _load_model(model_key, weights)
        pred_xy, pred_cf = _predict_batch(
            umodel, img_paths, device, batch=batch, imgsz=imgsz, half=use_half
        )
        del umodel
        for path, gt, pxy, pcf in zip(img_paths, gts, pred_xy, pred_cf):
            all_rows.append(
                _metrics_row(path.stem, model_label, False, gt, pxy, pcf)
            )

        log.info(f"[sahi] {model_label} com SAHI (slice {SLICE_SIZE}, overlap {OVERLAP_RATIO})...")
        sahi_model = _create_sahi_detection_model(weights, model_key, device)
        for i, (path, gt) in enumerate(zip(img_paths, gts)):
            if (i + 1) % 25 == 0 or i == 0:
                log.info(f"[sahi]   {model_label}+SAHI {i + 1}/{len(img_paths)}")
            pxy, pcf = _predict_sahi_image(sahi_model, path)
            all_rows.append(
                _metrics_row(path.stem, model_label, True, gt, pxy, pcf)
            )
        del sahi_model

    df = pd.DataFrame(all_rows)
    per_path = out_dir / "sahi_small_per_image.csv"
    df.to_csv(per_path, index=False, float_format="%.6f")
    log.info(f"[sahi] Salvo {per_path} ({len(df)} linhas)")

    summary = _build_summary(df)
    sum_path = out_dir / "sahi_small_summary.csv"
    summary.to_csv(sum_path, index=False, float_format="%.6f")

    mc_tbl, mc_matrix = _mcnemar_yolo_sahi_vs_detr_sahi(df)
    p_mc = float(mc_tbl["p_value"].iloc[0]) if len(mc_tbl) else None
    conclusion = _conclusion(summary, p_mc)

    print("\n" + "=" * 80)
    print("Tabela comparativa — faixa SMALL (medias no teste)")
    print("=" * 80)
    disp = summary.copy()
    for c in ("ap50", "ap5095", "precision", "recall", "f1", "detection_rate_iou03"):
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    print(disp.to_string(index=False))
    print(mc_matrix)
    print("\n" + conclusion)

    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SAHI na faixa small")
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
    run_sahi_evaluation(
        cfg,
        device=str(args.device),
        batch=int(args.batch),
        train_root=Path(args.train_root),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
