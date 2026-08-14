"""
statistical_tests.py — Wilcoxon e McNemar a partir de test_per_image.csv.

Lê results/evaluation/test_per_image.csv (evaluate_test.py), pareia por
image_id × bin e aplica correção de Bonferroni (k=12 Wilcoxon, k=3 McNemar).

McNemar IoU>=0.3 (--mcnemar-iou): inferência nos best.pt e critério clínico
conf>=0.25, IoU>=0.3 contra GT (split teste).

Uso:
    py -3.11 -m src.evaluation.statistical_tests
    py -3.11 -m src.evaluation.statistical_tests --mcnemar-iou --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.evaluation.evaluate_test import resolve_weights
from src.evaluation.per_image_scores import (
    _greedy_match_counts,
    _list_test_images,
    _load_gt_xyxy,
    _load_model,
    _parse_split_dirs,
    _predict_batch,
    safe_wilcoxon,
)
from src.preprocessing.stratify import BIN_ORDER
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

log = get_logger("statistical_tests")

ALPHA = 0.05
K_WILCOXON = 12  # 4 métricas × 3 faixas
K_MCNEMAR = 3
ALPHA_WILCOXON = ALPHA / K_WILCOXON
ALPHA_MCNEMAR = ALPHA / K_MCNEMAR

MODEL_DETR = "RT-DETR-L"
MODEL_YOLO = "YOLOv11m"
MODEL_KEYS = ("yolo", "detr")
WILCOXON_METRICS = ["ap50", "ap5095", "recall", "f1"]
IOU_HIT = 0.3
CONF_HIT = 0.25


def _bonferroni_p(p: Optional[float], k: int) -> Optional[float]:
    if p is None:
        return None
    return float(min(p * k, 1.0))


def _winner_from_means(
    mean_a: float, mean_b: float, significant: bool, label_a: str, label_b: str
) -> str:
    if not significant:
        return "n.s."
    if mean_a > mean_b:
        return label_a
    if mean_b > mean_a:
        return label_b
    return "tie"


def _pair_test_table(df: pd.DataFrame) -> pd.DataFrame:
    """Inner join YOLO vs DETR por (image_id, bin)."""
    d = df[df["model"] == MODEL_DETR].copy()
    y = df[df["model"] == MODEL_YOLO].copy()
    keys = ["image_id", "bin"]
    paired = d.merge(
        y,
        on=keys,
        suffixes=("_detr", "_yolo"),
        how="inner",
    )
    log.info(f"[statistical_tests] Pares image×bin: {len(paired)}")
    return paired


def run_wilcoxon_tests(paired: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for bin_name in BIN_ORDER:
        sub = paired[paired["bin"] == bin_name]
        if sub.empty:
            log.warning(f"[statistical_tests] Wilcoxon: sem pares em bin={bin_name}")
            continue
        for metric in WILCOXON_METRICS:
            col_d = f"{metric}_detr"
            col_y = f"{metric}_yolo"
            a = sub[col_d].to_numpy(dtype=np.float64)
            b = sub[col_y].to_numpy(dtype=np.float64)
            m = np.isfinite(a) & np.isfinite(b)
            a, b = a[m], b[m]
            stat, p_val = safe_wilcoxon(a, b)
            p_corr = _bonferroni_p(p_val, K_WILCOXON)
            sig = p_val is not None and p_val < ALPHA_WILCOXON
            mean_d = float(np.nanmean(a)) if len(a) else float("nan")
            mean_y = float(np.nanmean(b)) if len(b) else float("nan")
            winner = _winner_from_means(
                mean_d, mean_y, sig, MODEL_DETR, MODEL_YOLO
            )
            rows.append(
                {
                    "bin": bin_name,
                    "metric": metric,
                    "stat": stat,
                    "p_value": p_val,
                    "p_corrected": p_corr,
                    "significant": sig,
                    "winner": winner,
                }
            )
    return pd.DataFrame(rows)


def _contingency_counts(
    hit_detr: np.ndarray, hit_yolo: np.ndarray
) -> Tuple[int, int, int, int]:
    """n11 both+, n12 detr_only, n21 yolo_only, n22 both-."""
    hd = hit_detr.astype(int)
    hy = hit_yolo.astype(int)
    n11 = int(np.sum((hd == 1) & (hy == 1)))
    n12 = int(np.sum((hd == 1) & (hy == 0)))
    n21 = int(np.sum((hd == 0) & (hy == 1)))
    n22 = int(np.sum((hd == 0) & (hy == 0)))
    return n11, n12, n21, n22


def _mcnemar_from_contingency(
    n12: int, n21: int,
) -> Tuple[Optional[float], int, Optional[float]]:
    disc = n12 + n21
    if disc == 0:
        return 1.0, disc, 0.0
    if disc < 25:
        chi2 = (abs(n12 - n21) - 1) ** 2 / disc
    else:
        chi2 = (n12 - n21) ** 2 / disc
    p = float(1 - stats.chi2.cdf(chi2, df=1))
    return p, disc, float(chi2)


def _mcnemar_pvalue(hit_detr: np.ndarray, hit_yolo: np.ndarray) -> Tuple[Optional[float], int, Optional[float]]:
    _, n12, n21, _ = _contingency_counts(hit_detr, hit_yolo)
    return _mcnemar_from_contingency(n12, n21)


def _image_detection_hit(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float = IOU_HIT,
    conf_thr: float = CONF_HIT,
) -> int:
    """1 se existe predicao (conf>=thr) com IoU>=thr contra algum GT."""
    if gt_boxes.shape[0] == 0:
        return 0
    m = pred_scores >= conf_thr
    pb, ps = pred_boxes[m], pred_scores[m]
    if pb.shape[0] == 0:
        return 0
    _, _, matched = _greedy_match_counts(pb, ps, gt_boxes, iou_thr)
    return 1 if matched >= 1 else 0


def _collect_hits_for_model(
    cfg: dict,
    model_key: str,
    bin_name: str,
    weights: Path,
    device: str,
    batch: int,
) -> Dict[str, int]:
    """{image_stem: 0|1} no split teste, apenas imagens com GT>0."""
    hub = Path(cfg["paths"]["hub_yolo"])
    data_yaml = hub / f"data_{bin_name}.yaml"
    ev = cfg.get("evaluation", {})
    imgsz = int(ev.get("per_image_imgsz", cfg.get("image", {}).get("target_size", 640)))
    img_dir, lab_dir = _parse_split_dirs(data_yaml, "test")
    img_paths = _list_test_images(img_dir)
    if not img_paths:
        raise RuntimeError(f"Sem imagens de teste em {img_dir}")

    gts = [_load_gt_xyxy(lab_dir / f"{p.stem}.txt", img_size=imgsz) for p in img_paths]
    use_half = bool(ev.get("per_image_half", True)) and str(device).lower().startswith("cuda")

    log.info(
        f"[statistical_tests] Inferencia {model_key}/{bin_name} | "
        f"n={len(img_paths)} | {weights.name}"
    )
    model = _load_model(model_key, weights)
    pred_xy, pred_cf = _predict_batch(
        model, img_paths, device, batch=batch, imgsz=imgsz, half=use_half
    )
    del model

    hits: Dict[str, int] = {}
    for path, gt, pxy, pcf in zip(img_paths, gts, pred_xy, pred_cf):
        if gt.shape[0] == 0:
            continue
        hits[path.stem] = _image_detection_hit(pxy, pcf, gt, IOU_HIT, CONF_HIT)
    return hits


def run_mcnemar_iou03(
    cfg: dict,
    device: str = "cuda:0",
    batch: int = 1,
    train_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """McNemar com deteccao binaria IoU>=0.3, conf>=0.25 (inferencia nos best.pt)."""
    train_root = train_root or Path("runs/train")
    results_root = Path(cfg["paths"].get("results", "runs/detect/results"))
    out_dir = out_dir or Path("results/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    matrices: List[str] = []

    for bin_name in BIN_ORDER:
        hits: Dict[str, Dict[str, int]] = {}
        for model_key in MODEL_KEYS:
            weights = resolve_weights(model_key, bin_name, train_root, results_root)
            if weights is None:
                raise FileNotFoundError(f"Pesos ausentes: {model_key}/{bin_name}")
            hits[model_key] = _collect_hits_for_model(
                cfg, model_key, bin_name, weights, device, batch
            )

        common_ids = sorted(set(hits["detr"]) & set(hits["yolo"]))
        if not common_ids:
            log.warning(f"[statistical_tests] McNemar IoU: sem pares em {bin_name}")
            continue

        hd = np.array([hits["detr"][i] for i in common_ids], dtype=int)
        hy = np.array([hits["yolo"][i] for i in common_ids], dtype=int)
        n11, n12, n21, n22 = _contingency_counts(hd, hy)
        p_val, n_disc, chi2 = _mcnemar_from_contingency(n12, n21)
        p_corr = _bonferroni_p(p_val, K_MCNEMAR)
        sig = p_val is not None and p_val < ALPHA_MCNEMAR
        rate_d = float(np.mean(hd))
        rate_y = float(np.mean(hy))
        winner = _winner_from_means(rate_d, rate_y, sig, MODEL_DETR, MODEL_YOLO)

        rows.append(
            {
                "bin": bin_name,
                "n_both_pos": n11,
                "n_detr_only": n12,
                "n_yolo_only": n21,
                "n_both_neg": n22,
                "n_discord": n_disc,
                "chi2": chi2,
                "p_value": p_val,
                "p_corrected": p_corr,
                "significant": sig,
                "winner": winner,
            }
        )
        matrices.append(
            f"\n--- Matriz 2x2 | bin={bin_name} | n={len(common_ids)} imagens (GT>0) ---\n"
            f"                 YOLO+    YOLO-\n"
            f"  DETR+ (n11)    {n11:5d}    {n12:5d}   <- detr_only={n12}\n"
            f"  DETR- (n21)    {n21:5d}    {n22:5d}   <- yolo_only={n21}, both_neg={n22}\n"
            f"  Taxa deteccao: DETR={rate_d:.4f}  YOLO={rate_y:.4f}  |  "
            f"discordantes={n_disc}  p={p_val:.6f}  sig={sig}  winner={winner}"
        )

    df = pd.DataFrame(rows)
    out_path = out_dir / "mcnemar_iou03_results.csv"
    df.to_csv(out_path, index=False)
    log.info(f"[statistical_tests] Salvo {out_path}")

    print("\n" + "=" * 72)
    print("McNemar IoU>=0.3 (conf>=0.25) — inferencia nos best.pt")
    print("=" * 72)
    if not df.empty:
        out = df.copy()
        for col in ("chi2", "p_value", "p_corrected"):
            if col in out.columns:
                out[col] = out[col].apply(
                    lambda x: f"{x:.6f}" if x is not None and pd.notna(x) else ""
                )
        print(out.to_string(index=False))
    print(f"\nBonferroni McNemar alpha'={ALPHA_MCNEMAR:.6f} (k={K_MCNEMAR})")
    for block in matrices:
        print(block)

    return df


def run_mcnemar_tests(paired: pd.DataFrame) -> pd.DataFrame:
    """
    Detecção binária: positivo se ap50 > 0 (proxy de ≥1 acerto em AP@0.5).
    """
    rows: List[dict] = []
    for bin_name in BIN_ORDER:
        sub = paired[paired["bin"] == bin_name]
        if sub.empty:
            continue
        hit_d = (sub["ap50_detr"].to_numpy(dtype=np.float64) > 0).astype(int)
        hit_y = (sub["ap50_yolo"].to_numpy(dtype=np.float64) > 0).astype(int)
        p_val, n_disc, chi2 = _mcnemar_pvalue(hit_d, hit_y)
        p_corr = _bonferroni_p(p_val, K_MCNEMAR)
        sig = p_val is not None and p_val < ALPHA_MCNEMAR
        rate_d = float(np.mean(hit_d))
        rate_y = float(np.mean(hit_y))
        winner = _winner_from_means(rate_d, rate_y, sig, MODEL_DETR, MODEL_YOLO)
        rows.append(
            {
                "bin": bin_name,
                "n_discord": n_disc,
                "chi2": chi2,
                "p_value": p_val,
                "p_corrected": p_corr,
                "significant": sig,
                "winner": winner,
            }
        )
    return pd.DataFrame(rows)


def _format_table(df: pd.DataFrame, title: str) -> str:
    lines = [f"\n{'=' * 72}", title, "=" * 72]
    if df.empty:
        lines.append("(vazio)")
    else:
        out = df.copy()
        for col in ("stat", "p_value", "p_corrected", "chi2"):
            if col in out.columns:
                out[col] = out[col].apply(
                    lambda x: f"{x:.6f}" if x is not None and pd.notna(x) else ""
                )
        lines.append(out.to_string(index=False))
    lines.append(
        f"\nBonferroni: Wilcoxon alpha'={ALPHA_WILCOXON:.6f} (k={K_WILCOXON}) | "
        f"McNemar alpha'={ALPHA_MCNEMAR:.6f} (k={K_MCNEMAR})"
    )
    return "\n".join(lines)


def run_all(
    per_image_csv: Path,
    out_dir: Path,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_image_csv)
    required = {"image_id", "model", "bin", "ap50", "ap5095", "recall", "f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {per_image_csv}: {missing}")

    paired = _pair_test_table(df)
    df_w = run_wilcoxon_tests(paired)
    df_m = run_mcnemar_tests(paired)

    path_w = out_dir / "wilcoxon_results.csv"
    path_m = out_dir / "mcnemar_results.csv"
    df_w.to_csv(path_w, index=False)
    df_m.to_csv(path_m, index=False)
    log.info(f"[statistical_tests] Salvo {path_w}")
    log.info(f"[statistical_tests] Salvo {path_m}")

    print(_format_table(df_w, "Wilcoxon (pareado por image_id × bin)"))
    print(_format_table(df_m, "McNemar (detecção: ap50 > 0)"))

    return df_w, df_m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Testes estatísticos a partir de test_per_image.csv"
    )
    parser.add_argument(
        "--input",
        default="results/evaluation/test_per_image.csv",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out-dir", default="results/evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mcnemar-iou",
        action="store_true",
        help="McNemar com IoU>=0.3 via inferencia nos best.pt (split teste)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--train-root", default="runs/train")
    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.mcnemar_iou:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        run_mcnemar_iou03(
            cfg,
            device=str(args.device),
            batch=int(args.batch),
            train_root=Path(args.train_root),
            out_dir=Path(args.out_dir),
        )
        return

    run_all(Path(args.input), Path(args.out_dir), seed=args.seed)


if __name__ == "__main__":
    main()
