"""
per_image_scores.py — Scores pareados por imagem para Wilcoxon, McNemar e FROC.

Lê imagens do hub YOLO nos splits configurados em `evaluation.per_image_stat_splits`
(p.ex. só test ou test+val), executa inferência com os best.pt dos runs e calcula
métricas por imagem alinhadas entre RT-DETR-L e YOLOv11m.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
import yaml

from src.evaluation.cross_dataset import find_best_weights
from src.evaluation.metrics import compute_froc
from src.preprocessing.stratify import BIN_ORDER
from src.utils.logger import get_logger

log = get_logger("per_image_scores")


def _cuda_empty_cache() -> None:
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _load_model(model_key: str, weights_path: Path):
    if model_key == "detr":
        from ultralytics import RTDETR

        return RTDETR(str(weights_path))
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def _parse_split_dirs(data_yaml: Path, split_key: str) -> Tuple[Path, Path]:
    """Resolve pastas images/ e labels/ para train|val|test a partir do data_*.yaml."""
    with open(data_yaml, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if split_key not in d:
        raise KeyError(f"Chave '{split_key}' ausente em {data_yaml}")
    hub = Path(d["path"]).resolve()
    rel = d[split_key].replace("\\", "/").strip("/")
    img_dir = hub / rel
    parts = rel.split("/")
    if len(parts) >= 2 and parts[-1] == "images":
        label_rel = "/".join(parts[:-1] + ["labels"])
    else:
        label_rel = rel.replace("images", "labels")
    label_dir = hub / label_rel
    return img_dir, label_dir


def _parse_data_yaml(data_yaml: Path) -> Tuple[Path, Path]:
    """Compat: split de teste (YAML padrão Ultralytics)."""
    return _parse_split_dirs(data_yaml, "test")


def _stat_splits_from_cfg(ev: dict) -> List[str]:
    raw = ev.get("per_image_stat_splits", ["test"])
    if isinstance(raw, str):
        keys = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    else:
        keys = [str(x).strip() for x in raw if str(x).strip()]
    allowed = {"train", "val", "test"}
    out: List[str] = []
    for k in keys:
        if k not in allowed:
            log.warning(f"[per_image] per_image_stat_splits: ignorando '{k}'")
            continue
        if k not in out:
            out.append(k)
    return out or ["test"]


def _gather_images_for_stats(
    data_yaml: Path,
    split_keys: Sequence[str],
    imgsz: int,
) -> Tuple[List[Path], List[np.ndarray], List[str], str, str]:
    """
    Imagens + GT alinhados para inferência estatística.

    Returns:
        img_paths, gts, froc_image_ids (únicos entre splits), split_desc, leak_note
    """
    img_paths: List[Path] = []
    gts: List[np.ndarray] = []
    froc_ids: List[str] = []
    seen_resolved: set[str] = set()
    for sk in split_keys:
        img_dir, lab_dir = _parse_split_dirs(data_yaml, sk)
        for p in _list_test_images(img_dir):
            key = str(p.resolve())
            if key in seen_resolved:
                continue
            seen_resolved.add(key)
            img_paths.append(p)
            gts.append(_load_gt_xyxy(lab_dir / f"{p.stem}.txt", img_size=imgsz))
            froc_ids.append(f"{sk}:{p.stem}")
    split_desc = "+".join(split_keys)
    leak = ""
    if "val" in split_keys:
        leak = (
            "Aviso: inclui split val (usado no early stopping / melhor época). "
            "P-valores têm maior risco de otimismo; use [test] só para hipótese principal."
        )
    return img_paths, gts, froc_ids, split_desc, leak


def _list_test_images(test_img_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        p for p in test_img_dir.rglob("*") if p.suffix.lower() in exts
    )
    return paths


def _yolo_norm_to_xyxy(
    cx: float, cy: float, w: float, h: float, size: int = 640
) -> np.ndarray:
    x1 = (cx - w / 2.0) * size
    y1 = (cy - h / 2.0) * size
    x2 = (cx + w / 2.0) * size
    y2 = (cy + h / 2.0) * size
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _load_gt_xyxy(label_path: Path, img_size: int = 640) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float64)
    boxes: List[np.ndarray] = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append(_yolo_norm_to_xyxy(cx, cy, w, h, img_size))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float64)
    return np.vstack(boxes)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.size == 0 or gt.size == 0:
        return np.zeros((pred.shape[0], gt.shape[0]))
    out = np.zeros((pred.shape[0], gt.shape[0]))
    for i in range(pred.shape[0]):
        for j in range(gt.shape[0]):
            out[i, j] = _iou_xyxy(pred[i], gt[j])
    return out


def _average_precision_at_iou(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float,
) -> float:
    if gt_boxes.shape[0] == 0:
        return float("nan")
    if pred_boxes.shape[0] == 0:
        return 0.0
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    gt_matched = np.zeros(gt_boxes.shape[0], dtype=bool)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    for i in range(len(pred_boxes)):
        ious = _iou_matrix(pred_boxes[i : i + 1], gt_boxes)[0]
        j = int(np.argmax(ious))
        best = float(ious[j])
        if best >= iou_thr and not gt_matched[j]:
            tp[i] = 1.0
            gt_matched[j] = True
        else:
            fp[i] = 1.0
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / gt_boxes.shape[0]
    prec = np.divide(
        tp_c,
        np.maximum(tp_c + fp_c, 1e-9),
        out=np.zeros_like(tp_c),
        where=(tp_c + fp_c) > 0,
    )
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = recall >= t
        if np.any(mask):
            ap += float(np.max(prec[mask]))
    ap /= 101.0
    return ap


def _mean_ap_50_95(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
) -> float:
    if gt_boxes.shape[0] == 0:
        return float("nan")
    ious = np.arange(0.5, 1.0, 0.05)
    vals = [
        _average_precision_at_iou(pred_boxes, pred_scores, gt_boxes, thr)
        for thr in ious
    ]
    return float(np.nanmean(vals))


def _greedy_match_counts(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float,
) -> Tuple[int, int, int]:
    """Retorna (tp, fp, matched_gt_count) com greedy por score."""
    if pred_boxes.shape[0] == 0:
        return 0, 0, 0
    if gt_boxes.shape[0] == 0:
        return 0, int(pred_boxes.shape[0]), 0
    order = np.argsort(-pred_scores)
    gt_matched = np.zeros(gt_boxes.shape[0], dtype=bool)
    tp = fp = 0
    for i in order:
        ious = _iou_matrix(pred_boxes[i : i + 1], gt_boxes)[0]
        j = int(np.argmax(ious))
        best = float(ious[j])
        if best >= iou_thr and not gt_matched[j]:
            tp += 1
            gt_matched[j] = True
        else:
            fp += 1
    return tp, fp, int(gt_matched.sum())


def _recall_image(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float,
) -> float:
    if gt_boxes.shape[0] == 0:
        return float("nan")
    _, _, matched = _greedy_match_counts(pred_boxes, pred_scores, gt_boxes, iou_thr)
    return matched / gt_boxes.shape[0]


def _f1_at_conf(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float,
    conf_thr: float,
) -> float:
    if gt_boxes.shape[0] == 0:
        return float("nan")
    m = pred_scores >= conf_thr
    pb, ps = pred_boxes[m], pred_scores[m]
    tp, fp, matched = _greedy_match_counts(pb, ps, gt_boxes, iou_thr)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = matched / gt_boxes.shape[0]
    if prec + rec <= 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _predict_batch(
    model,
    paths: Sequence[Path],
    device: str,
    batch: int = 2,
    imgsz: int = 640,
    half: bool = False,
):
    """
    Inferência em lote com redução automática de batch em OOM (CUDA ou RAM).
    """
    str_paths = [str(p) for p in paths]
    current = max(1, int(batch))
    half_flag = bool(half) and str(device).lower().startswith("cuda")

    while True:
        xyxy_list: List[np.ndarray] = []
        conf_list: List[np.ndarray] = []
        try:
            for r in model.predict(
                source=str_paths,
                stream=True,
                batch=current,
                conf=0.001,
                device=device,
                verbose=False,
                imgsz=int(imgsz),
                half=half_flag,
            ):
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    conf = r.boxes.conf.cpu().numpy()
                else:
                    xyxy = np.zeros((0, 4), dtype=np.float64)
                    conf = np.zeros(0, dtype=np.float64)
                xyxy_list.append(xyxy)
                conf_list.append(conf)
            return xyxy_list, conf_list
        except Exception as e:
            msg = str(e).lower()
            oom = (
                "out of memory" in msg
                or isinstance(e, MemoryError)
                or "unable to allocate" in msg
            )
            if not oom or current <= 1:
                raise
            nxt = max(1, current // 2)
            log.warning(
                "[per_image] OOM/memória com batch=%s; tentando batch=%s.",
                current,
                nxt,
            )
            _cuda_empty_cache()
            current = nxt


def collect_paired_per_image_metrics(
    cfg: dict,
    bin_name: str,
    results_root: Path,
    iou_ap: float = 0.5,
    iou_hit: float = 0.3,
    conf_f1: float = 0.25,
    batch: int = 2,
) -> Optional[Dict[str, object]]:
    """
    Para um bin, retorna dict com séries pareadas por imagem e artefatos FROC.

    Séries alinhadas por índice da lista de imagens (apenas imagens com GT>0
    entram nas chaves ap50, map5095, recall, f1; listas completas para McNemar
    usam máscara gt_n>0).
    """
    hub = Path(cfg["paths"]["hub_yolo"])
    data_yaml = hub / f"data_{bin_name}.yaml"
    if not data_yaml.exists():
        log.error(f"[per_image] {data_yaml} não encontrado.")
        return None

    w_detr = find_best_weights(results_root, "detr", bin_name)
    w_yolo = find_best_weights(results_root, "yolo", bin_name)
    if w_detr is None or w_yolo is None:
        log.warning(
            f"[per_image] bin={bin_name}: pesos ausentes "
            f"(detr={w_detr is not None}, yolo={w_yolo is not None})."
        )
        return None

    ev = cfg.get("evaluation", {})
    device = str(
        ev.get("per_image_device")
        or cfg.get("hardware", {}).get("device", "cpu")
    ).strip() or "cpu"
    imgsz = int(
        ev.get("per_image_imgsz", cfg.get("image", {}).get("target_size", 640))
    )
    if str(device).lower().startswith("cpu"):
        cpu_cap = int(ev.get("per_image_cpu_max_batch", 1))
        batch = min(int(batch), max(1, cpu_cap))
    splits = _stat_splits_from_cfg(ev)
    try:
        img_paths, gts, froc_ids, split_desc, leak_note = _gather_images_for_stats(
            data_yaml, splits, imgsz
        )
    except KeyError as e:
        log.error(f"[per_image] {e}")
        return None
    if not img_paths:
        log.error(f"[per_image] bin={bin_name}: nenhuma imagem para splits={splits}")
        return None

    use_half = bool(ev.get("per_image_half", True)) and str(device).lower().startswith(
        "cuda"
    )

    # Um modelo por vez: libera VRAM/RAM antes do segundo modelo.
    log.info(
        f"[per_image] bin={bin_name} | device={device} | batch={batch} | "
        f"splits={split_desc} | n_infer={len(img_paths)} | inferência DETR…"
    )
    model_detr = _load_model("detr", w_detr)
    det_xy, det_cf = _predict_batch(
        model_detr, img_paths, device, batch=batch, imgsz=imgsz, half=use_half
    )
    del model_detr
    _cuda_empty_cache()

    log.info(f"[per_image] bin={bin_name} | inferência YOLO…")
    model_yolo = _load_model("yolo", w_yolo)
    yolo_xy, yolo_cf = _predict_batch(
        model_yolo, img_paths, device, batch=batch, imgsz=imgsz, half=use_half
    )
    del model_yolo
    _cuda_empty_cache()

    ap50_d, ap5095_d, rec_d, f1_d = [], [], [], []
    ap50_y, ap5095_y, rec_y, f1_y = [], [], [], []
    hit_d, hit_y = [], []

    preds_d_froc: List[dict] = []
    preds_y_froc: List[dict] = []
    gts_froc: List[dict] = []

    for i, fid in enumerate(froc_ids):
        gt = gts[i]
        for row in gt:
            gts_froc.append({"image_id": fid, "bbox": row.tolist()})

        dxy, dcf = det_xy[i], det_cf[i]
        yxy, ycf = yolo_xy[i], yolo_cf[i]

        for b, s in zip(dxy, dcf):
            preds_d_froc.append(
                {"image_id": fid, "bbox": b.tolist(), "score": float(s)}
            )
        for b, s in zip(yxy, ycf):
            preds_y_froc.append(
                {"image_id": fid, "bbox": b.tolist(), "score": float(s)}
            )

        if gt.shape[0] == 0:
            continue

        ap50_d.append(_average_precision_at_iou(dxy, dcf, gt, iou_ap))
        ap5095_d.append(_mean_ap_50_95(dxy, dcf, gt))
        rec_d.append(_recall_image(dxy, dcf, gt, iou_ap))
        f1_d.append(_f1_at_conf(dxy, dcf, gt, iou_ap, conf_f1))

        ap50_y.append(_average_precision_at_iou(yxy, ycf, gt, iou_ap))
        ap5095_y.append(_mean_ap_50_95(yxy, ycf, gt))
        rec_y.append(_recall_image(yxy, ycf, gt, iou_ap))
        f1_y.append(_f1_at_conf(yxy, ycf, gt, iou_ap, conf_f1))

        md, _, _ = _greedy_match_counts(dxy, dcf, gt, iou_hit)
        my, _, _ = _greedy_match_counts(yxy, ycf, gt, iou_hit)
        hit_d.append(1.0 if md >= 1 else 0.0)
        hit_y.append(1.0 if my >= 1 else 0.0)

    n_img = len(img_paths)
    froc_d = compute_froc(
        preds_d_froc, gts_froc, iou_threshold=iou_hit, n_images=n_img
    )
    froc_y = compute_froc(
        preds_y_froc, gts_froc, iou_threshold=iou_hit, n_images=n_img
    )

    return {
        "ap50_detr": np.array(ap50_d, dtype=np.float64),
        "ap5095_detr": np.array(ap5095_d, dtype=np.float64),
        "recall_detr": np.array(rec_d, dtype=np.float64),
        "f1_detr": np.array(f1_d, dtype=np.float64),
        "ap50_yolo": np.array(ap50_y, dtype=np.float64),
        "ap5095_yolo": np.array(ap5095_y, dtype=np.float64),
        "recall_yolo": np.array(rec_y, dtype=np.float64),
        "f1_yolo": np.array(f1_y, dtype=np.float64),
        "hit_detr": np.array(hit_d, dtype=np.float64),
        "hit_yolo": np.array(hit_y, dtype=np.float64),
        "froc_detr": froc_d,
        "froc_yolo": froc_y,
        "n_images_test": n_img,
        "n_images_with_gt": len(ap50_d),
        "stat_split_desc": split_desc,
        "n_infer_total": len(img_paths),
        "stat_leak_warning": leak_note,
    }


def _paired_mask(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m], m


def safe_wilcoxon(
    a: np.ndarray, b: np.ndarray
) -> Tuple[Optional[float], Optional[float]]:
    aa, bb, _ = _paired_mask(a, b)
    if len(aa) < 6:
        return None, None
    diff = aa - bb
    if np.allclose(diff, 0.0, atol=1e-12):
        return 0.0, 1.0
    from scipy import stats

    try:
        wstat, p = stats.wilcoxon(
            aa,
            bb,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
    except TypeError:
        wstat, p = stats.wilcoxon(aa, bb, alternative="two-sided", zero_method="wilcox")
    except ValueError as e:
        log.warning(f"[per_image] Wilcoxon indisponível: {e}")
        return None, None
    return float(wstat), float(p)


def wilcoxon_block_for_bin(
    pack: Dict[str, object],
) -> List[Tuple[str, Optional[float], int]]:
    """
    Retorna lista (nome_métrica, p_valor, n_pares) para um bin.
    """
    pairs = [
        ("mAP@0.5", "ap50_detr", "ap50_yolo"),
        ("mAP@0.5:0.95", "ap5095_detr", "ap5095_yolo"),
        ("Recall", "recall_detr", "recall_yolo"),
        ("F1", "f1_detr", "f1_yolo"),
    ]
    out: List[Tuple[str, Optional[float], int]] = []
    for label, kd, ky in pairs:
        a = pack[kd]  # type: ignore[index]
        b = pack[ky]  # type: ignore[index]
        aa, bb, _ = _paired_mask(a, b)  # type: ignore[arg-type]
        _, p = safe_wilcoxon(a, b)  # type: ignore[arg-type]
        out.append((label, p, int(len(aa))))
    return out


def mcnemar_for_bin(pack: Dict[str, object]) -> Tuple[Optional[float], int, str]:
    from scipy import stats

    hd = pack["hit_detr"].astype(int)  # type: ignore[union-attr]
    hy = pack["hit_yolo"].astype(int)  # type: ignore[union-attr]
    if len(hd) != len(hy) or len(hd) < 1:
        return None, 0, "sem pares"
    n11 = int(np.sum((hd == 1) & (hy == 1)))
    n12 = int(np.sum((hd == 1) & (hy == 0)))
    n21 = int(np.sum((hd == 0) & (hy == 1)))
    n22 = int(np.sum((hd == 0) & (hy == 0)))
    disc = n12 + n21
    if disc == 0:
        return 1.0, disc, "sem discordâncias"
    if disc < 25:
        chi2 = (abs(n12 - n21) - 1) ** 2 / (n12 + n21)
    else:
        chi2 = (n12 - n21) ** 2 / (n12 + n21)
    p = float(1 - stats.chi2.cdf(chi2, df=1))
    return p, disc, f"n11={n11} n12={n12} n21={n21} n22={n22}"


def run_per_image_evaluation_all_bins(
    cfg: dict,
    results_root: Path,
    bins: Optional[Sequence[str]] = None,
    batch: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    ev = cfg.get("evaluation", {})
    eff_batch = (
        int(batch)
        if batch is not None
        else int(ev.get("per_image_inference_batch", 2))
    )
    iou_ap = float(ev.get("iou_per_image_ap", 0.5))
    iou_hit = float(ev.get("iou_threshold_fusion", 0.3))
    conf_f1 = float(ev.get("conf_f1_per_image", 0.25))

    bins = list(bins) if bins is not None else list(BIN_ORDER)
    out: Dict[str, Dict[str, object]] = {}
    for bn in bins:
        pack = collect_paired_per_image_metrics(
            cfg,
            bn,
            results_root,
            iou_ap=iou_ap,
            iou_hit=iou_hit,
            conf_f1=conf_f1,
            batch=eff_batch,
        )
        if pack is not None:
            out[bn] = pack
    return out


def save_froc_tables(
    per_bin: Dict[str, Dict[str, object]], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    for bn, pack in per_bin.items():
        for model_key in ("detr", "yolo"):
            key = f"froc_{model_key}"
            if key not in pack:
                continue
            path = out_dir / f"froc_{model_key}_{bn}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(pack[key], f, indent=2)  # type: ignore[arg-type]
            log.info(f"[per_image] FROC salvo em {path}")
