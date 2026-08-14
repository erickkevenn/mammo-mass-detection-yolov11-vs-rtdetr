"""
heatmaps.py — Grad-CAM (YOLOv11m) e mapas de atenção (RT-DETR-L) por faixa.

Seleciona 3 casos por faixa (TP ambos, discordante DETR+, FP YOLO) e gera painéis.

Uso:
    py -3.11 -m src.evaluation.heatmaps --all --device cuda:0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from matplotlib.patches import Rectangle
from ultralytics.nn.modules.transformer import MSDeformAttn

from src.evaluation.evaluate_test import MODEL_LABELS, resolve_weights
from src.evaluation.per_image_scores import (
    _greedy_match_counts,
    _iou_matrix,
    _list_test_images,
    _load_gt_xyxy,
    _load_model,
    _parse_split_dirs,
    _predict_batch,
)
from src.evaluation.statistical_tests import CONF_HIT, IOU_HIT
from src.preprocessing.stratify import BIN_ORDER
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

log = get_logger("heatmaps")

CONF_THR = 0.25
IOU_THR = IOU_HIT
MODEL_YOLO = "YOLOv11m"
MODEL_DETR = "RT-DETR-L"
CASE_TITLES = {
    1: "Caso 1: TP ambos",
    2: "Caso 2: RT-DETR detecta / YOLO nao",
    3: "Caso 3: FP representativo YOLO",
}


@dataclass
class ImageMeta:
    image_id: str
    path: Path
    gt: np.ndarray
    yolo_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    yolo_cf: np.ndarray = field(default_factory=lambda: np.zeros(0))
    detr_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    detr_cf: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hit_yolo: int = 0
    hit_detr: int = 0

    @property
    def mean_conf_both(self) -> float:
        if self.yolo_cf.size == 0 and self.detr_cf.size == 0:
            return 0.0
        return float(np.mean(np.concatenate([self.yolo_cf, self.detr_cf])))


class YOLOWrapper(torch.nn.Module):
    """Wrapper documentado para integração com pytorch-grad-cam (inferência completa)."""

    def __init__(self, yolo) -> None:
        super().__init__()
        self.model = yolo.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out["scores"].max()
        if isinstance(out, (list, tuple)):
            return out[0].max()
        return out.max()


def ensure_grad_cam() -> None:
    try:
        import pytorch_grad_cam  # noqa: F401
    except ImportError:
        log.info("[heatmaps] Instalando grad-cam...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "grad-cam"])


def find_yolo_gradcam_layer(yolo_model) -> Tuple[str, torch.nn.Module]:
    convs = [
        (n, m)
        for n, m in yolo_model.named_modules()
        if isinstance(m, torch.nn.Conv2d)
        and n.startswith("model.")
        and int(n.split(".")[1]) <= 10
    ]
    if not convs:
        raise RuntimeError("Nenhuma Conv2d no backbone YOLO encontrada.")
    name, layer = convs[-1]
    log.info(f"[heatmaps] Grad-CAM target YOLO: {name} (indice backbone <= 10)")
    return name, layer


def find_detr_attn_layer(detr_model) -> Tuple[str, torch.nn.Module, str]:
    cross = [
        (n, m)
        for n, m in detr_model.named_modules()
        if isinstance(m, MSDeformAttn)
    ]
    attn_named = [
        (n, m)
        for n, m in detr_model.named_modules()
        if "CrossAttention" in m.__class__.__name__
        or "MultiheadAttention" in m.__class__.__name__
    ]
    note = ""
    if cross:
        name, layer = cross[-1]
        note = f"MSDeformAttn (cross-attention deformavel); {len(attn_named)} modulos MHA listados"
    elif attn_named:
        name, layer = attn_named[-1]
        note = f"Fallback MultiheadAttention: {name}"
    else:
        all_mod = list(detr_model.named_modules())
        name, layer = all_mod[-1]
        note = f"Fallback ultimo modulo: {name} ({layer.__class__.__name__})"
        log.warning("[heatmaps] CrossAttention nao encontrado. %s", note)
    log.info(f"[heatmaps] Attention target RT-DETR: {name} | {note}")
    return name, layer, note


def load_image_rgb(path: Path, size: int = 640) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
    return rgb


def image_to_tensor(rgb: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return t.to(device).requires_grad_(True)


def gradcam_yolo(
    yolo,
    layer: torch.nn.Module,
    tensor: torch.Tensor,
    layer_name: str,
) -> np.ndarray:
    """
    Grad-CAM no backbone (camadas 0-10) com gradiente de feat.max() ate a Conv2d alvo.
    Compatível com o grafo Ultralytics (sem passar pelo head).
    """
    conv_act: Dict[str, torch.Tensor] = {}
    layer.register_forward_hook(lambda _m, _i, o: conv_act.update(t=o))

    with torch.enable_grad():
        x = tensor
        feat = None
        for i, mod in enumerate(yolo.model.model):
            x = mod(x)
            if i == 10:
                feat = x
                break
        if feat is None:
            raise RuntimeError("Backbone YOLO nao terminou na camada 10.")
        grad, = torch.autograd.grad(
            feat.max(),
            conv_act["t"],
            retain_graph=False,
            allow_unused=False,
        )

    act = conv_act["t"][0].detach().cpu().numpy()
    g = grad[0].detach().cpu().numpy()
    w = g.mean(axis=(1, 2), keepdims=True)
    cam = np.maximum((w * act).sum(axis=0), 0)
    cam = cv2.resize(cam, (tensor.shape[3], tensor.shape[2]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    log.debug(f"[heatmaps] Grad-CAM {layer_name}: max={cam.max():.4f}")
    return cam.astype(np.float32)


@dataclass
class AttnExtractMeta:
    attn_shape: Tuple[int, ...]
    side: Optional[int]
    method: str


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def _finalize_attention_map(attn_2d: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Resize bicúbico + suavização progressiva (elimina grid, preserva localização)."""
    if attn_2d.shape[:2] != (img_h, img_w):
        attn_resized = cv2.resize(
            attn_2d.astype(np.float32),
            (img_w, img_h),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        attn_resized = attn_2d.astype(np.float32)
    attn_blurred = cv2.GaussianBlur(attn_resized, (31, 31), sigmaX=8.0)
    attn_blurred = cv2.GaussianBlur(attn_blurred, (15, 15), sigmaX=3.0)
    return _normalize_map(attn_blurred)


def _splat_gaussian(
    grid: np.ndarray, cx: float, cy: float, weight: float, sigma: float
) -> None:
    h, w = grid.shape
    radius = max(1, int(3 * sigma))
    x0, x1 = max(0, int(cx) - radius), min(w, int(cx) + radius + 1)
    y0, y1 = max(0, int(cy) - radius), min(h, int(cy) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    grid[y0:y1, x0:x1] += weight * g


def _msdeform_sampling_locations(
    module: MSDeformAttn,
    query: torch.Tensor,
    refer_bbox: torch.Tensor,
    value_shapes: list,
) -> torch.Tensor:
    bs, len_q = query.shape[:2]
    sampling_offsets = module.sampling_offsets(query).view(
        bs, len_q, module.n_heads, module.n_levels, module.n_points, 2
    )
    offset_normalizer = torch.as_tensor(
        value_shapes, dtype=query.dtype, device=query.device
    ).flip(-1)
    num_points = refer_bbox.shape[-1]
    if num_points == 2:
        add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        return refer_bbox[:, :, None, :, None, :] + add
    add = (
        sampling_offsets
        / module.n_points
        * refer_bbox[:, :, None, :, None, 2:]
        * 0.5
    )
    return refer_bbox[:, :, None, :, None, :2] + add


def _heatmap_msdeform_per_scale(
    attn_q: np.ndarray,
    locs_q: np.ndarray,
    shapes: List[Tuple[int, int]],
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Processa P3/P4/P5 separadamente; combina por média (sem concatenar keys)."""
    # attn_q: [heads, levels, points]; locs_q: [heads, levels, points, 2]
    level_maps: List[np.ndarray] = []
    for l, (lh, lw) in enumerate(shapes):
        level_map = np.zeros((lh, lw), dtype=np.float32)
        sigma = max(1.5, min(lh, lw) / 16.0)
        for hi in range(attn_q.shape[0]):
            for pi in range(attn_q.shape[2]):
                w = float(attn_q[hi, l, pi])
                if w <= 0:
                    continue
                loc = locs_q[hi, l, pi]
                cx = float(np.clip(loc[0], 0.0, 1.0)) * max(lw - 1, 0)
                cy = float(np.clip(loc[1], 0.0, 1.0)) * max(lh - 1, 0)
                _splat_gaussian(level_map, cx, cy, w, sigma)
        level_map = cv2.GaussianBlur(level_map, (5, 5), sigmaX=1.0)
        level_maps.append(
            cv2.resize(level_map, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        )
    merged = np.mean(np.stack(level_maps, axis=0), axis=0)
    return _normalize_map(merged)


def _heatmap_mha_square(attn: np.ndarray, img_h: int, img_w: int) -> Tuple[np.ndarray, int]:
    """Reshape quadrado para pesos MHA (batch, heads, queries, keys)."""
    attn_map = attn.mean(0).numpy()  # média sobre heads -> (queries, keys)
    if attn_map.ndim == 1:
        attn_map = attn_map[None, :]
    row = attn_map[0] if attn_map.shape[0] == 1 else attn_map.mean(0)
    n_keys = row.shape[0]
    side = int(np.sqrt(n_keys))
    if side * side != n_keys:
        row = row[: side * side]
    attn_2d = row.reshape(side, side)
    return _normalize_map(attn_2d), side


def eigen_cam_detr(detr, cross_attn_layer: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """Fallback EigenCAM na cross_attn (PCA do activation map, sem pesos MSDeform)."""
    ensure_grad_cam()
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    class DetrScalarWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.model(x)
            if isinstance(out, tuple):
                return out[0].max()
            return out.max()

    wrapper = DetrScalarWrapper(detr.model.float())
    cam = EigenCAM(model=wrapper, target_layers=[cross_attn_layer])
    inp = tensor.float().detach()
    grayscale = cam(input_tensor=inp, targets=[ClassifierOutputTarget(0)])[0]
    return _finalize_attention_map(grayscale, tensor.shape[2], tensor.shape[3])


def extract_detr_attention(
    detr,
    tensor: torch.Tensor,
    device: str,
    conf_threshold: float = 0.25,
    cross_attn_layer: Optional[torch.nn.Module] = None,
    pred_cf: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, object, AttnExtractMeta]:
    """
    Extrai mapa de atenção RT-DETR: hooks em MHA/MSDeformAttn;
    MSDeformAttn usa splat por escala (P3/P4/P5); fallback EigenCAM.
    """
    attention_maps: List[Tuple[int, Union[torch.Tensor, str], torch.Tensor]] = []
    msdeform_by_name: Dict[str, Dict[str, object]] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(scale_idx: int, module: torch.nn.Module, name: str):
        def hook_fn(_module, input, output) -> None:
            if isinstance(module, MSDeformAttn):
                query, refer_bbox, _value, value_shapes = input[0], input[1], input[2], input[3]
                bs, len_q = query.shape[:2]
                aw = module.attention_weights(query).view(
                    bs, len_q, module.n_heads, module.n_levels * module.n_points
                )
                aw = F.softmax(aw, dim=-1).view(
                    bs, len_q, module.n_heads, module.n_levels, module.n_points
                )
                locs = _msdeform_sampling_locations(
                    module, query, refer_bbox, value_shapes
                )
                msdeform_by_name[name] = {
                    "attn": aw.detach().cpu(),
                    "locs": locs.detach().cpu(),
                    "shapes": [tuple(int(x) for x in s) for s in value_shapes],
                }
                return
            if isinstance(output, tuple) and len(output) > 1:
                w = output[1]
                if w is not None:
                    attention_maps.append((scale_idx, name, w.detach().cpu()))

        return hook_fn

    attn_modules = [
        (n, m)
        for n, m in detr.model.named_modules()
        if any(
            t in m.__class__.__name__
            for t in ("MultiheadAttention", "MSDeformAttn", "CrossAttention")
        )
    ]
    for idx, (name, module) in enumerate(attn_modules):
        hooks.append(
            module.register_forward_hook(make_hook(idx, module, name))
        )

    inp = tensor.float()
    detr.model.float().to(device)
    with torch.no_grad():
        results = detr.predict(inp, verbose=False, conf=conf_threshold)

    for h in hooks:
        h.remove()

    img_h, img_w = tensor.shape[2], tensor.shape[3]
    confs = None
    if results and getattr(results[0], "boxes", None) is not None:
        confs = results[0].boxes.conf
    if confs is not None and len(confs) > 0:
        q_idx = int(confs.argmax())
    elif pred_cf is not None and pred_cf.size > 0:
        q_idx = int(np.argmax(pred_cf))
    else:
        q_idx = 0

    cross_names = [n for n in msdeform_by_name if n.endswith("cross_attn")]
    if cross_names:
        last_name = cross_names[-1]
        pack = msdeform_by_name[last_name]
        attn_t = pack["attn"]  # type: ignore[index]
        locs_t = pack["locs"]  # type: ignore[index]
        shapes = pack["shapes"]  # type: ignore[index]
        q_idx = min(q_idx, attn_t.shape[1] - 1)
        attn_q = attn_t[0, q_idx].numpy()
        locs_q = locs_t[0, q_idx].numpy()
        raw = _heatmap_msdeform_per_scale(attn_q, locs_q, shapes, img_h, img_w)
        attn_final = _finalize_attention_map(raw, img_h, img_w)
        meta = AttnExtractMeta(
            attn_shape=tuple(attn_t.shape),
            side=None,
            method=f"msdeform_per_scale:{last_name}",
        )
        return attn_final, results, meta

    if attention_maps:
        _, last_name, attn = attention_maps[-1]
        if attn.ndim == 4:
            q_idx = min(q_idx, attn.shape[2] - 1)
            attn_slice = attn[0, :, q_idx, :]
            n_keys = attn_slice.shape[-1]
            side = int(np.sqrt(n_keys))
            if side * side != n_keys:
                side = int(np.sqrt(n_keys))
                attn_map = attn_slice.mean(0).numpy()[: side * side]
            else:
                attn_map = attn_slice.mean(0).numpy()
            attn_2d, side_used = (
                attn_map.reshape(side, side),
                side,
            )
            attn_2d = _normalize_map(attn_2d)
            attn_final = _finalize_attention_map(attn_2d, img_h, img_w)
            meta = AttnExtractMeta(
                attn_shape=tuple(attn.shape),
                side=side_used,
                method=f"mha_square:{last_name}",
            )
            return attn_final, results, meta

    if cross_attn_layer is None:
        _, cross_attn_layer, _ = find_detr_attn_layer(detr.model)

    log.warning(
        "[heatmaps] Pesos MSDeform/MHA indisponiveis; fallback EigenCAM em %s",
        cross_attn_layer,
    )
    attn_final = eigen_cam_detr(detr, cross_attn_layer, tensor)
    meta = AttnExtractMeta(
        attn_shape=(0,),
        side=None,
        method="eigen_cam:cross_attn",
    )
    return attn_final, results, meta


def attention_map_detr(
    detr,
    attn_layer: torch.nn.Module,
    tensor: torch.Tensor,
    pred_xy: np.ndarray,
    pred_cf: np.ndarray,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, AttnExtractMeta]:
    attn, _results, meta = extract_detr_attention(
        detr,
        tensor,
        device=device,
        pred_cf=pred_cf,
        cross_attn_layer=attn_layer,
    )
    return attn, meta


def overlay_jet(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    heat_u8 = np.uint8(255 * np.clip(heat, 0, 1))
    jet = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    out = (alpha * jet + (1 - alpha) * gray_rgb).astype(np.uint8)
    return out


def filter_preds(
    xy: np.ndarray, cf: np.ndarray, conf: float = CONF_THR
) -> Tuple[np.ndarray, np.ndarray]:
    if xy.size == 0:
        return xy, cf
    m = cf >= conf
    return xy[m], cf[m]


def detection_hit(xy: np.ndarray, cf: np.ndarray, gt: np.ndarray) -> int:
    pb, ps = filter_preds(xy, cf)
    if gt.shape[0] == 0 or pb.shape[0] == 0:
        return 0
    _, _, matched = _greedy_match_counts(pb, ps, gt, IOU_THR)
    return 1 if matched >= 1 else 0


def yolo_fp_score(xy: np.ndarray, cf: np.ndarray, gt: np.ndarray) -> float:
    pb, ps = filter_preds(xy, cf)
    if pb.shape[0] == 0 or gt.shape[0] == 0:
        return 0.0
    best = 0.0
    for i in range(pb.shape[0]):
        ious = _iou_matrix(pb[i : i + 1], gt)[0]
        if float(ious.max()) < IOU_THR:
            best = max(best, float(ps[i]))
    return best


def build_metadata_bin(
    cfg: dict,
    bin_name: str,
    device: str,
    batch: int,
    train_root: Path,
    results_root: Path,
) -> Dict[str, ImageMeta]:
    hub = Path(cfg["paths"]["hub_yolo"])
    data_yaml = hub / f"data_{bin_name}.yaml"
    ev = cfg.get("evaluation", {})
    imgsz = int(ev.get("per_image_imgsz", cfg.get("image", {}).get("target_size", 640)))
    img_dir, lab_dir = _parse_split_dirs(data_yaml, "test")
    paths = _list_test_images(img_dir)
    gts = [_load_gt_xyxy(lab_dir / f"{p.stem}.txt", img_size=imgsz) for p in paths]

    w_y = resolve_weights("yolo", bin_name, train_root, results_root)
    w_d = resolve_weights("detr", bin_name, train_root, results_root)
    if w_y is None or w_d is None:
        raise FileNotFoundError(f"Pesos ausentes para bin={bin_name}")

    use_half = bool(ev.get("per_image_half", True)) and device.lower().startswith("cuda")
    my = _load_model("yolo", w_y)
    md = _load_model("detr", w_d)
    y_xy, y_cf = _predict_batch(my, paths, device, batch=batch, imgsz=imgsz, half=use_half)
    d_xy, d_cf = _predict_batch(md, paths, device, batch=batch, imgsz=imgsz, half=use_half)
    del my, md

    meta: Dict[str, ImageMeta] = {}
    for path, gt, yx, yc, dx, dc in zip(paths, gts, y_xy, y_cf, d_xy, d_cf):
        if gt.shape[0] == 0:
            continue
        m = ImageMeta(
            image_id=path.stem,
            path=path,
            gt=gt,
            yolo_xy=yx,
            yolo_cf=yc,
            detr_xy=dx,
            detr_cf=dc,
        )
        m.hit_yolo = detection_hit(yx, yc, gt)
        m.hit_detr = detection_hit(dx, dc, gt)
        meta[m.image_id] = m
    return meta


def select_cases(meta: Dict[str, ImageMeta]) -> Dict[int, ImageMeta]:
    items = list(meta.values())

    tp_both = [m for m in items if m.hit_yolo and m.hit_detr]
    tp_both.sort(key=lambda m: m.mean_conf_both, reverse=True)

    detr_only = [m for m in items if m.hit_detr and not m.hit_yolo]
    detr_only.sort(
        key=lambda m: float(filter_preds(m.detr_xy, m.detr_cf)[1].max())
        if filter_preds(m.detr_xy, m.detr_cf)[1].size
        else 0.0,
        reverse=True,
    )

    fp_yolo = [m for m in items if yolo_fp_score(m.yolo_xy, m.yolo_cf, m.gt) > 0]
    fp_yolo.sort(
        key=lambda m: yolo_fp_score(m.yolo_xy, m.yolo_cf, m.gt), reverse=True
    )

    selected: Dict[int, ImageMeta] = {}
    if tp_both:
        selected[1] = tp_both[0]
    if detr_only:
        selected[2] = detr_only[0]
    if fp_yolo:
        selected[3] = fp_yolo[0]

    for cid, pool, key_fn in [
        (1, tp_both, lambda m: m.mean_conf_both),
        (2, detr_only, lambda m: m.hit_detr),
        (3, fp_yolo, lambda m: yolo_fp_score(m.yolo_xy, m.yolo_cf, m.gt)),
    ]:
        if cid in selected or not pool:
            continue
        pool.sort(key=key_fn, reverse=True)
        selected[cid] = pool[0]
    return selected


def draw_boxes(
    ax,
    gt: np.ndarray,
    yolo_xy: np.ndarray,
    yolo_cf: np.ndarray,
    detr_xy: np.ndarray,
    detr_cf: np.ndarray,
    show_gt: bool = True,
    show_yolo: bool = True,
    show_detr: bool = True,
) -> None:
    if show_gt:
        for box in gt:
            x1, y1, x2, y2 = box
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1.5,
                    edgecolor="lime",
                    facecolor="none",
                    linestyle="--",
                )
            )
    yb, ys = filter_preds(yolo_xy, yolo_cf)
    if show_yolo:
        for box, sc in zip(yb, ys):
            x1, y1, x2, y2 = box
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1.5,
                    edgecolor="#1f77b4",
                    facecolor="none",
                )
            )
            ax.text(x1, max(0, y1 - 4), f"{sc:.2f}", color="#1f77b4", fontsize=7)
    db, ds = filter_preds(detr_xy, detr_cf)
    if show_detr:
        for box, sc in zip(db, ds):
            x1, y1, x2, y2 = box
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1.5,
                    edgecolor="#d62728",
                    facecolor="none",
                )
            )
            ax.text(x1, max(0, y1 - 14), f"{sc:.2f}", color="#d62728", fontsize=7)


def save_panel(
    out_path: Path,
    rgb: np.ndarray,
    grad_overlay: np.ndarray,
    attn_overlay: np.ndarray,
    meta: ImageMeta,
    bin_name: str,
    case_id: int,
    orig_passthrough: Optional[np.ndarray] = None,
    grad_passthrough: Optional[np.ndarray] = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    title = f"{bin_name.capitalize()} — {CASE_TITLES[case_id]}"
    fig.suptitle(title, fontsize=12)

    orig_img = orig_passthrough if orig_passthrough is not None else rgb
    grad_img = grad_passthrough if grad_passthrough is not None else grad_overlay
    panels = [
        (orig_img, "Original"),
        (grad_img, "Grad-CAM YOLOv11m"),
        (attn_overlay, "Attention RT-DETR-L"),
    ]
    for ax, (img, subt) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(subt, fontsize=10)
        ax.axis("off")
        if subt != "Original":
            plt.colorbar(
                plt.cm.ScalarMappable(cmap="jet"),
                ax=ax,
                fraction=0.046,
                pad=0.04,
            )
        if subt == "Original" and orig_passthrough is None:
            draw_boxes(ax, meta.gt, meta.yolo_xy, meta.yolo_cf, meta.detr_xy, meta.detr_cf)
        elif subt.startswith("Grad") and grad_passthrough is None:
            draw_boxes(
                ax, meta.gt, meta.yolo_xy, meta.yolo_cf, meta.detr_xy, meta.detr_cf,
                show_yolo=True, show_detr=False,
            )
        elif subt.startswith("Attention"):
            draw_boxes(
                ax, meta.gt, meta.yolo_xy, meta.yolo_cf, meta.detr_xy, meta.detr_cf,
                show_yolo=False, show_detr=True,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_panel_path(panel_path: Path) -> Tuple[str, int, str]:
    """Ex.: results/heatmaps/small/caso1_cdd_P280_L_DM_MLO_painel.png"""
    bin_name = panel_path.parent.name
    stem = panel_path.stem
    if not stem.endswith("_painel"):
        raise ValueError(f"Nome de painel invalido: {panel_path.name}")
    core = stem[: -len("_painel")]
    if not core.startswith("caso") or "_" not in core:
        raise ValueError(f"Nome de painel invalido: {panel_path.name}")
    case_part, image_id = core.split("_", 1)
    case_id = int(case_part.replace("caso", ""))
    return bin_name, case_id, image_id


def find_image_path(img_dir: Path, image_id: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = img_dir / f"{image_id}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"Imagem nao encontrada: {image_id} em {img_dir}")


def load_image_meta_single(
    cfg: dict,
    bin_name: str,
    image_id: str,
    device: str,
    batch: int,
    train_root: Path,
    results_root: Path,
    yolo=None,
    detr=None,
    use_half: bool = True,
) -> ImageMeta:
    hub = Path(cfg["paths"]["hub_yolo"])
    data_yaml = hub / f"data_{bin_name}.yaml"
    ev = cfg.get("evaluation", {})
    imgsz = int(ev.get("per_image_imgsz", cfg.get("image", {}).get("target_size", 640)))
    img_dir, lab_dir = _parse_split_dirs(data_yaml, "test")
    path = find_image_path(img_dir, image_id)
    gt = _load_gt_xyxy(lab_dir / f"{path.stem}.txt", img_size=imgsz)

    if yolo is None:
        w_y = resolve_weights("yolo", bin_name, train_root, results_root)
        yolo = _load_model("yolo", w_y)
        yolo.model.to(device).eval()
    if detr is None:
        w_d = resolve_weights("detr", bin_name, train_root, results_root)
        detr = _load_model("detr", w_d)
        detr.model.to(device).eval()

    eff_half = use_half and bool(ev.get("per_image_half", True)) and str(
        device
    ).lower().startswith("cuda")
    y_xy, y_cf = _predict_batch(
        yolo, [path], device, batch=batch, imgsz=imgsz, half=eff_half
    )
    d_xy, d_cf = _predict_batch(
        detr, [path], device, batch=batch, imgsz=imgsz, half=eff_half
    )

    meta = ImageMeta(
        image_id=path.stem,
        path=path,
        gt=gt,
        yolo_xy=y_xy[0],
        yolo_cf=y_cf[0],
        detr_xy=d_xy[0],
        detr_cf=d_cf[0],
    )
    meta.hit_yolo = detection_hit(meta.yolo_xy, meta.yolo_cf, meta.gt)
    meta.hit_detr = detection_hit(meta.detr_xy, meta.detr_cf, meta.gt)
    return meta


def _read_rgb_png(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def regen_attention_only(
    cfg: dict,
    device: str = "cuda:0",
    batch: int = 1,
    train_root: Optional[Path] = None,
    out_root: Optional[Path] = None,
) -> Tuple[List[Path], List[AttnExtractMeta]]:
    """Regenera apenas attention RT-DETR (individuais *_attn.png e paineis)."""
    set_global_seed(int(cfg.get("hardware", {}).get("seed", 42)))
    train_root = train_root or Path("runs/train")
    results_root = Path(cfg["paths"].get("results", "runs/detect/results"))
    out_root = out_root or Path("results/heatmaps")
    ind_dir = out_root / "individual"
    ind_dir.mkdir(parents=True, exist_ok=True)

    panel_paths = sorted(
        p for p in out_root.glob("*/*_painel.png") if p.parent.name in BIN_ORDER
    )
    if not panel_paths:
        raise FileNotFoundError(f"Nenhum painel em {out_root}")

    overwritten: List[Path] = []
    extract_metas: List[AttnExtractMeta] = []
    detr_by_bin: Dict[str, object] = {}
    attn_layer_by_bin: Dict[str, torch.nn.Module] = {}

    for panel_path in panel_paths:
        bin_name, case_id, image_id = parse_panel_path(panel_path)
        stem = f"{bin_name}_caso{case_id}_{image_id}"
        p_grad = ind_dir / f"{stem}_gradcam.png"
        p_orig = ind_dir / f"{stem}_original.png"
        p_attn = ind_dir / f"{stem}_attn.png"

        if bin_name not in detr_by_bin:
            w_d = resolve_weights("detr", bin_name, train_root, results_root)
            detr = _load_model("detr", w_d)
            detr.model.to(device).eval().float()
            detr_by_bin[bin_name] = detr
            _, attn_layer_by_bin[bin_name], _ = find_detr_attn_layer(detr.model)

        detr = detr_by_bin[bin_name]
        attn_layer = attn_layer_by_bin[bin_name]

        imeta = load_image_meta_single(
            cfg,
            bin_name,
            image_id,
            device,
            batch,
            train_root,
            results_root,
            detr=detr,
            use_half=False,
        )
        rgb = load_image_rgb(imeta.path)
        tensor_d = image_to_tensor(rgb, device)
        attn, attn_meta = attention_map_detr(
            detr, attn_layer, tensor_d, imeta.detr_xy, imeta.detr_cf, device=device
        )
        attn_overlay = overlay_jet(rgb, attn)
        log.info(
            "[heatmaps] attn %s shape=%s side=%s",
            attn_meta.method,
            attn_meta.attn_shape,
            attn_meta.side,
        )

        yb, ys = filter_preds(imeta.yolo_xy, imeta.yolo_cf)
        db, ds = filter_preds(imeta.detr_xy, imeta.detr_cf)

        drawn_attn = _draw_on_rgb(attn_overlay, imeta.gt, yb, ys, db, ds, mode="detr")
        cv2.imwrite(str(p_attn), cv2.cvtColor(drawn_attn, cv2.COLOR_RGB2BGR))
        overwritten.append(p_attn)

        orig_pt = _read_rgb_png(p_orig) if p_orig.is_file() else None
        grad_pt = _read_rgb_png(p_grad) if p_grad.is_file() else None
        save_panel(
            panel_path,
            rgb,
            grad_overlay=rgb,
            attn_overlay=attn_overlay,
            meta=imeta,
            bin_name=bin_name,
            case_id=case_id,
            orig_passthrough=orig_pt,
            grad_passthrough=grad_pt,
        )
        overwritten.append(panel_path)
        extract_metas.append(attn_meta)

        log.info(f"[heatmaps] regen-attn: {panel_path.name}")

    if extract_metas:
        last = extract_metas[-1]
        print("\n" + "=" * 72)
        print("RT-DETR attention — diagnostico (ultima imagem)")
        print("=" * 72)
        print(f"  Metodo: {last.method}")
        print(f"  Tensor capturado (batch, heads, queries, keys): {last.attn_shape}")
        print(f"  Side no reshape: {last.side if last.side is not None else 'N/A (msdeform por escala)'}")
        print(
            "  Padrao em blocos: esperado eliminado com splat P3/P4/P5 separados "
            "(inspecionar paineis *_attn.png)"
        )

    for detr in detr_by_bin.values():
        del detr
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return overwritten, extract_metas


def case_justification(case_id: int, meta: ImageMeta) -> str:
    if case_id == 1:
        return (
            f"TP ambos (IoU>={IOU_THR}, conf>={CONF_THR}); "
            f"media conf={meta.mean_conf_both:.3f}"
        )
    if case_id == 2:
        dc = float(filter_preds(meta.detr_xy, meta.detr_cf)[1].max()) if meta.detr_cf.size else 0
        return f"McNemar detr_only; conf RT-DETR max={dc:.3f}"
    sc = yolo_fp_score(meta.yolo_xy, meta.yolo_cf, meta.gt)
    return f"YOLO FP (conf>={CONF_THR}, IoU<{IOU_THR}); conf FP={sc:.3f}"


def run_heatmaps(
    cfg: dict,
    device: str = "cuda:0",
    batch: int = 1,
    train_root: Optional[Path] = None,
    out_root: Optional[Path] = None,
    bins: Optional[List[str]] = None,
) -> None:
    ensure_grad_cam()
    set_global_seed(int(cfg.get("hardware", {}).get("seed", 42)))

    train_root = train_root or Path("runs/train")
    results_root = Path(cfg["paths"].get("results", "runs/detect/results"))
    out_root = out_root or Path("results/heatmaps")
    (out_root / "individual").mkdir(parents=True, exist_ok=True)
    bins = bins or list(BIN_ORDER)

    layer_log: List[str] = []
    selection_log: List[str] = []
    saved_files: List[Path] = []

    for bin_name in bins:
        log.info(f"[heatmaps] === Faixa {bin_name} ===")
        meta = build_metadata_bin(cfg, bin_name, device, batch, train_root, results_root)
        cases = select_cases(meta)

        w_y = resolve_weights("yolo", bin_name, train_root, results_root)
        w_d = resolve_weights("detr", bin_name, train_root, results_root)
        yolo = _load_model("yolo", w_y)
        detr = _load_model("detr", w_d)
        yolo.model.to(device)
        detr.model.to(device)
        yolo.model.eval()
        detr.model.eval()
        yolo_name, yolo_layer = find_yolo_gradcam_layer(yolo.model)
        detr_name, detr_layer, detr_note = find_detr_attn_layer(detr.model)
        layer_log.append(
            f"{bin_name} | YOLO Grad-CAM: {yolo_name} | RT-DETR: {detr_name} ({detr_note})"
        )

        for case_id, imeta in sorted(cases.items()):
            rgb = load_image_rgb(imeta.path)
            tensor = image_to_tensor(rgb, device)

            grad = gradcam_yolo(yolo, yolo_layer, tensor, yolo_name)
            grad_overlay = overlay_jet(rgb, grad)

            tensor_d = image_to_tensor(rgb, device)
            attn, _attn_meta = attention_map_detr(
                detr, detr_layer, tensor_d, imeta.detr_xy, imeta.detr_cf, device=device
            )
            attn_overlay = overlay_jet(rgb, attn)

            yb, ys = filter_preds(imeta.yolo_xy, imeta.yolo_cf)
            db, ds = filter_preds(imeta.detr_xy, imeta.detr_cf)

            bin_dir = out_root / bin_name
            ind_dir = out_root / "individual"
            stem = f"{bin_name}_caso{case_id}_{imeta.image_id}"

            p_orig = ind_dir / f"{stem}_original.png"
            p_grad = ind_dir / f"{stem}_gradcam.png"
            p_attn = ind_dir / f"{stem}_attn.png"
            p_panel = bin_dir / f"caso{case_id}_{imeta.image_id}_painel.png"

            ind_dir.mkdir(parents=True, exist_ok=True)
            for path, img, mode in (
                (p_orig, rgb, "orig"),
                (p_grad, grad_overlay, "yolo"),
                (p_attn, attn_overlay, "detr"),
            ):
                drawn = _draw_on_rgb(
                    img, imeta.gt, yb, ys, db, ds, mode=mode
                )
                ok = cv2.imwrite(str(path), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
                if not ok:
                    log.warning(f"[heatmaps] Falha ao salvar {path}")

            save_panel(p_panel, rgb, grad_overlay, attn_overlay, imeta, bin_name, case_id)

            for p in (p_orig, p_grad, p_attn, p_panel):
                saved_files.append(p)

            just = case_justification(case_id, imeta)
            selection_log.append(f"{bin_name} | caso{case_id} | {imeta.image_id} | {just}")

        del yolo, detr
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("Camadas usadas (Grad-CAM / Attention)")
    print("=" * 72)
    for line in layer_log:
        print(line)

    print("\n" + "=" * 72)
    print("Imagens selecionadas")
    print("=" * 72)
    for line in selection_log:
        print(line)

    print("\n" + "=" * 72)
    print("Arquivos salvos")
    print("=" * 72)
    for p in sorted(saved_files):
        kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {p} ({kb:.1f} KB)")


def _draw_on_rgb(
    rgb: np.ndarray,
    gt: np.ndarray,
    yolo_xy: Optional[np.ndarray],
    yolo_cf: Optional[np.ndarray],
    detr_xy: Optional[np.ndarray],
    detr_cf: Optional[np.ndarray],
    mode: str,
) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]

    def draw_rect(box, color, dashed=False):
        x1, y1, x2, y2 = map(int, box)
        if dashed:
            for xs in range(x1, x2, 6):
                cv2.line(out, (xs, y1), (min(xs + 3, x2), y1), color, 1)
                cv2.line(out, (xs, y2), (min(xs + 3, x2), y2), color, 1)
            for ys in range(y1, y2, 6):
                cv2.line(out, (x1, ys), (x1, min(ys + 3, y2)), color, 1)
                cv2.line(out, (x2, ys), (x2, min(ys + 3, y2)), color, 1)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    if mode in ("orig", "yolo", "detr"):
        for b in gt:
            draw_rect(b, (0, 255, 0), dashed=True)
    if mode in ("orig", "yolo") and yolo_xy is not None and yolo_cf is not None:
        for b, s in zip(yolo_xy, yolo_cf):
            draw_rect(b, (31, 119, 180))
            cv2.putText(out, f"{s:.2f}", (int(b[0]), max(12, int(b[1]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (31, 119, 180), 1)
    if mode in ("orig", "detr") and detr_xy is not None and detr_cf is not None:
        for b, s in zip(detr_xy, detr_cf):
            draw_rect(b, (214, 39, 40))
            cv2.putText(out, f"{s:.2f}", (int(b[0]), max(12, int(b[3]) + 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (214, 39, 40), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Heatmaps Grad-CAM e atencao")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--train-root", default="runs/train")
    parser.add_argument("--out-dir", default="results/heatmaps")
    parser.add_argument("--all", action="store_true", help="Todas as faixas")
    parser.add_argument("--bins", default=None, help="small,medium,large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regen-attn",
        action="store_true",
        help="Regenera apenas attention RT-DETR (paineis + individuais *_attn.png)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.regen_attn:
        paths, _metas = regen_attention_only(
            cfg,
            device=str(args.device),
            batch=int(args.batch),
            train_root=Path(args.train_root),
            out_root=Path(args.out_dir),
        )
        panels = sorted(p for p in paths if p.name.endswith("_painel.png"))
        print("\n" + "=" * 72)
        print("Arquivos sobrescritos (--regen-attn)")
        print("=" * 72)
        for p in sorted(paths):
            kb = p.stat().st_size / 1024 if p.exists() else 0
            print(f"  {p} ({kb:.1f} KB)")
        print("\n--- Paineis regenerados (tamanho KB) ---")
        for p in panels:
            kb = p.stat().st_size / 1024
            print(f"  {p.parent.name}/{p.name}: {kb:.1f} KB")
        return

    bins = list(BIN_ORDER) if args.all or not args.bins else [
        b.strip() for b in args.bins.split(",") if b.strip()
    ]
    run_heatmaps(
        cfg,
        device=str(args.device),
        batch=int(args.batch),
        train_root=Path(args.train_root),
        out_root=Path(args.out_dir),
        bins=bins,
    )


if __name__ == "__main__":
    main()
