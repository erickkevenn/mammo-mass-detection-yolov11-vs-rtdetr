"""
run_pipeline.py — Runner principal do pipeline de pré-processamento.

Executa todas as etapas na ordem obrigatória para os três datasets:

  1. Carregamento e parsing específico por dataset
  2. Normalização de profundidade de bits → uint8
  3. Cálculo de size_mm e validação de bboxes
  4. Crop da região mamária + recálculo de bboxes
  5. Aplicação de CLAHE
  6. Redimensionamento para 640×640 + escala de bboxes
  7. Estratificação por faixa de tamanho
  8. Geração de splits por paciente (com validação de leakage)
  9. Exportação YOLO (.txt) + COCO JSON + data_{bin}.yaml

Uso:
    python -m src.preprocessing.run_pipeline --config config.yaml
    python -m src.preprocessing.run_pipeline --config config.yaml --dataset mias
    python -m src.preprocessing.run_pipeline --config config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import yaml
from PIL import Image
from tqdm import tqdm

from src.preprocessing.annotations import compute_annotations, normalize_to_uint8_percentile
from src.preprocessing.clahe import apply_clahe
from src.preprocessing.crop import crop_breast_region
from src.preprocessing.export import (
    copy_image_to_hub,
    export_annotations_csv,
    export_coco_json,
    export_yolo_dataset_yaml,
    export_yolo_label,
)
from src.preprocessing.loaders import (
    MammographyCase,
    load_cdd_cesm,
    load_inbreast,
    load_vindr_mammo,
)
from src.preprocessing.resize_norm import resize_and_scale_annotations, save_processed_image
from src.preprocessing.split import generate_splits, save_splits
from src.preprocessing.stratify import BIN_ORDER, EXPORT_BINS, filter_by_bin, stratify_cases
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

log = get_logger("run_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de imagem bruta
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Carrega a imagem bruta como array numpy 2D (grayscale).

    Suporta DICOM (.dcm), PGM e formatos PIL (PNG, JPEG, etc.).

    Returns:
        Array numpy uint8/uint16 ou None se falhar.
    """
    suffix = image_path.suffix.lower()
    try:
        if suffix in (".dcm", ".dicom"):
            dcm = pydicom.dcmread(str(image_path))
            arr = dcm.pixel_array
            # Aplicar RescaleSlope/Intercept se disponível
            slope = float(getattr(dcm, "RescaleSlope", 1))
            intercept = float(getattr(dcm, "RescaleIntercept", 0))
            arr = (arr * slope + intercept).astype(np.int32)
            arr = np.clip(arr, 0, None).astype(np.uint16)
            # Inverter se MONOCHROME1 (fundo branco → fundo preto)
            photo_interp = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
            if photo_interp == "MONOCHROME1":
                arr = arr.max() - arr
            return arr
        else:
            img = Image.open(image_path).convert("L")
            return np.array(img)
    except Exception as e:
        log.warning(f"Erro ao carregar {image_path}: {e}")
        return None


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline por caso
# ─────────────────────────────────────────────────────────────────────────────

def process_case(
    case: MammographyCase,
    cfg: dict,
    processed_dir: Path,
    save_visuals: bool = True,
    dry_run: bool = False,
) -> Optional[Tuple[MammographyCase, Path]]:
    """
    Executa as etapas 1-6 do pipeline para um único caso.

    Returns:
        (case_atualizado, caminho_png) ou None se falhar.
    """
    logs_dir = Path(cfg["paths"]["logs"])

    # ── Etapa 1: carregar imagem bruta ───────────────────────────────────
    raw = load_raw_image(case.image_path)
    if raw is None:
        log.warning(f"[pipeline] {case.case_id}: imagem não carregada. Pulando.")
        return None

    h, w = raw.shape[:2]

    # ── Etapa 2: normalizar bits → uint8 (percentis — plano v3) ────────────
    img_u8 = normalize_to_uint8_percentile(raw)

    # ── Etapa 3: calcular size_mm e validar bboxes ───────────────────────
    case = compute_annotations(case, img_w=w, img_h=h)

    # ── Etapa 4: crop da mama + recálculo de bboxes ──────────────────────
    img_u8, case, crop_info = crop_breast_region(
        image=img_u8,
        case=case,
        threshold=cfg["crop"]["threshold"],
        margin_px=cfg["crop"]["margin_px"],
        save_dir=logs_dir / "crop_examples" if save_visuals else None,
        save_visual=save_visuals,
    )

    # ── Etapa 5: CLAHE ───────────────────────────────────────────────────
    img_u8 = apply_clahe(
        image=img_u8,
        tile_grid_size=cfg["clahe"]["tile_grid_size"],
        clip_limit=cfg["clahe"]["clip_limit"],
        case_id=case.case_id,
        save_dir=logs_dir / "clahe_examples" if save_visuals else None,
        save_visual=save_visuals,
    )

    # ── Etapa 6: letterbox 640 + escala de bboxes ─────────────────────────
    img_u8, case = resize_and_scale_annotations(
        image=img_u8,
        case=case,
        target_size=cfg["image"]["target_size"],
    )

    if dry_run:
        return (case, Path("__dry_run__.png"))

    dataset_processed_dir = processed_dir / case.dataset
    dataset_processed_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_processed_image(
        image=img_u8,
        case=case,
        out_dir=dataset_processed_dir,
        target_size=cfg["image"]["target_size"],
    )

    return (case, png_path)


# ─────────────────────────────────────────────────────────────────────────────
# Runner principal
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    cfg: dict,
    datasets: Optional[List[str]] = None,
    dry_run: bool = False,
    save_visuals: bool = True,
    max_visuals: int = 10,
) -> None:
    """
    Executa o pipeline completo de pré-processamento.

    Args:
        cfg:          Configurações carregadas de config.yaml.
        datasets:     Lista de datasets a processar. None = todos.
        dry_run:      Se True, valida sem salvar arquivos.
        save_visuals: Se True, salva exemplos visuais (primeiros max_visuals).
        max_visuals:  Número máximo de exemplos visuais por etapa.
    """
    set_global_seed(cfg["hardware"]["seed"])
    start = time.time()

    processed_dir = Path(cfg["paths"]["processed"])
    hub_yolo = Path(cfg["paths"]["hub_yolo"])
    hub_coco = Path(cfg["paths"]["hub_coco"])
    hub_root = Path(cfg["paths"].get("hub_root", hub_yolo.parent))

    # ── Carregar datasets (plano v3: INbreast, CDD-CESM LE, VinDr) ─────────
    all_cases: List[MammographyCase] = []
    datasets_to_run = datasets or cfg.get("datasets_to_run", ["inbreast", "cdd_cesm", "vindr"])
    vindr_include_bg = bool(cfg.get("vindr_negatives", False))

    if "inbreast" in datasets_to_run:
        log.info("--- Carregando INbreast ------------------------------")
        try:
            all_cases.extend(load_inbreast(cfg["paths"]["inbreast_raw"]))
        except Exception as e:
            log.error(f"Falha ao carregar INbreast: {e}")

    if "cdd_cesm" in datasets_to_run:
        log.info("--- Carregando CDD-CESM (LE) -------------------------")
        try:
            all_cases.extend(load_cdd_cesm(cfg["paths"]["cdd_cesm_raw"]))
        except Exception as e:
            log.error(f"Falha ao carregar CDD-CESM: {e}")

    if "vindr" in datasets_to_run:
        log.info("--- Carregando VinDr-Mammo ---------------------------")
        try:
            all_cases.extend(
                load_vindr_mammo(
                    cfg["paths"]["vindr_raw"],
                    include_background=vindr_include_bg,
                )
            )
        except Exception as e:
            log.error(f"Falha ao carregar VinDr-Mammo: {e}")

    if "cbis" in datasets_to_run and cfg["paths"].get("cbis_raw"):
        log.info("--- Carregando CBIS-DDSM -----------------------------")
        try:
            from src.preprocessing.loaders import load_cbis_ddsm

            all_cases.extend(load_cbis_ddsm(cfg["paths"]["cbis_raw"]))
        except Exception as e:
            log.error(f"Falha ao carregar CBIS-DDSM: {e}")

    if "mias" in datasets_to_run and cfg["paths"].get("mias_raw"):
        log.info("--- Carregando MIAS ----------------------------------")
        try:
            from src.preprocessing.loaders import load_mias

            all_cases.extend(load_mias(cfg["paths"]["mias_raw"]))
        except Exception as e:
            log.error(f"Falha ao carregar MIAS: {e}")

    log.info(f"Total de casos carregados: {len(all_cases)}")

    # ── Processar imagens (etapas 1-6) ───────────────────────────────────
    log.info("--- Processando imagens ----------------------------------")
    processed: List[Tuple[MammographyCase, Path]] = []
    checksums: Dict[str, str] = {}
    visual_count = 0

    for case in tqdm(all_cases, desc="Processando"):
        do_visual = save_visuals and visual_count < max_visuals
        result = process_case(
            case=case,
            cfg=cfg,
            processed_dir=processed_dir,
            save_visuals=do_visual,
            dry_run=dry_run,
        )
        if result is not None:
            case_out, png_path = result
            processed.append((case_out, png_path))
            if not dry_run and png_path.exists():
                checksums[case_out.case_id] = md5_file(png_path)
            if do_visual:
                visual_count += 1

    log.info(f"Casos processados com sucesso: {len(processed)}")

    processed = [(c, p) for c, p in processed if c.annotations]
    if not processed and not dry_run:
        log.error("Nenhum caso com anotações válidas após o processamento. Abortando.")
        return
    if not processed and dry_run:
        log.warning("[DRY RUN] Nenhum caso com anotações.")
        return

    log.info(f"Casos com pelo menos 1 lesão para o hub: {len(processed)}")

    if dry_run:
        log.info("[DRY RUN] Pipeline validado. Nenhum arquivo foi salvo.")
        return

    # Salvar checksums
    cksum_path = hub_root / "checksums.json"
    cksum_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cksum_path, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)
    log.info(f"Checksums salvos em {cksum_path}")

    # ── Estratificação (etapa 7) ──────────────────────────────────────────
    log.info("--- Estratificando por faixa de tamanho ------------------")
    cases_only = [c for c, _ in processed]
    cases_only = stratify_cases(cases_only, bins=cfg["nodule_size_bins"])

    # Reconstruir lista com paths
    case_id_to_png = {c.case_id: p for c, p in processed}
    processed = [(c, case_id_to_png[c.case_id]) for c in cases_only if c.case_id in case_id_to_png]

    # ── Split por paciente (etapa 8) ──────────────────────────────────────
    log.info("--- Gerando splits por paciente --------------------------")
    split_map = generate_splits(
        cases=cases_only,
        train_ratio=cfg["splits"]["train"],
        val_ratio=cfg["splits"]["val"],
        seed=cfg["splits"]["seed"],
    )
    save_splits(split_map, hub_root / "splits.json")
    export_annotations_csv(cases_only, split_map, hub_root / "annotations.csv")

    # ── Exportação (etapa 9) ──────────────────────────────────────────────
    log.info("--- Exportando por faixa de tamanho ----------------------")
    min_samples = cfg["preprocessing"]["min_samples_per_bin"]
    neg_ratio = cfg["preprocessing"].get("negative_sample_ratio", 0.0)

    for bin_name in EXPORT_BINS:
        if bin_name == "full" and neg_ratio <= 0:
            log.info("Bin 'full' omitido (estudo sem negativos / cenário agregado).")
            continue
        bin_cases = filter_by_bin(cases_only, bin_name, neg_ratio=neg_ratio)
        if not bin_cases:
            log.warning(f"Bin '{bin_name}': nenhum caso. Pulando.")
            continue

        # Verificar amostras mínimas no teste
        test_cases = [c for c in bin_cases if split_map.get(c.case_id) == "test"]
        if len(test_cases) < min_samples:
            log.warning(
                f"Bin '{bin_name}': {len(test_cases)} casos no teste "
                f"(mínimo={min_samples}) → análise exploratória com ressalva."
            )

        # Organizar por split
        coco_input: Dict[str, List[Tuple[MammographyCase, Path]]] = defaultdict(list)

        for case in bin_cases:
            split = split_map.get(case.case_id, "train")
            png_path = case_id_to_png.get(case.case_id)
            if png_path is None:
                continue

            # YOLO
            img_dir   = hub_yolo / bin_name / split / "images"
            label_dir = hub_yolo / bin_name / split / "labels"
            dst_png = copy_image_to_hub(png_path, img_dir)
            export_yolo_label(case, label_dir)
            coco_input[split].append((case, dst_png))

        # COCO JSON
        coco_out_dir = hub_coco / bin_name
        export_coco_json(dict(coco_input), coco_out_dir)

        # YAML do YOLO
        export_yolo_dataset_yaml(bin_name, hub_yolo)

    elapsed = time.time() - start
    log.info(f"--- Pipeline concluído em {elapsed:.1f}s -----------------")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de pré-processamento — DETR vs YOLO Mamografia"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Caminho para config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--dataset", type=str, nargs="+",
        choices=["inbreast", "cdd_cesm", "vindr", "cbis", "mias"],
        default=None,
        help="Dataset(s) a processar. Padrão: config.datasets_to_run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Valida o pipeline sem salvar arquivos."
    )
    parser.add_argument(
        "--no-visuals", action="store_true",
        help="Não salva exemplos visuais de crop e CLAHE."
    )
    parser.add_argument(
        "--max-visuals", type=int, default=10,
        help="Número máximo de exemplos visuais por etapa (default: 10)."
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_pipeline(
        cfg=cfg,
        datasets=args.dataset,
        dry_run=args.dry_run,
        save_visuals=not args.no_visuals,
        max_visuals=args.max_visuals,
    )


if __name__ == "__main__":
    main()
