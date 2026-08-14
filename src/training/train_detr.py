"""
train_detr.py — Treinamento do RT-DETR-L por faixa de tamanho.

Usa a interface Ultralytics RTDETR, que aceita o mesmo formato
data_{bin}.yaml do YOLO — garantindo mesmo pipeline de dados.

Bins: small, medium, large (por tamanho em mm), full (dataset agregado + NORM),
ou all para treinar todos em sequência.

Uso:
    py -3.11 -m src.training.train_detr --config config.yaml --bin small
    py -3.11 -m src.training.train_detr --config config.yaml --bin full
    py -3.11 -m src.training.train_detr --config config.yaml --bin all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import RTDETR

from src.preprocessing.stratify import EXPORT_BINS
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

log = get_logger("train_detr")


def train_detr(bin_name: str, cfg: dict) -> None:
    """
    Treina RT-DETR-L em uma faixa de tamanho específica.

    Args:
        bin_name: Nome da faixa: 'small', 'medium', 'large' ou 'full'.
        cfg:      Configurações carregadas de config.yaml.
    """
    set_global_seed(cfg["hardware"]["seed"])

    hub_yolo = Path(cfg["paths"]["hub_yolo"])
    data_yaml = hub_yolo / f"data_{bin_name}.yaml"

    if not data_yaml.exists():
        log.error(
            f"[DETR] data_{bin_name}.yaml não encontrado em {hub_yolo}. "
            "Execute o pipeline de pré-processamento primeiro."
        )
        return

    detr_cfg  = cfg["training"]["detr"]
    train_cfg = cfg["training"]

    log.info(f"[DETR] Iniciando treinamento — bin='{bin_name}'")
    log.info(f"[DETR] model={detr_cfg['model']} | epochs={train_cfg['epochs']} "
             f"| batch={train_cfg['batch_size']} | imgsz={train_cfg['img_size']}")

    model = RTDETR(detr_cfg["model"])

    results = model.train(
        data=str(data_yaml),
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch_size"],
        imgsz=train_cfg["img_size"],
        optimizer=detr_cfg["optimizer"],
        lr0=detr_cfg["lr0"],
        weight_decay=detr_cfg["weight_decay"],
        warmup_epochs=detr_cfg["warmup_epochs"],
        patience=train_cfg["early_stopping_patience"],
        device=cfg["hardware"]["device"],
        seed=cfg["hardware"]["seed"],
        project=str(Path(cfg["paths"].get("results", "results")) / "detr"),
        name=f"rtdetr_l_{bin_name}",
        # Augmentation — flip_vertical=False é crítico em mamografia
        fliplr=cfg["augmentation"]["flip_horizontal"],
        flipud=cfg["augmentation"]["flip_vertical"],  # 0.0 = desativado
        degrees=cfg["augmentation"]["rotation_degrees"],
        scale=cfg["augmentation"]["scale_limit"],
        mosaic=0.0,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    log.info(f"[DETR] Treinamento concluído — bin='{bin_name}'")
    log.info(f"[DETR] mAP@0.5 (val) = {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino RT-DETR-L por faixa de tamanho")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--bin",
        required=True,
        choices=EXPORT_BINS + ["all"],
        metavar="BIN",
        help=(
            "Faixa: small | medium | large | full — ou 'all' para "
            f"{', '.join(EXPORT_BINS)} em sequência."
        ),
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    bins_to_train = EXPORT_BINS if args.bin == "all" else [args.bin]
    for bin_name in bins_to_train:
        train_detr(bin_name, cfg)


if __name__ == "__main__":
    main()
