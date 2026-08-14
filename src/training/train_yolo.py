"""
train_yolo.py — Treinamento do YOLOv11m por faixa de tamanho.

Uso:
    python -m src.training.train_yolo --config config.yaml --bin small
    python -m src.training.train_yolo --config config.yaml --bin all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.preprocessing.stratify import BIN_ORDER, EXPORT_BINS
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

log = get_logger("train_yolo")


def train_yolo(bin_name: str, cfg: dict) -> None:
    """
    Treina YOLOv11m em uma faixa de tamanho específica.

    Args:
        bin_name: Nome da faixa ('tiny','small','medium','large').
        cfg:      Configurações carregadas de config.yaml.
    """
    set_global_seed(cfg["hardware"]["seed"])

    hub_yolo = Path(cfg["paths"]["hub_yolo"])
    data_yaml = hub_yolo / f"data_{bin_name}.yaml"

    if not data_yaml.exists():
        log.error(
            f"[YOLO] data_{bin_name}.yaml não encontrado em {hub_yolo}. "
            "Execute o pipeline de pré-processamento primeiro."
        )
        return

    yolo_cfg = cfg["training"]["yolo"]
    train_cfg = cfg["training"]

    log.info(f"[YOLO] Iniciando treinamento — bin='{bin_name}'")
    log.info(f"[YOLO] model={yolo_cfg['model']} | epochs={train_cfg['epochs']} "
             f"| batch={train_cfg['batch_size']} | imgsz={train_cfg['img_size']}")

    model = YOLO(yolo_cfg["model"])

    results = model.train(
        data=str(data_yaml),
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch_size"],
        imgsz=train_cfg["img_size"],
        optimizer=yolo_cfg["optimizer"],
        lr0=yolo_cfg["lr0"],
        weight_decay=yolo_cfg["weight_decay"],
        warmup_epochs=yolo_cfg["warmup_epochs"],
        patience=train_cfg["early_stopping_patience"],
        device=cfg["hardware"]["device"],
        seed=cfg["hardware"]["seed"],
        project=str(Path(cfg["paths"].get("results", "results")) / "yolo"),
        name=f"yolo11m_{bin_name}",
        # Augmentation — flip_vertical=False é crítico em mamografia
        fliplr=cfg["augmentation"]["flip_horizontal"],
        flipud=cfg["augmentation"]["flip_vertical"],  # 0.0 = desativado
        degrees=cfg["augmentation"]["rotation_degrees"],
        scale=cfg["augmentation"]["scale_limit"],
        # Desativar mosaico por padrão (pode distorcer lesões pequenas)
        mosaic=0.0,
        # Salvar pesos e métricas
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    log.info(f"[YOLO] Treinamento concluído — bin='{bin_name}'")
    log.info(f"[YOLO] mAP@0.5 (val) = {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino YOLOv11m por faixa de tamanho")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--bin", required=True,
        choices=EXPORT_BINS + ["all"],
        help="Faixa de tamanho a treinar, ou 'all' para todas."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    bins_to_train = EXPORT_BINS if args.bin == "all" else [args.bin]
    for bin_name in bins_to_train:
        train_yolo(bin_name, cfg)


if __name__ == "__main__":
    main()
