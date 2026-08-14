"""
cross_dataset.py — Avaliação cross-dataset (treinar em A, testar em B).

Matriz completa para RT-DETR-L e YOLOv11m:
  INbreast → {CBIS, MIAS}
  CBIS     → {INbreast, MIAS}
  MIAS     → {INbreast, CBIS}

Uso:
    python -m src.evaluation.cross_dataset --config config.yaml --bin small
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.preprocessing.stratify import BIN_ORDER
from src.utils.logger import get_logger

log = get_logger("cross_dataset")

DATASETS  = ["inbreast", "cbis", "mias"]
MODELS    = ["yolo", "detr"]
MODEL_CLS = {"yolo": "YOLO", "detr": "RTDETR"}


def _load_model(model_key: str, weights_path: Path):
    """Carrega modelo Ultralytics pelo tipo."""
    if model_key == "detr":
        from ultralytics import RTDETR
        return RTDETR(str(weights_path))
    else:
        from ultralytics import YOLO
        return YOLO(str(weights_path))


def find_best_weights(results_root: Path, model_key: str, bin_name: str) -> Optional[Path]:
    """Localiza best.pt de um run de treinamento."""
    model_dir = results_root / model_key
    patterns = [f"*{bin_name}*/weights/best.pt"]
    for pattern in patterns:
        matches = list(model_dir.glob(pattern))
        if matches:
            return sorted(matches)[-1]
    return None


def find_dataset_yaml(hub_yolo: Path, bin_name: str, dataset: str) -> Optional[Path]:
    """
    Localiza o YAML de um dataset específico dentro do bin.

    O data_{bin}.yaml aponta para todos os datasets misturados.
    Para cross-dataset, precisamos de um YAML apontando apenas
    para as imagens de teste de um dataset específico.

    Se não existir, gera um temporário.
    """
    specific = hub_yolo / f"data_{bin_name}_{dataset}_test.yaml"
    if specific.exists():
        return specific

    # Gerar YAML temporário de teste por dataset
    base_yaml = hub_yolo / f"data_{bin_name}.yaml"
    if not base_yaml.exists():
        return None

    import yaml as _yaml
    with open(base_yaml) as f:
        base = _yaml.safe_load(f)

    # Substituir test path por subdiretório do dataset
    test_path_orig = base.get("test", "")
    # Espera-se que as imagens já estejam organizadas por dataset
    # em subdiretórios: {bin}/{split}/images/{dataset}/
    test_filtered = str(test_path_orig).replace(
        "test/images", f"test/images/{dataset}"
    )
    base["test"] = test_filtered
    base["_source_dataset"] = dataset

    with open(specific, "w") as f:
        _yaml.dump(base, f, default_flow_style=False)

    return specific


def evaluate_cross_dataset(
    model_key: str,
    train_dataset: str,
    test_dataset: str,
    bin_name: str,
    cfg: dict,
) -> Optional[Dict]:
    """
    Avalia o modelo treinado em train_dataset e testado em test_dataset.

    Args:
        model_key:     'yolo' ou 'detr'.
        train_dataset: Dataset de treinamento ('inbreast','cbis','mias').
        test_dataset:  Dataset de teste.
        bin_name:      Faixa de tamanho.
        cfg:           Configurações.

    Returns:
        Dicionário com métricas ou None se falhar.
    """
    results_root = Path(cfg["paths"].get("results", "results"))
    hub_yolo     = Path(cfg["paths"]["hub_yolo"])
    device       = cfg["hardware"]["device"]

    # Localizar pesos do modelo treinado no train_dataset
    # Convenção de nome: {model}_{bin}_{train_dataset}
    model_dir = results_root / model_key
    pattern   = f"*{bin_name}*{train_dataset}*/weights/best.pt"
    matches   = list(model_dir.glob(pattern))

    if not matches:
        # Tentar sem especificação de dataset (modelo treinado em todos)
        matches = list(model_dir.glob(f"*{bin_name}*/weights/best.pt"))

    if not matches:
        log.warning(
            f"[cross] Pesos não encontrados: {model_key}/{bin_name}/{train_dataset}"
        )
        return None

    weights_path = sorted(matches)[-1]
    model = _load_model(model_key, weights_path)

    # Localizar YAML de teste para test_dataset
    data_yaml = find_dataset_yaml(hub_yolo, bin_name, test_dataset)
    if data_yaml is None:
        log.warning(f"[cross] YAML não encontrado: {bin_name}/{test_dataset}")
        return None

    log.info(
        f"[cross] {MODEL_CLS[model_key]} | train={train_dataset} | "
        f"test={test_dataset} | bin={bin_name}"
    )

    try:
        results = model.val(
            data=str(data_yaml),
            split="test",
            device=device,
            verbose=False,
        )
        m = results.results_dict
        return {
            "model":          MODEL_CLS[model_key],
            "bin":            bin_name,
            "train_dataset":  train_dataset,
            "test_dataset":   test_dataset,
            "mAP@0.5":        round(float(m.get("metrics/mAP50(B)", 0.0)), 4),
            "mAP@0.5:0.95":   round(float(m.get("metrics/mAP50-95(B)", 0.0)), 4),
            "Precision":      round(float(m.get("metrics/precision(B)", 0.0)), 4),
            "Recall":         round(float(m.get("metrics/recall(B)", 0.0)), 4),
        }
    except Exception as e:
        log.error(f"[cross] Erro na avaliação: {e}")
        return None


def run_cross_dataset_matrix(
    cfg: dict,
    bin_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Executa a matriz completa de avaliação cross-dataset.

    Args:
        cfg:       Configurações.
        bin_names: Faixas a avaliar. None = todas.

    Returns:
        DataFrame com resultados de todos os pares (train, test).
    """
    bins = bin_names or BIN_ORDER
    rows: List[Dict] = []

    for model_key in MODELS:
        for bin_name in bins:
            for train_ds in DATASETS:
                for test_ds in DATASETS:
                    if train_ds == test_ds:
                        continue  # in-domain — já coberto pelo treino normal
                    result = evaluate_cross_dataset(
                        model_key, train_ds, test_ds, bin_name, cfg
                    )
                    if result:
                        rows.append(result)

    df = pd.DataFrame(rows)
    if not df.empty:
        log.info(f"[cross] Matriz completa: {len(df)} avaliações.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação cross-dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--bin", nargs="+", choices=BIN_ORDER + ["all"], default=["all"])
    parser.add_argument("--out", default="results/tables/cross_dataset.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bins = BIN_ORDER if "all" in args.bin else args.bin
    df = run_cross_dataset_matrix(cfg, bin_names=bins)

    if df.empty:
        log.warning("[cross] Nenhum resultado. Execute os treinamentos primeiro.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
