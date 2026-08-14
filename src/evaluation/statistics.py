"""
statistics.py — Testes estatísticos para comparação RT-DETR vs YOLOv11m.

Teste de Wilcoxon signed-rank pareado por imagem (scores derivados de AP,
recall e F1 por imagem). Correção de Bonferroni: k = 4 métricas × número de
faixas efetivamente avaliadas nesta execução (ex.: só medium → k=4).

Teste de McNemar: detecção binária (≥1 GT correspondente a IoU clínico).
Bonferroni para McNemar usa o número de faixas com pacote pareado nesta run.

Uso:
    python -m src.evaluation.statistics --config config.yaml
    python -m src.evaluation.statistics --config config.yaml --device cpu
    python -m src.evaluation.statistics --config config.yaml --bins medium
    python -m src.evaluation.statistics --config config.yaml --skip-per-image
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.preprocessing.stratify import BIN_ORDER
from src.utils.logger import get_logger

log = get_logger("statistics")

ALPHA = 0.05
METRICS_FOR_TEST = ["mAP@0.5", "mAP@0.5:0.95", "Recall", "F1"]
N_WILCOXON_PLANNED = len(METRICS_FOR_TEST) * len(BIN_ORDER)  # 12 — teto se todas as faixas
N_MCNEMAR_PLANNED = len(BIN_ORDER)  # 3


def wilcoxon_bonferroni_k(
    per_bin_packs: Optional[Dict[str, Dict[str, object]]],
) -> int:
    """k para Bonferroni no Wilcoxon: 4 métricas × faixas com inferência nesta execução."""
    if not per_bin_packs:
        return N_WILCOXON_PLANNED
    bins_done = [b for b in BIN_ORDER if b in per_bin_packs]
    return max(len(METRICS_FOR_TEST) * len(bins_done), 1)


def mcnemar_bonferroni_k(
    per_bin_packs: Optional[Dict[str, Dict[str, object]]],
) -> int:
    """k para Bonferroni no McNemar: uma linha por faixa com pacote nesta execução."""
    if not per_bin_packs:
        return N_MCNEMAR_PLANNED
    bins_done = [b for b in BIN_ORDER if b in per_bin_packs]
    return max(len(bins_done), 1)


def parse_bins_list(s: Optional[str]) -> List[str]:
    """CLI --bins: 'medium' ou 'small,large'."""
    if s is None or not str(s).strip():
        return list(BIN_ORDER)
    out: List[str] = []
    for part in str(s).replace(";", ",").split(","):
        b = part.strip().lower()
        if not b:
            continue
        if b not in BIN_ORDER:
            raise ValueError(
                f"Faixa inválida em --bins: {b!r}. Use: {', '.join(BIN_ORDER)}"
            )
        if b not in out:
            out.append(b)
    return out or list(BIN_ORDER)
def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Aplica Wilcoxon signed-rank test entre dois conjuntos de scores.

    Args:
        scores_a: Scores do modelo A (ex: RT-DETR-L) por imagem.
        scores_b: Scores do modelo B (ex: YOLOv11m) por imagem.
        alternative: 'two-sided' | 'greater' | 'less'.

    Returns:
        (statistic, p_value)
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Tamanhos diferentes: len(A)={len(scores_a)}, len(B)={len(scores_b)}"
        )
    if len(scores_a) < 10:
        log.warning(
            f"[statistics] Menos de 10 amostras ({len(scores_a)}) — "
            "resultado do teste pode não ser confiável."
        )

    stat, p = stats.wilcoxon(scores_a, scores_b, alternative=alternative)
    return float(stat), float(p)


def bonferroni_correction(p_values: List[float], alpha: float = ALPHA) -> List[bool]:
    """
    Aplica correção de Bonferroni para múltiplos testes.

    Args:
        p_values: Lista de p-valores.
        alpha:    Nível de significância global (default 0.05).

    Returns:
        Lista de booleans (True = significativo após correção).
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values]


def bonferroni_significant_fixed_k(
    p_value: Optional[float], k_planned: int, alpha: float = ALPHA
) -> Optional[bool]:
    """Significativo se p < alpha/k_planned (Bonferroni para k testes planejados)."""
    if p_value is None:
        return None
    return bool(p_value < (alpha / max(k_planned, 1)))


def run_statistical_comparison(
    metrics_df: pd.DataFrame,
    min_samples: int = 30,
    per_bin_packs: Optional[Dict[str, Dict[str, object]]] = None,
) -> pd.DataFrame:
    """
    Comparação RT-DETR-L vs YOLOv11m por faixa.

    Se `per_bin_packs` for fornecido (saída de per_image_scores), preenche
    p_valor e significativo via Wilcoxon pareado por imagem + Bonferroni (k
    conforme número de faixas avaliadas nesta execução).
    """
    results: List[Dict] = []

    metric_to_pack_keys = {
        "mAP@0.5": ("ap50_detr", "ap50_yolo"),
        "mAP@0.5:0.95": ("ap5095_detr", "ap5095_yolo"),
        "Recall": ("recall_detr", "recall_yolo"),
        "F1": ("f1_detr", "f1_yolo"),
    }

    k_wilcox = wilcoxon_bonferroni_k(per_bin_packs)
    for bin_name in BIN_ORDER:
        df_bin = metrics_df[metrics_df["bin"] == bin_name]
        if df_bin.empty:
            continue

        detr_row = df_bin[df_bin["model"].str.contains("DETR", case=False)]
        yolo_row = df_bin[df_bin["model"].str.contains("YOLO", case=False)]

        if detr_row.empty or yolo_row.empty:
            log.warning(
                f"[statistics] Bin '{bin_name}': dados ausentes para um dos modelos."
            )
            continue

        pack = (per_bin_packs or {}).get(bin_name)

        for metric in METRICS_FOR_TEST:
            if metric not in df_bin.columns:
                continue

            val_detr = float(detr_row[metric].iloc[0])
            val_yolo = float(yolo_row[metric].iloc[0])
            delta = val_detr - val_yolo

            p_val: Optional[float] = None
            sig: Optional[bool] = None
            nota = ""
            n_pairs = None

            if pack is not None:
                kd, ky = metric_to_pack_keys[metric]
                a = pack[kd]  # type: ignore[index]
                b = pack[ky]  # type: ignore[index]
                m = np.isfinite(a) & np.isfinite(b)  # type: ignore[operator]
                n_pairs = int(m.sum())
                if n_pairs < min_samples:
                    nota = f"n_pares={n_pairs} < min_samples={min_samples}"
                else:
                    from src.evaluation.per_image_scores import safe_wilcoxon

                    _, p_val = safe_wilcoxon(a, b)  # type: ignore[arg-type]
                    sig = bonferroni_significant_fixed_k(
                        p_val, k_wilcox, ALPHA
                    )
                    extra = ""
                    sd = pack.get("stat_split_desc")  # type: ignore[union-attr]
                    ni = pack.get("n_infer_total")  # type: ignore[union-attr]
                    if sd:
                        extra += f" | splits_inferência={sd} N_total={ni}"
                    lw = pack.get("stat_leak_warning")  # type: ignore[union-attr]
                    if lw:
                        extra += f" | {lw}"
                    alpha_prime = ALPHA / max(k_wilcox, 1)
                    nota = (
                        f"Wilcoxon pareado por imagem (n={n_pairs}); "
                        f"Bonferroni k={k_wilcox} (α'={alpha_prime:.5f}){extra}"
                    )

            results.append(
                {
                    "bin": bin_name,
                    "metric": metric,
                    "RT-DETR-L": round(val_detr, 4),
                    "YOLOv11m": round(val_yolo, 4),
                    "delta": round(delta, 4),
                    "p_valor": round(p_val, 6) if p_val is not None else None,
                    "significativo": sig,
                    "n_pairs": n_pairs,
                    "nota": nota
                    or (
                        "inferência pareada não disponível (--skip-per-image ou falha nos pesos)"
                        if pack is None
                        else ""
                    ),
                }
            )

    return pd.DataFrame(results)


def mcnemar_rows_from_packs(
    per_bin_packs: Dict[str, Dict[str, object]],
    k_mcnemar: Optional[int] = None,
) -> pd.DataFrame:
    """Uma linha por bin: taxa de detecção + McNemar."""
    from src.evaluation.per_image_scores import mcnemar_for_bin

    k_mc = k_mcnemar if k_mcnemar is not None else mcnemar_bonferroni_k(per_bin_packs)
    rows: List[Dict] = []
    for bin_name in BIN_ORDER:
        pack = per_bin_packs.get(bin_name)
        if pack is None:
            continue
        hd = pack["hit_detr"]  # type: ignore[index]
        hy = pack["hit_yolo"]  # type: ignore[index]
        rate_d = float(np.mean(hd)) if len(hd) else 0.0
        rate_y = float(np.mean(hy)) if len(hy) else 0.0
        p_m, disc, note = mcnemar_for_bin(pack)
        sig = bonferroni_significant_fixed_k(p_m, k_mc, ALPHA)
        rows.append(
            {
                "bin": bin_name,
                "metric": "McNemar_detection_IoU_hit",
                "taxa_acerto_RT-DETR-L": round(rate_d, 4),
                "taxa_acerto_YOLOv11m": round(rate_y, 4),
                "delta_taxa": round(rate_d - rate_y, 4),
                "p_valor": round(p_m, 6) if p_m is not None else None,
                "significativo": sig,
                "n_discordantes": disc,
                "nota": f"{note} | Bonferroni k={k_mc} (α'={ALPHA/max(k_mc,1):.5f})",
            }
        )
    return pd.DataFrame(rows)


def compare_per_image(
    scores_detr: Dict[str, List[float]],
    scores_yolo: Dict[str, List[float]],
    min_samples: int = 30,
) -> pd.DataFrame:
    """
    Executa Wilcoxon signed-rank test usando scores por imagem (legado).

    Espera dicionários {bin_name: [scores...]} alinhados por posição.
    """
    results: List[Dict] = []
    all_p_values: List[float] = []

    for bin_name in BIN_ORDER:
        sa = scores_detr.get(bin_name, [])
        sb = scores_yolo.get(bin_name, [])

        if len(sa) < min_samples or len(sb) < min_samples:
            note = (
                f"análise exploratória (n_detr={len(sa)}, n_yolo={len(sb)} "
                f"< min={min_samples})"
            )
            results.append(
                {
                    "bin": bin_name,
                    "n_detr": len(sa),
                    "n_yolo": len(sb),
                    "stat": None,
                    "p_valor": None,
                    "significativo": None,
                    "nota": note,
                }
            )
            continue

        try:
            n = min(len(sa), len(sb))
            stat, p = wilcoxon_test(sa[:n], sb[:n])
            all_p_values.append(p)
            results.append(
                {
                    "bin": bin_name,
                    "n_detr": len(sa),
                    "n_yolo": len(sb),
                    "stat": round(stat, 4),
                    "p_valor": round(p, 4),
                    "significativo": None,
                    "nota": "",
                }
            )
        except Exception as e:
            log.warning(f"[statistics] Erro no teste para bin '{bin_name}': {e}")

    sig_flags = bonferroni_correction(all_p_values)
    sig_idx = 0
    for row in results:
        if row["p_valor"] is not None:
            row["significativo"] = sig_flags[sig_idx]
            sig_idx += 1

    return pd.DataFrame(results)


def mcnemar_test(
    detections_a: List[bool],
    detections_b: List[bool],
) -> Dict:
    """
    Teste de McNemar para comparação binária de detectores.

    Avalia se dois modelos diferem significativamente em termos de
    detecção correta por imagem (detectou/não detectou o nódulo).

    Args:
        detections_a: Lista de bool — True se modelo A detectou o nódulo na imagem i.
        detections_b: Lista de bool — True se modelo B detectou o nódulo na imagem i.

    Returns:
        Dict com: contingency_table, statistic, p_value, significant.
    """
    if len(detections_a) != len(detections_b):
        raise ValueError(
            f"Tamanhos diferentes: A={len(detections_a)}, B={len(detections_b)}"
        )

    n11 = sum(1 for a, b in zip(detections_a, detections_b) if a and b)
    n12 = sum(1 for a, b in zip(detections_a, detections_b) if a and not b)
    n21 = sum(1 for a, b in zip(detections_a, detections_b) if not a and b)
    n22 = sum(1 for a, b in zip(detections_a, detections_b) if not a and not b)

    n_discordant = n12 + n21

    if n_discordant == 0:
        log.warning("[McNemar] Nenhuma discordância — modelos idênticos nesta análise.")
        return {
            "contingency_table": [[n11, n12], [n21, n22]],
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "n_discordant": 0,
            "nota": "Sem discordâncias — teste não aplicável.",
        }

    if n_discordant < 25:
        chi2 = (abs(n12 - n21) - 1) ** 2 / (n12 + n21)
        nota = "Correção de continuidade aplicada (n_discordant < 25)"
    else:
        chi2 = (n12 - n21) ** 2 / (n12 + n21)
        nota = ""

    p_value = float(1 - stats.chi2.cdf(chi2, df=1))

    significant = p_value < ALPHA

    log.info(
        f"[McNemar] n11={n11} n12={n12} n21={n21} n22={n22} | "
        f"chi2={chi2:.4f} p={p_value:.4f} sig={significant}"
    )

    return {
        "contingency_table": [[n11, n12], [n21, n22]],
        "statistic": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significant": significant,
        "n_discordant": n_discordant,
        "nota": nota,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Testes estatísticos RT-DETR vs YOLOv11m")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--metrics-csv",
        default="results/tables/metrics_all.csv",
        help="CSV de métricas gerado por metrics.py",
    )
    parser.add_argument("--out", default="results/tables/statistical_tests.csv")
    parser.add_argument(
        "--out-mcnemar",
        default="results/tables/statistical_mcnemar.csv",
        help="McNemar por bin (detecção IoU clínico)",
    )
    parser.add_argument(
        "--skip-per-image",
        action="store_true",
        help="Não roda inferência pareada (Wilcoxon/McNemar/FROC por imagem).",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default=None,
        metavar="LISTA",
        help="Faixas a inferir: ex. medium ou small,large. Padrão: small,medium,large.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch inferência Ultralytics (default: evaluation.per_image_inference_batch no YAML)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Dispositivo só na inferência pareada (ex.: cpu, cuda:0). Sobrescreve evaluation.per_image_device.",
    )
    args = parser.parse_args()

    try:
        bins_to_run = parse_bins_list(args.bins)
    except ValueError as e:
        parser.error(str(e))

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.device:
        ev = dict(cfg.get("evaluation", {}))
        ev["per_image_device"] = str(args.device).strip()
        cfg["evaluation"] = ev
        log.info(f"[statistics] Inferência pareada forçada em device={ev['per_image_device']!r}")

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        log.error(f"Arquivo de métricas não encontrado: {metrics_path}")
        return

    df = pd.read_csv(metrics_path)
    min_samples = cfg["evaluation"]["min_samples_statistical_test"]
    inf_batch = args.batch
    if inf_batch is None:
        inf_batch = int(
            cfg.get("evaluation", {}).get("per_image_inference_batch", 2)
        )

    per_bin_packs: Optional[Dict[str, Dict[str, object]]] = None
    if not args.skip_per_image:
        from src.evaluation.per_image_scores import (
            run_per_image_evaluation_all_bins,
            save_froc_tables,
        )

        results_root = Path(cfg["paths"].get("results", "results"))
        log.info(
            "[statistics] Avaliação pareada por imagem (GPU/CPU pode demorar)… "
            f"faixas={bins_to_run}"
        )
        per_bin_packs = run_per_image_evaluation_all_bins(
            cfg, results_root, bins=bins_to_run, batch=inf_batch
        )
        if per_bin_packs:
            froc_dir = Path(args.out).parent
            save_froc_tables(per_bin_packs, froc_dir)
            mcnemar_df = mcnemar_rows_from_packs(per_bin_packs)
            mcnemar_path = Path(args.out_mcnemar)
            mcnemar_path.parent.mkdir(parents=True, exist_ok=True)
            mcnemar_df.to_csv(mcnemar_path, index=False)
            log.info(f"[statistics] McNemar salvo em {mcnemar_path}")
            print("\n--- McNemar (detecção, imagens com GT) ---")
            print(mcnemar_df.to_string(index=False))
        else:
            log.warning(
                "[statistics] Nenhum pacote pareado gerado (pesos ou dados ausentes)."
            )

    result_df = run_statistical_comparison(
        df, min_samples=min_samples, per_bin_packs=per_bin_packs
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print("\n--- Wilcoxon / agregados ---")
    print(result_df.to_string(index=False))
    log.info(f"[statistics] Tabela salva em {out_path}")


if __name__ == "__main__":
    main()
