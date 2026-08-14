"""
stratify.py — Estratificação por faixa de tamanho real do nódulo.

Atribui o campo size_bin em cada NoduleAnnotation e no MammographyCase
com base no tamanho real em mm definido em config.yaml.

Faixas (baseadas no ACR BI-RADS Atlas e TNM - AJCC):
  small:  < 10 mm  — Engloba Muito Pequeno (<5mm) e Pequeno (5-10mm) (T1a/T1b)
  medium: 10–20 mm — Médio (estadiamento T1c)
  large:  > 20 mm  — Engloba Grande (20-50mm) e Muito Grande (>50mm) (T2/T3)

O size_bin do caso é determinado pela MAIOR lesão presente.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.preprocessing.loaders import MammographyCase
from src.utils.logger import get_logger

log = get_logger("stratify")

# Tipo dos bins: Dict[nome, (min_mm_inclusive, max_mm_exclusive)]
SizeBins = Dict[str, Tuple[float, float]]

DEFAULT_BINS: SizeBins = {
    "small":  (0.0,  10.0),
    "medium": (10.0, 20.0),
    "large":  (20.0, 9999.0),
}

BIN_ORDER = ["small", "medium", "large"]
EXPORT_BINS = BIN_ORDER + ["full"]


def assign_size_bin(size_mm: float, bins: SizeBins) -> str:
    """
    Retorna o nome da faixa de tamanho para um dado size_mm.

    Args:
        size_mm: Tamanho da lesão em mm.
        bins:    Dicionário de faixas {nome: (min, max)}.

    Returns:
        Nome da faixa (ex: 'small'). Retorna 'large' se nenhuma faixa cobrir.
    """
    for bin_name in BIN_ORDER:
        if bin_name not in bins:
            continue
        low, high = bins[bin_name]
        if low <= size_mm < high:
            return bin_name
    # Fallback para o último bin
    return BIN_ORDER[-1]


def stratify_cases(
    cases: List[MammographyCase],
    bins: SizeBins | None = None,
) -> List[MammographyCase]:
    """
    Atribui size_bin a todas as anotações e ao caso (pelo maior nódulo).

    Casos sem anotações (imagens NORM) recebem size_bin=''.
    Casos com anotações de tamanhos diferentes recebem o bin do maior nódulo.

    Args:
        cases: Lista de MammographyCase com size_mm calculado.
        bins:  Dicionário de faixas. Se None, usa DEFAULT_BINS.

    Returns:
        Lista de casos com size_bin preenchido.
    """
    if bins is None:
        bins = DEFAULT_BINS

    stats: Dict[str, int] = {b: 0 for b in BIN_ORDER}
    stats["norm"] = 0
    no_size_mm = 0

    for case in cases:
        if not case.annotations:
            case.birads = case.birads or 1
            stats["norm"] += 1
            continue

        max_size_mm = 0.0
        for ann in case.annotations:
            if ann.size_mm <= 0:
                no_size_mm += 1
                ann.size_bin = "unknown"
                continue
            ann.size_bin = assign_size_bin(ann.size_mm, bins)
            max_size_mm = max(max_size_mm, ann.size_mm)

        if max_size_mm > 0:
            # O bin do caso é determinado pelo maior nódulo
            case_bin = assign_size_bin(max_size_mm, bins)
            # Propagar para annotations sem size_mm definido
            for ann in case.annotations:
                if ann.size_bin == "unknown":
                    ann.size_bin = case_bin
            stats[case_bin] += 1
        else:
            stats["norm"] += 1

    if no_size_mm > 0:
        log.warning(
            f"[stratify] {no_size_mm} anotações com size_mm=0 — "
            "verifique pixel_spacing e bbox."
        )

    log.info(
        f"[stratify] Distribuição de casos por faixa: "
        + " | ".join(f"{k}={v}" for k, v in stats.items())
    )
    return cases


def filter_by_bin(
    cases: List[MammographyCase],
    bin_name: str,
    neg_ratio: float = 0.20,
) -> List[MammographyCase]:
    """
    Filtra casos que pertencem a uma faixa específica.

    Inclui apenas casos onde pelo menos uma anotação pertence ao bin.
    Casos NORM (sem anotações) podem ser incluídos como amostras negativas no bin 'full'.

    Args:
        cases:     Lista de MammographyCase estratificados.
        bin_name:  Nome da faixa (ex: 'small').
        neg_ratio: Proporção de imagens negativas (NORM) em relação às positivas
                   (apenas para o bin 'full'). Default 0.20 (20%).

    Returns:
        Subset de casos.
    """
    if bin_name == "full":
        positives = [c for c in cases if c.annotations]
        negatives = [c for c in cases if not c.annotations]
        
        if neg_ratio > 0 and negatives:
            import random
            # Usar seed fixa para reprodutibilidade
            random.seed(42)
            num_neg = int(len(positives) * neg_ratio)
            selected_neg = random.sample(negatives, min(num_neg, len(negatives)))
            log.info(
                f"[stratify] Bin 'full': incluindo {len(positives)} positivas e "
                f"{len(selected_neg)} negativas (ratio={neg_ratio})."
            )
            return positives + selected_neg
        
        return positives

    result = []
    for case in cases:
        # Filtrar as anotações do bin
        anns_in_bin = [a for a in case.annotations if a.size_bin == bin_name]
        if not anns_in_bin:
            continue
        # Criar cópia do caso com apenas as anotações do bin
        import copy
        case_copy = copy.deepcopy(case)
        case_copy.annotations = anns_in_bin
        result.append(case_copy)

    log.info(f"[stratify] Bin '{bin_name}': {len(result)} casos com anotações.")
    return result
