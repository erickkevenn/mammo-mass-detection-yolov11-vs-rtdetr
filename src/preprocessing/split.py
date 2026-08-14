"""
split.py — Geração de splits treino/val/teste balanceados por paciente.

PONTO CRÍTICO DE VALIDADE CIENTÍFICA:
  O split é feito em nível de PATIENT_ID, não de imagem.
  Isso garante que imagens do mesmo paciente nunca apareçam em
  splits diferentes (evita data leakage).

Estratégia:
  1. GroupShuffleSplit com groups=patient_id
  2. Estratificação por label composto: (size_bin, birads_group)
     birads_group: 'low' (1-3) vs 'high' (4-5)
  3. Proporções: 70% treino / 15% val / 15% teste
  4. Validação automática: AssertionError se patient_id aparecer em 2 splits

Saída:
  splits.json mapeando case_id → 'train' | 'val' | 'test'
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from sklearn.model_selection import GroupShuffleSplit

from src.preprocessing.loaders import MammographyCase
from src.utils.logger import get_logger

log = get_logger("split")


def _birads_group(birads: int) -> str:
    """Agrupa BI-RADS em 'low' (1-3) ou 'high' (4-5) para estratificação."""
    return "high" if birads >= 4 else "low"


def _get_case_label(case: MammographyCase) -> str:
    """
    Rótulo composto para estratificação.
    Usa o bin do maior nódulo do caso.
    """
    if not case.annotations:
        return f"norm_{_birads_group(case.birads)}"

    # Bin dominante (maior nódulo)
    bins_present = [a.size_bin for a in case.annotations if a.size_bin]
    from src.preprocessing.stratify import BIN_ORDER
    dominant_bin = "unknown"
    for b in reversed(BIN_ORDER):   # large > medium > small > tiny
        if b in bins_present:
            dominant_bin = b
            break

    return f"{dominant_bin}_{_birads_group(case.birads)}"


def generate_splits(
    cases: List[MammographyCase],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Gera splits treino/val/teste com split por paciente.

    Args:
        cases:       Lista de MammographyCase estratificados.
        train_ratio: Fração para treino (default 0.70).
        val_ratio:   Fração para validação (default 0.15).
        seed:        Semente aleatória (default 42).

    Returns:
        Dicionário {case_id: split} onde split ∈ {'train','val','test'}.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Soma dos ratios deve ser 1.0"

    patient_ids = [c.patient_id for c in cases]
    labels      = [_get_case_label(c) for c in cases]
    indices     = list(range(len(cases)))

    # ── Passo 1: separar test ────────────────────────────────────────────
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    trainval_idx, test_idx = next(
        gss_test.split(indices, labels, groups=patient_ids)
    )

    # ── Passo 2: separar val dentro de trainval ──────────────────────────
    # val_frac relativo ao trainval
    val_frac = val_ratio / (train_ratio + val_ratio)

    trainval_patients = [patient_ids[i] for i in trainval_idx]
    trainval_labels   = [labels[i] for i in trainval_idx]

    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=seed
    )
    train_local_idx, val_local_idx = next(
        gss_val.split(trainval_idx, trainval_labels, groups=trainval_patients)
    )

    train_global = set(trainval_idx[i] for i in train_local_idx)
    val_global   = set(trainval_idx[i] for i in val_local_idx)
    test_global  = set(test_idx)

    # ── Montar resultado ─────────────────────────────────────────────────
    split_map: Dict[str, str] = {}
    for i, case in enumerate(cases):
        if i in train_global:
            split_map[case.case_id] = "train"
        elif i in val_global:
            split_map[case.case_id] = "val"
        else:
            split_map[case.case_id] = "test"

    # ── Validação: nenhum patient_id em dois splits ──────────────────────
    _validate_no_patient_leakage(cases, split_map)

    # ── Log de distribuição ──────────────────────────────────────────────
    counts: Dict[str, int] = defaultdict(int)
    for s in split_map.values():
        counts[s] += 1
    log.info(
        f"[split] Distribuição: "
        f"train={counts['train']} | val={counts['val']} | test={counts['test']} "
        f"(total={len(cases)})"
    )

    return split_map


def _validate_no_patient_leakage(
    cases: List[MammographyCase],
    split_map: Dict[str, str],
) -> None:
    """
    Verifica que nenhum patient_id aparece em mais de um split.
    Lança AssertionError com detalhes se encontrar leakage.
    """
    patient_to_splits: Dict[str, Set[str]] = defaultdict(set)
    for case in cases:
        split = split_map.get(case.case_id)
        if split:
            patient_to_splits[case.patient_id].add(split)

    leaked = {
        pid: splits
        for pid, splits in patient_to_splits.items()
        if len(splits) > 1
    }

    if leaked:
        details = "\n".join(f"  {pid}: {splits}" for pid, splits in leaked.items())
        raise AssertionError(
            f"[split] DATA LEAKAGE DETECTADO — "
            f"{len(leaked)} paciente(s) em múltiplos splits:\n{details}"
        )

    log.info(
        f"[split] Validação de leakage OK — "
        f"{len(patient_to_splits)} pacientes únicos distribuídos sem leakage."
    )


def save_splits(
    split_map: Dict[str, str],
    out_path: str | Path,
) -> None:
    """
    Salva o mapeamento case_id → split em JSON.

    Args:
        split_map: Dicionário {case_id: split}.
        out_path:  Caminho de saída (ex: 'data/splits.json').
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2, sort_keys=True)
    log.info(f"[split] splits.json salvo em {out_path}")


def load_splits(path: str | Path) -> Dict[str, str]:
    """Carrega splits.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
