"""
Pacote de carregamento por dataset — alinhado ao plano v2.

Importações estáveis: ``from src.preprocessing.loaders import load_inbreast``.
Implementação compartilhada em :mod:`src.preprocessing.mammo_loaders`.
"""

from __future__ import annotations

from src.preprocessing.loaders.cdd_cesm_loader import load_cdd_cesm
from src.preprocessing.mammo_loaders import (
    MammographyCase,
    NoduleAnnotation,
    load_cbis_ddsm,
    load_inbreast,
    load_mias,
    load_vindr_mammo,
)

__all__ = [
    "MammographyCase",
    "NoduleAnnotation",
    "load_inbreast",
    "load_cbis_ddsm",
    "load_mias",
    "load_vindr_mammo",
    "load_cdd_cesm",
]
