"""
normalize.py — Etapas 1–2 do plano v2 (DICOM rescale + uniformização de profundidade).

A normalização por **percentil** para uint8 (recomendada no plano) será alinhada ao
pré-processamento completo; até lá reutilizamos a função existente do projeto.
"""

from __future__ import annotations

from src.preprocessing.annotations import normalize_bit_depth

__all__ = ["normalize_bit_depth"]
