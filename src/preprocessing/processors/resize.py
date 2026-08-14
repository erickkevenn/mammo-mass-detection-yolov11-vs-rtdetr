"""
processors.resize — redimensionamento para o tamanho de treino (:mod:`resize_norm`).

.. note::
   O plano v2 exige **letterbox** (padding) com escala de bbox alinhada ao Ultralytics;
   a implementação atual pode usar *stretch* — ver ``resize_norm.py``.
"""

from __future__ import annotations

from src.preprocessing.resize_norm import resize_and_scale_annotations, save_processed_image

__all__ = ["resize_and_scale_annotations", "save_processed_image"]
