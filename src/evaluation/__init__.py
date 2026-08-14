# src/evaluation/__init__.py
#
# Imports lazy: evita RuntimeWarning ao usar
#   python -m src.evaluation.metrics
# (o pacote não carrega submódulos antes do __main__ do metrics).

from __future__ import annotations

import importlib
from typing import Any, List

__all__ = [
    "collect_all_metrics",
    "evaluate_coco_metrics",
    "save_metrics_table",
    "collect_paired_per_image_metrics",
    "run_per_image_evaluation_all_bins",
    "save_froc_tables",
    "wilcoxon_test",
    "run_statistical_comparison",
    "compare_per_image",
    "run_cross_dataset_matrix",
    "late_fusion",
    "Detection",
    "CaseDetections",
]

_LAZY = {
    "collect_all_metrics": (".metrics", "collect_all_metrics"),
    "evaluate_coco_metrics": (".metrics", "evaluate_coco_metrics"),
    "save_metrics_table": (".metrics", "save_metrics_table"),
    "collect_paired_per_image_metrics": (".per_image_scores", "collect_paired_per_image_metrics"),
    "run_per_image_evaluation_all_bins": (".per_image_scores", "run_per_image_evaluation_all_bins"),
    "save_froc_tables": (".per_image_scores", "save_froc_tables"),
    "wilcoxon_test": (".statistics", "wilcoxon_test"),
    "run_statistical_comparison": (".statistics", "run_statistical_comparison"),
    "compare_per_image": (".statistics", "compare_per_image"),
    "run_cross_dataset_matrix": (".cross_dataset", "run_cross_dataset_matrix"),
    "late_fusion": (".fusion", "late_fusion"),
    "Detection": (".fusion", "Detection"),
    "CaseDetections": (".fusion", "CaseDetections"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submod, attr = _LAZY[name]
    mod = importlib.import_module(submod, __package__)
    return getattr(mod, attr)


def __dir__() -> List[str]:
    return sorted(__all__)
