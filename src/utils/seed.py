"""
seed.py — Reprodutibilidade global do experimento.

Define seeds fixas para Python, NumPy, PyTorch (CPU e CUDA).
Deve ser chamado como primeira linha de qualquer script de treino/avaliação.
"""

import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """
    Fixa seeds em todos os geradores de números aleatórios relevantes.

    Args:
        seed: Semente global. Padrão 42 (definido em config.yaml).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch não instalado — contexto de pré-processamento apenas
