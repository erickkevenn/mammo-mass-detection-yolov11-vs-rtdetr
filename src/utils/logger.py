"""
logger.py — Configuração centralizada de logging para o pipeline.

Cria um logger com saída simultânea para console e arquivo de log.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str | Path = "logs/preprocessing") -> logging.Logger:
    """
    Retorna um logger configurado com handler de arquivo e console.

    Args:
        name:    Nome do logger (ex: 'loaders', 'crop', 'clahe').
        log_dir: Diretório onde o arquivo de log será salvo.

    Returns:
        Logger configurado.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # já configurado

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de arquivo — um arquivo por execução
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"pipeline_{timestamp}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Handler de console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
