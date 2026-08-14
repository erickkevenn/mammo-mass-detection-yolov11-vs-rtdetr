"""
birads.py — Derivação de BI-RADS para o dataset MIAS.

O MIAS não possui BI-RADS nativo. Este módulo implementa o critério
de mapeamento documentado no artigo (Seção Materiais e Métodos):

  Entrada:  classe de anormalidade (CIRC, SPIC, MISC, ARCH, ASYM, CALC)
            caráter (B = benigno, M = maligno)
            raio da lesão em pixels

  Saída:    BI-RADS inteiro (1–5) conforme ACR BI-RADS Atlas 5ª ed. (2013)

Referência: American College of Radiology. ACR BI-RADS® Atlas, Breast
Imaging Reporting and Data System. 5th ed. Reston, VA: ACR; 2013.
"""

from __future__ import annotations

from src.utils.logger import get_logger

log = get_logger("birads")

# Limiar de raio (px) para distinguir lesões MISC pequenas/grandes
_MISC_RADIUS_THRESHOLD = 30


def derive_birads_mias(
    abnormality_class: str,
    character: str,
    radius_px: int,
) -> int:
    """
    Deriva o BI-RADS para uma lesão do dataset MIAS.

    Tabela de mapeamento (documentada no artigo):

    | Classe | Caráter | Raio (px) | BI-RADS | Justificativa                        |
    |--------|---------|-----------|---------|--------------------------------------|
    | CIRC   | B       | qualquer  | 2       | Massa bem definida benigna           |
    | CIRC   | M       | qualquer  | 5       | Massa circunscrita maligna           |
    | SPIC   | B       | qualquer  | 4A      | Espiculação benigna (suspeição baixa)|
    | SPIC   | M       | qualquer  | 5       | Espiculada maligna (alta suspeição)  |
    | MISC   | B       | < 30      | 3       | Massa ill-def. pequena benigna       |
    | MISC   | B       | >= 30     | 4A      | Massa ill-def. grande benigna        |
    | MISC   | M       | qualquer  | 4       | Suspeição intermediária-alta         |
    | ARCH   | B       | qualquer  | 4A      | Distorção arquitetural benigna       |
    | ARCH   | M       | qualquer  | 5       | Distorção arquitetural maligna       |
    | ASYM   | B       | qualquer  | 3       | Assimetria benigna                   |
    | ASYM   | M       | qualquer  | 4       | Assimetria maligna                   |
    | CALC   | B       | qualquer  | 3       | Calcificação benigna                 |
    | CALC   | M       | qualquer  | 4       | Calcificação maligna (suspeita)      |
    | NORM   | —       | —         | 1       | Normal                               |

    Nota: BI-RADS 4A é representado internamente como 4 (inteiro).
    A distinção 4A/4B/4C pode ser recuperada pela combinação
    (classe, caráter) e é reportada no artigo em texto.

    Args:
        abnormality_class: Uma de {CIRC, SPIC, MISC, ARCH, ASYM, CALC, NORM}.
        character:         'B' (benigno) ou 'M' (maligno). '' para NORM.
        radius_px:         Raio da lesão em pixels.

    Returns:
        BI-RADS como inteiro (1–5).
    """
    cls = abnormality_class.strip().upper()
    char = character.strip().upper()

    if cls == "NORM" or cls == "":
        return 1

    if cls == "CIRC":
        return 2 if char == "B" else 5

    if cls == "SPIC":
        return 4 if char == "B" else 5

    if cls == "MISC":
        if char == "B":
            return 3 if radius_px < _MISC_RADIUS_THRESHOLD else 4
        return 4  # M

    if cls == "ARCH":
        return 4 if char == "B" else 5

    if cls == "ASYM":
        return 3 if char == "B" else 4

    if cls == "CALC":
        return 3 if char == "B" else 4

    # Classe desconhecida — registrar e retornar 0 (desconhecido)
    log.warning(f"[BIRADS] Classe desconhecida: '{cls}'. Retornando 0.")
    return 0
