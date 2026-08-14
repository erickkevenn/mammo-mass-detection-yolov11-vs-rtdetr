"""
loaders.py — Carregamento específico por dataset.

Funções públicas:
    load_inbreast(raw_dir, cfg)    -> List[MammographyCase]
    load_cbis_ddsm(raw_dir, cfg)   -> List[MammographyCase]
    load_mias(raw_dir, cfg)        -> List[MammographyCase]
    load_vindr_mammo(raw_dir, cfg) -> List[MammographyCase]

Dataclasses compartilhadas:
    NoduleAnnotation
    MammographyCase
"""

from __future__ import annotations

import hashlib
import plistlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import pydicom

from src.preprocessing.bbox_ops import clip_bbox, compute_size_mm, is_valid_bbox
from src.utils.logger import get_logger

log = get_logger("mammo_loaders")


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NoduleAnnotation:
    """Anotação de um nódulo em coordenadas absolutas da imagem original."""
    bbox_abs: tuple[int, int, int, int]  # (x1, y1, x2, y2) px absolutos
    size_mm: float                        # tamanho real estimado em mm
    size_bin: str = ""                    # preenchido em stratify.py
    malignant: bool = False
    birads: int = 0                       # 0 = desconhecido


@dataclass
class MammographyCase:
    """Representa uma imagem mamográfica com suas anotações."""
    case_id: str
    patient_id: str
    dataset: str            # 'inbreast' | 'cbis' | 'mias'
    view: str               # 'CC' | 'MLO' | 'unknown'
    image_path: Path
    pixel_spacing_mm: float
    annotations: List[NoduleAnnotation] = field(default_factory=list)
    birads: int = 0         # BI-RADS do caso (máximo entre anotações)
    laterality: str = ""    # 'LEFT' | 'RIGHT' | ''


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _get_pixel_spacing(dcm: pydicom.Dataset) -> float:
    """Extrai pixel spacing em mm/px do header DICOM.

    Tenta os atributos em ordem de precedência:
      1. PixelSpacing (0028,0030)
      2. ImagerPixelSpacing (0018,1164)
    Retorna -1.0 se não encontrado (deve ser tratado pelo caller).
    """
    for tag in ["PixelSpacing", "ImagerPixelSpacing"]:
        val = getattr(dcm, tag, None)
        if val is not None:
            try:
                return float(val[0])
            except (IndexError, TypeError, ValueError):
                pass
    return -1.0


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# INbreast
# ─────────────────────────────────────────────────────────────────────────────

def _inbreast_point_px_to_xy(point_px: List) -> Tuple[List[float], List[float]]:
    """Converte lista Point_px do plist INbreast (strings '(x, y)') em xs, ys."""
    xs: List[float] = []
    ys: List[float] = []
    for raw in point_px or []:
        if not isinstance(raw, str):
            continue
        m = re.match(r"\s*\(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)", raw.strip())
        if not m:
            continue
        xs.append(float(m.group(1)))
        ys.append(float(m.group(2)))
    return xs, ys


def _parse_inbreast_xml(xml_path: Path) -> List[NoduleAnnotation]:
    """
    Parseia o XML plist do INbreast (Release 1.0) e extrai ROIs ``Mass``.

    Estrutura: plist → ``Images`` → ``ROIs`` com ``Point_px`` (polígono).
    """
    annotations: List[NoduleAnnotation] = []
    try:
        with open(xml_path, "rb") as f:
            pl = plistlib.load(f)
    except Exception as e:
        log.warning(f"Erro ao parsear plist INbreast {xml_path}: {e}")
        return annotations

    for img in pl.get("Images", []):
        for roi in img.get("ROIs", []):
            if roi.get("Name") != "Mass":
                continue
            xs, ys = _inbreast_point_px_to_xy(roi.get("Point_px", []))
            if len(xs) < 3:
                continue
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            if x2 <= x1 or y2 <= y1:
                continue
            annotations.append(
                NoduleAnnotation(
                    bbox_abs=(x1, y1, x2, y2),
                    size_mm=0.0,
                )
            )

    return annotations


def load_inbreast(raw_dir: str | Path) -> List[MammographyCase]:
    """
    Carrega todos os casos do INbreast.

    Args:
        raw_dir: Caminho para 'archive/INbreast Release 1.0'.

    Returns:
        Lista de MammographyCase.
    """
    raw_dir = Path(raw_dir)
    dicom_dir = raw_dir / "AllDICOMs"
    xml_dir   = raw_dir / "AllXML"
    csv_path  = raw_dir / "INbreast.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"INbreast.csv não encontrado em {raw_dir}")

    meta = pd.read_csv(csv_path, sep=";", encoding="latin-1")
    # Normalizar nomes de coluna
    meta.columns = [c.strip() for c in meta.columns]
    log.info(f"[INbreast] {len(meta)} linhas no CSV. Colunas: {list(meta.columns)}")

    cases: List[MammographyCase] = []

    for _, row in meta.iterrows():
        file_name = str(row.get("File Name", "")).strip()
        if not file_name:
            continue

        # Localizar DICOM (nome do arquivo = File Name sem extensão)
        dcm_candidates = list(dicom_dir.glob(f"{file_name}*.dcm"))
        if not dcm_candidates:
            log.warning(f"[INbreast] DICOM não encontrado para File Name={file_name}")
            continue
        dcm_path = dcm_candidates[0]

        # Ler header DICOM para pixel spacing
        try:
            dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            pixel_spacing = _get_pixel_spacing(dcm)
            if pixel_spacing <= 0:
                log.warning(f"[INbreast] PixelSpacing inválido em {dcm_path.name}. "
                             "Usando 0.07 mm/px (estimativa FFDM típica).")
                pixel_spacing = 0.07
        except Exception as e:
            log.warning(f"[INbreast] Erro ao ler DICOM {dcm_path.name}: {e}")
            pixel_spacing = 0.07

        # Paciente por plano v2: prefixo do ficheiro (ex.: 20586908_...)
        patient_raw = str(row.get("Patient ID", "")).strip()
        pid_from_fn = file_name.split("_")[0] if "_" in file_name else Path(file_name).stem
        patient_id = pid_from_fn
        if patient_raw and patient_raw.lower() != "removed":
            patient_id = patient_raw
        view = str(row.get("View", "")).strip().upper()
        view = view if view in ("CC", "MLO") else "unknown"
        laterality = str(row.get("Laterality", "")).strip().upper()
        birads_raw = row.get("Bi-Rads", 0)
        try:
            birads = int(birads_raw)
        except (ValueError, TypeError):
            birads = 0

        # Localizar XML de anotações
        xml_candidates = list(xml_dir.glob(f"{file_name}*.xml"))
        annotations = _parse_inbreast_xml(xml_candidates[0]) if xml_candidates else []

        case = MammographyCase(
            case_id=file_name,
            patient_id=patient_id,
            dataset="inbreast",
            view=view,
            image_path=dcm_path,
            pixel_spacing_mm=pixel_spacing,
            annotations=annotations,
            birads=birads,
            laterality=laterality,
        )
        cases.append(case)

    log.info(f"[INbreast] {len(cases)} casos carregados.")
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# CBIS-DDSM
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_cbis_image(raw_dir: Path, dicom_rel_path: str) -> Optional[Path]:
    """
    Resolve o caminho da imagem original no dataset Kaggle (JPEG).
    A versão Kaggle salva a imagem em: raw_dir/jpeg/<UID>/1-xxx.jpg
    O UID é o penúltimo componente do caminho DICOM original no CSV.
    """
    parts = Path(dicom_rel_path.strip().strip('"')).parts
    if len(parts) < 2:
        return None
    uid = parts[-2]
    
    target_dir = raw_dir / "jpeg" / uid
    if not target_dir.exists():
        return None
        
    jpgs = list(target_dir.glob("*.jpg"))
    if jpgs:
        return jpgs[0]
    return None


def _resolve_cbis_mask(raw_dir: Path, dicom_rel_path: str) -> Optional[Path]:
    """
    Resolve o caminho da máscara. No Kaggle, a pasta do UID da máscara
    costuma conter duas imagens: a máscara original e um crop pequeno.
    Retornamos a imagem de maior área (que é a máscara completa).
    """
    import PIL.Image
    
    parts = Path(dicom_rel_path.strip().strip('"')).parts
    if len(parts) < 2:
        return None
    uid = parts[-2]
    
    target_dir = raw_dir / "jpeg" / uid
    if not target_dir.exists():
        return None
        
    jpgs = list(target_dir.glob("*.jpg"))
    if not jpgs:
        return None
    if len(jpgs) == 1:
        return jpgs[0]
        
    largest_jpg = None
    max_area = -1
    for jpg in jpgs:
        try:
            with PIL.Image.open(jpg) as img:
                w, h = img.size
                area = w * h
                if area > max_area:
                    max_area = area
                    largest_jpg = jpg
        except Exception:
            pass
            
    return largest_jpg


def _extract_cbis_bbox_from_mask(mask_path: Path) -> Optional[tuple[int, int, int, int]]:
    """
    Extrai bounding box da máscara ROI (DICOM ou JPEG binário).
    Retorna (x1, y1, x2, y2) ou None se falhar.
    """
    import cv2
    import numpy as np

    try:
        # Tentar como DICOM
        if mask_path.suffix.lower() in (".dcm",):
            dcm = pydicom.dcmread(str(mask_path))
            mask = dcm.pixel_array
        else:
            import PIL.Image
            mask = np.array(PIL.Image.open(mask_path).convert("L"))

        # Binarizar
        _, binary = cv2.threshold(mask.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Maior contorno
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return (x, y, x + w, y + h)
    except Exception as e:
        log.warning(f"[CBIS] Erro ao extrair bbox da máscara {mask_path.name}: {e}")
        return None


def load_cbis_ddsm(raw_dir: str | Path) -> List[MammographyCase]:
    """
    Carrega casos do CBIS-DDSM (apenas masses — nódulos).

    Usa mass_case_description_train_set.csv + mass_case_description_test_set.csv.
    Split original do CBIS é ignorado — será re-splitado por paciente.

    Tratamento de duplicatas:
      - Mesma imagem com múltiplas anotações (abnormality_id diferente) = nódulos distintos
      - Mesma máscara referenciada por linhas diferentes = deduplicada por MD5
    """
    raw_dir = Path(raw_dir)
    csv_dir = raw_dir / "csv"

    csvs = [
        csv_dir / "mass_case_description_train_set.csv",
        csv_dir / "mass_case_description_test_set.csv",
    ]
    dfs = []
    for p in csvs:
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            log.warning(f"[CBIS] CSV não encontrado: {p}")
    if not dfs:
        raise FileNotFoundError(f"Nenhum CSV de masses encontrado em {csv_dir}")

    df = pd.concat(dfs, ignore_index=True)
    log.info(f"[CBIS] {len(df)} linhas totais nos CSVs de masse.")

    # Agrupar por imagem principal (image file path)
    grouped = df.groupby(["patient_id", "left or right breast", "image view", "image file path"])

    cases: List[MammographyCase] = []
    mask_md5_seen: set[str] = set()  # deduplicação de máscaras

    for (patient_id, laterality, view, img_path_str), group in grouped:
        view = str(view).strip().upper()
        laterality = str(laterality).strip().upper()
        img_path_str = str(img_path_str).strip()

        # Resolver caminho da imagem
        img_path = _resolve_cbis_image(raw_dir, img_path_str)
        if img_path is None:
            log.warning(f"[CBIS] Imagem não encontrada: {img_path_str}")
            continue

        # Pixel spacing: tentar ler do DICOM, fallback estimado
        pixel_spacing = -1.0
        if img_path.suffix.lower() == ".dcm":
            try:
                dcm = pydicom.dcmread(str(img_path), stop_before_pixels=True)
                pixel_spacing = _get_pixel_spacing(dcm)
            except Exception:
                pass
        if pixel_spacing <= 0:
            # CBIS-DDSM: filme digitalizado a ~50µm típico → 0.05 mm/px
            pixel_spacing = 0.05
            log.debug(f"[CBIS] PixelSpacing não disponível em {img_path.name}. "
                      "Usando 0.05 mm/px.")

        # BI-RADS do caso: usar o valor da primeira linha (consistente por imagem)
        birads_raw = group["assessment"].iloc[0]
        try:
            birads = int(birads_raw)
        except (ValueError, TypeError):
            birads = 0

        # Pathology
        pathology = str(group["pathology"].iloc[0]).strip().upper()
        malignant = pathology == "MALIGNANT"

        # Extrair anotações por abnormality_id
        annotations: List[NoduleAnnotation] = []
        for _, row in group.iterrows():
            mask_path_str = str(row.get("ROI mask file path", "")).strip().strip('"')
            mask_path = _resolve_cbis_mask(raw_dir, mask_path_str)
            if mask_path is None:
                log.warning(f"[CBIS] Máscara não encontrada: {mask_path_str}")
                continue

            # Deduplicação por MD5
            md5 = _md5(mask_path)
            if md5 in mask_md5_seen:
                log.debug(f"[CBIS] Máscara duplicada ignorada: {mask_path.name}")
                continue
            mask_md5_seen.add(md5)

            bbox = _extract_cbis_bbox_from_mask(mask_path)
            if bbox is None:
                continue

            annotations.append(NoduleAnnotation(
                bbox_abs=bbox,
                size_mm=0.0,   # calculado em annotations.py
                malignant=malignant,
                birads=birads,
            ))

        if not annotations:
            log.debug(f"[CBIS] Nenhuma anotação válida para {patient_id} {view}.")
            continue

        case_id = f"cbis_{patient_id}_{laterality}_{view}"
        case = MammographyCase(
            case_id=case_id,
            patient_id=patient_id,
            dataset="cbis",
            view=view if view in ("CC", "MLO") else "unknown",
            image_path=img_path,
            pixel_spacing_mm=pixel_spacing,
            annotations=annotations,
            birads=birads,
            laterality=laterality,
        )
        cases.append(case)

    log.info(f"[CBIS] {len(cases)} casos carregados.")
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# MIAS
# ─────────────────────────────────────────────────────────────────────────────

def load_mias(raw_dir: str | Path) -> List[MammographyCase]:
    """
    Carrega casos do MIAS.

    Fonte: all-mias/Info.txt + arquivos mdb*.pgm

    Pixel spacing: 0.200 mm/px (confirmado no Info.txt: "200 micron pixel edge").

    Sistema de coordenadas: origem no canto inferior esquerdo (MIAS) →
    convertido para origem superior esquerdo (padrão imagem):
        y_img = 1024 - y_mias

    Pares de imagens: mdb(2n-1) e mdb(2n) formam um paciente (nota 1 do Info.txt).
    Vista: não distinguível CC/MLO → registrada como 'unknown'.

    Casos *NOTE 3* (calcificações difusas sem coordenadas): excluídos das anotações.
    """
    raw_dir = Path(raw_dir)
    info_path = raw_dir / "Info.txt"

    if not info_path.exists():
        raise FileNotFoundError(f"Info.txt não encontrado em {raw_dir}")

    PIXEL_SPACING_MM = 0.200  # confirmado no Info.txt

    # ── Parsear Info.txt ──────────────────────────────────────────────────
    # Formato (após o cabeçalho):
    # mdb001 G CIRC B 535 425 197
    # mdb003 D NORM
    # mdb005 F CIRC B 477 133 30
    # mdb005 F CIRC B 500 168 26  ← múltiplas lesões na mesma imagem
    # Linhas com *NOTE 3* têm coordenadas ausentes

    raw_annotations: dict[str, list[dict]] = {}  # key = mdb_id

    with open(info_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "====" in line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mdb_id = parts[0]
            if not mdb_id.startswith("mdb"):
                continue

            background = parts[1] if len(parts) > 1 else ""
            abn_class  = parts[2] if len(parts) > 2 else "NORM"
            severity   = parts[3] if len(parts) > 3 else ""

            if abn_class == "NORM":
                raw_annotations.setdefault(mdb_id, [])
                continue

            # Verificar NOTE 3 (coordenadas ausentes)
            line_upper = line.upper()
            if "*NOTE" in line_upper:
                log.info(f"[MIAS] {mdb_id}: NOTE 3 — calcificações difusas sem coords. "
                         "Excluído das anotações.")
                raw_annotations.setdefault(mdb_id, [])
                continue

            # Coordenadas opcionais
            try:
                cx     = int(parts[4])
                cy     = int(parts[5])
                radius = int(parts[6])
            except (IndexError, ValueError):
                log.warning(f"[MIAS] {mdb_id}: coordenadas ausentes/inválidas. Ignorado.")
                raw_annotations.setdefault(mdb_id, [])
                continue

            raw_annotations.setdefault(mdb_id, []).append({
                "abn_class": abn_class,
                "severity":  severity.upper(),
                "cx": cx, "cy": cy, "radius": radius,
            })

    # ── Construir casos ───────────────────────────────────────────────────
    from src.preprocessing.birads import derive_birads_mias

    cases: List[MammographyCase] = []
    all_mdb_ids = sorted(raw_annotations.keys())

    for mdb_id in all_mdb_ids:
        pgm_path = raw_dir / f"{mdb_id}.pgm"
        if not pgm_path.exists():
            log.warning(f"[MIAS] PGM não encontrado: {pgm_path}")
            continue

        anns_raw = raw_annotations[mdb_id]

        # Derivar patient_id pelo número do arquivo:
        # mdb001/mdb002 → paciente 1, mdb003/mdb004 → paciente 2, etc.
        num = int(mdb_id.replace("mdb", ""))
        patient_id = f"mias_p{(num + 1) // 2:03d}"

        annotations: List[NoduleAnnotation] = []
        case_birads = 1  # padrão NORM

        for ann in anns_raw:
            abn_class = ann["abn_class"]
            severity  = ann["severity"]
            cx, cy, radius = ann["cx"], ann["cy"], ann["radius"]

            # Converter coordenadas: origem inferior-esquerda → superior-esquerda
            cy_img = 1024 - cy

            # Bounding box quadrada a partir do círculo
            x1 = max(0, cx - radius)
            y1 = max(0, cy_img - radius)
            x2 = min(1024, cx + radius)
            y2 = min(1024, cy_img + radius)

            size_px = 2 * radius
            size_mm = size_px * PIXEL_SPACING_MM

            birads = derive_birads_mias(abn_class, severity, radius)
            malignant = severity == "M"
            case_birads = max(case_birads, birads)

            annotations.append(NoduleAnnotation(
                bbox_abs=(x1, y1, x2, y2),
                size_mm=size_mm,
                malignant=malignant,
                birads=birads,
            ))

        case = MammographyCase(
            case_id=mdb_id,
            patient_id=patient_id,
            dataset="mias",
            view="unknown",   # MIAS não distingue CC/MLO
            image_path=pgm_path,
            pixel_spacing_mm=PIXEL_SPACING_MM,
            annotations=annotations,
            birads=case_birads,
            laterality="",
        )
        cases.append(case)

    log.info(f"[MIAS] {len(cases)} casos carregados.")
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# VinDr-Mammo
# ─────────────────────────────────────────────────────────────────────────────

def _parse_vindr_birads(birads_str: str) -> int:
    """Converte string BI-RADS do VinDr para inteiro. Ex: 'BI-RADS 4' → 4."""
    if not isinstance(birads_str, str):
        return 0
    birads_str = birads_str.strip()
    for digit in ("1", "2", "3", "4", "5"):
        if digit in birads_str:
            return int(digit)
    return 0


def _parse_vindr_pixel_spacing(spacing_str: str) -> float:
    """Parseia pixel spacing do metadata.csv. Ex: '[0.085, 0.085]' → 0.085."""
    if not isinstance(spacing_str, str):
        return -1.0
    try:
        # Formato: "[0.085, 0.085]"
        cleaned = spacing_str.strip().strip("[]")
        parts = [p.strip() for p in cleaned.split(",")]
        return float(parts[0])
    except (ValueError, IndexError):
        return -1.0


def load_vindr_mammo(
    raw_dir: str | Path,
    max_background_ratio: float = 0.25,
    include_background: bool = True,
) -> List[MammographyCase]:
    """
    Carrega casos do VinDr-Mammo v1.0.0.

    Fonte:
      - finding_annotations.csv: bboxes, finding_categories, birads, split
      - metadata.csv: pixel spacing (Imager Pixel Spacing)
      - images/<study_id>/<image_id>.dicom

    Filtragem:
      - Somente finding_categories contendo 'Mass' geram NoduleAnnotation.
      - Imagens 'No Finding' são incluídas como background, limitadas a
        max_background_ratio × nº de imagens com Mass (evita processar
        todos os 18k+ DICOMs e reduz desbalanceamento de dataset).
      - Se ``include_background=False`` (estudo sem negativos), retorna apenas casos com Mass.

    Patient ID:
      - Usa study_id como patient_id (cada study = 1 paciente no VinDr).
      - Isso garante que vistas CC/MLO do mesmo paciente fiquem no mesmo split.

    Args:
        raw_dir:                Caminho para o diretório raiz do VinDr-Mammo.
        max_background_ratio:   Proporção máxima de imagens background em relação
                                às imagens com Mass. Default 0.25 (25%).
        include_background:     Se False, não inclui imagens sem lesão (plano v3).

    Returns:
        Lista de MammographyCase.
    """
    raw_dir = Path(raw_dir)
    finding_csv = raw_dir / "finding_annotations.csv"
    metadata_csv = raw_dir / "metadata.csv"

    if not finding_csv.exists():
        raise FileNotFoundError(f"finding_annotations.csv não encontrado em {raw_dir}")

    # ── Carregar CSVs ────────────────────────────────────────────────────
    df_findings = pd.read_csv(finding_csv)
    log.info(f"[VinDr] {len(df_findings)} linhas no finding_annotations.csv")

    # Carregar metadata para pixel spacing
    pixel_spacing_map: dict[str, float] = {}
    if metadata_csv.exists():
        df_meta = pd.read_csv(metadata_csv)
        # Coluna: "Imager Pixel Spacing" ou primeira ocorrência de "SOP Instance UID"
        id_col = "SOP Instance UID"
        spacing_col = "Imager Pixel Spacing"
        if id_col in df_meta.columns and spacing_col in df_meta.columns:
            for _, row in df_meta.iterrows():
                img_id = str(row[id_col]).strip()
                spacing = _parse_vindr_pixel_spacing(str(row[spacing_col]))
                if spacing > 0:
                    pixel_spacing_map[img_id] = spacing
        log.info(f"[VinDr] Pixel spacing carregado para {len(pixel_spacing_map)} imagens.")
    else:
        log.warning("[VinDr] metadata.csv não encontrado. Usando pixel spacing padrão.")

    DEFAULT_PIXEL_SPACING = 0.085  # mm/px — típico do Siemens Mammomat Inspiration

    # ── Agrupar findings por image_id ────────────────────────────────────
    # Cada image_id pode ter múltiplas findings (ou apenas "No Finding")
    grouped = df_findings.groupby("image_id")

    cases_mass: List[MammographyCase] = []
    cases_background: List[MammographyCase] = []

    for image_id, group in grouped:
        image_id = str(image_id).strip()
        first_row = group.iloc[0]

        study_id = str(first_row["study_id"]).strip()
        laterality = str(first_row.get("laterality", "")).strip().upper()
        view = str(first_row.get("view_position", "")).strip().upper()
        view = view if view in ("CC", "MLO") else "unknown"
        breast_birads = _parse_vindr_birads(str(first_row.get("breast_birads", "")))

        # Pixel spacing
        pixel_spacing = pixel_spacing_map.get(image_id, DEFAULT_PIXEL_SPACING)

        # Path da imagem DICOM + dimensões (para clip de bbox — plano v3)
        dicom_path = raw_dir / "images" / study_id / f"{image_id}.dicom"
        if not dicom_path.exists():
            log.debug(f"[VinDr] DICOM não encontrado: {dicom_path}")
            continue

        img_w, img_h = 0, 0
        try:
            dcm_hdr = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
            img_h = int(getattr(dcm_hdr, "Rows", 0) or 0)
            img_w = int(getattr(dcm_hdr, "Columns", 0) or 0)
        except Exception:
            pass
        if img_w <= 0 or img_h <= 0:
            wr, hr = first_row.get("width"), first_row.get("height")
            if pd.notna(wr) and pd.notna(hr):
                img_w, img_h = int(wr), int(hr)

        # ── Extrair anotações de Mass ────────────────────────────────────
        annotations: List[NoduleAnnotation] = []
        has_mass = False

        for _, row in group.iterrows():
            finding_cat = str(row.get("finding_categories", ""))

            # Verificar se esta finding é Mass
            if "Mass" not in finding_cat:
                continue

            # Verificar se tem bbox válida
            xmin = row.get("xmin")
            ymin = row.get("ymin")
            xmax = row.get("xmax")
            ymax = row.get("ymax")

            if pd.isna(xmin) or pd.isna(ymin) or pd.isna(xmax) or pd.isna(ymax):
                continue

            x1 = int(round(float(xmin)))
            y1 = int(round(float(ymin)))
            x2 = int(round(float(xmax)))
            y2 = int(round(float(ymax)))

            if img_w <= 0 or img_h <= 0:
                log.warning(f"[VinDr] Sem dimensões de imagem para clip: {image_id}")
                continue

            bbox_raw = (x1, y1, x2, y2)
            bbox = clip_bbox(bbox_raw, img_w, img_h)

            if not is_valid_bbox(bbox):
                log.debug(f"[VinDr] bbox degenerada após clip {image_id}: {bbox_raw} → {bbox}")
                continue

            size_mm = compute_size_mm(bbox, pixel_spacing)
            if not (0 < size_mm <= 150):
                continue

            # BI-RADS da finding
            finding_birads = _parse_vindr_birads(str(row.get("finding_birads", "")))
            malignant = finding_birads >= 4

            has_mass = True
            annotations.append(NoduleAnnotation(
                bbox_abs=bbox,
                size_mm=size_mm,
                malignant=malignant,
                birads=finding_birads,
            ))

        # ── Decidir se incluir este caso ─────────────────────────────────
        # Casos com Mass: sempre incluir
        # Casos "No Finding": coletar separadamente para limitar depois
        is_no_finding = all(
            "No Finding" in str(r.get("finding_categories", ""))
            for _, r in group.iterrows()
        )

        if not has_mass and not is_no_finding:
            # Findings sem Mass (ex: Calcification, Asymmetry) → ignorar
            continue

        case_birads = breast_birads
        if annotations:
            case_birads = max(case_birads, max(a.birads for a in annotations))

        case = MammographyCase(
            case_id=f"vindr_{image_id}",
            patient_id=f"vindr_{study_id}",
            dataset="vindr",
            view=view,
            image_path=dicom_path,
            pixel_spacing_mm=pixel_spacing,
            annotations=annotations,
            birads=case_birads,
            laterality=laterality,
        )
        if has_mass:
            cases_mass.append(case)
        else:
            cases_background.append(case)

    # ── Limitar background para evitar desbalanceamento e I/O ────────────
    import random

    random.seed(42)
    if not include_background:
        cases = cases_mass
        log.info(f"[VinDr] Somente imagens com Mass (sem negativos): {len(cases)} casos.")
    else:
        max_bg = int(len(cases_mass) * max_background_ratio)
        if len(cases_background) > max_bg:
            cases_background = random.sample(cases_background, max_bg)
            log.info(
                f"[VinDr] Background limitado a {max_bg} imagens "
                f"(ratio={max_background_ratio} × {len(cases_mass)} Mass)."
            )
        cases = cases_mass + cases_background

    # ── Estatísticas ─────────────────────────────────────────────────────
    n_total_anns = sum(len(c.annotations) for c in cases_mass)
    log.info(
        f"[VinDr] {len(cases)} casos carregados "
        f"(mass: {len(cases_mass)}, "
        f"{n_total_anns} anotações de Mass no subconjunto mass)."
    )
    return cases
