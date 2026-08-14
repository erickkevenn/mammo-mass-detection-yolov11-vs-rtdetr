# Detecção de Massas Mamárias Estratificada por Tamanho: YOLOv11m vs. RT-DETR-L

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-%E2%89%A58.2-042AFF)](https://github.com/ultralytics/ultralytics)
[![CUDA](https://img.shields.io/badge/CUDA-recomendado-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Comparação sistemática entre **YOLOv11m** (detector CNN de estágio único, baseado em âncoras livres) e
**RT-DETR-L** (detector *end-to-end* baseado em Transformers) na detecção de **massas mamárias** em
mamografia digital, **estratificada por tamanho da lesão** (small / medium / large) usando três bases
públicas: **INbreast**, **CDD-CESM** (*low-energy*) e **VinDr-Mammo**.

> Repositório de pesquisa. O código cobre o ciclo completo: ingestão DICOM → pré-processamento →
> estratificação → treino → avaliação formal → testes estatísticos → explicabilidade.

---

## Sumário

- [Visão geral do experimento](#visão-geral-do-experimento)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Requisitos](#requisitos)
- [1. Instalação](#1-instalação)
- [2. Preparação dos datasets](#2-preparação-dos-datasets)
- [3. Pré-processamento](#3-pré-processamento)
- [4. Treinamento](#4-treinamento)
- [5. Avaliação e testes estatísticos](#5-avaliação-e-testes-estatísticos)
- [6. Explicabilidade (Grad-CAM e atenção)](#6-explicabilidade-grad-cam-e-atenção)
- [Pipeline completo em um bloco](#pipeline-completo-em-um-bloco)
- [Configuração (`config.yaml`)](#configuração-configyaml)
- [Saídas geradas](#saídas-geradas)
- [Resultados de referência](#resultados-de-referência)
- [Reprodutibilidade](#reprodutibilidade)
- [Solução de problemas](#solução-de-problemas)
- [Módulos internos (sem CLI)](#módulos-internos-sem-cli)
- [Licença e uso dos dados](#licença-e-uso-dos-dados)

---

## Visão geral do experimento

### Contribuições

1. **Comparação arquitetural estratificada por tamanho** — CNN (YOLOv11m) vs. Transformer (RT-DETR-L),
   com modelos treinados de forma independente em cada faixa de tamanho.
2. **Explicabilidade** — Grad-CAM (YOLOv11m) e mapas de atenção cruzada (RT-DETR-L) nos mesmos casos.
3. **Análise FROC estrita** + **SAHI** (*Slicing Aided Hyper Inference*) para isolar variáveis de
   confusão ligadas à resolução de entrada e ao tamanho das lesões pequenas.
4. **Testes pareados** de Wilcoxon (signed-rank) e McNemar com correção de **Bonferroni**.

### Estratificação por tamanho

As faixas são **tercis empíricos (P33/P67)** da distribuição agregada do tamanho das lesões, fixados
após a auditoria dos três datasets (com *clamp* de bounding box e filtro `]0, 150] mm`):

| Faixa    | Tamanho da lesão | Lesões | Imagens (teste) |
|:---------|:-----------------|-------:|----------------:|
| `small`  | 0 – 19,4 mm      |    775 |             107 |
| `medium` | 19,4 – 36,3 mm   |    796 |             129 |
| `large`  | 36,3 – 150 mm    |    770 |             155 |

O `size_bin` de cada imagem é determinado pela **maior lesão** presente nela.

### Dados após o pré-processamento

| Dataset      | Imagens | Lesões |
|:-------------|--------:|-------:|
| VinDr-Mammo  |   1.112 |  1.225 |
| CDD-CESM (LE)|     494 |  1.000 |
| INbreast     |     107 |    116 |
| **Total**    | **1.713** | **2.341** |

937 pacientes · *split* **por paciente** (70 / 15 / 15) com validação de vazamento · `seed = 42`.

### Pré-processamento aplicado

DICOM/RAW → normalização percentílica para `uint8` → cálculo de `size_mm` e validação de bbox →
recorte da região mamária (com recálculo das bboxes) → **CLAHE** (`clip=2.0`, `tile=8×8`) →
*letterbox* para **640×640** → estratificação → *split* por paciente → export **YOLO** + **COCO**.

> **Nota clínica:** *flip vertical* está **desativado** (`flip_vertical: 0.0`) e o **mosaic também**
> (`mosaic: 0.0`), para preservar a orientação anatômica e a escala real das lesões.

---

## Estrutura do repositório

```
ProjetoYoloxDter/
├── config.yaml                  # Configuração central: caminhos, hiperparâmetros, splits, faixas
├── requirements.txt             # Dependências Python
├── README.md
│
├── ds/                          # Datasets BRUTOS (não versionados — baixe manualmente)
│   ├── CDD-CESM/
│   ├── INBREAST/
│   └── vindr-mammo-.../
│
├── data/                        # Gerado pelo pipeline (não versionado)
│   ├── processed/               # PNGs 640×640 pós-CLAHE, por dataset
│   └── hub/
│       ├── yolo/                # data_{bin}.yaml + {bin}/{split}/{images,labels}
│       ├── coco/                # anotações COCO por faixa
│       ├── annotations.csv      # tabela mestre (case, paciente, size_mm, bin, split, bbox)
│       ├── splits.json          # mapa case_id → train/val/test
│       └── checksums.json       # MD5 de cada imagem processada
│
├── src/
│   ├── preprocessing/
│   │   ├── run_pipeline.py      # ▶ runner do pré-processamento (etapas 1–9)
│   │   ├── audit_datasets.py    # ▶ auditoria/QA dos datasets brutos
│   │   ├── mammo_loaders.py     # loaders INbreast / CDD-CESM / VinDr
│   │   ├── annotations.py       # size_mm, normalização, validação de bbox
│   │   ├── crop.py              # recorte da região mamária
│   │   ├── clahe.py             # equalização adaptativa
│   │   ├── resize_norm.py       # letterbox 640 + reescala de bboxes
│   │   ├── stratify.py          # atribuição de size_bin
│   │   ├── split.py             # split por paciente (anti-leakage)
│   │   └── export.py            # export YOLO / COCO / data_{bin}.yaml
│   │
│   ├── training/
│   │   ├── train_yolo.py        # ▶ treino YOLOv11m por faixa
│   │   └── train_detr.py        # ▶ treino RT-DETR-L por faixa
│   │
│   ├── evaluation/
│   │   ├── evaluate_test.py     # ▶ avaliação formal no split de teste
│   │   ├── statistical_tests.py # ▶ Wilcoxon + McNemar (Bonferroni)
│   │   ├── froc.py              # ▶ curvas FROC por faixa
│   │   ├── sahi_inference.py    # ▶ SAHI na faixa small
│   │   ├── heatmaps.py          # ▶ Grad-CAM + mapas de atenção
│   │   ├── metrics.py           # ▶ coleta de métricas dos runs Ultralytics
│   │   ├── statistics.py        # ▶ pipeline estatístico alternativo (legado)
│   │   ├── cross_dataset.py     # ▶ avaliação cross-dataset
│   │   ├── per_image_scores.py  #   biblioteca (sem CLI)
│   │   └── fusion.py            #   biblioteca (sem CLI) — late fusion CC+MLO
│   │
│   └── utils/                   # logger, seeds globais
│
├── runs/                        # Saídas do Ultralytics (não versionado)
│   └── detect/results/{yolo,detr}/...
├── results/                     # Tabelas, curvas e heatmaps finais
│   ├── audit/  evaluation/  tables/  heatmaps/
├── logs/                        # Logs + exemplos visuais de crop e CLAHE
└── artigo/                      # Rascunhos e figuras do artigo científico
```

`▶` = módulo executável via linha de comando.

---

## Requisitos

### Software

| Item | Versão |
|:-----|:-------|
| Python | **3.11** (recomendado; o `config.yaml` assume `py -3.11`) |
| PyTorch | ≥ 2.1 com suporte CUDA |
| Ultralytics | ≥ 8.2 (fornece `YOLO` e `RTDETR`) |

Dependências principais (`requirements.txt`): `torch`, `torchvision`, `ultralytics`, `pydicom`,
`opencv-python`, `Pillow`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `pyyaml`, `tqdm`,
`matplotlib`, `seaborn`.

**Extras instalados sob demanda** (os próprios scripts fazem `pip install` na primeira execução):

| Pacote | Usado por |
|:-------|:----------|
| `sahi` | `src/evaluation/sahi_inference.py` |
| `grad-cam` | `src/evaluation/heatmaps.py` |

Para instalar antes, evitando o download no meio da execução:

```bash
pip install sahi grad-cam
```

### Hardware

- **GPU NVIDIA com CUDA é obrigatória na prática** para o treino (100 épocas × 6 modelos).
  `config.yaml` define `device: "cuda:0"`.
- A avaliação por imagem roda em CPU por padrão (`evaluation.per_image_device: cpu`), mas é bem mais
  rápida em GPU — use `--device cuda:0`.
- Espaço em disco: reserve **~500 GB** se for baixar os três datasets brutos (o VinDr-Mammo sozinho
  passa de 300 GB em DICOM).

---

## 1. Instalação

```bash
git clone https://github.com/<seu-usuario>/<seu-repo>.git
cd <seu-repo>
```

<details open>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch com CUDA 12.1 (ajuste conforme sua GPU/driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Linux / macOS</strong></summary>

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
</details>

Verifique a instalação:

```bash
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available())"
```

> **Windows:** `albumentations` está comentado no `requirements.txt` porque exige o Microsoft C++
> Build Tools. Não é necessário — o Ultralytics já aplica a augmentation equivalente
> (`fliplr`, `flipud`, `degrees`, `scale`).

> Todos os comandos abaixo usam `python -m`. No Windows, se você tiver várias versões de Python
> instaladas, troque por `py -3.11 -m` (é o `python_launcher` definido no `config.yaml`).
> **Execute sempre a partir da raiz do projeto** — os módulos são importados como pacote (`src.`).

---

## 2. Preparação dos datasets

Os datasets **não são versionados** e devem ser baixados nas fontes oficiais (todos exigem aceite de
termos de uso):

| Dataset | Fonte | Pasta esperada |
|:--------|:------|:---------------|
| **INbreast** | Repositório oficial / mediante solicitação | `ds/INBREAST/INbreast Release 1.0/` |
| **CDD-CESM** | [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/) | `ds/CDD-CESM/` |
| **VinDr-Mammo** | [PhysioNet](https://physionet.org/content/vindr-mammo/) | `ds/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/` |

Estrutura mínima esperada:

```
ds/
├── INBREAST/
│   └── INbreast Release 1.0/
│       ├── AllDICOMs/          # imagens .dcm
│       ├── AllXML/             # contornos das lesões
│       └── INbreast.csv
├── CDD-CESM/
│   ├── Low energy images of CDD-CESM/
│   └── Radiology_hand_drawn_segmentations_v2.csv
└── vindr-mammo-.../
    ├── images/                 # DICOM organizados por study_id
    ├── finding_annotations.csv
    └── metadata.csv
```

Os caminhos são configuráveis em `config.yaml → paths`. Se sua estrutura for diferente, edite lá em
vez de mover os arquivos.

### 2.1 Auditoria (opcional, mas recomendado)

Antes de processar tudo, valide a integridade dos datasets brutos:

```bash
python -m src.preprocessing.audit_datasets --config config.yaml
```

Saída: `results/audit/audit_report.csv` — verifica bboxes inválidas, `size_mm` fora de faixa,
metadados de espaçamento de pixel ausentes e casos sem anotação.

---

## 3. Pré-processamento

Executa as 9 etapas na ordem obrigatória e popula `data/processed/` e `data/hub/`:

```bash
python -m src.preprocessing.run_pipeline --config config.yaml
```

| Flag | Padrão | Descrição |
|:-----|:-------|:----------|
| `--config` | `config.yaml` | Caminho do arquivo de configuração |
| `--dataset` | `config.datasets_to_run` | Um ou mais de `inbreast cdd_cesm vindr cbis mias` |
| `--dry-run` | desligado | Valida o pipeline **sem gravar nada** em disco |
| `--no-visuals` | desligado | Não salva exemplos visuais de crop/CLAHE |
| `--max-visuals` | `10` | Número máximo de exemplos visuais por etapa |

**Sempre comece com um teste seco** — ele percorre todo o pipeline sem escrever arquivos:

```bash
python -m src.preprocessing.run_pipeline --config config.yaml --dry-run
```

Processar um dataset por vez (útil para depurar):

```bash
python -m src.preprocessing.run_pipeline --config config.yaml --dataset inbreast
python -m src.preprocessing.run_pipeline --config config.yaml --dataset cdd_cesm
python -m src.preprocessing.run_pipeline --config config.yaml --dataset vindr
```

> ⚠️ Rodar por dataset **sobrescreve** os splits e o `annotations.csv` com apenas aquele subconjunto.
> Para o experimento oficial, rode uma única vez com os três datasets juntos.

Ao final você deve ter:

```
data/hub/yolo/data_small.yaml
data/hub/yolo/data_medium.yaml
data/hub/yolo/data_large.yaml
data/hub/yolo/{small,medium,large}/{train,val,test}/{images,labels}/
data/hub/coco/{small,medium,large}/
data/hub/{annotations.csv, splits.json, checksums.json}
```

---

## 4. Treinamento

Um modelo é treinado **por faixa de tamanho** — 3 faixas × 2 arquiteturas = **6 treinos**.
Os pesos pré-treinados (`yolo11m.pt`, `rtdetr-l.pt`) devem estar na raiz do projeto; o Ultralytics
os baixa automaticamente se não existirem.

### YOLOv11m

```bash
python -m src.training.train_yolo --config config.yaml --bin small
python -m src.training.train_yolo --config config.yaml --bin medium
python -m src.training.train_yolo --config config.yaml --bin large
```

### RT-DETR-L

```bash
python -m src.training.train_detr --config config.yaml --bin small
python -m src.training.train_detr --config config.yaml --bin medium
python -m src.training.train_detr --config config.yaml --bin large
```

### Flags

| Flag | Valores | Descrição |
|:-----|:--------|:----------|
| `--config` | caminho | Padrão `config.yaml` |
| `--bin` | `small` · `medium` · `large` · `full` · `all` | **Obrigatório.** `all` roda as faixas em sequência |

> `full` (dataset agregado com negativos) **não é usado neste estudo** — `negative_sample_ratio: 0.0`
> faz o pipeline pular a exportação desse bin, então não existe `data_full.yaml`.

Hiperparâmetros (em `config.yaml → training`):

| | YOLOv11m | RT-DETR-L |
|:--|:--|:--|
| Otimizador | SGD | AdamW |
| `lr0` | 0.01 | 0.0001 |
| `weight_decay` | 0.0005 | 0.0001 |
| Épocas / paciência | 100 / 20 | 100 / 20 |
| Batch / imgsz | 8 / 640 | 8 / 640 |

**Saídas:** `runs/detect/results/yolo/yolo11m_{bin}/` e `runs/detect/results/detr/rtdetr_l_{bin}/`,
cada uma com `weights/best.pt`, `results.csv`, curvas PR/F1 e matriz de confusão.

---

## 5. Avaliação e testes estatísticos

Execute **nesta ordem** — cada etapa consome as saídas da anterior.

### 5.1 Avaliação formal no split de teste

```bash
python -m src.evaluation.evaluate_test --config config.yaml --device cuda:0
```

Carrega o `best.pt` de cada modelo × faixa, infere no split de teste e grava métricas por imagem.

| Flag | Padrão | Descrição |
|:-----|:-------|:----------|
| `--device` | `hardware.device` | `cuda:0` ou `cpu` |
| `--batch` | `evaluation.per_image_inference_batch` | Batch de inferência |
| `--train-root` | `runs/train` | Onde procurar `{yolo\|detr}_{bin}/weights/best.pt` |
| `--out-dir` | `results/evaluation` | Diretório de saída |

> **Resolução dos pesos:** procura primeiro em `--train-root/{modelo}_{bin}/weights/best.pt`; se não
> achar, cai automaticamente para `paths.results` do `config.yaml`
> (`runs/detect/results/{yolo,detr}/*{bin}*/weights/best.pt`). Com o layout padrão deste repositório
> o *fallback* é o que será usado — isso é esperado, não é erro.

**Gera:** `results/evaluation/test_per_image.csv` e `test_summary.csv`.

### 5.2 Testes estatísticos pareados

```bash
# Wilcoxon + McNemar a partir do CSV por imagem
python -m src.evaluation.statistical_tests --config config.yaml

# McNemar clínico (conf ≥ 0.25, IoU ≥ 0.3) — reexecuta inferência
python -m src.evaluation.statistical_tests --mcnemar-iou --device cuda:0
```

Wilcoxon signed-rank pareado por imagem (Bonferroni k=12) e McNemar por faixa (Bonferroni k=3).

**Gera:** `results/evaluation/wilcoxon_results.csv`, `mcnemar_results.csv`,
`mcnemar_iou03_results.csv`.

### 5.3 Curvas FROC

```bash
python -m src.evaluation.froc --config config.yaml --device cuda:0
```

Inferência com `conf=0.001`, pareamento com `IoU ≥ 0.3`, considerando apenas imagens com GT > 0.

**Gera:** `results/evaluation/froc_curves.png`, `froc_curves.pdf`, `froc_values.csv`,
`froc_operational_points.csv` (sensibilidade em 0,125 / 0,25 / 0,5 / 1 / 2 FP por imagem).

### 5.4 SAHI na faixa `small`

```bash
python -m src.evaluation.sahi_inference --config config.yaml --device cuda:0
```

Compara inferência com e sem *tiling* (SAHI) no teste da faixa `small`, com McNemar entre
YOLO+SAHI e RT-DETR+SAHI.

**Gera:** `results/evaluation/sahi_small_per_image.csv`, `sahi_small_summary.csv`.

### 5.5 Coleta consolidada de métricas

```bash
python -m src.evaluation.metrics --config config.yaml

# Reexecuta val() no teste para obter AR@100 e AP_small/medium/large (COCO)
python -m src.evaluation.metrics --config config.yaml --coco-val
```

**Gera:** `results/tables/metrics_all.csv` e `metrics_all.json`.

### 5.6 Cross-dataset (análise secundária)

```bash
python -m src.evaluation.cross_dataset --config config.yaml --bin all
```

**Gera:** `results/tables/cross_dataset.csv`.

> `src/evaluation/statistics.py` é um pipeline estatístico **alternativo/legado** que junta coleta de
> métricas + Wilcoxon + McNemar em uma chamada só. Foi mantido para reprodutibilidade de execuções
> anteriores; para o fluxo atual do artigo use `evaluate_test.py` → `statistical_tests.py`.

---

## 6. Explicabilidade (Grad-CAM e atenção)

```bash
python -m src.evaluation.heatmaps --all --device cuda:0
```

Seleciona 3 casos por faixa (TP em ambos os modelos, discordante pró-RT-DETR, FP do YOLO) e gera
painéis comparativos com Grad-CAM (YOLOv11m) e mapas de atenção cruzada (RT-DETR-L).

| Flag | Padrão | Descrição |
|:-----|:-------|:----------|
| `--all` | — | Processa todas as faixas |
| `--bins` | — | Subconjunto, ex.: `small,medium` |
| `--regen-attn` | desligado | Regenera **apenas** os mapas de atenção do RT-DETR |
| `--out-dir` | `results/heatmaps` | Diretório de saída |

**Gera:** `results/heatmaps/{small,medium,large}/*_painel.png` e
`results/heatmaps/individual/*_{original,gradcam,attn}.png`.

---

## Pipeline completo em um bloco

```bash
# 0 — auditoria dos datasets brutos
python -m src.preprocessing.audit_datasets --config config.yaml

# 1 — pré-processamento (teste seco, depois pra valer)
python -m src.preprocessing.run_pipeline --config config.yaml --dry-run
python -m src.preprocessing.run_pipeline --config config.yaml

# 2 — treino (6 runs)
python -m src.training.train_yolo --config config.yaml --bin small
python -m src.training.train_yolo --config config.yaml --bin medium
python -m src.training.train_yolo --config config.yaml --bin large
python -m src.training.train_detr --config config.yaml --bin small
python -m src.training.train_detr --config config.yaml --bin medium
python -m src.training.train_detr --config config.yaml --bin large

# 3 — avaliação formal
python -m src.evaluation.evaluate_test      --config config.yaml --device cuda:0
python -m src.evaluation.metrics            --config config.yaml --coco-val

# 4 — estatística
python -m src.evaluation.statistical_tests  --config config.yaml
python -m src.evaluation.statistical_tests  --config config.yaml --mcnemar-iou --device cuda:0

# 5 — FROC e SAHI
python -m src.evaluation.froc               --config config.yaml --device cuda:0
python -m src.evaluation.sahi_inference     --config config.yaml --device cuda:0

# 6 — explicabilidade
python -m src.evaluation.heatmaps           --config config.yaml --all --device cuda:0
```

---

## Configuração (`config.yaml`)

Tudo é controlado por um único arquivo. As chaves mais relevantes:

| Chave | Valor atual | O que faz |
|:------|:------------|:----------|
| `datasets_to_run` | `[inbreast, cdd_cesm, vindr]` | Quais datasets o pipeline carrega |
| `vindr_negatives` | `false` | Exclui casos "No Finding" do VinDr |
| `paths.*` | ver arquivo | Caminhos de entrada, hub e resultados |
| `image.target_size` | `640` | Resolução final (letterbox) |
| `crop.threshold` / `margin_px` | `10` / `10` | Recorte da região mamária |
| `clahe.clip_limit` / `tile_grid_size` | `2.0` / `[8,8]` | Parâmetros do CLAHE |
| `nodule_size_bins` | tercis P33/P67 | Limites das faixas em mm |
| `augmentation.flip_vertical` | `0.0` | **Nunca ativar** em mamografia |
| `augmentation.mosaic` | `0.0` | Desativado — preserva a escala das lesões |
| `splits` | `0.70 / 0.15 / 0.15`, `seed 42` | Divisão por paciente |
| `training.epochs` / `early_stopping_patience` | `100` / `20` | Orçamento de treino |
| `evaluation.iou_per_image_ap` | `0.5` | IoU para AP por imagem |
| `evaluation.conf_f1_per_image` | `0.25` | Confiança do critério clínico |
| `evaluation.min_samples_statistical_test` | `30` | n mínimo por teste estatístico |
| `hardware.device` / `seed` | `cuda:0` / `42` | Dispositivo e semente global |

**Rodando sem GPU:** altere `hardware.device` para `cpu` e reduza `training.epochs`. Treinar os 6
modelos em CPU é inviável na prática; use CPU apenas para validar o pipeline ponta a ponta.

---

## Saídas geradas

| Arquivo | Gerado por | Conteúdo |
|:--------|:-----------|:---------|
| `results/audit/audit_report.csv` | `audit_datasets` | QA dos datasets brutos |
| `data/hub/annotations.csv` | `run_pipeline` | Tabela mestre: caso, paciente, `size_mm`, bin, split, bbox |
| `data/hub/splits.json` | `run_pipeline` | Mapa `case_id → split` |
| `data/hub/checksums.json` | `run_pipeline` | MD5 de cada imagem processada |
| `runs/detect/results/**/weights/best.pt` | `train_yolo` / `train_detr` | Pesos treinados |
| `results/evaluation/test_per_image.csv` | `evaluate_test` | Métricas por imagem (pareadas) |
| `results/evaluation/test_summary.csv` | `evaluate_test` | Resumo por modelo × faixa |
| `results/evaluation/wilcoxon_results.csv` | `statistical_tests` | Wilcoxon + Bonferroni |
| `results/evaluation/mcnemar_iou03_results.csv` | `statistical_tests --mcnemar-iou` | McNemar clínico |
| `results/evaluation/froc_*.{csv,png,pdf}` | `froc` | Curvas FROC e pontos operacionais |
| `results/evaluation/sahi_small_*.csv` | `sahi_inference` | Efeito do SAHI na faixa small |
| `results/tables/metrics_all.{csv,json}` | `metrics` | mAP, P, R, F1 por modelo × faixa |
| `results/heatmaps/**` | `heatmaps` | Painéis Grad-CAM e atenção |

---

## Resultados de referência

Valores da execução de referência deste repositório (`results/`), split de **teste**.
Use-os para verificar se sua reprodução está consistente.

### Métricas no teste (`results/evaluation/test_summary.csv`)

| Modelo | Faixa | n | AP@0.5 | AP@0.5:0.95 | Precision | Recall | F1 |
|:-------|:------|--:|-------:|------------:|----------:|-------:|---:|
| YOLOv11m | small | 126 | 0,466 | **0,242** | **0,339** | 0,376 | **0,347** |
| RT-DETR-L | small | 126 | 0,463 | 0,210 | 0,132 | **0,569** | 0,204 |
| YOLOv11m | medium | 343 | **0,629** | **0,315** | **0,453** | 0,531 | 0,475 |
| RT-DETR-L | medium | 343 | 0,610 | 0,305 | 0,436 | **0,615** | **0,486** |
| YOLOv11m | large | 282 | **0,608** | **0,306** | 0,353 | 0,386 | 0,358 |
| RT-DETR-L | large | 282 | 0,575 | 0,282 | **0,514** | **0,523** | **0,506** |

### Testes pareados

**Wilcoxon** (Bonferroni k=12) — RT-DETR-L vence em **recall** nas três faixas; YOLOv11m vence em
**F1** na faixa `small`; RT-DETR-L vence em F1 na `large`. AP@0.5 e AP@0.5:0.95 não diferem
significativamente em nenhuma faixa.

**McNemar** (conf ≥ 0.25, IoU ≥ 0.3, Bonferroni k=3) — RT-DETR-L detecta significativamente mais
lesões em **todas** as faixas:

| Faixa | Só RT-DETR | Só YOLO | χ² | p corrigido |
|:------|-----------:|--------:|---:|------------:|
| small | 29 | 2 | 23,52 | 3,7 × 10⁻⁶ |
| medium | 52 | 12 | 25,00 | 1,7 × 10⁻⁶ |
| large | 82 | 8 | 60,84 | 1,9 × 10⁻¹⁴ |

### FROC — sensibilidade a 1 FP/imagem

| Faixa | YOLOv11m | RT-DETR-L |
|:------|---------:|----------:|
| small | **0,456** | 0,338 |
| medium | 0,627 | **0,662** |
| large | 0,595 | **0,673** |

### SAHI na faixa `small`

| Configuração | AP@0.5 | Recall | Taxa de detecção (IoU ≥ 0,3) |
|:-------------|-------:|-------:|-----------------------------:|
| YOLOv11m sem SAHI | 0,466 | 0,376 | 0,429 |
| YOLOv11m com SAHI | 0,400 | 0,419 | 0,468 |
| RT-DETR-L sem SAHI | 0,463 | 0,569 | 0,643 |
| RT-DETR-L com SAHI | 0,418 | 0,597 | **0,730** |

SAHI aumenta a sensibilidade nas lesões pequenas, mas ao custo de precisão — daí a queda no AP.

---

## Reprodutibilidade

- **Semente global 42** aplicada em `random`, `numpy` e `torch` via `src/utils/seed.py`, em todos os
  runners (pré-processamento, treino, avaliação).
- **Split por paciente** com validação explícita de vazamento — nenhuma paciente aparece em mais de
  um split.
- **`checksums.json`** com o MD5 de cada imagem processada permite confirmar que o pré-processamento
  produziu exatamente os mesmos pixels.
- `annotations.csv` e `splits.json` são pequenos e **podem ser versionados** para que outra pessoa
  reproduza os splits sem redistribuir imagens.

Pequenas variações de terceira casa decimal entre execuções são esperadas por conta de não
determinismo do cuDNN.

---

## Solução de problemas

<details>
<summary><strong><code>ModuleNotFoundError: No module named 'src'</code></strong></summary>

Você não está na raiz do projeto, ou está chamando o arquivo direto. Use sempre
`python -m src.<pacote>.<modulo>` a partir da pasta que contém `config.yaml`.
</details>

<details>
<summary><strong><code>data_{bin}.yaml não encontrado</code></strong></summary>

O pré-processamento não rodou (ou rodou com `--dry-run`). Execute
`python -m src.preprocessing.run_pipeline --config config.yaml` e confirme que
`data/hub/yolo/data_small.yaml` existe.
</details>

<details>
<summary><strong><code>best.pt não encontrado para {modelo}/{faixa}</code></strong></summary>

Nem `runs/train/` nem `paths.results` contêm os pesos. Ou o treino daquela faixa não terminou, ou os
pesos estão em outro lugar — nesse caso passe `--train-root <caminho>` apontando para a pasta que
contém `{yolo|detr}_{bin}/weights/best.pt`.
</details>

<details>
<summary><strong>CUDA out of memory</strong></summary>

Reduza `training.batch_size` no `config.yaml` (de 8 para 4 ou 2). Na avaliação, use
`--batch 1`. Se persistir, reduza `training.img_size` — mas isso muda o experimento e invalida a
comparação com os resultados de referência.
</details>

<details>
<summary><strong><code>Bin 'X': N casos no teste (mínimo=30)</code></strong></summary>

Aviso, não erro. Aquela faixa ficou com poucos casos no teste e os testes estatísticos daquela faixa
devem ser tratados como exploratórios. Ocorre quando você processa apenas um dataset.
</details>

<details>
<summary><strong>Erro instalando <code>albumentations</code> no Windows</strong></summary>

Não instale. Está comentado no `requirements.txt` de propósito — exige o Microsoft C++ Build Tools e
não é usado: a augmentation vem do Ultralytics.
</details>

<details>
<summary><strong>O pré-processamento demora horas</strong></summary>

Esperado. São ~1.700 imagens DICOM em alta resolução com CLAHE e recorte. Use `--no-visuals` para
economizar E/S e `--dataset` para processar em etapas durante a depuração.
</details>

---

## Módulos internos (sem CLI)

Estes são importados por outros scripts e **não têm ponto de entrada** — chamá-los com `python -m`
não faz nada:

| Módulo | Função |
|:-------|:-------|
| `src/evaluation/per_image_scores.py` | Cálculo pareado de scores por imagem (usado por `statistics.py`) |
| `src/evaluation/fusion.py` | *Late fusion* de detecções CC + MLO por caso clínico |
| `src/preprocessing/mammo_loaders.py`, `crop.py`, `clahe.py`, `export.py`, ... | Etapas chamadas por `run_pipeline.py` |

---

## Licença e uso dos dados

O código é distribuído sob **AGPL-3.0**, por compatibilidade com o [Ultralytics](https://github.com/ultralytics/ultralytics)
(YOLO11 / RT-DETR), que é AGPL-3.0.

**Os datasets não são redistribuídos neste repositório.** INbreast, CDD-CESM e VinDr-Mammo têm
licenças e termos de uso próprios, com restrições para uso clínico e comercial. Obtenha cada base
diretamente na fonte oficial e cite os autores originais em qualquer publicação.

> ⚠️ **Aviso:** este é um projeto de pesquisa. Os modelos aqui treinados **não são dispositivos
> médicos** e não devem ser usados para diagnóstico.

### Modelos base

- **YOLOv11m** — [Ultralytics](https://github.com/ultralytics/ultralytics)
- **RT-DETR-L** — *Real-Time DEtection TRansformer*, via integração Ultralytics
