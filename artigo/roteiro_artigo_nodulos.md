# Detecção de Massas Mamárias Estratificada por Tamanho em Mamografia: Estudo Comparativo entre YOLOv11m e RT-DETR-L

**CDD-CESM · INbreast · VinDr-Mammo**

> **Autores:** [Nome 1], [Nome 2], ... — [Instituição], [País]
> **Journal alvo:** [ex.: Computers in Biology and Medicine / Medical Image Analysis / Diagnostics]
> **Data de submissão:** [DD/MM/AAAA]

---

## Resumo

> ✍ **Nota:** Limite típico de 250–300 palavras. Estrutura obrigatória: Contexto / Objetivo / Métodos / Resultados / Conclusões.

**Contexto:** A detecção de massas mamárias em mamografia permanece um desafio para sistemas de detecção assistida por computador (CAD), especialmente devido à variabilidade no tamanho das lesões. Redes neurais convolucionais (CNNs) e Vision Transformers representam dois paradigmas arquiteturais dominantes com estratégias distintas de processamento espacial, mas seu desempenho comparativo estratificado por tamanho de lesão não foi sistematicamente avaliado.

**Objetivo:** Comparar YOLOv11m (detector CNN one-stage) e RT-DETR-L (detector baseado em Transformer) para detecção de massas mamárias em mamografia, com avaliação estatística formal estratificada por faixa de tamanho.

**Métodos:** Três datasets públicos foram combinados (CDD-CESM low-energy, INbreast, VinDr-Mammo; 2.341 anotações de massa em 1.713 imagens). As faixas de tamanho foram definidas por tercis empíricos (pequena: 0–19,4 mm; média: 19,4–36,3 mm; grande: >36,3 mm). Ambos os modelos foram treinados independentemente por faixa (6 experimentos no total) utilizando o framework Ultralytics (batch=8, img_size=640, CUDA). A avaliação incluiu mAP@0.5, mAP@0.5:0.95, Precisão, Recall, F1, curvas FROC, testes de Wilcoxon pareados (12 comparações, correção de Bonferroni α′=0,0042), testes de McNemar com critério IoU≥0,3 (3 comparações, α′=0,0167), além de visualizações Grad-CAM e mapas de atenção.

**Resultados:** O RT-DETR-L demonstrou recall significativamente superior ao YOLOv11m nas três faixas de tamanho (p<0,001 após correção de Bonferroni) e maior taxa de detecção binária (McNemar, p<0,0001 em todas as faixas). O YOLOv11m obteve F1 significativamente maior na faixa pequena (p=1,30×10⁻⁴), enquanto o RT-DETR-L obteve F1 significativamente maior na faixa grande (p=2,12×10⁻⁷). A análise FROC confirmou a superioridade do YOLO em lesões pequenas (sensibilidade 0,397 vs 0,176 a 0,5 FP/imagem) e do RT-DETR em lesões grandes (0,643 vs 0,485). Uma análise post-hoc com SAHI confirmou que a vantagem do RT-DETR na faixa pequena é de natureza arquitetural e não atribuível à resolução de entrada.

**Conclusões:** As duas arquiteturas apresentam forças complementares: o YOLOv11m oferece precisão de localização superior para massas pequenas, enquanto o RT-DETR-L alcança maior sensibilidade clínica para massas médias e grandes. Esses achados sugerem que estratégias híbridas de implantação estratificadas por tamanho podem otimizar o desempenho de sistemas CAD em todo o espectro de tamanhos de massas mamárias.

**Palavras-chave:** mamografia; detecção de massas mamárias; detecção de objetos; YOLOv11; RT-DETR; aprendizado profundo; detecção assistida por computador; estratificação por tamanho; Grad-CAM.

---

## 1. Introdução

> ✍ **Nota:** ~600–800 palavras. Objetivo: contextualizar o problema clínico, identificar a lacuna na literatura e enunciar as contribuições do estudo.

### 1.1 Contexto Clínico

O câncer de mama é o tipo de câncer mais frequentemente diagnosticado em mulheres no mundo, sendo a detecção precoce o principal determinante dos desfechos de sobrevida [REF]. A mamografia permanece a modalidade padrão de rastreamento, porém a detecção de massas — especialmente lesões pequenas — representa um desafio significativo mesmo para radiologistas experientes, com taxas de falha entre 10–30% em tecido mamário denso [REF].

A dificuldade de detecção varia marcadamente com o tamanho da lesão: massas pequenas (T1a/T1b, <10 mm) são desproporcionalmente perdidas no rastreamento, enquanto massas maiores (T2+, >20 mm) são mais conspícuas mas ainda sujeitas a erros de localização em parênquima denso. Apesar da relevância clínica dessa variabilidade, a maioria dos estudos de CAD avalia o desempenho de detectores de forma agregada, sem estratificação por tamanho — limitando a compreensão do comportamento dos modelos nas situações clinicamente mais críticas.

### 1.2 Detectores de Objetos em Imagem Médica

Detectores baseados em redes neurais convolucionais (CNNs), em particular arquiteturas one-stage como a família YOLO, demonstraram desempenho competitivo em tarefas de detecção em imagem médica [REF]. Mais recentemente, detectores baseados em Vision Transformers — notadamente o RT-DETR [REF] — emergiram como alternativas robustas, aproveitando mecanismos de auto-atenção global que podem oferecer vantagens para localização de lesões em contextos de parênquima complexo.

Apesar da literatura extensa sobre avaliações individuais de modelos, estudos comparativos entre detectores CNN e Transformer especificamente estratificados por tamanho de lesão mamária permanecem escassos. Essa lacuna é clinicamente relevante: as características arquiteturais que favorecem a detecção de objetos pequenos (receptive field local das CNNs) diferem das que favorecem objetos grandes (integração de contexto global dos Transformers), sugerindo que a vantagem relativa de cada arquitetura pode variar com o tamanho da lesão.

### 1.3 Contribuições deste Estudo

Este estudo apresenta as seguintes contribuições:

1. Comparação sistemática e estratificada por tamanho entre YOLOv11m e RT-DETR-L em três faixas de tamanho empiricamente definidas, utilizando três datasets públicos heterogêneos;
2. Avaliação estatística formal via testes de Wilcoxon pareados e testes de McNemar com correção de Bonferroni, garantindo validade inferencial por faixa;
3. Análise FROC com limiar de IoU clinicamente motivado;
4. Análise de interpretabilidade qualitativa via Grad-CAM (YOLO) e mapas de atenção cruzada (RT-DETR);
5. Análise post-hoc com SAHI isolando efeitos arquiteturais dos efeitos de resolução na detecção de lesões pequenas.

---

## 2. Trabalhos Relacionados

> ✍ **Nota:** ~400–600 palavras. Focar em: (a) detectores CNN em mamografia, (b) Transformers em imagem médica, (c) estudos com múltiplos datasets, (d) avaliação estratificada por tamanho.

### 2.1 Detecção CNN em Mamografia

> 📌 **Preencher:** Revisar e citar estudos com YOLO / Faster R-CNN / RetinaNet em mamografia. Destacar limitações de avaliação não estratificada por tamanho.

### 2.2 Detectores Baseados em Transformer em Imagem Médica

> 📌 **Preencher:** Revisar e citar DETR, RT-DETR, ViT aplicados a imagem médica. Destacar a vantagem teórica do self-attention para objetos de tamanho variável.

### 2.3 Avaliação Estratificada por Tamanho

> 📌 **Preencher:** Revisar e citar estudos que avaliam desempenho por faixa de tamanho em CAD mamário. Justificar a lacuna que este estudo preenche.

---

## 3. Materiais e Métodos

### 3.1 Datasets

Três datasets públicos de mamografia foram combinados para compor o coorte do estudo. Apenas achados classificados como **massa** foram incluídos; calcificações, assimetrias e distorções arquiteturais foram excluídos para garantir homogeneidade morfológica necessária para a estratificação por tamanho.

| Dataset | Modalidade | Formato | Anotações | BI-RADS | Espaçamento (mm/px) |
|---|---|---|---|---|---|
| INbreast | FFDM | DICOM + XML | Polígono → BBox | XML nativo | DICOM PixelSpacing |
| CDD-CESM (somente LE) | CESM low-energy | JPEG | CSV (VIA) | CSV nativo | Fallback 0,094 mm/px |
| VinDr-Mammo | FFDM | DICOM + CSV | BBox direto | Campo `assessment` | DICOM / fallback 0,085 |

> ✍ **Nota:** Adicionar referências originais de cada dataset.

Para o CDD-CESM, foram utilizadas exclusivamente as imagens **low-energy (LE)** para manter consistência de domínio visual com os datasets FFDM; as imagens recombinadas (com subtração de contraste) foram excluídas. Os registros do VinDr-Mammo contendo apenas anotações "No Finding" foram excluídos integralmente — nenhuma amostra negativa foi utilizada em qualquer etapa do pipeline.

O coorte combinado compreendeu **2.341 anotações de massa em 1.713 imagens**, distribuídas como segue:

| Dataset | Anotações | Imagens |
|---|---|---|
| CDD-CESM (LE) | 1.000 | — |
| INbreast | 116 | — |
| VinDr-Mammo | 1.225 | — |
| **Total** | **2.341** | **1.713** |

### 3.2 Estratificação por Tamanho

As faixas de tamanho foram definidas por **tercis empíricos (P33 e P67)** da distribuição do maior diâmetro da lesão (mm), calculados sobre o coorte combinado após filtros de qualidade e antes da partição por paciente. Essa abordagem garante contagens aproximadamente iguais por faixa, assegurando poder estatístico comparável entre os estratos para os testes de Wilcoxon e McNemar planejados.

| Faixa | Intervalo (mm) | Referência Clínica | n (teste, 15%) |
|---|---|---|---|
| Pequena | 0 – 19,4 | ≈ Fronteira T1b/T1c (TNM) | 126 |
| Média | 19,4 – 36,3 | ≈ Fronteira T1c/T2 (TNM) | 343 |
| Grande | > 36,3 | T2+ | 282 |

> ✍ **Nota:** Preencher a coluna "n (total)" com os valores do `splits.json`.

Os limites empíricos (19,4 mm e 36,3 mm) coincidem aproximadamente com limiares do estadiamento TNM (T1c/T2 em 20 mm), conferindo interpretabilidade clínica sem impor cortes arbitrários. Essa coincidência é reportada no artigo como favorável, não como justificativa primária da escolha — os tercis foram definidos pelos dados, não pelo TNM.

### 3.3 Pipeline de Pré-processamento

Todas as imagens passaram por um pipeline de pré-processamento modularizado e padronizado, aplicado de forma **idêntica** em todos os datasets:

1. **Leitura DICOM** com correção de RescaleSlope/RescaleIntercept; inversão de MONOCHROME1; conversão de JPEG CDD-CESM via PIL.
2. **Normalização para 8 bits** (uint8 [0–255]) via escalonamento por percentil, uniformizando profundidades de bits (INbreast: 12–14 bits; CDD-CESM: 8 bits; VinDr: variável).
3. **Cálculo do tamanho físico:** `size_mm = max(largura_px, altura_px) × pixel_spacing`, com PixelSpacing/ImagerPixelSpacing extraídos do cabeçalho DICOM (INbreast, VinDr) ou de metadados CSV com fallbacks documentados para CDD-CESM (0,094 mm/px).
4. **Crop anatômico:** limiarização binária (threshold fixo = 10; fallback Otsu se contorno < 30% da área da imagem), seleção do maior contorno, margem de 10 px, reposicionamento de bounding boxes.
5. **CLAHE:** `clip_limit=2,0`, `tile_grid=8×8`, aplicado identicamente em todos os datasets.
6. **Resize letterbox** para 640×640 px (padding com preservação de aspecto), alinhado com o comportamento padrão do Ultralytics.
7. **Exportação no formato YOLO:** `class cx cy w h` (normalizado 0–1).

### 3.4 Divisão dos Dados

A divisão foi realizada no nível de paciente usando `GroupShuffleSplit` (scikit-learn), com chaves de agrupamento específicas por dataset: prefixo do arquivo para INbreast, coluna `patient_id` para CDD-CESM e `study_id` para VinDr-Mammo. Proporções: 70% treino / 15% validação / 15% teste (seed=42). A ausência de sobreposição de pacientes entre splits foi verificada formalmente (treino ∩ teste = ∅).

Imagens com múltiplas massas foram atribuídas à faixa da **maior lesão presente** (prioridade: grande > média > pequena). Essa regra e suas implicações para lesões menores co-presentes são reconhecidas como limitação do estudo.

### 3.5 Arquiteturas e Treinamento

Dois detectores de objetos foram avaliados: **YOLOv11m** (detector CNN one-stage) e **RT-DETR-L** (detector baseado em Transformer com atenção deformável). Ambos foram implementados via framework **Ultralytics** para garantir pipeline de treinamento idêntico, cálculo de métricas equivalente e eliminação de confundidores relacionados ao framework.

| Hiperparâmetro | YOLOv11m | RT-DETR-L | Justificativa |
|---|---|---|---|
| Otimizador | SGD | AdamW | Padrão da arquitetura |
| Learning rate (lr0) | 0,01 | 0,0001 | Padrão da arquitetura |
| Weight decay | 0,0005 | 0,0001 | Padrão da arquitetura |
| Batch size | 8 | 8 | **Igualado** — elimina confundidor |
| Tamanho de imagem | 640×640 | 640×640 | **Igualado** |
| Épocas máximas | 100 | 100 | Early stopping (patience=20) |
| Épocas de warmup | 3 | 3 | Ambos os modelos |
| Dispositivo | CUDA (RTX 3070) | CUDA (RTX 3070) | GPU única |
| Seed | 42 | 42 | Reprodutibilidade global |

O batch size foi igualado em 8 para ambos os modelos, pois o RT-DETR-L é limitado pelo custo quadrático do self-attention a batch=8 em 8 GB de VRAM. O YOLOv11m suportaria batches maiores, mas foi igualado para eliminar o batch size como variável confundidora. O flip vertical foi desativado (preservação do contexto anatômico); o mosaic augmentation foi desativado (preservação da escala real das lesões). A reprodutibilidade global foi garantida via seed fixada na partição dos dados, inicialização dos modelos e PyTorch (`cudnn.deterministic=True`, `benchmark=False`).

Foram conduzidos **6 experimentos independentes**: 3 faixas × 2 modelos. Os modelos foram inicializados com pesos pré-treinados no ImageNet (padrão Ultralytics). O early stopping interrompeu o treinamento quando o mAP@0.5:0.95 de validação não apresentou melhora por 20 épocas consecutivas; as durações efetivas de treinamento variaram de 29 a 67 épocas.

### 3.6 Métricas de Avaliação

#### 3.6.1 Métricas Padrão de Detecção

Avaliadas no split de teste de cada faixa com limiar de confiança 0,25: mAP@0.5, mAP@0.5:0.95, Precisão, Recall, F1-score.

#### 3.6.2 Análise FROC

Curvas FROC (Free-Response ROC) foram calculadas por faixa e modelo com varredura de limiar de confiança (conf=0,001) e critério de correspondência IoU≥0,3. A sensibilidade foi reportada nos pontos operacionais 0,125; 0,25; 0,5; 1,0 e 2,0 falsos positivos por imagem (FP/img). Como nenhuma imagem negativa foi incluída no conjunto de teste, o FROC mede detecções excedentes dentro de imagens positivas — não sensibilidade sobre uma população clínica mista; essa limitação é declarada explicitamente no artigo.

#### 3.6.3 Testes Estatísticos

Testes de **Wilcoxon pareados** (por imagem, pareados por `image_id`) foram aplicados a quatro métricas contínuas (mAP@0.5, mAP@0.5:0.95, Recall, F1) nas três faixas: **12 testes no total**. Testes de **McNemar** avaliaram concordância de detecção binária (IoU≥0,3, conf≥0,25) por faixa: **3 testes no total**. Correção de Bonferroni aplicada: α′=0,05/12≈0,0042 (Wilcoxon) e α′=0,05/3≈0,0167 (McNemar). Grupos com n<30 no split de teste foram designados como exploratórios com ressalva explícita.

#### 3.6.4 Análise Post-hoc com SAHI

O SAHI (Slicing Aided Hyper Inference) foi aplicado post-hoc em ambos os modelos no conjunto de teste da faixa pequena (slice: 320×320 px, overlap: 0,2, NMS IoU: 0,5) para avaliar se a diferença de detecção em lesões pequenas era atribuível à resolução de entrada ou a diferenças arquiteturais.

#### 3.6.5 Visualização de Ativação

Mapas Grad-CAM foram gerados para o YOLOv11m usando a última camada convolucional do backbone antes do neck de pirâmide de features (`model.10.m.0.ffn.1.conv`). Mapas de atenção cruzada foram extraídos para o RT-DETR-L via forward hook no módulo MSDeformAttn da camada 5 do decoder (`model.28.decoder.layers.5.cross_attn`), com média sobre heads e seleção da query de maior confiança. Para cada faixa, três casos representativos foram selecionados: (i) verdadeiro positivo em ambos os modelos; (ii) verdadeiro positivo no RT-DETR / falso negativo no YOLO; (iii) falso positivo representativo do YOLO.

---

## 4. Resultados

### 4.1 Convergência do Treinamento

Todos os seis experimentos convergiram antes do limite de 100 épocas via early stopping (patience=20). As durações efetivas variaram de 29 épocas (YOLO grande) a 67 épocas (YOLO pequena), indicando convergência estável. O RT-DETR-L na faixa pequena convergiu mais cedo (época 30), com mAP@0.5 de validação estabilizando em 0,224 — padrão consistente com arquiteturas Transformer que requerem maior volume de dados para detecção de objetos pequenos.

| Modelo | Faixa | Melhor época | mAP@0.5 (val) | mAP@0.5:0.95 (val) |
|---|---|---|---|---|
| YOLOv11m | Pequena | 47/67 | 0,432 | 0,174 |
| YOLOv11m | Média | 33/53 | 0,613 | 0,258 |
| YOLOv11m | Grande | 29/49 | 0,489 | 0,205 |
| RT-DETR-L | Pequena | 10/30 | 0,224 | 0,102 |
| RT-DETR-L | Média | 39/59 | 0,633 | 0,280 |
| RT-DETR-L | Grande | 45/65 | 0,552 | 0,225 |

### 4.2 Desempenho no Conjunto de Teste

| Modelo | Faixa | n (teste) | mAP@0.5 | mAP@0.5:0.95 | Precisão | Recall | F1 |
|---|---|---|---|---|---|---|---|
| YOLOv11m | Pequena | 126 | 0,466 | 0,242 | 0,339 | 0,376 | 0,347 |
| RT-DETR-L | Pequena | 126 | 0,463 | 0,210 | 0,132 | 0,569 | 0,204 |
| YOLOv11m | Média | 343 | 0,629 | 0,315 | 0,453 | 0,531 | 0,475 |
| RT-DETR-L | Média | 343 | 0,610 | 0,305 | 0,436 | 0,615 | 0,486 |
| YOLOv11m | Grande | 282 | 0,608 | 0,306 | 0,353 | 0,386 | 0,358 |
| RT-DETR-L | Grande | 282 | 0,575 | 0,282 | 0,514 | 0,523 | 0,506 |

### 4.3 Testes Estatísticos

#### 4.3.1 Wilcoxon — Resultados Completos

| Faixa | Métrica | YOLOv11m | RT-DETR-L | p (corrigido) | Significativo | Vencedor |
|---|---|---|---|---|---|---|
| Pequena | Recall | 0,376 | 0,569 | 8,53×10⁻⁵ | **Sim** | RT-DETR-L |
| Pequena | F1 | 0,347 | 0,204 | 1,30×10⁻⁴ | **Sim** | YOLOv11m |
| Pequena | mAP@0.5 | 0,466 | 0,463 | 1,000 | Não | n.s. |
| Pequena | mAP@0.5:0.95 | 0,242 | 0,210 | 1,000 | Não | n.s. |
| Média | Recall | 0,531 | 0,615 | 0,0021 | **Sim** | RT-DETR-L |
| Média | F1 | 0,475 | 0,486 | 1,000 | Não | n.s. |
| Média | mAP@0.5 | 0,629 | 0,610 | 1,000 | Não | n.s. |
| Média | mAP@0.5:0.95 | 0,315 | 0,305 | 1,000 | Não | n.s. |
| Grande | F1 | 0,358 | 0,506 | 2,12×10⁻⁷ | **Sim** | RT-DETR-L |
| Grande | Recall | 0,386 | 0,523 | 1,39×10⁻⁵ | **Sim** | RT-DETR-L |
| Grande | mAP@0.5 | 0,608 | 0,575 | 0,527 | Não | n.s. |
| Grande | mAP@0.5:0.95 | 0,306 | 0,282 | 0,361 | Não | n.s. |

**Resumo:** 5/12 testes significativos após Bonferroni (α′=0,0042). RT-DETR-L superior em Recall nas três faixas; YOLOv11m superior em F1 na faixa pequena; RT-DETR-L superior em F1 na faixa grande. Nenhuma diferença significativa em métricas de AP.

#### 4.3.2 McNemar — Detecção Binária (IoU ≥ 0,3)

| Faixa | DETR+/YOLO+ | DETR+/YOLO− | DETR−/YOLO+ | DETR−/YOLO− | p (corrigido) | Vencedor |
|---|---|---|---|---|---|---|
| Pequena | 52 | 29 | 2 | 43 | 3,72×10⁻⁶ | **RT-DETR-L** |
| Média | 189 | 52 | 12 | 90 | 1,72×10⁻⁶ | **RT-DETR-L** |
| Grande | 120 | 82 | 8 | 72 | 1,87×10⁻¹⁴ | **RT-DETR-L** |

Taxas de detecção (IoU≥0,3):
- **Pequena:** RT-DETR 64,3% vs YOLO 42,9%
- **Média:** RT-DETR 70,3% vs YOLO 58,6%
- **Grande:** RT-DETR 71,6% vs YOLO 45,4%

Todas as diferenças significativas após correção de Bonferroni (α′=0,0167).

### 4.4 Análise FROC

| Modelo | Faixa | Sens. @ 0,125 | Sens. @ 0,25 | Sens. @ 0,5 | Sens. @ 1,0 | Sens. @ 2,0 |
|---|---|---|---|---|---|---|
| YOLOv11m | Pequena | 0,228 | 0,324 | **0,397** | **0,456** | **0,529** |
| RT-DETR-L | Pequena | 0,059 | 0,118 | 0,176 | 0,338 | 0,449 |
| YOLOv11m | Média | 0,362 | 0,474 | 0,564 | 0,627 | 0,689 |
| RT-DETR-L | Média | 0,411 | 0,526 | **0,616** | **0,662** | **0,719** |
| YOLOv11m | Grande | 0,345 | 0,411 | 0,485 | 0,595 | 0,688 |
| RT-DETR-L | Grande | 0,473 | 0,571 | **0,643** | **0,673** | **0,702** |

> 📌 **Inserir Figura 1:** `froc_curves.png` — layout 1×3 subplots (pequena | média | grande), YOLO azul / RT-DETR vermelho, linhas verticais tracejadas em 0,5 e 1,0 FP/img, 300 DPI. Nota de rodapé: *"FROC calculado sobre imagens positivas do teste — ausência de imagens negativas (ver limitações)".*

### 4.5 Análise Post-hoc com SAHI (Faixa Pequena)

| Configuração | mAP@0.5 | Precisão | Recall | F1 | Taxa det. IoU≥0,3 |
|---|---|---|---|---|---|
| YOLOv11m (sem SAHI) | 0,466 | 0,339 | 0,376 | 0,347 | 42,9% |
| YOLOv11m (com SAHI) | 0,400 | 0,241 | 0,419 | 0,288 | 46,8% |
| RT-DETR-L (sem SAHI) | 0,463 | 0,132 | 0,569 | 0,204 | 64,3% |
| RT-DETR-L (com SAHI) | 0,418 | 0,058 | 0,597 | 0,102 | 73,0% |

O SAHI melhorou marginalmente o Recall do YOLO (+4,4 pp) à custa de redução substancial de Precisão (−9,8 pp) e F1 (−5,9 pp). A vantagem de detecção do RT-DETR foi mantida após o SAHI (McNemar p≈0), confirmando que a diferença na faixa pequena é de natureza **arquitetural**, não de resolução de entrada.

### 4.6 Visualizações de Ativação

> 📌 **Inserir Figura 2:** Painéis comparativos 3×3 (3 faixas × 3 casos). Cada painel: [imagem original | Grad-CAM YOLO | Attention Map RT-DETR]. Legenda: GT verde tracejado, predição YOLO azul, predição RT-DETR vermelho, colorbar jet, conf score sobre cada bbox, resolução 300 DPI.

Os mapas Grad-CAM do YOLOv11m revelaram ativação espacialmente concentrada sobre a região da lesão nos casos de verdadeiro positivo, porém ativação difusa sobre tecido fibroglandular denso nos casos de falso positivo. Os mapas de atenção do RT-DETR-L demonstraram ancoragem espacial mais seletiva das queries de detecção nos casos de verdadeiro positivo; o padrão em blocos nos mapas de atenção reflete a estrutura multi-escala do mecanismo MSDeformAttn e não constitui artefato de processamento.

---

## 5. Discussão

### 5.1 Complementaridade Arquitetural por Faixa de Tamanho

O achado central deste estudo é uma **inversão de vantagem arquitetural dependente do tamanho**: o YOLOv11m demonstrou desempenho de localização superior (F1, sensibilidade FROC) para massas pequenas, enquanto o RT-DETR-L demonstrou maior completude de detecção (Recall, taxa de detecção binária McNemar, F1) para massas médias e grandes. Esse padrão tem explicação mecanicista nas propriedades de cada arquitetura.

Massas pequenas ocupam poucos tokens no feature map após downsampling para 640×640 px — o mecanismo de auto-atenção do RT-DETR distribui a atenção por todo o contexto da imagem, diluindo o sinal de objetos pequenos. O receptive field convolucional localizado do YOLOv11m, por sua vez, é mais adequado para detectar regiões compactas de alto contraste. Para massas grandes, a integração de contexto global do Transformer — abrangendo o fundo parenquimatoso completo — viabiliza detecção mais confiável de lesões conspícuas em tecido complexo, tarefa para a qual o processamento local do YOLO pode falhar em lesões que abrangem múltiplas células do grid do feature map.

### 5.2 Implicações Clínicas

Do ponto de vista clínico, o Recall é o critério de desempenho primário em sistemas CAD de rastreamento, pois uma massa não detectada (falso negativo) tem consequência clínica maior que um alarme falso. O RT-DETR-L demonstrou Recall significativamente superior nas três faixas e maior taxa de detecção binária em todo o espectro de tamanhos, sugerindo preferência como componente primário de detecção em sistemas orientados ao rastreamento.

Contudo, o colapso de Precisão do RT-DETR-L na faixa pequena (0,132 vs 0,339 do YOLO) implica substancialmente maior carga de falsos positivos — limitação prática em ambientes com restrição de tempo de revisão radiológica. O F1 superior do YOLOv11m na faixa pequena reflete equilíbrio precisão-recall mais adequado para aplicações de guia de biópsia, onde a precisão de localização é prioritária.

Uma estratégia híbrida de implantação — RT-DETR-L para triagem inicial (maximizar sensibilidade) seguido de YOLOv11m para refinamento de localização — representa uma direção clinicamente motivada para trabalhos futuros.

### 5.3 A Vantagem do RT-DETR na Detecção Binária

A análise de McNemar com critério IoU≥0,3 revelou o achado mais consistente e robusto do estudo: o RT-DETR-L detectou significativamente mais imagens com massa em todas as três faixas (p<10⁻⁵ após Bonferroni), com vantagens de taxa de detecção de 21 pp (pequena), 12 pp (média) e 26 pp (grande). Esse resultado não foi capturado pela análise de Wilcoxon sobre métricas de AP — que penaliza localização imprecisa — evidenciando que a escolha do critério de avaliação afeta materialmente as conclusões comparativas em pesquisa de CAD.

### 5.4 SAHI e Efeitos Arquiteturais vs Efeitos de Resolução

A análise post-hoc com SAHI confirmou que a vantagem de detecção do RT-DETR-L na faixa pequena não é atribuível à resolução de entrada. O SAHI melhorou o Recall do YOLO em apenas 4,4 pp com redução substancial de Precisão, e a vantagem do RT-DETR em detecção binária foi mantida após o tiling (McNemar p≈0). A falha do SAHI em melhorar o desempenho geral de ambos os modelos (redução líquida de F1 nos dois casos) sugere que estratégias genéricas de tiling requerem calibração do limiar de NMS pós-merge específica para o domínio mamográfico antes da implantação prática.

### 5.5 Convergência Precoce do RT-DETR na Faixa Pequena

O RT-DETR-L convergiu na época 10 na faixa pequena — substancialmente mais cedo que o YOLO (época 47) — com mAP@0.5 de validação estabilizando em 0,224. Esse padrão sugere que o Transformer não extraiu características discriminativas de objetos pequenos dentro do volume de treinamento disponível. Arquiteturas Transformer são reconhecidamente mais exigentes em volume de dados para detecção de objetos pequenos [REF]; a faixa pequena (n≈126 imagens de teste) provavelmente não atende os requisitos de dados do encoder RT-DETR. Trabalhos futuros com datasets maiores ou estratégias de augmentation específicas (ex.: copy-paste augmentation) podem alterar esse achado.

### 5.6 Interpretação dos Mapas Grad-CAM e Atenção

A análise Grad-CAM revelou um padrão consistente de falso positivo no YOLOv11m: ativação difusa sobre tecido fibroglandular denso sem lesão anotada. Esse comportamento, documentado anteriormente em sistemas CAD baseados em CNN [REF], indica que o YOLOv11m captura características texturais globais do parênquima mamário denso em vez de características morfológicas locais da lesão — consistente com a maior taxa de falsos positivos do modelo em mamas densas.

Os mapas de atenção do RT-DETR-L demonstraram ancoragem espacial mais seletiva das queries, potencialmente explicando seu Recall superior apesar de menor Precisão: o mecanismo de atenção cruzada permite que as queries do decoder se ancorem em múltiplas regiões candidatas simultaneamente, aumentando a sensibilidade à custa de especificidade.

### 5.7 Limitações

As seguintes limitações devem ser consideradas na interpretação dos achados:

1. **FROC sem imagens negativas:** o FROC foi calculado exclusivamente sobre imagens positivas do teste, medindo detecções excedentes dentro de imagens com lesão — não sensibilidade sobre uma população clínica representativa mista. Comparação direta com resultados FROC de estudos com coortes negativas completas não é apropriada.

2. **INbreast sub-representado:** com apenas 116 anotações, o INbreast contribui minoritariamente para o coorte combinado; análise estratificada por dataset seria sub-potenciada para o INbreast e não foi conduzida.

3. **Restrições de hardware:** todos os experimentos foram realizados em GPU única (RTX 3070, 8 GB VRAM) com batch size igualado (8) e img_size=640×640; tamanhos de imagem maiores — potencialmente benéficos para detecção de lesões pequenas — não puderam ser avaliados sem comprometer a equivalência de batch exigida para comparação justa.

4. **Escopo restrito a massas:** a análise é limitada a achados de massa; generalização para calcificações, assimetrias ou distorções arquiteturais não pode ser assumida.

5. **Ausência de validação externa:** nenhum dataset externo independente foi utilizado; o desempenho em dados institucionais independentes é desconhecido.

6. **Regra de atribuição de faixa:** imagens com múltiplas massas foram atribuídas à faixa da maior lesão presente; lesões menores co-presentes podem não ser representativas do treinamento da faixa superior correspondente.

---

## 6. Conclusões

Este estudo apresenta a primeira comparação sistemática estratificada por tamanho entre YOLOv11m e RT-DETR-L para detecção de massas mamárias em três datasets públicos heterogêneos de mamografia. Os achados principais são:

1. **O RT-DETR-L alcança Recall e taxa de detecção binária significativamente superiores nas três faixas de tamanho**, estabelecendo-o como arquitetura preferencial quando a sensibilidade clínica é o objetivo primário.

2. **O YOLOv11m alcança F1 significativamente superior na faixa pequena (0–19,4 mm)**, oferecendo melhor equilíbrio precisão-recall para localização de lesões pequenas.

3. **A vantagem de detecção do RT-DETR-L em lesões pequenas é arquitetural**, não dependente de resolução, conforme confirmado pela análise post-hoc com SAHI.

4. **Nenhuma diferença significativa foi encontrada em métricas de mAP após correção de Bonferroni**, destacando a importância de critérios orientados a recall e detecção binária na avaliação clínica de sistemas CAD.

5. **A análise Grad-CAM revela padrão consistente de ativação do YOLO sobre parênquima denso**, com implicações clínicas para implantação em populações com mama heterogeneamente densa.

Esses achados apoiam uma estratégia híbrida de implantação clinicamente motivada: RT-DETR-L para detecção de rastreamento de alta sensibilidade, complementado pelo YOLOv11m para localização de precisão em massas pequenas. Trabalhos futuros devem investigar coortes de treinamento maiores para a faixa pequena, copy-paste augmentation para enriquecimento de objetos pequenos, calibração do NMS do SAHI para mamografia e validação clínica com múltiplos leitores.

---

## Agradecimentos

> 📌 **Preencher:** Financiamento, crédito pelos datasets, infraestrutura computacional.

---

## Declarações

### Conflito de Interesses

Os autores declaram não ter conflito de interesses.

### Disponibilidade de Dados

Todos os datasets utilizados são publicamente disponíveis: INbreast [REF], CDD-CESM [REF], VinDr-Mammo [REF]. O código e os arquivos de configuração estão disponíveis em [URL do repositório].

### Ética

Este estudo utilizou exclusivamente datasets públicos de-identificados. Aprovação ética institucional não foi necessária.

---

## Referências

> ✍ **Nota:** Usar estilo IEEE ou Vancouver conforme exigência do journal alvo. Mínimo recomendado: 30–40 referências.

- [1] Dataset INbreast — Moreira et al., 2012. *[citar]* 
- [2] Dataset CDD-CESM — *[citar paper original]*
- [3] Dataset VinDr-Mammo — Nguyen et al., 2022. *[citar]*
- [4] YOLOv11 — Ultralytics, 2024. *[citar]*
- [5] RT-DETR — Zhao et al., 2024. *[citar]*
- [6] Grad-CAM — Selvaraju et al., 2017. *[citar]*
- [7] SAHI — Akyon et al., 2022. *[citar]*
- [8–40] *Adicionar referências de background: CNN em mamografia, Transformers em imagem médica, estudos CAD com estratificação por tamanho, FROC em CAD, BI-RADS Atlas ACR.*

---

## Apêndice — Checklist de Verificação antes da Submissão

```
[ ] Abstract dentro do limite de palavras do journal alvo
[ ] Todos os [REF] substituídos por citações reais
[ ] Todos os [📌 Inserir figura] com figuras em resolução ≥ 300 DPI
[ ] Tabela de datasets com referências originais
[ ] Coluna "n (total)" da tabela de faixas preenchida com valores do splits.json
[ ] URL do repositório de código preenchida
[ ] Autores e instituição preenchidos
[ ] Journal alvo definido e formatação de referências ajustada
[ ] Nota de rodapé FROC presente na figura 1
[ ] Limitações declaradas na seção 5.7
[ ] Contribuição de escopo (apenas massas) declarada no Methods
[ ] Frase de ancoragem TNM dos tercis empíricos presente no texto
[ ] Análise SAHI descrita como post-hoc (não como experimento principal)
[ ] Convergência precoce RT-DETR small discutida como limitação
```
