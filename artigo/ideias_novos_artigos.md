# Propostas de Novos Artigos

Este documento detalha duas ideias de artigos científicos que podem ser desenvolvidos aproveitando o *pipeline*, os dados e os modelos já treinados (YOLOv11m e RT-DETR-L) do projeto atual de detecção de massas mamárias.

---

## Ideia 1: O Impacto do "Domain Shift" (Generalização Cruzada) em Mamografia: CNNs vs Transformers

### Contexto e Motivação
No estudo anterior, os três *datasets* (INbreast, CDD-CESM e VinDr-Mammo) foram combinados antes da partição de treino e teste. No entanto, na prática clínica real, um modelo treinado nos dados de um hospital (ou país) frequentemente sofre uma queda drástica de desempenho quando aplicado a dados de outro hospital, devido a diferenças no equipamento, protocolos de aquisição e demografia da população. Esse fenômeno é conhecido como *Domain Shift*.

### Pergunta de Pesquisa
Qual arquitetura de aprendizado profundo é mais robusta e sofre menor degradação de desempenho ao enfrentar *Domain Shift* em mamografia: redes com campos receptivos locais (CNNs - YOLOv11m) ou redes com atenção global (Transformers - RT-DETR-L)?

### Metodologia Proposta
1. **Treinamento Isolado:** Treinar o YOLOv11m e o RT-DETR-L usando exclusivamente os dados de um *dataset* (ex: VinDr-Mammo, que possui maior volume de dados).
2. **Avaliação Zero-Shot:** Testar os modelos treinados diretamente nos outros *datasets* (INbreast e CDD-CESM) sem nenhum re-treinamento (*fine-tuning*).
3. **Métricas:** Comparar a queda de sensibilidade (Recall) e precisão, avaliando as curvas FROC para cada modelo no novo domínio.

### Esforço de Implementação (Baixo)
- Não exige código novo estrutural.
- Requer apenas a modificação da lógica de separação de treino/teste para isolar o particionamento dos dados por fonte (possivelmente via `config.yaml` ou ajustes no `split`).
- Re-executar os scripts de treinamento e avaliação já existentes.

---

## Ideia 2: Estratégia Híbrida (Ensemble / Late Fusion): Combinando o Melhor dos Dois Mundos

### Contexto e Motivação
A conclusão do artigo anterior revelou uma complementaridade arquitetural clara: o YOLOv11m é superior em precisão para lesões pequenas, enquanto o RT-DETR-L é superior em sensibilidade geral para lesões médias e grandes. Uma progressão natural dessa descoberta é a implementação de um sistema CAD híbrido que combine as predições de ambos os modelos para maximizar a performance clínica.

### Pergunta de Pesquisa
A fusão tardia (*Late Fusion*) das predições de um detector *One-Stage* (YOLOv11m) e de um *Transformer* (RT-DETR-L) consegue gerar uma curva FROC superior ao desempenho de ambas as arquiteturas isoladamente, mitigando o alto número de falsos positivos do Transformer em lesões pequenas?

### Metodologia Proposta
1. **Extração de Predições:** Utilizar os pesos já treinados para gerar *bounding boxes* e níveis de confiança para todas as imagens do conjunto de teste combinado.
2. **Fusão de Caixas (Ensemble):** Implementar algoritmos de fusão, como *Non-Maximum Suppression* (NMS) ou *Weighted Boxes Fusion* (WBF), para unificar as predições do YOLO e do DETR em uma única saída por imagem.
3. **Avaliação:** Submeter as predições fundidas ao pipeline de avaliação existente (mAP, FROC, Teste de McNemar) e comparar estatisticamente com os modelos base.

### Esforço de Implementação (Moderado)
- Os modelos já estão treinados e as inferências base podem ser extraídas diretamente.
- Exige a criação de um novo script (ex: `src/evaluation/ensemble.py`) para ler os arquivos de predição, aplicar a lógica de NMS/WBF e salvar as predições finais combinadas.
- Executar os scripts de avaliação `froc.py` e `statistics.py` sobre as predições fundidas para gerar os resultados e gráficos.

---

## Ideia 3: Segmentação Guiada por Prompt Semântico (BI-RADS) e Classificação de Malignidade Baseada em Radiômica

### Contexto e Motivação
A sua excelente ideia aborda uma das maiores limitações dos modelos de detecção puros (como YOLO e RT-DETR): eles funcionam como "caixas pretas" (*black-boxes*). Eles fornecem a localização da lesão, mas não explicam o *porquê* de considerá-la suspeita. Na prática clínica, radiologistas analisam a **margem** da lesão (circunscrita, obscurecida, microlobulada, indistinta, espiculada) segundo o léxico BI-RADS para prever a malignidade. Ao extrair características radiômicas (que quantificam textura e forma a nível de pixel) de uma lesão muito bem segmentada, é possível criar um modelo altamente acurado e, mais importante, **interpretável**.

### A Ideia Refinada (Pipeline do Artigo)
A proposta consiste em um sistema de diagnóstico em 3 estágios, utilizando os modelos atuais como ponto de partida:

1. **Segmentação Guiada por Text/Box Prompt (Foundation Models):** 
   - **Como funciona:** Em vez de usar apenas caixas delimitadoras, combinamos a caixa (gerada pelo YOLO/DETR) com um *text prompt* contendo o descritor BI-RADS da margem (ex: "Segmentar nódulo de margem espiculada").
   - **Modelo sugerido:** Utilizar modelos de fundação adaptados para área médica (ex: MedSAM - *Segment Anything for Medical Images*). O texto força o modelo a focar nos detalhes finos da borda, gerando uma máscara (mask) extremamente precisa do nódulo, capturando exatamente as espiculações ou lobulações.

2. **Extração de Características (Radiomics):**
   - **Como funciona:** Usar a biblioteca genérica de radiômica (ex: `pyradiomics`) para analisar a máscara gerada no passo anterior.
   - **O que é extraído:** Fatores matemáticos invisíveis a olho nu, como heterogeneidade de textura, curtose, entropia, e medidas precisas de irregularidade de forma.

3. **Classificação Híbrida e Interpretável:**
   - **Como funciona:** Treinar um classificador final (como XGBoost, Random Forest ou uma rede neural simples).
   - **Input:** As características de Radiômica + a categoria da margem BI-RADS.
   - **Output:** Predição final (Benigno vs. Maligno) ou grau de suspeita.

### Pergunta de Pesquisa
"A utilização de descritores semânticos de margem do BI-RADS como *prompts* para refinar a segmentação de nódulos mamários melhora significativamente a extração de assinaturas radiômicas e a acurácia final da predição de malignidade?"

### Por que esta é uma publicação forte?
- **Inovação:** O uso de descritores clínicos como *prompts* para modelos de fundação em visão computacional médica (Vision-Language) é o "estado da arte" absoluto (tendência de 2024/2025).
- **Interpretabilidade (XAI):** O modelo deixará de dizer apenas "Probabilidade 90%". Ele fundamentará a decisão: "Malignidade 90% baseada em contorno microlobulado e alta entropia de textura". A aceitação clínica para esse tipo de artigo é altíssima.
- **Progressão Natural:** Você aproveita o seu YOLO/RT-DETR para gerar a localização inicial (*bounding box*), tornando este artigo a continuação perfeita do seu trabalho atual.

### Esforço de Implementação (Alto)
- Requer a integração de um novo modelo para segmentação baseada em *prompt* (como o SAM).
- Exige o mapeamento/criação de anotações a nível de pixel (máscaras) caso o modelo precise de *fine-tuning* (embora o zero-shot com SAM seja possível).
- Requer a implementação do pipeline de extração de radiômica (`pyradiomics`) e do classificador final.
