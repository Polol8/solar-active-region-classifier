# Fluxo do Projeto: Solar Classifier (YOLOv11)

Este documento descreve a arquitetura de dados e o fluxo de processamento do sistema de detecção e classificação de Regiões Ativas (ARs) solares utilizando magnetogramas do SDO/HMI.

## 1. Visão Geral
O objetivo do projeto é treinar um modelo de visão computacional (**YOLOv11**) para localizar regiões ativas no Sol e classificá-las segundo a **classificação de Mount Wilson**, que indica a complexidade magnética da região e sua probabilidade de gerar erupções solares (flares).

---

## 2. Pipeline de Dados (Arquitetura)

O fluxo de dados é linear e modular, transformando dados astronômicos brutos em um dataset formatado para Deep Learning:

`JSOC (FITS)` $\xrightarrow{download.py}$ `Local Storage` $\xrightarrow{preprocess.py}$ `PNG + WCS JSON` $\xrightarrow{labels.py}$ `YOLO Labels (.txt)` $\xrightarrow{train.py}$ `YOLOv11 Model`

---

## 3. Detalhamento das Etapas

### Etapa 1: Aquisição de Dados (`src/download.py`)
Os dados são obtidos do **JSOC (Joint Science Operations Center)**, especificamente a série `hmi.M_720s` (magnetogramas de linha de visão).

-   **Protocolo de Exportação**: O download segue um fluxo de 3 etapas via HTTP:
    1.  **Request**: Solicita a exportação de um intervalo de datas.
    2.  **Polling**: Aguarda o JSOC processar a solicitação.
    3.  **Streaming**: Baixa os arquivos FITS resultantes.
-   **Estratégia**: Implementação assíncrona com `aiohttp` e divisão de grandes períodos em blocos de 90 dias para evitar timeouts no servidor.

### Etapa 2: Pré-processamento (`src/preprocess.py`)
Arquivos FITS (formato astronômico) não podem ser lidos diretamente por redes neurais. Esta etapa converte a física em imagem.

-   **Correção de Orientação**: Verifica e corrige o ângulo `CROTA2` para garantir que a imagem esteja sempre com o **Norte Solar para cima**.
-   **Normalização Magnética**:
    -   O campo magnético (em Gauss) é clipado em $\pm 1000$ G.
    -   Valores são mapeados linearmente para `uint8 [0, 255]`, onde o "sol calmo" (0 G) torna-se cinza médio ($\approx 127$).
-   **Redimensionamento**: Imagens são redimensionadas para $1024 \times 1024$ usando interpolação de Lanczos para preservar detalhes de manchas solares pequenas.
-   **Sidecar WCS (World Coordinate System)**: Como o redimensionamento altera a geometria, é gerado um arquivo `.json` com os parâmetros de escala e referência. Isso é vital para converter coordenadas do catálogo HEK (em arcsegundos) para pixels da imagem PNG.

### Etapa 3: Geração de Labels (`src/labels.py`)
As "verdades de campo" (ground truth) são extraídas do **HEK (Heliophysics Event Knowledgebase)**.

-   **Classificações (Classes YOLO)**:
    -   `0: Alpha` (Unipolar, baixa complexidade)
    -   `1: Beta` (Bipolar simples)
    -   `2: BetaGamma` (Bipolar complexo)
    -   `3: BetaGammaDelta` (Complexidade máxima, alta probabilidade de flares)
-   **Estratégia de Consulta**: Para evitar milhares de requisições HTTP, o sistema faz **uma query por dia**, armazena os eventos em cache e associa as imagens aos eventos que ocorreram dentro de uma janela de $\pm 30$ minutos.
-   **Geometria da Bounding Box**:
    1.  Prioriza dados do pipeline **SHARP**, que fornece a extensão real do patch magnético.
    2.  Fallback para **SPoCA** via proximidade espacial.
    3.  Último recurso: estimativa de tamanho baseada na área da região (em MSH - Millionths of Solar Hemisphere).

### Etapa 4: Treinamento e Avaliação (`src/train.py`)
Utiliza-se a arquitetura **YOLOv11m** (medium) com transfer learning a partir do dataset COCO.

-   **Restrições Físicas de Augmentação**: Diferente de fotos comuns, magnetogramas possuem semântica física rigorosa:
    -   `fliplr=0` e `flipud=0`: Desativados. Inverter a imagem horizontalmente inverteria a polaridade Leste-Oeste; verticalmente inverteria a Lei de Joy (inclinação das manchas), criando dados fisicamente impossíveis.
    -   `hsv_h=0, hsv_s=0`: Desativados, pois os dados são em escala de cinza.
-   **Otimização**: Uso de *Cosine Learning Rate Schedule* e *Early Stopping* para evitar overfitting em datasets solares pequenos.
-   **Hardware**: Suporte a CUDA (Nvidia) e DirectML (AMD GPU) via `torch-directml`.

---

## 4. Referências Técnicas para Apresentação

| Conceito | Referência no Código | Significado |
| :--- | :--- | :--- |
| **Séries HMI** | `src/download.py` | Dados de magnetometria do SDO. |
| **WCS Projection** | `src/preprocess.py` $\rightarrow$ `sidecar` | Transformação de coordenadas esféricas $\rightarrow$ pixels. |
| **Mount Wilson** | `src/labels.py` $\rightarrow$ `MTWILSON_MAP` | Escala de complexidade magnética. |
| **Joy's Law** | `src/train.py` $\rightarrow$ `flipud=0` | Regra física que dita a inclinação das regiões ativas. |
| **mAP50** | `src/train.py` $\rightarrow$ `evaluate()` | Métrica de precisão média para detecção de objetos. |
