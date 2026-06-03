# Solar Active Region Classifier / Classificador de Regiões Ativas Solares

YOLOv11-based detector that localises and classifies solar active regions (ARs) in full-disk line-of-sight magnetograms from NASA's Solar Dynamics Observatory (SDO/HMI).
Detector baseado em YOLOv11 que localiza e classifica regiões ativas (RAs) solares em magnetogramas de disco inteiro do Observatório de Dinâmica Solar da NASA (SDO/HMI).

---

## Overview / Visão Geral

| Stage / Etapa | Tool / Ferramenta | Output / Saída |
|---|---|---|
| Download | SunPy Fido → JSOC | `data/raw/*.fits` |
| Preprocess / Pré-processamento | astropy / Pillow | `data/images/*.png` + WCS sidecar |
| Label / Anotação | HEK REST API | `data/labels/*.txt` (YOLO format) |
| Split / Divisão | chronological 70/15/15 | `datasets/{train,val,test}/` |
| Train / Treinamento | YOLOv11m (Ultralytics) | `runs/solar_ar/weights/best.pt` |
| Predict / Predição | YOLOv11 inference | annotated PNGs + `detections.csv` |

### Classification scheme — Mount Wilson / Esquema de classificação — Mount Wilson

| Class ID | Label | Magnetic complexity / Complexidade magnética |
|---|---|---|
| 0 | Alpha | Unipolar — lowest flare risk / menor risco de flare |
| 1 | Beta | Bipolar |
| 2 | BetaGamma | Bipolar with complex inversion line / com linha de inversão complexa |
| 3 | BetaGammaDelta | Strong delta spots — highest flare risk / manchas delta intensas — maior risco |

---

## Requirements / Requisitos

- Python 3.10+
- A free [JSOC account](http://jsoc.stanford.edu/ajax/register_email.html) — email required for data export.
  Uma [conta JSOC](http://jsoc.stanford.edu/ajax/register_email.html) gratuita — e-mail necessário para exportação de dados.
- GPU strongly recommended for training (CUDA 11.8+).
  GPU recomendada para treinamento (CUDA 11.8+).
- [uv](https://docs.astral.sh/uv/) for dependency management.
  [uv](https://docs.astral.sh/uv/) para gerenciamento de dependências.

```bash
# Install uv / Instalar o uv
curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS/Linux
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create virtual environment and install all dependencies
# Criar ambiente virtual e instalar todas as dependências
uv sync

# Install only development dependencies (no torch/sunpy — for CI or quick setup)
# Instalar apenas dependências de desenvolvimento (sem torch/sunpy — para CI ou setup rápido)
uv sync --only-group dev
```

---

## Quick Start / Início Rápido

### Full pipeline / Pipeline completo

```bash
uv run python main.py \
  --email your@email.com \
  --start 2014-01-01 \
  --end   2014-06-30
```

### Step by step / Passo a passo

```bash
# 1 — Download HMI magnetograms (6-hour cadence)
#     Baixar magnetogramas HMI (cadência de 6 horas)
uv run python -m src.download --email your@email.com \
                              --start 2014-01-01 --end 2014-06-30 \
                              --cadence 6

# 2 — Convert FITS → normalised PNG (±1000 G clipped, 1024×1024 px)
#     Converter FITS → PNG normalizado (±1000 G cortado, 1024×1024 px)
uv run python -m src.preprocess

# 3 — Generate YOLO labels from the HEK catalogue
#     Gerar labels YOLO a partir do catálogo HEK
uv run python -m src.labels

# 4 — Split into train / val / test (chronological, no shuffle)
#     Dividir em treino / validação / teste (cronológico, sem embaralhamento)
uv run python -m src.dataset

# 5 — Fine-tune YOLOv11m / Ajuste fino do YOLOv11m
uv run python -m src.train --epochs 100 --batch 8

# 6 — Evaluate on test set / Avaliar no conjunto de teste
uv run python -m src.train --eval --weights runs/solar_ar/weights/best.pt

# 7 — Inference on new images / Inferência em novas imagens
uv run python -m src.predict --weights runs/solar_ar/weights/best.pt \
                             --source data/images/ --output results/
```

---

## Project Structure / Estrutura do Projeto

```
solar-classifier/
├── main.py                  # Full pipeline CLI / CLI do pipeline completo
├── pyproject.toml           # Project metadata and dependencies (uv)
│                            # Metadados e dependências do projeto (uv)
├── uv.lock                  # Pinned dependency tree / Árvore de dependências fixadas
├── .python-version          # Pinned Python version (3.12) / Versão do Python fixada
├── configs/
│   └── solar.yaml           # YOLO dataset config (nc=4, class names)
│                            # Configuração do dataset YOLO (nc=4, nomes das classes)
├── data/
│   ├── raw/                 # Downloaded FITS files / Arquivos FITS baixados
│   ├── images/              # Processed PNGs + WCS sidecar JSONs
│   │                        # PNGs processados + sidecars WCS em JSON
│   └── labels/              # YOLO .txt label files / Arquivos de label YOLO
├── datasets/
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   └── test/{images,labels}/
├── runs/                    # Training outputs (auto-created by YOLO)
│                            # Saídas de treinamento (criadas automaticamente pelo YOLO)
├── src/
│   ├── download.py          # JSOC download via SunPy Fido
│   ├── preprocess.py        # FITS → PNG + WCS sidecar
│   ├── labels.py            # HEK query → YOLO labels
│   ├── dataset.py           # Chronological train/val/test split
│   ├── train.py             # YOLOv11 training + evaluation
│   ├── predict.py           # Inference → annotated PNG + CSV
│   └── log.py               # Rich terminal output utilities
└── tests/
    ├── conftest.py
    ├── test_labels.py
    ├── test_preprocess.py
    └── test_dataset.py
```

---

## Key Design Decisions / Decisões de Projeto

### No horizontal/vertical flip augmentation / Sem augmentação por espelhamento

HMI magnetograms encode physical magnetic polarity: white pixels are positive-polarity field (pointing toward the observer) and black pixels are negative-polarity field. Flipping an image horizontally mirrors the leading and trailing sunspot configuration of an active region, producing a signature that is physically impossible or that belongs to the opposite solar hemisphere. Vertical flips place the solar north pole at the bottom, inverting latitude-dependent features such as Joy's Law (the statistical tilt of bipolar groups). For these reasons, flips are disabled (`fliplr=0`, `flipud=0`).
Os magnetogramas HMI codificam polaridade magnética física: pixels brancos são campo de polaridade positiva (apontando para o observador) e pixels pretos são campo de polaridade negativa. Espelhar horizontalmente inverte a configuração de manchas líderes e seguidoras de uma região ativa, produzindo uma assinatura fisicamente impossível ou pertencente ao hemisfério solar oposto. Espelhamentos verticais colocam o polo norte solar na base, invertendo características dependentes de latitude como a Lei de Joy (a inclinação estatística de grupos bipolares). Por esses motivos, os flips estão desativados (`fliplr=0`, `flipud=0`).

### Chronological split / Divisão cronológica

Active regions are not isolated events: a single region can persist and evolve for 10–30 days, appearing in many consecutive magnetograms. A random shuffle would place frames from the same active region in both the training and test sets, inflating generalisation metrics. The split preserves the original time order so that the test set contains only regions that appeared *after* the training period, simulating real-world deployment where the model is used to detect future events.
Regiões ativas não são eventos isolados: uma única região pode persistir e evoluir por 10 a 30 dias, aparecendo em muitos magnetogramas consecutivos. Um embaralhamento aleatório colocaria frames da mesma região ativa no treino e no teste, inflando as métricas de generalização. A divisão preserva a ordem temporal original para que o conjunto de teste contenha apenas regiões que apareceram *após* o período de treinamento, simulando o uso real do modelo para detectar eventos futuros.

### Magnetogram normalisation / Normalização do magnetograma

Raw HMI line-of-sight magnetograms contain values from roughly −3000 G to +3000 G. However, most of the solar disk is "quiet sun" with field strengths below 100 G, while scientifically interesting active-region field concentrations range from a few hundred to ~2000 G. Clipping to ±1000 G before mapping to [0, 255] compresses the quiet-sun noise into a neutral grey midpoint (~127) and uses the full contrast range for the active-region signal. The exact clip threshold is a trade-off: lower values (e.g. ±500 G) would better separate weak fields but saturate large delta groups; higher values (e.g. ±2000 G) preserve more field-strength information but reduce contrast in typical ARs.
Os magnetogramas brutos de linha de visada do HMI contêm valores de aproximadamente −3000 G a +3000 G. Porém, a maior parte do disco solar é "sol quieto" com intensidades de campo abaixo de 100 G, enquanto concentrações de campo em regiões ativas de interesse científico variam de algumas centenas a ~2000 G. O corte em ±1000 G antes de mapear para [0, 255] comprime o ruído do sol quieto para um cinza neutro (~127) e usa toda a faixa de contraste para o sinal das regiões ativas. O limiar exato de corte é uma compensação: valores menores (ex.: ±500 G) separam melhor campos fracos, mas saturam grandes grupos delta; valores maiores (ex.: ±2000 G) preservam mais informação de intensidade de campo, mas reduzem o contraste em RAs típicas.

### Bounding box estimation from MSH area / Estimativa da bounding box pela área em MSH

The HEK does not supply bounding rectangles for active regions — only the centroid in heliocentric projected coordinates (HPC, in arcseconds from disk centre) and the area in Millionths of Solar Hemisphere (MSH). One MSH equals 10⁻⁶ of one solar hemisphere, so the formula converts area to pixels by scaling through the plate scale of the instrument (arcsec/pixel), then approximates the active region as a circle with a 1.5× safety margin:
O HEK não fornece retângulos delimitadores para regiões ativas — apenas o centroide em coordenadas heliocentricamente projetadas (HPC, em arcsegundos a partir do centro do disco) e a área em Milionésimos do Hemisfério Solar (MSH). Um MSH equivale a 10⁻⁶ de um hemisfério solar; portanto, a fórmula converte área em pixels escalando pela escala de placa do instrumento (arcseg/pixel) e aproxima a região ativa como um círculo com margem de segurança de 1,5×:

```
R_px = sqrt(area_MSH × 1e-6 × π × R_sun_px²  /  π)  ×  1.5  (margin / margem)
```

---

## Data Source / Fonte dos Dados

- **Images / Imagens**: [SDO/HMI](https://hmiwww.ssl.berkeley.edu/), series `hmi.M_720s` (one magnetogram every 12 minutes), accessed via [JSOC](http://jsoc.stanford.edu/). We sample at 6-hour cadence, yielding ~4 images per day.
  [SDO/HMI](https://hmiwww.ssl.berkeley.edu/), série `hmi.M_720s` (um magnetograma a cada 12 minutos), acessada via [JSOC](http://jsoc.stanford.edu/). Amostramos em cadência de 6 horas, resultando em ~4 imagens por dia.

- **Labels / Anotações**: [Heliophysics Event Knowledgebase (HEK)](https://www.lmsal.com/hek/), AR events from the NOAA SWPC Observer and SHARP (Space Weather HMI Active Region Patches) pipeline. We query one day at a time and match events to image timestamps with a ±30-minute window.
  [Heliophysics Event Knowledgebase (HEK)](https://www.lmsal.com/hek/), eventos de RA do NOAA SWPC Observer e do pipeline SHARP (Space Weather HMI Active Region Patches). Consultamos um dia por vez e associamos eventos aos timestamps das imagens com janela de ±30 minutos.

---

## Running the Tests / Executando os Testes

```bash
# Run all tests / Executar todos os testes
uv run pytest

# With coverage report / Com relatório de cobertura
uv run pytest --cov
```

The test suite mocks all network dependencies (sunpy, astropy, HEK) so it runs offline without any scientific packages installed.
A suíte de testes simula todas as dependências de rede (sunpy, astropy, HEK) para rodar offline sem nenhum pacote científico instalado.

---

## References / Referências

### Instrumentation and data / Instrumentação e dados

Pesnell, W.D., Thompson, B.J., & Chamberlin, P.C. (2012). The Solar Dynamics Observatory (SDO). *Solar Physics*, 275(1–2), 3–15. https://doi.org/10.1007/s11207-011-9841-3

Scherrer, P.H., et al. (2012). The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO). *Solar Physics*, 275(1–2), 207–227. https://doi.org/10.1007/s11207-011-9834-2

Schou, J., et al. (2012). Design and Ground Calibration of the Helioseismic and Magnetic Imager (HMI) Instrument on the Solar Dynamics Observatory (SDO). *Solar Physics*, 275(1–2), 229–259. https://doi.org/10.1007/s11207-011-9842-2

### Event catalogues / Catálogos de eventos

Hurlburt, N., et al. (2012). Heliophysics Event Knowledgebase for the Solar Dynamics Observatory (SDO) and Beyond. *Solar Physics*, 275(1–2), 67–78. https://doi.org/10.1007/s11207-010-9624-2

Bobra, M.G., et al. (2014). The Helioseismic and Magnetic Imager (HMI) Vector Magnetic Field Pipeline: SHARPs — Space Weather HMI Active Region Patches. *Solar Physics*, 289(9), 3549–3578. https://doi.org/10.1007/s11207-014-0529-3

### Active region classification / Classificação de regiões ativas

Hale, G.E., Ellerman, F., Nicholson, S.B., & Joy, A.H. (1919). The Magnetic Polarity of Sun-Spots. *Astrophysical Journal*, 49, 153–178. https://doi.org/10.1086/142452

Künzel, H. (1960). Die Flare-Häufigkeit in Fleckengruppen unterschiedlicher Klasse und magnetischer Struktur. *Astronomische Nachrichten*, 285(5), 271–273. https://doi.org/10.1002/asna.19602850516

McIntosh, P.S. (1990). The Classification of Sunspot Groups. *Solar Physics*, 125(2), 251–267. https://doi.org/10.1007/BF00158405

### Solar physics and space weather / Física solar e clima espacial

Schrijver, C.J. (2007). A Characteristic Magnetic Field Pattern Associated with All Major Solar Flares and Its Use in Flare Forecasting. *The Astrophysical Journal Letters*, 655(2), L117–L120. https://doi.org/10.1086/511857

Benz, A.O. (2017). Flare Observations. *Living Reviews in Solar Physics*, 14(1), 2. https://doi.org/10.1007/s41116-016-0004-3

van Driel-Gesztelyi, L., & Green, L.M. (2015). Evolution of Active Regions. *Living Reviews in Solar Physics*, 12(1), 1. https://doi.org/10.1007/lrsp-2015-1

### Software

The SunPy Community, et al. (2020). The SunPy Project: Open Source Development and Status of the Version 1.0 Core Package. *The Astrophysical Journal*, 890(1), 68. https://doi.org/10.3847/1538-4357/ab4f7a

Astropy Collaboration, et al. (2022). The Astropy Project: Sustaining and Growing a Community-developed Open-source Project and Status of the v5.0 Core Package. *The Astrophysical Journal*, 935(2), 167. https://doi.org/10.3847/1538-4357/ac7c74

Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0). Zenodo. https://doi.org/10.5281/zenodo.8347048

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 779–788. https://doi.org/10.1109/CVPR.2016.91
