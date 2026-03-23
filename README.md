<div align="center">

<h1>Self-Supervised Temporal Phenotype Trajectory Analysis of ICU Sepsis Patients</h1>

<p>
  <strong>Wang Ruike</strong><br>
  Department of Computer Science &middot; Advanced Programming Course
</p>

<p>
  <a href="docs/RESEARCH_PAPER.pdf"><img src="https://img.shields.io/badge/Research%20Paper-PDF-B31B1B?style=for-the-badge" alt="Research Paper PDF"></a>
  <a href="docs/RESEARCH_PAPER.md"><img src="https://img.shields.io/badge/Manuscript-Markdown-1F6FEB?style=for-the-badge" alt="Manuscript Markdown"></a>
  <a href="docs/EXPERIMENT_REGISTRY.md"><img src="https://img.shields.io/badge/Experiments-E001--E017-0E8A16?style=for-the-badge" alt="Experiment Registry"></a>
  <a href="docs/DECISIONS.md"><img src="https://img.shields.io/badge/Design%20Log-D001--D016-6F42C1?style=for-the-badge" alt="Design Decisions"></a>
</p>

<p>
  A research codebase for ICU sepsis phenotyping that moves from static clustering,
  to self-supervised temporal representation learning, to descriptive phenotype trajectory analysis
  on 11,986 multi-center PhysioNet 2012 patients, with supplementary downstream mortality validation
  on frozen learned embeddings.
</p>

<img src="docs/figures/summary_dashboard.png" alt="Project summary dashboard" width="920">

</div>

---

## At A Glance

<table>
  <tr>
    <td width="25%" align="center"><strong>27.7 pp</strong><br>mortality range across temporally stable phenotypes</td>
    <td width="25%" align="center"><strong>35.2%</strong><br>of patients show at least one phenotype transition</td>
    <td width="25%" align="center"><strong>6 / 6</strong><br>cross-center validation criteria satisfied</td>
    <td width="25%" align="center"><strong>14.2%</strong><br>verified in-hospital mortality from outcomes files</td>
  </tr>
</table>

## Why This Repository Matters

Most sepsis phenotyping papers stop at static patient clusters. This project goes one step further: it learns temporal patient representations from sparse ICU time series, then asks how phenotype membership evolves during the first 48 ICU hours.

The repo is organized as a full research artifact, not just a model dump. It includes the data pipeline, experiment scripts, diagnostics, manuscript sources, figures, audit logs, and a compiled paper.

## Core Result

The strongest result is not just that four phenotypes exist, but that their temporal trajectories are clinically structured:

- Stable temporal phenotypes stratify mortality from `4.0%` to `31.7%`.
- Nearly one-third of the cohort moves between phenotypes within 48 hours.
- The most frequent transitions move toward lower-risk states.
- The same mortality ordering is preserved on held-out Center B within the PhysioNet 2012 multi-center cohort.

> Caveat: this is cross-center validation within the same source dataset, not full external validation on an independently collected ICU database.

## Three-Stage Pipeline

```text
Stage 0 / Data Foundation
  PhysioNet 2012 raw files
    -> aligned hourly tensors
    -> observation masks
    -> verified in-hospital mortality labels

Stage 1 / Static Baseline
  48h time series
    -> 378 statistical features
    -> PCA (32d)
    -> K-Means phenotypes

Stage 1.5 / Self-Supervised Representation Learning
  values + masks
    -> Transformer encoder
    -> masked value prediction
    -> contrastive window objective
    -> 128d patient embeddings

Stage 2 / Temporal Phenotype Trajectories
  rolling 24h windows, stride 6h
    -> per-window embeddings
    -> K-Means on window states
    -> stability, transitions, prevalence shift, mortality analysis

Stage 3 / Cross-Center Validation
  train on Center A
    -> evaluate phenotype structure on Center B
```

## Supplementary Downstream Validation

```text
Frozen S1.5 embeddings
  -> logistic regression mortality classifier
  -> threshold tuning on Center A validation split
  -> held-out Center B accuracy / balanced accuracy / recall / AUROC
```

## Visual Overview

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/figures/pipeline_diagram.png" alt="Pipeline diagram" width="100%">
      <br>
      <sub>Pipeline from preprocessing to temporal phenotype analysis</sub>
    </td>
    <td align="center" width="50%">
      <img src="docs/figures/sankey_transitions.png" alt="Phenotype transitions" width="100%">
      <br>
      <sub>Descriptive transition flow across five rolling windows</sub>
    </td>
  </tr>
</table>

## Methods Snapshot

### Stage 1: Static Baseline

- Uses 48-hour summary features as a conventional reference point.
- Establishes that the cohort contains clinically meaningful heterogeneity before deep learning.
- Baseline `K=4` result: silhouette `0.061`, mortality range `29.2 pp`.

### Stage 1.5: Mask-Aware Self-Supervised Encoder

- Input is `concat([values, masks])`, so missingness is treated as signal, not discarded.
- Pretraining combines:
  - masked value prediction on observed entries
  - temporal contrastive learning on stochastic 30-hour overlapping windows
- Best representation: `S1.5`, selected for center stability, missingness robustness, and rolling-window suitability.

### Stage 2: Temporal Phenotype Trajectories

- Extracts `5` rolling windows per patient: `[0,24)`, `[6,30)`, `[12,36)`, `[18,42)`, `[24,48)`.
- Clusters each window embedding into one of four phenotype states.
- Classifies patients as `stable`, `single-transition`, or `multi-transition`.

## Main Quantitative Results

### Representation Comparison

| Method | Silhouette | Mortality Range | Center L1 | Mortality Probe AUROC | Density \|r\| |
|--------|-----------:|----------------:|----------:|----------------------:|--------------:|
| PCA (32d) | 0.061 | 29.2% | 0.027 | 0.825 | 0.231 |
| S1 masked (128d) | 0.087 | 17.6% | 0.024 | 0.825 | 0.247 |
| **S1.5 mask + contrastive (128d)** | **0.080** | **24.6%** | **0.016** | **0.830** | **0.148** |
| S1.6 lambda=0.2 (128d) | 0.079 | 25.1% | 0.021 | 0.825 | 0.148 |

### Stable Temporal Phenotypes

| Phenotype | Patients | In-Hospital Mortality |
|-----------|---------:|----------------------:|
| P0 | 2,216 | 4.0% |
| P3 | 1,891 | 9.7% |
| P1 | 2,547 | 22.5% |
| P2 | 1,110 | 31.7% |

### Cross-Center Validation

| Metric | Center A | Center B |
|--------|---------:|---------:|
| Patients | 7,989 | 3,997 |
| Stable fraction | 65.0% | 64.4% |
| Non-self transition proportion | 10.3% | 10.6% |
| Mortality ordering | `[P0, P3, P1, P2]` | `[P0, P3, P1, P2]` |
| Highest-risk phenotype | `P2 (32.6%)` | `P2 (30.0%)` |
| Mean prevalence L1 | - | 0.022 |

### Supplementary Downstream Mortality Validation

| Operating Point | Test Accuracy | Test Balanced Accuracy | Test Recall | Test AUROC |
|----------------|--------------:|-----------------------:|------------:|-----------:|
| Balanced threshold (`thr=0.55`) | 0.784 | 0.745 | 0.691 | 0.829 |
| Accuracy-optimized threshold (`thr=0.85`) | 0.865 | 0.623 | 0.280 | 0.829 |
| Majority-class baseline | 0.854 | 0.500 | 0.000 | - |

Because held-out mortality prevalence is only `14.6%`, plain accuracy is misleading on its own. The accuracy-optimized operating point barely beats the majority baseline while missing most deaths, whereas the balanced threshold preserves the same AUROC with much stronger recall.

### Improved Downstream Models

Using more of the already-available cohort information than the embedding-only linear probe:

- `HGB + statistics + masks + proxy + static` reaches `test accuracy=0.791`, `balanced accuracy=0.780`, `AUROC=0.862`
- `HGB ensemble (fused + stats views)` reaches `balanced accuracy=0.785`, `recall=0.812`, `AUROC=0.865`

These models learn from more data modalities already present in the repository: 48h summary statistics, missingness patterns, proxy indicators, demographics, and optionally S1.5 embeddings.

## Repository Map

| Path | Role |
|------|------|
| [`s0/`](s0) | Data extraction, preprocessing, schema, splits, and verified outcomes |
| [`s1/`](s1) | Masked reconstruction encoder and embedding extraction |
| [`s15/`](s15) | Contrastive pretraining, diagnostics, and multi-method comparison |
| [`s2light/`](s2light) | Rolling embeddings, temporal clustering, transitions, visualization |
| [`scripts/`](scripts) | Reproducible entry points for every stage |
| [`data/`](data) | Raw PhysioNet files plus generated reports and artifacts |
| [`docs/`](docs) | Paper, logs, patch history, decisions, and supporting docs |
| [`tests/`](tests) | Unit tests for core pipeline components |
| [`src/`](src) | Legacy V1 pipeline kept for reference only |

<details>
<summary><strong>Show Full Project Layout</strong></summary>

```text
project/
|-- README.md
|-- requirements.txt
|-- config/
|-- s0/
|-- s1/
|-- s15/
|-- s2light/
|-- scripts/
|-- data/
|-- docs/
|-- tests/
`-- src/
```

</details>

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Reproduce The Main Pipeline

```bash
# Optional environment settings (macOS / local BLAS conflicts)
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# Stage 0: prepare aligned and processed PhysioNet 2012 tensors
python scripts/s0_prepare.py

# Stage 1.5: train and evaluate the selected self-supervised encoder
python scripts/s15_pretrain.py --epochs 50 --device cpu
python scripts/s15_extract.py
python scripts/s15_compare.py
python scripts/s15_diagnostics.py
python scripts/s15_train_classifier.py
python scripts/s15_train_advanced_classifier.py --model-type hgb --feature-set stats_mask_proxy_static

# Stage 2: temporal phenotype trajectory analysis
python scripts/s2_extract_rolling.py
python scripts/s2_cluster_and_analyze.py

# Sensitivity analysis
python scripts/s2_sensitivity_stride12.py

# Stage 3: cross-center validation
python scripts/s3_cross_center_validation.py

# Optional: retrain a fresh S1.5 model in an isolated output directory
python scripts/s15_pretrain.py --config config/s15_trainval_config.yaml --device cpu
python scripts/s15_extract.py --config config/s15_trainval_config.yaml --device cpu
python scripts/s15_train_classifier.py --config config/s15_trainval_config.yaml
python scripts/s15_train_advanced_classifier.py --config config/s15_trainval_config.yaml --model-type hgb --feature-set stats_mask_proxy_static
python scripts/s15_train_advanced_classifier.py --config config/s15_trainval_config.yaml --model-type hgb_ensemble
```

### 3. Compile The Paper

```bash
cd docs
pdflatex -interaction=nonstopmode RESEARCH_PAPER.tex
pdflatex -interaction=nonstopmode RESEARCH_PAPER.tex
```

## Dataset

The project is built around the **PhysioNet/CinC 2012 Challenge** ICU database:

- `11,986` retained patients
- `21` continuous clinical variables
- `48` hourly timesteps per stay
- `73.3%` overall missingness before imputation
- `14.2%` verified in-hospital mortality from outcomes files

Center split used in the project:

- **Center A** = `set-a + set-b` (`7,989` patients), used for training and development
- **Center B** = `set-c` (`3,997` patients), used for held-out cross-center evaluation

Raw challenge files are stored under [`data/external/`](data/external). Large derived arrays such as `.npy` tensors are excluded from version control where appropriate.

## Documentation

| Document | Purpose |
|----------|---------|
| [Research paper PDF](docs/RESEARCH_PAPER.pdf) | Full paper with methods, results, discussion, and figures |
| [Research paper source](docs/RESEARCH_PAPER.tex) | LaTeX manuscript source |
| [Experiment registry](docs/EXPERIMENT_REGISTRY.md) | Logged experiments, configurations, and artifact paths |
| [Decisions log](docs/DECISIONS.md) | Major design decisions and rationale |
| [Manuscript patch list](docs/MANUSCRIPT_PATCHLIST.md) | Tracked paper revisions |
| [Next steps](docs/NEXT_STEPS.md) | Current status and future work |
| [Worklog](docs/WORKLOG.md) | Chronological implementation record |

## Manuscript Status

The manuscript is currently **submission-ready**.

- Compiled PDF: [`docs/RESEARCH_PAPER.pdf`](docs/RESEARCH_PAPER.pdf)
- Current size: `17` pages
- Main content: `6` tables, `4` main figures, `5` supplementary figures
- Claims are tied back to logged experiments in [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md)

## Reproducibility Notes

- All reported mortality values use verified outcomes files, not proxy labels.
- Temporal findings are described as **descriptive trajectories**, not causal treatment effects.
- The stride=`12h` sensitivity analysis preserves the same phenotype risk ordering.
- Cross-center results should be interpreted as **within-cohort multi-center validation**, not external database validation.

## Selected References

1. Rudd et al. (2020). Global sepsis incidence and mortality. *The Lancet*
2. Seymour et al. (2019). Clinical phenotypes for sepsis. *JAMA*
3. Silva et al. (2012). PhysioNet/CinC Challenge 2012. *Computing in Cardiology*
4. Zheng et al. (2025). Self-supervised representation learning for clinical EHR. *npj Digital Medicine*
5. Feng et al. (2025). Deep temporal graph clustering for sepsis. *EClinicalMedicine*

Full references are listed in the [paper](docs/RESEARCH_PAPER.pdf).
