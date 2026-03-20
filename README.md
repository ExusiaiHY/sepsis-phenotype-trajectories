# Self-Supervised Temporal Phenotype Trajectory Analysis of ICU Sepsis Patients

> **Wang Ruike** | Department of Computer Science, Advanced Programming Course

A three-stage computational framework that progresses from static clustering to self-supervised representation learning to descriptive temporal phenotype trajectory analysis, applied to 11,986 multi-center ICU patients from the PhysioNet 2012 database.

---

## Highlights

- **27.7 pp** mortality range across temporally stable phenotypes (4.0% - 31.7%)
- **35.2%** of patients undergo phenotype transitions within 48 ICU hours
- **6/6** cross-center validation criteria passed (train Center A, evaluate Center B)
- **14.2%** ground-truth in-hospital mortality (verified outcomes, not proxy labels)

---

## Three-Stage Framework

```
Stage 1 ─ Static Baseline
   Raw Data ──> Preprocessing ──> 378 Features ──> PCA (32d) ──> K-Means (K=4)

Stage 2 ─ Self-Supervised Representation Learning
   Time Series + Masks ──> Transformer Encoder ──> Masked Prediction
                                                  + Contrastive Window Loss
                                                  ──> 128d Embeddings

Stage 3 ─ Temporal Phenotype Trajectories
   Rolling Windows (24h, stride 6h) ──> Per-Window Embedding ──> K-Means
                                     ──> Trajectory Classification
                                     ──> Transition Analysis
```

### Stage 1: Static Phenotyping Baseline

PCA-based clustering on statistical features. Establishes the reference point: K=4 phenotypes with silhouette = 0.061 and mortality range 29.2 pp.

### Stage 2: Self-Supervised Encoder (S1.5)

A Transformer encoder (2-layer, 4-head, 128d) trained via:
- **Masked value prediction** (15% masking on observed values)
- **Temporal contrastive window objective** (NT-Xent on stochastic 30h views, lambda warmup 0 -> 0.5)

Four representations systematically compared:

| Method | Silhouette | Mortality Range | Center L1 | Mort. Probe AUROC | Density \|r\| |
|--------|-----------|----------------|-----------|-------------------|--------------|
| PCA (32d) | 0.061 | 29.2% | 0.027 | 0.825 | 0.231 |
| S1: masked only (128d) | 0.087 | 17.6% | 0.024 | 0.825 | 0.247 |
| **S1.5: mask+contrastive (128d)** | **0.080** | **24.6%** | **0.016** | **0.830** | **0.148** |
| S1.6: lambda=0.2 (128d) | 0.079 | 25.1% | 0.021 | 0.825 | 0.148 |

S1.5 selected for temporal analysis based on superior center stability, lowest missingness sensitivity, and highest mortality probe AUROC.

### Stage 3: Temporal Phenotype Trajectories

Rolling-window embeddings (5 windows of 24h, 6h stride) clustered via K-Means:

| Stable Phenotype | N | In-Hospital Mortality |
|-----------------|---|----------------------|
| Phenotype 0 (lowest risk) | 2,216 | 4.0% |
| Phenotype 3 | 1,891 | 9.7% |
| Phenotype 1 | 2,547 | 22.5% |
| Phenotype 2 (highest risk) | 1,110 | **31.7%** |

**Key findings:**
- 64.8% of patients remain in a stable phenotype across all windows
- 35.2% exhibit at least one phenotype transition
- Most common transitions move toward lower-acuity phenotypes (P1->P0, P1->P3, P3->P0)
- Single-transition patients have lower mortality (11.4%) than stable patients (15.4%)
- Stride=12h sensitivity analysis confirms identical mortality ordering (conservative estimate)

### Cross-Center Validation (S3)

Encoder and K-Means trained exclusively on Center A (7,989 patients), evaluated on held-out Center B (3,997 patients):

| Metric | Center A (train) | Center B (test) |
|--------|-----------------|-----------------|
| Stable fraction | 65.0% | 64.4% |
| Mortality ordering | [P0, P3, P1, P2] | [P0, P3, P1, P2] |
| Highest-risk phenotype | P2 (32.6%) | P2 (30.0%) |
| Mean prevalence L1 | --- | 0.022 |

> **Caveat:** Both centers derive from the same PhysioNet 2012 source database. This is cross-center temporal validation within a multi-center cohort, not full external validation on independently collected databases.

---

## Project Structure

```
project/
├── CLAUDE.md                   # Claude Code workflow instructions
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── config/                     # YAML configurations for each stage
│   ├── config.yaml             # Global settings
│   ├── s0_config.yaml          # S0 data pipeline
│   ├── s1_config.yaml          # S1 masked encoder
│   ├── s15_config.yaml         # S1.5 contrastive encoder
│   ├── s16_config.yaml         # S1.6 lambda ablation
│   └── s2_config.yaml          # S2 temporal trajectories
│
├── s0/                         # Stage 0: Data pipeline + real outcomes
│   ├── data_loader.py          # PhysioNet 2012 data loading
│   ├── preprocess.py           # Hourly resampling, imputation, masks
│   ├── feature_engineering.py  # 378 statistical features
│   └── schema.py               # CLIF-inspired data schema
│
├── s1/                         # Stage 1: Masked reconstruction encoder
│   ├── encoder.py              # Transformer encoder architecture
│   ├── pretrain.py             # Masked value prediction training
│   └── extract.py              # Embedding extraction
│
├── s15/                        # Stage 1.5: Contrastive encoder
│   ├── contrastive_encoder.py  # Encoder + projection head + NT-Xent
│   ├── pretrain_contrastive.py # Combined masked + contrastive training
│   ├── diagnostics.py          # Probes (mortality, center, LOS, density)
│   └── compare_three.py        # Multi-representation comparison
│
├── s2light/                    # Stage 2: Temporal phenotype trajectories
│   ├── rolling_embeddings.py   # Rolling-window extraction
│   ├── temporal_clustering.py  # Per-window K-Means
│   ├── transition_analysis.py  # Trajectory classification + transitions
│   └── visualization.py        # Sankey, prevalence, mortality plots
│
├── scripts/                    # Executable scripts for each stage
│   ├── s0_preprocess.py
│   ├── s1_pretrain.py
│   ├── s15_pretrain.py / s15_extract.py / s15_compare.py
│   ├── s16_run_all.py
│   ├── s2_extract_rolling.py / s2_cluster_and_analyze.py
│   ├── s2_sensitivity_stride12.py
│   └── s3_cross_center_validation.py
│
├── src/                        # Legacy V1 pipeline (reference only)
│
├── data/                       # Data directory (.npy files gitignored)
│   ├── external/               # Raw PhysioNet 2012 text files + Outcomes
│   ├── s0/                     # Preprocessed tensors + static.csv + splits
│   ├── s1/ s15/ s16/           # Embeddings + checkpoints + reports
│   ├── s2/                     # Rolling embeddings + trajectory stats
│   └── s3/                     # Cross-center validation report
│
├── docs/                       # Documentation + Manuscript
│   ├── RESEARCH_PAPER.tex      # LaTeX manuscript (submission-ready)
│   ├── RESEARCH_PAPER.md       # Markdown version
│   ├── RESEARCH_PAPER.pdf      # Compiled PDF (13 pages)
│   ├── WORKLOG.md              # Chronological work log
│   ├── DECISIONS.md            # Design decision log (D001-D016)
│   ├── EXPERIMENT_REGISTRY.md  # Experiment tracking (E001-E015)
│   ├── MANUSCRIPT_PATCHLIST.md # Manuscript revision tracking (P001-P014)
│   ├── NEXT_STEPS.md           # Project status + future directions
│   └── figures/                # All manuscript figures (9 total)
│
└── tests/                      # Unit tests
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt
```

### Reproduce the Full Pipeline

```bash
# Set environment (macOS)
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# Stage 0: Preprocess PhysioNet 2012 data
python scripts/s0_preprocess.py

# Stage 1.5: Train self-supervised encoder (50 epochs)
python scripts/s15_pretrain.py --epochs 50 --device cpu
python scripts/s15_extract.py
python scripts/s15_compare.py

# Stage 2: Temporal phenotype trajectories
python scripts/s2_extract_rolling.py
python scripts/s2_cluster_and_analyze.py

# Stage 2 sensitivity: Stride=12h robustness check
python scripts/s2_sensitivity_stride12.py

# Stage 3: Cross-center validation
python scripts/s3_cross_center_validation.py

# Compile manuscript
cd docs && pdflatex -interaction=nonstopmode RESEARCH_PAPER.tex && pdflatex -interaction=nonstopmode RESEARCH_PAPER.tex
```

---

## Data

**PhysioNet 2012 Challenge** — 12,000 ICU patients across 4 hospitals, organized into 2 centers:
- **Center A** (set-a + set-b): 7,989 patients (used for training)
- **Center B** (set-c): 3,997 patients (held-out for cross-center validation)

21 continuous clinical variables (vital signs + labs), 48-hour observation windows, 73.3% overall missing rate. Ground-truth in-hospital mortality from PhysioNet Outcomes files (14.2%).

> The raw `.txt` files are included in `data/external/`. Large processed arrays (`.npy`) are excluded from git via `.gitignore`.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Deep Learning | PyTorch (Transformer encoder, NT-Xent loss) |
| ML & Clustering | scikit-learn (K-Means, PCA, linear probes) |
| Data Processing | pandas, NumPy, SciPy |
| Visualization | Matplotlib (figures, Sankey diagrams) |
| Manuscript | LaTeX (pdflatex, natbib) |
| Version Control | Git + GitHub |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [WORKLOG.md](docs/WORKLOG.md) | Chronological log of all work sessions |
| [DECISIONS.md](docs/DECISIONS.md) | Design decisions with rationale (D001-D016) |
| [EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md) | All experiments with configs and results (E001-E015) |
| [MANUSCRIPT_PATCHLIST.md](docs/MANUSCRIPT_PATCHLIST.md) | Manuscript revision tracking (P001-P014) |
| [NEXT_STEPS.md](docs/NEXT_STEPS.md) | Current status and future directions |

---

## Manuscript Status

**SUBMISSION-READY** (audit completed 2026-03-20)

- 13 pages, 4 tables, 4 main figures + 5 supplementary
- 20 references (7 from 2025)
- 14 patches applied and verified
- All claims traceable to logged experiments
- No overclaims, no proxy mortality, no causal language

---

## References

1. Rudd et al. (2020). Global sepsis incidence and mortality. *The Lancet*
2. Seymour et al. (2019). Clinical phenotypes for sepsis. *JAMA*
3. Silva et al. (2012). PhysioNet/CinC Challenge 2012. *Computing in Cardiology*
4. Zheng et al. (2025). Self-supervised representation learning for clinical EHR. *npj Digital Medicine*
5. Feng et al. (2025). Deep temporal graph clustering for sepsis. *EClinicalMedicine*

See full reference list in the [manuscript](docs/RESEARCH_PAPER.pdf).
