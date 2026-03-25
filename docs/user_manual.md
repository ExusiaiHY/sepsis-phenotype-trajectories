# Software User Manual

# ICU Sepsis Temporal Phenotype Trajectory Analysis System

---

## 1. Software Overview

This software implements a three-stage computational framework for self-supervised temporal phenotype trajectory analysis of ICU sepsis patients. Applied to the PhysioNet 2012 multi-center ICU database (11,986 patients), the system performs:

- **Stage 0**: Multi-center data extraction, preprocessing (forward fill, median imputation, outlier clipping, Z-score normalization), and cross-center split generation
- **Stage 1.5**: Self-supervised Transformer encoder training via masked value prediction + temporal contrastive learning
- **Stage 2--3**: Rolling-window embedding extraction, temporal clustering, and descriptive phenotype trajectory analysis
- **Downstream**: Leakage-aware OOF stacking mortality classifier with probability calibration optimization
- **External Transfer**: Frozen-transfer temporal analysis on MIMIC-IV and eICU-CRD

Key capabilities:

- Self-supervised representation learning from sparse, multi-center ICU time-series with explicit missingness modeling
- Descriptive temporal phenotype trajectories revealing within-stay dynamics (35.2% of patients undergo phenotype transitions)
- Calibrated mortality prediction (Brier 0.090, ECE 0.023, AUROC 0.873, Recall 83.8%)
- Cross-center validation and external database transfer

## 2. System Requirements

| Item | Requirement |
|------|-------------|
| Operating System | macOS / Linux / Windows |
| Python Version | 3.9 or above (tested with 3.14) |
| Memory | 8 GB minimum (16 GB recommended for external cohorts) |
| Disk Space | 2 GB minimum (including dependencies and data) |
| GPU | Optional (accelerates Stage 1.5 pretraining; CPU works) |

## 3. Installation Steps

### 3.1 Obtain the Project Code

```bash
git clone https://github.com/ExusiaiHY/sepsis-phenotype-trajectories.git
cd sepsis-phenotype-trajectories
```

### 3.2 Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Purpose |
|---------|---------|
| numpy, pandas, scipy | Numerical computation and data processing |
| scikit-learn | Clustering, classification, evaluation |
| torch (PyTorch) | Transformer encoder training |
| pyyaml | Configuration file parsing |
| matplotlib | Visualization |
| lifelines | Survival analysis (optional) |
| umap-learn | Legacy V1 dimensionality reduction (optional) |

### 3.3 Verify Installation

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s0_smoke_test.py
```

## 4. Project Directory Structure

```
project/
├── config/                     # YAML configuration files
│   ├── s0_config.yaml          # Data extraction & preprocessing
│   ├── s1_config.yaml          # Masked pretraining (Stage 1)
│   ├── s15_config.yaml         # Contrastive pretraining (Stage 1.5)
│   ├── s15_trainval_config.yaml # S1.5 with trainval paths
│   ├── s2_config.yaml          # Rolling window & temporal clustering
│   └── config.yaml             # Legacy V1 pipeline config
├── s0/                         # Stage 0: Data pipeline
│   ├── physionet2012_extractor.py   # PhysioNet 2012 data extraction
│   ├── preprocessor.py              # Forward fill, median, clip, Z-score
│   ├── schema.py                    # 21-feature + proxy + static schema
│   ├── splits.py                    # Cross-center and random splits
│   ├── external_temporal_builder.py # MIMIC-IV / eICU to S0 format
│   └── dataset.py                   # PyTorch Dataset wrapper
├── s1/                         # Stage 1: Masked pretraining
│   ├── encoder.py              # ICUTransformerEncoder (2-layer, 4-head, d=128)
│   ├── pretrain.py             # Masked value prediction training
│   └── extract_embeddings.py   # Patient embedding extraction
├── s15/                        # Stage 1.5: Contrastive + downstream
│   ├── contrastive_encoder.py  # NT-Xent + projection head
│   ├── pretrain_contrastive.py # Combined masked + contrastive training
│   ├── classification_eval.py  # Frozen embedding classifier
│   ├── advanced_classifier.py  # Multi-view HGB/LR classifiers
│   ├── stacking_classifier.py  # OOF stacking mortality classifier
│   ├── stacking_validation.py  # Bootstrap CI + calibration analysis
│   ├── calibration.py          # Post-hoc calibration methods (5 calibrators)
│   ├── calibrated_stacking.py  # Calibration-aware stacking classifier
│   ├── calibration_losses.py   # FocalLoss, BrierLoss, SoftECELoss
│   ├── finetune_supervised.py  # End-to-end attention-pooled fine-tuning
│   └── sepsis2019_bridge.py    # PhysioNet 2019 auxiliary bridge
├── s2light/                    # Stage 2-3: Temporal analysis
│   ├── rolling_embeddings.py   # Rolling-window embedding extraction
│   ├── temporal_clustering.py  # Per-window KMeans clustering
│   ├── transition_analysis.py  # Trajectory classification & transitions
│   └── visualization.py       # Sankey diagrams, prevalence plots
├── scripts/                    # Entry-point runner scripts
│   ├── s0_prepare.py           # Prepare PhysioNet 2012 data
│   ├── s1_pretrain.py          # Stage 1 masked pretraining
│   ├── s15_pretrain.py         # Stage 1.5 contrastive pretraining
│   ├── s15_extract.py          # Extract S1.5 embeddings
│   ├── s15_train_stacking_classifier.py     # Train OOF stacking
│   ├── s15_validate_stacking_classifier.py  # Validate + calibration
│   ├── s15_calibrate.py                     # Post-hoc calibration
│   ├── s15_train_calibrated_stacking.py     # Train calibrated model
│   ├── s15_calibration_hparam_search.py     # Hparam search (30 configs)
│   ├── s15_calibration_comparison.py        # 8-model comparison
│   ├── s2_extract_rolling.py   # Extract rolling-window embeddings
│   ├── s2_cluster_and_analyze.py # Temporal clustering + analysis
│   ├── s3_cross_center_validation.py # Cross-center validation
│   └── run_external_temporal_stage3.py # External temporal transfer
├── src/                        # Legacy V1 static pipeline
│   └── main.py                 # Legacy entry point (simulated/MIMIC/eICU)
├── docs/                       # Documentation
├── data/                       # Data artifacts (gitignored for large files)
├── outputs/                    # Generated figures and reports
└── tests/                      # Pytest test suite
```

## 5. Input Data Format

### 5.1 PhysioNet 2012 (Primary)

Place the PhysioNet 2012 data files in `data/raw_aligned/`:
- `set-a/`, `set-b/`, `set-c/` directories with patient `.txt` files
- `Outcomes-a.txt`, `Outcomes-b.txt`, `Outcomes-c.txt`

Run data preparation:
```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s0_prepare.py
```

This produces:
- `data/s0/processed/continuous.npy` -- (11986, 48, 21) preprocessed tensor
- `data/s0/processed/masks_continuous.npy` -- (11986, 48, 21) observation masks
- `data/s0/processed/proxy_indicators.npy` -- (11986, 48, 2) proxy features
- `data/s0/static.csv` -- Patient demographics + mortality labels
- `data/s0/splits.json` -- Train/val/test indices (cross-center split)

### 5.2 External Databases (Supplementary)

For MIMIC-IV or eICU frozen-transfer analysis:
```bash
python3 scripts/run_external_temporal_stage3.py --source mimic --device auto
python3 scripts/run_external_temporal_stage3.py --source eicu --device auto
```

## 6. Configuration Files

All configuration is in `config/` using YAML format.

### 6.1 Encoder Configuration (`s15_config.yaml`)

```yaml
encoder:
  n_features: 21       # Input feature channels
  d_model: 128          # Transformer hidden dimension
  n_heads: 4            # Attention heads
  n_layers: 2           # Transformer layers
  d_ff: 256             # Feed-forward dimension
  dropout: 0.2
  max_seq_len: 48       # Hours in observation window

contrastive:
  view_len: 30          # Stochastic window length (hours)
  mask_ratio: 0.15      # Masked value prediction ratio
  temperature: 0.1      # NT-Xent temperature
  proj_dim: 64          # Projection head output
  max_lambda: 0.5       # Maximum contrastive weight
  warmup_epochs: 10     # Lambda warmup period

pretraining:
  epochs: 50
  batch_size: 64
  lr: 1.0e-3
  weight_decay: 1.0e-5
  patience: 15
  grad_clip: 1.0
  seed: 42
```

### 6.2 Rolling Window Configuration (`s2_config.yaml`)

```yaml
rolling:
  window_len: 24        # Window size in hours
  stride: 6             # Stride between windows
  seq_len: 48           # Total sequence length

clustering:
  k: 4                  # Number of temporal phenotypes
```

## 7. How to Run

**Important**: Always prefix Python commands with:
```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
```

### 7.1 Full Pipeline (Stages 0 to 3)

```bash
cd project

# Stage 0: Data preparation
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s0_prepare.py

# Stage 1.5: Self-supervised pretraining
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s15_pretrain.py

# Extract embeddings
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s15_extract.py

# Stage 2: Rolling-window embeddings
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s2_extract_rolling.py

# Stage 3: Temporal clustering + trajectory analysis
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s2_cluster_and_analyze.py

# Cross-center validation
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s3_cross_center_validation.py
```

### 7.2 Mortality Prediction with Calibration

```bash
# Train OOF stacking classifier
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  python3 scripts/s15_train_stacking_classifier.py

# Validate with bootstrap CIs and calibration analysis
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  python3 scripts/s15_validate_stacking_classifier.py \
    --model-dir data/s15_trainval/stacking_accuracy

# Post-hoc calibration (applies 5 methods to existing model)
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  python3 scripts/s15_calibrate.py \
    --model-dir data/s15_trainval/stacking_accuracy

# Train calibrated stacking model (recommended)
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  python3 scripts/s15_train_calibrated_stacking.py

# Run hyperparameter search for calibration optimization
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  python3 scripts/s15_calibration_hparam_search.py

# Full 8-model comparison report
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  python3 scripts/s15_calibration_comparison.py
```

### 7.3 External Temporal Transfer

```bash
# Run frozen S1.5 + Stage 3 on MIMIC-IV and eICU
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  python3 scripts/run_external_temporal_stage3.py --source all --device auto
```

### 7.4 Legacy V1 Pipeline (Simulated Data)

```bash
cd project
python src/main.py --n-patients 500 --method kmeans
```

## 8. Output Description

### 8.1 Stage 1.5 Outputs (`data/s15_trainval/`)

| File | Description |
|------|-------------|
| `checkpoints/pretrain_best.pt` | Best pretraining checkpoint |
| `embeddings_s15.npy` | Patient embeddings (11986, 128) |
| `pretrain_log.json` | Training loss history |

### 8.2 Stacking Classifier Outputs (`data/s15_trainval/stacking_accuracy/`)

| File | Description |
|------|-------------|
| `stacking_mortality_classifier.pkl` | Trained model bundle |
| `stacking_mortality_classifier_report.json` | Training report with operating points |
| `stacking_validation_report.json` | Bootstrap CIs + calibration + importance |

### 8.3 Calibration Outputs

| Directory | Key Files |
|-----------|-----------|
| `stacking_accuracy/calibration/` | `calibration_report.json`, `calibrators.pkl`, `test_probs_*.npy` |
| `calibrated_stacking/` | `calibrated_stacking_classifier.pkl`, `calibrated_stacking_report.json` |
| `calibration_hparam_search/` | `hparam_search_report.json` |
| `calibration_comparison/` | `calibration_comparison_report.json` |

### 8.4 Temporal Analysis Outputs (`data/s2light/`)

| File | Description |
|------|-------------|
| `rolling_embeddings.npy` | (11986, 5, 128) per-window embeddings |
| `cluster_assignments.npy` | (11986, 5) per-window cluster labels |
| `trajectory_stats.json` | Stability, transitions, mortality stratification |

### 8.5 Visualization Outputs (`outputs/figures/`)

| File | Description |
|------|-------------|
| `per_window_prevalence.png` | Phenotype prevalence across rolling windows |
| `sankey_transitions.png` | Phenotype transition flow diagram |
| `mortality_by_trajectory.png` | Mortality rates by trajectory category |
| `pipeline_diagram.png` | Three-stage framework architecture |

## 9. Frequently Asked Questions

**Q1: "ModuleNotFoundError" occurs at runtime**

A: Install all dependencies: `pip install -r requirements.txt`. For PyTorch, follow pytorch.org for your platform.

**Q2: Pretraining is slow on CPU**

A: Stage 1.5 pretraining benefits from GPU acceleration. On CPU it takes about 25 min, vs about 8 min on GPU. Use `--device cpu` or `--device cuda` to control.

**Q3: How to reproduce the calibration optimization?**

A: Run the full calibration pipeline:
```bash
# 1. Train original stacking model (if not already done)
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s15_train_stacking_classifier.py

# 2. Apply post-hoc calibration to original model
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s15_calibrate.py --model-dir data/s15_trainval/stacking_accuracy

# 3. Train calibrated stacking model (structural fix)
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s15_train_calibrated_stacking.py

# 4. Compare all methods
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s15_calibration_comparison.py
```

**Q4: How to use a custom mortality prior for Bayesian calibration?**

A: Pass `--prior-rate 0.20` to `scripts/s15_calibrate.py` to use 20% instead of 14.2%.

**Q5: What threshold should I use for clinical deployment?**

A: The calibrated model supports multiple operating points:
- **Triage (high sensitivity)**: threshold=0.05, captures >90% of deaths
- **Balanced**: threshold=0.09, recall 83.8%, balanced accuracy 80.1%
- **Resource allocation (high specificity)**: threshold=0.30, precision approximately 60%

**Q6: Results differ between runs**

A: Ensure `seed: 42` in config files. All random processes use deterministic seeding.

**Q7: How to run external temporal transfer on MIMIC-IV / eICU?**

A: First prepare the source-specific data with `src/main.py`, then:
```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  python3 scripts/run_external_temporal_stage3.py --source all --device auto
```

## 10. Important Notes

1. Always use `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` prefix for Python commands to avoid threading conflicts
2. The calibrated stacking model (`data/s15_trainval/calibrated_stacking/`) is the recommended model for clinical applications
3. All mortality outcomes use verified PhysioNet Outcomes files (14.2% in-hospital mortality rate)
4. External transfer results (MIMIC-IV, eICU) use the frozen PhysioNet-trained encoder and should be interpreted as supplementary frozen-transfer analyses
5. Model artifacts use serialized format for sklearn compatibility -- only load model files produced by this project
6. Large data files (`.npy`, `.pt`) are gitignored; re-run the pipeline to regenerate them
