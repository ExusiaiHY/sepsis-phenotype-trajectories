# Software User Manual

# ICU Sepsis Temporal Phenotype Trajectory Analysis System

---

## 1. Software Overview

This software provides a comprehensive framework for ICU sepsis patient analysis through multiple modeling stages:

| Stage | Component | Purpose |
|-------|-----------|---------|
| **S0** | Data Pipeline | Multi-center data extraction, preprocessing, and cross-center splits |
| **S1.5** | Self-Supervised Encoder | Masked value prediction + temporal contrastive learning for patient representations |
| **S2-S3** | Temporal Analysis | Rolling-window embeddings, clustering, and phenotype trajectory analysis |
| **Downstream** | Mortality Classifier | Leakage-aware OOF stacking with probability calibration |
| **S4** | Treatment-Aware Modeling | Treatment-phenotype interactions and observational causal analysis |
| **S5** | Bedside Dashboard | Real-time distilled student classifier with HTML dashboard generation |
| **S6** | Advanced Optimization | Foundation models (TimesFM), causal ML, and domain adaptation |

---

## 2. System Requirements

| Item | Requirement |
|------|-------------|
| Operating System | macOS / Linux / Windows |
| Python Version | 3.9+ (tested with 3.14) |
| Memory | 8 GB minimum, 16 GB recommended for external cohorts |
| Disk Space | 2 GB minimum |
| GPU | Optional for S0-S5 (accelerates training); Required for S6 (TimesFM) |

---

## 3. Installation

### 3.1 Clone Repository

```bash
git clone https://github.com/ExusiaiHY/sepsis-phenotype-trajectories.git
cd sepsis-phenotype-trajectories
```

### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `numpy`, `pandas`, `scipy` - Data processing
- `scikit-learn` - Clustering and classification
- `torch` - Transformer encoder training
- `pyyaml` - Configuration management

**Stage 6 Dependencies (optional):**
- `timesfm` - Foundation model features
- `SAITS` - Missingness imputation
- `causalml`, `dowhy` - Causal inference

### 3.3 Verify Installation

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s0_smoke_test.py
```

---

## 4. Quick Start

### 4.1 Basic Pipeline (Stages 0-3)

```bash
cd project
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

# Stage 0: Data preparation
python3 scripts/s0_prepare.py

# Stage 1.5: Self-supervised pretraining
python3 scripts/s15_pretrain.py --epochs 50 --device cpu

# Extract embeddings
python3 scripts/s15_extract.py

# Stage 2-3: Temporal analysis
python3 scripts/s2_extract_rolling.py
python3 scripts/s2_cluster_and_analyze.py

# Cross-center validation
python3 scripts/s3_cross_center_validation.py
```

### 4.2 Mortality Prediction with Calibration

```bash
# Train calibrated classifier (recommended)
python3 scripts/s15_train_calibrated_stacking.py

# Validate with bootstrap CIs
python3 scripts/s15_validate_stacking_classifier.py \
    --model-dir data/s15_trainval/calibrated_stacking
```

### 4.3 External Database Analysis

```bash
# MIMIC-IV or eICU frozen-transfer analysis
python3 scripts/run_external_temporal_stage3.py --source mimic --device auto
python3 scripts/run_external_temporal_stage3.py --source eicu --device auto
```

---

## 5. Stage 4: Treatment-Aware Analysis

### 5.1 Prepare Treatment Features

```bash
python3 scripts/s4_prepare_treatments.py \
    --source mimic \
    --prepared-dir data/processed_mimic_real \
    --raw-dir mimic-iv-3.1 \
    --output-dir data/s4/mimic_treatments
```

### 5.2 Train Treatment-Aware Model

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/s4_train_treatment_aware.py \
    --device cuda \
    --s0-dir data/external_temporal/mimic/s0 \
    --treatment-dir data/s4/mimic_treatments \
    --output-dir data/s4/mimic_treatment_aware
```

### 5.3 Causal Analysis (PSM + DML)

```bash
python3 scripts/s4_run_causal_analysis.py \
    --s0-dir data/external_temporal/mimic/s0 \
    --treatment-dir data/s4/mimic_treatments \
    --embeddings data/external_temporal/mimic/s15/embeddings_s15.npy \
    --phenotype-labels data/external_temporal/mimic/s2/window_labels.npy \
    --output-dir data/s4/mimic_causal
```

**Note:** Treatment effect estimates are source-specific and observational. Use for hypothesis generation only.

---

## 6. Stage 5: Bedside Dashboard

### 6.1 Generate Patient Dashboard

```python
from s5.dashboard import render_clinical_dashboard_html
from pathlib import Path

snapshots = [
    {"risk_probability": 0.15, "phenotype": "P0", "hours_seen": 6, "risk_alert": False},
    {"risk_probability": 0.31, "phenotype": "P1", "hours_seen": 18, "risk_alert": True},
]

render_clinical_dashboard_html(
    patient_id="ICU-001",
    snapshots=snapshots,
    output_path=Path("outputs/dashboards/patient_001.html"),
    model_meta={"version": "S5-v1.0", "threshold": 0.25}
)
```

---

## 7. Stage 6: Advanced Optimization

### 7.1 Run Full Optimization

```bash
# Requires GPU for TimesFM
python3 scripts/s6_run_full_optimization.py \
    --config config/s6_config_round7.yaml
```

### 7.2 External Generalization with Domain Adaptation

```bash
python3 scripts/s6_run_external_generalization.py \
    --config config/s6_config_round7_external_fast.yaml \
    --sources mimic \
    --output-root data/s6_external_mimic
```

**Use Cases for Stage 6:**
- Need phenotype separation with mortality range > 0.45
- Multi-domain data requiring alignment
- Causal-validated phenotype-treatment associations

---

## 8. Configuration

### 8.1 Encoder Config (`config/s15_config.yaml`)

```yaml
encoder:
  n_features: 21
  d_model: 128
  n_heads: 4
  n_layers: 2
  dropout: 0.2

contrastive:
  view_len: 30
  mask_ratio: 0.15
  temperature: 0.1
  max_lambda: 0.5

pretraining:
  epochs: 50
  batch_size: 64
  lr: 1.0e-3
```

### 8.2 Rolling Window Config (`config/s2_config.yaml`)

```yaml
rolling:
  window_len: 24
  stride: 6
  seq_len: 48

clustering:
  k: 4
```

---

## 9. Output Files

### 9.1 Embeddings and Models

| Path | Description |
|------|-------------|
| `data/s15_trainval/checkpoints/pretrain_best.pt` | S1.5 encoder checkpoint |
| `data/s15_trainval/embeddings_s15.npy` | Patient embeddings (11986, 128) |
| `data/s15_trainval/calibrated_stacking/calibrated_stacking_classifier.pkl` | Calibrated mortality classifier |

### 9.2 Temporal Analysis

| Path | Description |
|------|-------------|
| `data/s2light/rolling_embeddings.npy` | Per-window embeddings (11986, 5, 128) |
| `data/s2light/trajectory_stats.json` | Trajectory statistics |

### 9.3 Treatment Analysis

| Path | Description |
|------|-------------|
| `data/s4/*/treatment_aware_report.json` | Treatment-aware model metrics |
| `data/s4/*/causal_analysis_report.json` | PSM/DML causal estimates |

### 9.4 Dashboards

| Path | Description |
|------|-------------|
| `outputs/dashboards/patient_*.html` | Bedside HTML dashboards |

---

## 10. FAQ

**Q: How to select operating threshold for clinical use?**

| Scenario | Threshold | Characteristics |
|----------|-----------|-----------------|
| Triage (high sensitivity) | 0.05 | Captures >90% of deaths |
| Balanced | 0.09 | Recall 83.8%, balanced accuracy 80.1% |
| Resource allocation | 0.30 | Precision ~60% |

**Q: How to reproduce calibration optimization?**

```bash
python3 scripts/s15_train_stacking_classifier.py
python3 scripts/s15_calibrate.py --model-dir data/s15_trainval/stacking_accuracy
python3 scripts/s15_train_calibrated_stacking.py
python3 scripts/s15_calibration_comparison.py
```

**Q: Stage 6 domain adaptation requirements?**

Requires multi-domain data (e.g., multi-center with explicit center labels). Single-domain runs skip adaptation automatically.

---

## 11. Important Notes

1. Always use `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` prefix
2. The calibrated stacking model is recommended for clinical applications
3. All mortality outcomes use verified PhysioNet Outcomes files (14.2% rate)
4. External transfer uses frozen PhysioNet-trained encoder
5. S4 treatment effects are observational; use for hypothesis generation only
