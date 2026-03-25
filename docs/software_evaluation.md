# Software Evaluation

# ICU Sepsis Temporal Phenotype Trajectory Analysis System -- Evaluation Report

---

## 1. Software Objectives

This software implements a three-stage computational framework for self-supervised temporal phenotype trajectory analysis of ICU sepsis patients. The specific objectives include:

1. Performing standardized preprocessing on multi-center ICU time-series data (PhysioNet 2012, 11,986 patients), including structured missingness handling, outlier clipping, and Z-score normalization
2. Training a Transformer-based self-supervised encoder via masked value prediction and temporal contrastive learning (Stage 2)
3. Extracting rolling-window embeddings and performing descriptive temporal phenotype trajectory analysis via per-window clustering (Stage 3)
4. Building a leakage-aware OOF stacking mortality classifier with probability calibration optimization
5. Providing comprehensive evaluation covering clustering quality, mortality stratification, cross-center validation, calibration analysis, and bootstrap confidence intervals
6. Supporting frozen-transfer temporal analysis on external databases (MIMIC-IV, eICU-CRD)

## 2. Testing Environment

| Item | Configuration |
|------|---------------|
| Operating System | macOS Darwin 25.3.0 |
| Python Version | 3.14.0a2 |
| CPU | Apple Silicon |
| Memory | 8 GB |
| Key Dependencies | scikit-learn 1.6.1, numpy 2.2.1, pandas 2.2.3, PyTorch 2.5.1, scipy 1.15.0 |

## 3. Test Data Description

### 3.1 Primary Dataset: PhysioNet 2012 Multi-Center ICU Database

| Parameter | Value |
|-----------|-------|
| Total Patients | 11,986 (after quality filtering from 12,000) |
| Center A (set-a + set-b) | 7,989 patients (training + validation) |
| Center B (set-c) | 3,997 patients (held-out test) |
| Time Window | 48 hours (hourly resolution) |
| Continuous Features | 21 (vital signs + laboratory + blood gas) |
| Observation Masks | Binary per-variable per-hour (21 channels) |
| Proxy Indicators | 2 (MAP < 65, FiO2 > 0.21) |
| Static Features | 14 (age, sex, ICU type, height, weight, etc.) |
| In-Hospital Mortality | 14.2% (verified from PhysioNet Outcomes files) |
| Overall Missing Rate | 73.3% |
| Core Hemodynamic Missingness | 9.8%--11.9% (HR, SBP, DBP, MAP) |
| Laboratory Missingness | 93.8% mean (bilirubin 98.3%, lactate 95.9%) |

### 3.2 Supplementary External Datasets

| Database | Cohort Size | Mapped Channels | Purpose |
|----------|-------------|-----------------|---------|
| MIMIC-IV 3.1 | 94,458 ICU stays | 15/21 | Frozen-transfer temporal replication |
| eICU-CRD 2.0 | 200,859 ICU stays | 12/21 | Frozen-transfer temporal replication |
| PhysioNet/CinC 2019 | 40,331 sepsis stays | 18/21 | Auxiliary supervision bridge |

### 3.3 Legacy Simulated Data (V1 Pipeline)

The system also includes a built-in simulated data generator (500 patients, 4 subtypes, AR(1) dynamics) for development validation of the legacy static clustering pipeline.

## 4. Functional Test Results

### 4.1 Stage 0: Data Pipeline Tests

| Function | Test Method | Result | Status |
|----------|-------------|--------|--------|
| PhysioNet 2012 Extraction | Extract set-a/b/c | (11986, 48, 21) continuous tensor + masks | Passed |
| Outcome Label Verification | Audit vs. Outcomes files | 14.2% mortality, PPV audit passed | Passed |
| Forward Fill Imputation | Max 6h gap fill | Applied correctly, masks preserved | Passed |
| Global Median Imputation | Per-feature median | No residual NaN after imputation | Passed |
| Outlier Clipping | 4-sigma clipping | Values within ±4σ range | Passed |
| Z-score Normalization | Per-feature standardization | Mean ≈ 0, std ≈ 1 per feature | Passed |
| Cross-Center Split | center_a → train+val, center_b → test | Stratified by mortality, correct indices | Passed |
| External Cohort Alignment | MIMIC-IV / eICU → S0 schema | Channel mapping + reuse of preprocessing stats | Passed |

### 4.2 Stage 1--1.5: Self-Supervised Pretraining Tests

| Function | Test Method | Result | Status |
|----------|-------------|--------|--------|
| Masked Value Prediction | 15% mask, MSE on observed | Reconstruction loss converges | Passed |
| Temporal Contrastive Learning | NT-Xent (τ=0.1), stochastic 30h views | cos_pos > cos_neg, alignment improves | Passed |
| Combined Training | L_masked + λ(t)·L_contrastive, λ warmup | Total loss monotonically decreases | Passed |
| Encoder Embedding Quality | 128d patient embeddings | Center stability L1=0.016, density |r|=0.148 | Passed |
| Representation Comparison | PCA vs S1 vs S1.5 vs S1.6 | S1.5 best on center stability + density robustness | Passed |

### 4.3 Stage 2--3: Temporal Clustering and Trajectory Tests

| Function | Test Method | Result | Status |
|----------|-------------|--------|--------|
| Rolling Window Extraction | 24h window, 6h stride, 5 windows | (11986, 5, 128) embedding tensor | Passed |
| Per-Window KMeans (K=4) | Fit on train windows, apply to all | 59,930 window assignments | Passed |
| Trajectory Classification | Stable / single-transition / multi-transition | 64.8% / 29.3% / 5.9% | Passed |
| Mortality Stratification | Stable phenotype mortality rates | 4.0%, 9.7%, 22.5%, 31.7% (range 27.7pp) | Passed |
| Stride Sensitivity | Stride=12h (3 windows, 50% overlap) | Identical mortality ordering, range 28.0pp | Passed |
| Cross-Center Validation | Train Center A, test Center B | All 6 criteria satisfied | Passed |
| External Temporal Transfer | MIMIC-IV + eICU frozen S1.5 | Meaningful clusters and transitions recovered | Passed |

### 4.4 Downstream Mortality Classification Tests

| Function | Test Method | Result | Status |
|----------|-------------|--------|--------|
| Frozen S1.5 Probe | Logistic regression on embeddings | AUROC 0.829, balanced acc 0.745 | Passed |
| Advanced Classifier | HGB on fused features | AUROC 0.862, balanced acc 0.780 | Passed |
| OOF Stacking Committee | 5-fold CV, 3 base learners, meta-LR | AUROC 0.873, accuracy 0.880 | Passed |
| Bootstrap Confidence Intervals | 500 bootstrap samples | AUROC 95% CI: 0.858--0.888 | Passed |
| Calibration Analysis | 10-bin ECE, Brier score | Original: Brier 0.144, ECE 0.222 | Passed |
| **Calibration Optimization** | **Calibrated stacking + Platt scaling** | **Brier 0.090, ECE 0.023, AUROC 0.873** | **Passed** |

### 4.5 Calibration Pipeline Tests

| Function | Test Method | Result | Status |
|----------|-------------|--------|--------|
| Temperature Scaling | Fit T on val logits | T=0.43, ECE 0.222→0.137 | Passed |
| Platt Scaling | Logistic regression on logits | a=2.72, b=-3.69, ECE→0.093 | Passed |
| Isotonic Regression | Non-parametric calibration | ECE→0.095, AUROC reduced to 0.838 | Passed |
| Bayesian Prior Calibration | 14.2% mortality prior | strength=0.45, ECE→0.123 | Passed |
| Composite (Temp+Bayesian) | Two-stage calibration | ECE→0.073, Brier→0.095 | Passed |
| Calibrated Stacking | Structural fix + Platt | **ECE 0.023, Brier 0.090, AUROC 0.873** | Passed |
| Hparam Search | 30 configs, depth×lr×iter×C | Optimal: d=3, lr=0.03, iter=300, C=0.05 | Passed |

## 5. Model Performance Evaluation

### 5.1 Self-Supervised Representation Quality (Stage 2)

| Method | Silhouette | Mort. Range | Center L1 ↓ | Mort. Probe | |r|_density ↓ |
|--------|-----------|------------|------------|------------|--------------|
| PCA (32d) | 0.061 | 29.2% | 0.027 | 0.825 | 0.231 |
| S1: masked (128d) | 0.087 | 17.6% | 0.024 | 0.825 | 0.247 |
| **S1.5: mask+contr. (128d)** | **0.080** | **24.6%** | **0.016** | **0.830** | **0.148** |
| S1.6: λ=0.2 (128d) | 0.079 | 25.1% | 0.021 | 0.825 | 0.148 |

S1.5 was selected for temporal analysis based on best center stability (L1=0.016) and lowest missingness sensitivity (|r|=0.148).

### 5.2 Temporal Phenotype Trajectories (Stage 3)

| Metric | Primary (stride=6h) | Sensitivity (stride=12h) |
|--------|---------------------|--------------------------|
| Stable patient fraction | 64.8% | 65.6% |
| Non-self transition rate | 10.4% | 19.1% |
| Mortality ordering | [P0, P3, P1, P2] | [P0, P3, P1, P2] |
| Highest-risk phenotype | P2 (31.7%) | P2 (31.9%) |
| Stable mortality range | 27.7 pp | 28.0 pp |

### 5.3 Downstream Mortality Classification

| Model / Operating Point | Acc. | Bal. Acc. | Prec. | Recall | F1 | AUROC |
|------------------------|------|-----------|-------|--------|-----|-------|
| Frozen S1.5 probe | 0.784 | 0.745 | 0.372 | 0.691 | 0.484 | 0.829 |
| Feature-fusion HGB | 0.791 | 0.780 | 0.391 | 0.764 | 0.517 | 0.862 |
| End-to-end fine-tune + aux | 0.795 | 0.753 | 0.388 | 0.692 | 0.498 | 0.842 |
| OOF stacking (accuracy thr) | 0.880 | 0.653 | 0.682 | 0.333 | 0.448 | 0.873 |
| OOF stacking (balanced thr) | 0.803 | 0.792 | 0.409 | 0.776 | 0.536 | 0.873 |
| **Calibrated stacking** | **0.766** | **0.801** | **0.356** | **0.838** | **0.499** | **0.873** |
| Majority baseline | 0.854 | 0.500 | --- | 0.000 | --- | --- |

### 5.4 Calibration Optimization Results

| Method | Brier ↓ | ECE ↓ | MCE ↓ | AUROC | Recall | Mean Pred |
|--------|---------|-------|-------|-------|--------|-----------|
| Original uncalibrated | 0.144 | 0.222 | 0.423 | 0.873 | 0.836 | 0.369 |
| + Temperature scaling | 0.142 | 0.137 | 0.533 | 0.873 | 0.814 | --- |
| + Platt scaling | 0.107 | 0.093 | 0.400 | 0.873 | 0.677 | --- |
| + Isotonic regression | 0.108 | 0.095 | 0.454 | 0.838 | 0.699 | --- |
| + Bayesian prior (14.2%) | 0.104 | 0.123 | 0.143 | 0.873 | 0.785 | --- |
| + Composite (Temp+Bayesian) | 0.095 | 0.073 | 0.236 | 0.873 | 0.790 | --- |
| **Calibrated stacking (optimal)** | **0.090** | **0.023** | **0.098** | **0.873** | **0.838** | **0.141** |

**Root cause**: `class_weight="balanced"` in the meta-learner inflated positive-class probabilities by ~6x, resulting in mean predicted probability (36.9%) far exceeding the observed mortality rate (14.6%).

**Resolution**: Removing class-weight balancing, reducing base learner depth (5→3), and applying lightweight Platt scaling produced well-calibrated probabilities (mean prediction 14.1% vs observed 14.6%) with ECE=0.023 and fully preserved AUROC.

### 5.5 Cross-Center Temporal Validation

| Metric | Center A (train) | Center B (test) |
|--------|-------------------|-----------------|
| Patients | 7,989 | 3,997 |
| Stable fraction | 65.0% | 64.4% |
| Mortality ordering | [P0, P3, P1, P2] | [P0, P3, P1, P2] |
| Highest-risk phenotype | P2 (32.6%) | P2 (30.0%) |
| Stable mortality range | 28.7 pp | 25.8 pp |
| Mean prevalence L1 | --- | 0.022 |

All six cross-center validation criteria were satisfied.

## 6. Runtime Efficiency Analysis

| Pipeline Component | Cohort Size | Time | Notes |
|-------------------|-------------|------|-------|
| S0: PhysioNet 2012 extraction | 11,986 | ~5s | One-time data preparation |
| S0: Preprocessing (ffill+median+clip+zscore) | 11,986 | ~2s | Cached to disk |
| S1.5: Contrastive pretraining (50 epochs) | 7,989 train | ~8 min | GPU recommended; CPU ~25 min |
| S1.5: Embedding extraction | 11,986 | ~3s | Single forward pass |
| S2: Rolling window embedding extraction | 59,930 windows | ~15s | Memory-mapped output |
| S2: KMeans clustering (K=4) | 59,930 windows | ~2s | Fit on train, predict all |
| S3: Transition analysis + visualization | 11,986 | ~1s | Lightweight |
| Stacking classifier training (5-fold CV) | 7,989 dev | ~25s | 3 base learners × 5 folds |
| Calibration pipeline (6 methods) | 3,997 test | ~1s | Post-hoc, no retraining |
| Calibrated stacking training | 7,989 dev | ~30s | With hparam-optimized specs |
| Calibration hparam search (30 configs) | 7,989 dev | ~2 min | Full grid |

Full pipeline from raw data to calibrated predictions completes in under 15 minutes on CPU.

## 7. Strengths and Limitations

### 7.1 Strengths

1. **Three-Stage Progressive Architecture**: Static baseline → self-supervised representations → temporal trajectory analysis, each stage building on the previous
2. **Mask-Aware Self-Supervised Learning**: Explicit observation masks prevent conflating missing values with normality; contrastive window alignment enables rolling-window reuse
3. **Comprehensive Calibration Framework**: Five post-hoc methods + structural calibration-aware stacking with 30-config hyperparameter search; final ECE=0.023
4. **Clinically Meaningful Outputs**: Probability predictions aligned with true mortality rates (14.1% vs 14.6%), suitable for bedside risk communication
5. **Cross-Center Validation**: Identical phenotype mortality ordering between training and held-out centers (within PhysioNet 2012 cohort)
6. **External Transfer Capability**: Frozen S1.5 encoder processes MIMIC-IV (94K) and eICU (201K) stays with meaningful trajectory recovery
7. **Leakage-Aware Evaluation**: OOF stacking with bootstrap CIs, permutation importance, and stratified calibration analysis
8. **Modular, Reproducible Design**: YAML-configured, script-driven pipeline with deterministic seeds; each stage independently runnable

### 7.2 Limitations

1. **Descriptive Temporal Analysis**: Per-window clustering rather than latent-state inference (no transition probability modeling)
2. **Single-Dataset Encoder Training**: S1.5 encoder trained only on PhysioNet 2012; external transfers use frozen weights under incomplete channel overlap
3. **Missing Treatment Data**: PhysioNet 2012 lacks true treatment records; proxy indicators are physiological thresholds, not interventions
4. **Observational Associations**: Lower mortality in single-transition patients is descriptive, not causal
5. **Window Overlap Sensitivity**: 75% overlap at stride=6h smooths transitions; sensitivity analysis confirms robustness but exact rates depend on stride
6. **Data Sparsity Ceiling**: 73.3% overall missingness, with key biomarkers (bilirubin 98.3%, lactate 95.9%) rarely observed

## 8. Verification Against Design Targets

| Target | Required | Achieved | Status |
|--------|----------|----------|--------|
| ECE (calibration) | ≤ 0.10 | **0.023** | Exceeded |
| Brier score | Significant reduction | **0.090** (37% ↓ from 0.144) | Achieved |
| AUROC (discrimination) | ≥ 0.85 | **0.873** | Achieved |
| Recall (sensitivity) | ≥ 75% | **83.8%** | Exceeded |
| Cross-center stability | Identical mortality ordering | **Confirmed** | Achieved |
| Mean predicted probability | Aligned with base rate | **14.1% vs 14.6%** | Achieved |

## 9. Future Improvement Directions

1. **Treatment-Aware Temporal Phenotyping**: Extend to MIMIC-IV / eICU with true intervention timestamps for phenotype-treatment interaction analysis
2. **Source-Specific Encoder Retraining**: Compare frozen transfer vs multi-source pretraining to quantify channel mismatch effects
3. **State-Space Modeling**: Replace per-window clustering with HMMs or switching state-space models for probabilistic transition modeling
4. **Online Prediction**: Deploy calibrated model for real-time 6h/12h window mortality risk updates at bedside
5. **Richer Phenotype Interpretation**: Post-hoc attribution (SHAP), multimodal augmentation, and linkage to organ-support variables
6. **Prospective Validation**: Test temporal phenotype trajectories in a prospective clinical study
