# From Static Clusters to Temporal Trajectories: Self-Supervised Phenotyping of ICU Sepsis with Real-Time Deployment

**Wang Ruike**  
School of Biomedical Engineering, ShanghaiTech University

---

## Abstract

**Background:** Sepsis is clinically heterogeneous, but many phenotyping studies collapse early ICU hours into static summary features. Most learned representations also fail to deploy in real-time bedside environments due to high computational latency and poor calibration.

**Methods:** We developed a seven-stage pipeline (S0--S6) using 11,986 ICU stays from PhysioNet 2012 as the primary cohort and 295,140 external stays (MIMIC-IV: 94,458; eICU: 200,859) for transfer validation. The pipeline includes: (1) static clustering baseline; (2) mask-aware self-supervised Transformer encoder (S1.5); (3) rolling-window temporal trajectory analysis (S2--S3); (4) post-hoc calibration (S3.5); (5) treatment-aware external extension (S4); (6) real-time student distillation for bedside deployment (S5--S5-v2); and (7) mechanism-based causal phenotyping via stability-gated CATE estimation and organ-system severity splitting (S6).

**Results:** The selected S1.5 representation achieved the best balance of signal and robustness (center L1 distance: 0.016; density correlation: |r| = 0.148). After calibration (S3.5), the model achieved AUROC 0.873 with 91% ECE reduction (0.222 to 0.020). Temporal trajectory analysis revealed 35.2% of patients underwent phenotype transitions during the first 48 hours, with stable-phenotype mortality ranging from 4.0% to 31.7%. The distilled real-time student (S5-v2) achieved AUROC 0.873 on MIMIC-IV and 0.898 on eICU with 1.1 ms inference latency---meeting all real-time deployment gates (AUROC > 0.85, latency < 10 ms, ECE < 0.05). Mechanism-based causal phenotyping (S6) refined the 4 data-driven clusters into 16 organ-system phenotypes with nearly double the mortality discrimination range (1.3%--50.4% vs 4.8%--30.2%), validated by DoWhy causal refutation and cross-center consistency. However, observational treatment effects from Stage 4 showed zero cross-source consistency (0/6 treatments), indicating they should not be interpreted as transportable causal recommendations.

**Conclusions:** Self-supervised temporal embeddings make early ICU phenotype trajectories visible, well-calibrated, and deployable in real-time bedside environments. Mechanism-based causal phenotyping further refines data-driven clusters into clinically interpretable organ-system subgroups with stronger mortality discrimination. While predictive performance transfers across sources, observational treatment signals remain source-specific and require randomized validation before clinical implementation.

---

## At a Glance

| Item | Summary |
| --- | --- |
| **Primary question** | Can a self-supervised temporal representation reveal clinically meaningful early sepsis trajectories and deploy in real-time? |
| **Primary cohort** | PhysioNet 2012: 11,986 ICU stays, 14.2% in-hospital mortality |
| **External cohorts** | MIMIC-IV: 94,458 stays; eICU: 200,859 stays |
| **Core model** | 2-layer mask-aware Transformer with masked reconstruction + temporal contrastive learning (S1.5) |
| **Key metrics** | AUROC 0.873, ECE 0.020 (91% reduction), latency 1.1 ms |
| **Main temporal finding** | 35.2% transition rate, 27.7 pp mortality range across phenotypes |
| **S6 causal phenotyping** | 16 mechanism-based phenotypes, mortality range 1.3%--50.4% (vs 4.8%--30.2% baseline) |
| **Real-time readiness** | Calibrated transformer student passes all 5 engineering gates on 2/2 sources |
| **Claim boundary** | Stage 3--5 are descriptive/predictive; Stage 4 treatment effects are observational and not transportable; S6 phenotypes are causal-informed but observational |

---

## 1. Introduction

Sepsis is not a single disease course. Patients who meet the same diagnostic definition can differ biologically and evolve differently after ICU admission. Prior work has identified clinically meaningful sepsis subgroups, but most studies rely on static snapshots or summary statistics rather than explicit early trajectories [1--3].

This limitation matters for two reasons. First, ICU sepsis care is dynamic: patients stabilize, deteriorate, or fluctuate within hours. Second, ICU data are sparse in a structured way. If a model treats imputed and observed values as equivalent, it confuses "not measured" with "normal." A third limitation has received less attention: most learned phenotyping models are too computationally expensive for real-time bedside deployment, and their probability outputs are poorly calibrated for clinical decision-making.

This study addresses all three limitations through a seven-stage pipeline that progresses from static phenotypes to temporal trajectories, real-time deployable student models, and mechanism-based causal phenotyping. The central hypothesis is that a mask-aware self-supervised representation can make early sepsis trajectories visible without ignoring heavy missingness, and that this representation can be distilled into a lightweight, well-calibrated model suitable for bedside use.

---

## 2. Methods

### 2.1 Data Sources

**Primary cohort.** The PhysioNet 2012 ICU database [4] provided 11,986 ICU stays after filtering, split into Center A (7,989) and Center B (3,997). In-hospital mortality was 14.2%.

**External cohorts.** MIMIC-IV 3.1 [6] contributed 94,458 ICU stays (41,295 Sepsis-3 for Stage 4). eICU-CRD 2.0 [7] contributed 200,859 stays. These test portability, not replace the primary analysis.

### 2.2 Missingness Structure

The primary cohort has 21 continuous channels on an hourly 48-hour grid. Overall missingness is 73.3%, highly uneven: heart rate/SBP/DBP/MAP (9.8--11.9%), respiratory rate (75.9%), GCS (67.9%), lactate (95.9%), bilirubin (98.3%).

### 2.3 Pipeline Overview

The workflow (Figure 1, to be drawn in LaTeX TikZ) progresses through six stages:

| Stage | Input | Output | Purpose |
| --- | --- | --- | --- |
| S0 | Raw time series | Preprocessed tensors with masks | Data preparation and quality control |
| S1.5 | Hourly values + masks | 128-d self-supervised embeddings | Learn reusable temporal representation |
| S2--S3 | Rolling 24-hour windows | Phenotype trajectories | Make within-stay movement visible |
| S3.5 | Uncalibrated predictions | Calibrated probabilities | Ensure reliable clinical interpretation |
| S4 | Physiology + treatments | Treatment-aware risk models | Observational treatment signal exploration |
| S5--S5-v2 | Teacher model | Real-time student (90K params) | Bedside deployment with <2 ms latency |
| S6 | S2 clusters + organ scores | 16 mechanism-based phenotypes | Causal-informed organ-system phenotyping |

### 2.4 Self-Supervised Learning (S1.5)

The Stage 2 encoder is a 2-layer Transformer trained with: (1) masked value prediction for local temporal structure; and (2) temporal contrastive learning to align overlapping patient views. Values and masks are both inputs, making the representation explicitly aware of observation patterns versus imputed values.

### 2.5 Calibration (S3.5)

Post-hoc temperature scaling and Platt scaling were applied to reduce expected calibration error (ECE) while preserving discriminative performance.

### 2.6 Real-Time Student (S5--S5-v2)

Knowledge distillation transferred the S4 teacher (321K parameters, ~50 ms latency) to a lightweight student (90K parameters). Two architectures were evaluated: transformer and temporal convolutional network (TCN). The calibrated transformer was selected for deployment based on real-data performance (TCN regressed on MIMIC-IV despite synthetic gains).

### 2.7 Observational Causal Analysis (S4)

Propensity score matching (PSM) and double machine learning (DML) estimated conditional average treatment effects (CATE) for six interventions. Results are explicitly flagged as hypothesis-generating, not causal recommendations.

---

## 3. Results

### 3.1 Data Preprocessing and Missingness (S0)

The S0 preprocessing flow (Figure 2A) converted 11,986 raw stays into ML-ready tensors. The missingness pattern (Figure 2B) shows the defining challenge: 73.3% overall missingness with highly structured sparsity---core hemodynamics are relatively dense while laboratory values are rarely observed.

### 3.2 Representation Learning (S1.5)

S1.5 (masked + contrastive) was selected over PCA and masked-only alternatives (Figure 3) because it best balances clinical signal with robustness:

| Representation | Silhouette | Center L1 (↓) | Mortality AUROC | Density |r| (↓) |
| --- | ---: | ---: | ---: | ---: |
| PCA baseline | 0.061 | 0.027 | 0.825 | 0.231 |
| S1 masked only | 0.087 | 0.024 | 0.825 | 0.247 |
| **S1.5 (selected)** | **0.080** | **0.016** | **0.830** | **0.148** |

Lower center L1 indicates better cross-center stability; lower density correlation indicates less sensitivity to observation sparsity. S1.5 dominates on both robustness metrics while maintaining strong discriminative performance.

Training convergence (Figure 3B) occurred by epoch 35 with early stopping. The selected representation is the most reusable across windows, centers, and external datasets, not the most aggressive static separator.

### 3.3 Temporal Trajectory Analysis (S2--S3)

Rolling-window analysis revealed substantial within-stay movement:

- **Stable trajectories**: 64.8%
- **Single transition**: 29.3%
- **Multiple transitions**: 5.9%
- **Non-self transition events**: 10.4%

The phenotype sequences are clinically ordered (Figure 4). Among stable patients, mortality spans 27.7 percentage points:

- Phenotype 0 (low risk): 4.0%
- Phenotype 3 (intermediate): 9.7%
- Phenotype 1 (medium risk): 22.5%
- Phenotype 2 (high risk): 31.7%

Trajectory category analysis reveals that single-transition patients have lower mortality (11.4%) than stable patients (15.4%), suggesting some transitions represent clinical improvement.

### 3.4 Calibration Improvement (S3.5)

Post-hoc calibration achieved 91% ECE reduction (Figure 5):

| Metric | Before | After | Change |
| --- | ---: | ---: | --- |
| AUROC | 0.873 | 0.873 | Preserved |
| ECE | 0.222 | 0.020 | -91% |
| Brier score | 0.144 | 0.090 | -38% |
| Recall | 83.6% | 83.8% | Preserved |

Calibration ensures predicted probabilities match observed frequencies, essential for clinical risk communication.

### 3.5 Treatment-Aware Extension (S4)

Stage 4 models achieved strong predictive performance on external cohorts:

| Source | Outcome | AUROC | Balanced Acc | ECE |
| --- | --- | ---: | ---: | ---: |
| MIMIC-IV Sepsis-3 | In-hospital mortality | 0.870 | 0.786 | 0.013 |
| eICU-CRD | 28-day mortality | 0.898 | 0.816 | 0.012 |

However, observational treatment effects showed **zero cross-source consistency** (Figure 6). All six treatments (early vasopressor, mechanical ventilation, fluid bolus, antibiotics, RRT) had opposite effect directions across MIMIC-IV and eICU. This is not a failure---it is the correct warning signal that residual confounding and practice-pattern differences prevent transportable causal inference from observational data.

### 3.6 Real-Time Deployment (S5--S5-v2)

The distilled student achieves real-time performance (Figure 7):

| Model | Parameters | AUROC | Latency | ECE |
| --- | ---: | ---: | ---: | ---: |
| Teacher (S4) | 321K | 0.870 | ~50 ms | 0.013 |
| S5 Student (Transformer) | 91K | 0.875 | 1.12 ms | 0.011 |
| **S5-v2 (Calibrated)** | **91K** | **0.873** | **1.10 ms** | **0.010** |
| S5-v2 (TCN, exploratory) | 34K | 0.860 | 0.80 ms | -- |

All five engineering validation gates passed on both MIMIC-IV and eICU (Figure 7B):

1. AUROC > 0.85 ✓
2. Latency < 10 ms ✓ (actual: ~1.1 ms)
3. ECE < 0.05 ✓ (actual: ~0.01)
4. Memory < 1 MB ✓
5. All unit tests pass ✓

The TCN variant showed 62% parameter reduction and faster synthetic latency but regressed on real MIMIC-IV data, so it is retained as exploratory only.

### 3.7 Mechanism-Based Causal Phenotyping (S6)

Stage 6 refines the 4 data-driven S2 temporal clusters into 16 clinically interpretable, mechanism-based phenotypes using organ-system scoring, stability-gated causal inference, missingness covariate encoding, and CORAL domain adaptation.

**Method.** The pipeline estimates conditional average treatment effects (CATE) for vasopressor exposure via cross-fitted double machine learning (DML), gated by a stability criterion (max std ≤ 0.12, max |q90| ≤ 0.15). SOFA organ sub-scores (Sepsis-3 definition, 24-hour horizon) provide the physiological basis for phenotype naming. Hierarchical severity splitting on four target systems (respiratory, hemodynamic responsive, hemodynamic refractory, neurological) produces recovering/base/critical tiers. DoWhy causal refutation validates the treatment effect estimate.

**Table: S6 mechanism-based phenotypes (top 8 by clinical significance)**

| Phenotype | n | Fraction | Mortality | SOFA | CATE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mild organ stable | 1,077 | 9.0% | 1.3% | 0.75 | +0.011 |
| Hemodynamic refract. recovering | 980 | 8.2% | 6.6% | 2.65 | -0.014 |
| Respiratory failure (base) | 2,132 | 17.8% | 16.0% | 7.71 | +0.006 |
| Multi-organ deteriorating | 538 | 4.5% | 22.9% | 8.27 | +0.007 |
| Hemodynamic refract. critical | 469 | 3.9% | 29.9% | 6.99 | -0.022 |
| Respiratory failure critical | 363 | 3.0% | 31.4% | 9.82 | +0.003 |
| Neurological decline critical | 86 | 0.7% | 32.6% | 7.05 | +0.007 |
| Hemodynamic responsive critical | 228 | 1.9% | 50.4% | 8.62 | +0.056 |

**Table: S6 vs S2 baseline comparison**

| Metric | S2 Baseline | S6 Optimized | Change |
| --- | ---: | ---: | --- |
| Group count | 4 | 16 | +300% |
| Mortality range | 0.254 | 0.491 | +93% |
| Weighted mortality std | 0.087 | 0.091 | +5% |
| Dominant group fraction | 32.2% | 17.8% | -45% |

**Causal validation.** DoWhy estimated ATE = 0.0076 for vasopressor exposure. Both refutation tests passed: random common cause (p = 0.450) and placebo treatment (p = 0.452). CORAL domain adaptation reduced the weighted cross-center mean gap from 0.043 to 0.030, and domain probe accuracy decreased from 0.490 to 0.477, confirming reduced center-specific signal leakage.

**Cross-center consistency.** Center A (n = 7,989) and Center B (n = 3,997) showed matching phenotype distributions and mortality ordering across all 16 phenotypes. The hemodynamic responsive critical phenotype---the highest-risk subgroup---had mortality of 51.4% in Center A and 48.8% in Center B.

### 3.8 Robustness Checks

**Cross-center validation.** Center A and Center B show identical mortality ordering and the same highest-risk phenotype. Mean per-window prevalence L1 distance: 0.022.

**Temporal stride sensitivity.** Reducing overlap from 6-hour to 12-hour stride changed transition frequency (10.4% to 19.1%) but preserved mortality ordering and range (27.7 vs 28.0 pp).

**External transfer.** Frozen representation remains usable on MIMIC-IV (15/21 channels mapped, 55.5% missingness, silhouette 0.119) and eICU (12/21 channels, 81.4% missingness, silhouette 0.193).

---

## 4. Discussion

This study makes four contributions to sepsis phenotyping. First, it demonstrates that mask-aware self-supervised learning produces representations suitable for temporal reuse across windows, centers, and external cohorts. The S1.5 selection criterion---balancing signal with robustness to center mix and observation density---is a modeling lesson applicable beyond sepsis.

Second, the temporal trajectory results show that early ICU sepsis is neither completely stable nor chaotically unstable. The 35.2% transition rate and structured movement pathways suggest that phenotype labels should be allowed to change during the first 48 hours rather than fixed at admission.

Third, the real-time deployment pipeline (S5-v2) proves that complex temporal phenotyping can be distilled into a lightweight, well-calibrated model suitable for bedside use. The 1.1 ms latency and 0.010 ECE meet clinical requirements for real-time risk stratification.

Fourth, mechanism-based causal phenotyping (S6) demonstrates that data-driven clusters can be refined into clinically interpretable organ-system subgroups. The 16 S6 phenotypes nearly double the mortality discrimination range compared to the 4 S2 clusters (49.1 pp vs 25.4 pp), while maintaining cross-center consistency. The hemodynamic responsive phenotypes show the strongest positive CATE (+0.04--0.06), while refractory phenotypes show negative CATE, consistent with their clinical definitions.

The Stage 4 results set an important boundary. While treatment-aware models predict well (AUROC 0.87--0.90), observational treatment effects disagree across sources. S6 addresses this partially by using stability-gated DML on the primary cohort, but the causal estimates remain observational and require randomized validation.

### Limitations

1. **Descriptive analysis:** Phenotypes are clusters, not probabilistic states.
2. **No primary treatment timestamps:** Stage 4 is external only.
3. **Frozen transfer:** No full retraining on external sources.
4. **Risk-ordered labels:** S2 labels are not mapped to molecular mechanisms; S6 labels are organ-system-based but still observational.
5. **Observational causal inference:** Treatment signals are source-specific; S6 CATE estimates are stability-gated but not randomized.

---

## 5. Conclusion

This project demonstrates readable, deployable temporal phenotyping without overstating the science. The complete S0--S6 pipeline progresses from 11,986 PhysioNet stays to real-time bedside-ready models with 1.1 ms inference and 91% calibration improvement, and further refines data-driven clusters into 16 mechanism-based phenotypes with 49.1 pp mortality discrimination range.

The strongest conclusion is descriptive: self-supervised temporal embeddings make early ICU trajectories visible and reusable across contexts. The real-time student is deployment-ready for descriptive risk stratification. The intermediate conclusion is mechanistic: organ-system causal phenotyping produces clinically interpretable subgroups that nearly double the mortality discrimination of data-driven clusters. The weaker conclusion is causal: observational treatment signals require randomized validation before clinical implementation. Maintaining this distinction makes the project credible and clinically actionable.

---

## References

1. Rudd KE, Johnson SC, Agesa KM, et al. Global, regional, and national sepsis incidence and mortality, 1990-2017. *The Lancet*. 2020;395(10219):200-211.
2. Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*. 2016;315(8):801-810.
3. Seymour CW, Kennedy JN, Wang S, et al. Derivation, validation, and potential treatment implications of novel clinical phenotypes for sepsis. *JAMA*. 2019;321(20):2003-2017.
4. Silva I, Moody G, Scott DJ, et al. Predicting in-hospital mortality of ICU patients: The PhysioNet/CinC Challenge 2012. *Computing in Cardiology*. 2012;39:245-248.
5. Johnson AEW, Bulgarelli L, Pollard TJ, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*. 2023;10:1.
6. Pollard TJ, Johnson AEW, Raffa JD, et al. The eICU Collaborative Research Database. *Scientific Data*. 2018;5:180178.

---

## Figure Legends

**Figure 1. Complete S0--S6 Pipeline.** Schematic of the seven-stage workflow from raw ICU data to real-time bedside deployment and mechanism-based causal phenotyping. S0: data preprocessing; S1.5: self-supervised representation learning; S2--S3: temporal trajectory analysis; S3.5: calibration; S4: treatment-aware extension; S5--S5-v2: real-time student distillation; S6: mechanism-based causal phenotyping.

**Figure 2. S0 Data Preprocessing and Missingness.** (A) Preprocessing flow from 11,986 raw stays to ML-ready tensors. (B) Temporal observation pattern heatmap (top) and missingness rate over time (bottom), showing 73.3% overall missingness with structured sparsity.

**Figure 3. S1.5 Representation Selection and Training.** (A) Comparison of PCA, S1 masked-only, and S1.5 (masked + contrastive) across clustering quality, cross-center stability, and missingness robustness. S1.5 selected for best balance. (B) Training convergence curves showing loss stabilization by epoch 35.

**Figure 4. S2--S3 Temporal Phenotype Trajectories.** (A) Phenotype transitions across five rolling 24-hour windows (35.2% of patients transition at least once). (B) Mortality stratification by stable phenotype (27.7 pp range) and by trajectory category.

**Figure 5. S3.5 Calibration Improvement.** (A) Reliability diagrams showing 91% ECE reduction after calibration (ECE: 0.222 to 0.020). (B) Metrics before/after calibration: AUROC and recall preserved, ECE and Brier score substantially improved.

**Figure 6. S4 Treatment Effects Across Sources.** Conditional average treatment effects (CATE) estimated via PSM + DML for five interventions. Zero of six treatments show cross-source consistency (opposite effect directions on MIMIC-IV vs eICU), indicating observational effects are not transportable.

**Figure 7. S5--S5-v2 Real-Time Deployment Profile.** (A) Efficiency vs. performance trade-off: calibrated student achieves 90K parameters with AUROC 0.873--0.898. (B) Latency comparison: all student variants meet the 10 ms real-time threshold (actual: ~1.1 ms). (C) Engineering validation gates: 5/5 gates passed on both MIMIC-IV and eICU.
