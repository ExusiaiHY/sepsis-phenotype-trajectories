# Self-Supervised Temporal Phenotype Trajectory Analysis of ICU Sepsis Patients from Multi-Center Time-Series Data

**Wang Ruike**

Department of Computer Science, Advanced Programming Course

---

## Abstract

**Background.** Sepsis exhibits substantial clinical heterogeneity, yet most phenotyping approaches rely on static summary features that compress temporal dynamics into single-timepoint representations. Whether learned temporal representations can reveal clinically meaningful phenotype trajectories within the first 48 ICU hours remains underexplored.

**Methods.** We developed a three-stage computational framework applied to 11,986 ICU patients from the PhysioNet 2012 multi-center database (4 hospitals, 2 centers). First, we established a static phenotyping baseline using statistical features and PCA. Second, we trained a Transformer-based encoder via self-supervised learning (masked value prediction combined with a temporal contrastive window objective) and compared four representation methods across clustering quality, mortality stratification, center stability, and missingness robustness. Third, using the selected self-supervised representation, we extracted rolling-window embeddings (24-hour windows, 6-hour stride) and performed descriptive temporal phenotype trajectory analysis via K-Means clustering on per-window embeddings. All mortality outcomes were obtained from verified PhysioNet Outcomes files (in-hospital mortality rate: 14.2%).

**Results.** Static K=4 clustering on PCA-reduced features identified phenotypes with mortality rates ranging from 6.8% to 36.0%. The self-supervised encoder achieved superior center stability (cluster distribution L1 distance = 0.016 vs 0.027 for PCA) and reduced missingness sensitivity (density-norm correlation |r| = 0.15 vs 0.23), while PCA retained stronger static mortality separation. Temporal trajectory analysis revealed that 64.8% of patients maintained a stable phenotype across all windows, while 35.2% exhibited at least one phenotype transition. Stable phenotypes showed strongly stratified mortality (4.0%, 9.7%, 22.5%, 31.7%; range 27.7 percentage points). Single-transition patients had lower mortality (11.4%) than stable patients (15.4%), and phenotype prevalence shifted toward lower-acuity states over the 48-hour window. Sensitivity analysis with reduced overlap (stride=12h) confirmed identical mortality ordering and range (28.0pp), with transition rates increasing under reduced overlap, indicating the primary analysis provides a conservative estimate. Cross-center temporal validation within the PhysioNet 2012 multi-center cohort (train on Center A, evaluate on Center B) confirmed identical mortality ordering, the same highest-risk phenotype, and a 25.8 percentage-point mortality range on the held-out center.

**Conclusions.** Self-supervised temporal representations enable descriptive phenotype trajectory analysis that reveals clinically meaningful within-stay dynamics beyond static clustering. The identified phenotypes show strong mortality stratification with ground-truth outcomes and cross-center temporal validation within the PhysioNet 2012 multi-center cohort. Both centers derive from the same source database; full external validation requires independently collected ICU cohorts. These findings support further investigation of temporal phenotyping for sepsis precision medicine, though causal treatment-response claims require additional study designs.

**Keywords**: sepsis phenotyping, self-supervised learning, temporal trajectories, ICU time-series, multi-center, representation learning

---

## 1. Introduction

### 1.1 Background

Sepsis affects approximately 49 million individuals worldwide annually and is associated with mortality rates ranging from 20% to over 50% in intensive care units (ICUs) [1]. Despite decades of clinical research, sepsis remains one of the most challenging conditions in critical care medicine. A key contributor to this challenge is the recognition that sepsis is not a single homogeneous disease but rather a syndrome encompassing multiple distinct biological phenotypes [2].

The landmark study by Seymour et al. (2019) identified four sepsis phenotypes — designated α, β, γ, and δ — with markedly different clinical trajectories, biomarker profiles, and treatment responses [3]. The δ subtype, characterized by elevated lactate, liver dysfunction, and shock, exhibited the highest 28-day mortality (39%) and the greatest benefit from early aggressive resuscitation. More recently, Antcliffe et al. (2025) argued that the true value of subphenotyping lies in predicting heterogeneous treatment effects (theragnostics) rather than prognosis alone [15]. Feng et al. (2025) demonstrated that organ interaction trajectories can be modeled via deep temporal graph clustering with external validation [17].

### 1.2 Problem Statement

Existing approaches to sepsis subtyping face several challenges:
1. **Static representations**: Most methods compress 48-hour trajectories into summary statistics, discarding temporal dynamics that may distinguish improving from deteriorating patients
2. **Data heterogeneity**: Different ICU databases use different variable nomenclature, measurement frequencies, and data formats
3. **High missingness**: Clinical time-series data are inherently sparse, with laboratory values often measured only once or twice daily
4. **Cross-center generalizability**: Subtypes discovered in a single center may not transfer to other hospitals due to differences in clinical practice and patient populations
5. **Outcome label quality**: Many studies rely on proxy mortality labels rather than verified discharge outcomes

### 1.3 Contributions

This work makes the following contributions:
- A **three-stage phenotyping framework** progressing from static clustering to self-supervised representation learning to descriptive temporal phenotype trajectory analysis
- A **self-supervised Transformer encoder** trained via masked value prediction and temporal contrastive learning, with systematic comparison against PCA baselines
- **Descriptive temporal phenotype trajectories** showing that 35.2% of ICU sepsis patients undergo phenotype transitions within 48 hours, with transitions predominantly toward lower-acuity states
- A **validated data pipeline** using ground-truth in-hospital mortality outcomes from PhysioNet Outcomes files, replacing previously used proxy labels
- **Robustness validation** including stride sensitivity analysis confirming that temporal findings are conservative, and center probe analysis confirming no center information leakage
- **Cross-center temporal validation** within the PhysioNet 2012 multi-center cohort, confirming that phenotype structure and mortality stratification transfer from the training center to a held-out center

---

## 2. Related Work

### 2.1 Sepsis Phenotyping

The concept of sepsis heterogeneity was first systematically investigated by Seymour et al. (2019), who applied latent class analysis (LCA) to 20,189 sepsis patients from three multi-center datasets [3]. Their four-phenotype model has been replicated and extended by subsequent studies [4,5]. More recently, machine learning approaches including deep temporal clustering [6], contrastive learning [7], and organ interaction trajectory modeling [17] have been proposed for subtype discovery. Zhang et al. (2024) demonstrated that time-series k-means with dynamic time warping (DTW) can identify sepsis subphenotypes with external validation and treatment heterogeneity analysis [18]. Antcliffe et al. (2025) provided a comprehensive review of the theragnostic potential of sepsis subphenotypes [15].

### 2.2 Self-Supervised Representation Learning for Clinical Data

Self-supervised representation learning (SSRL) has emerged as a powerful approach for extracting patient representations from unlabeled clinical data. Zheng et al. (2025) conducted a scoping review of 46 SSRL studies for clinical decision-making, identifying Transformer-based (43%), autoencoder-based (28%), and graph-based (17%) model families as dominant approaches [16]. Amirahmadi et al. (2025) introduced trajectory-ordered objectives for BERT-style pretraining on EHR sequences, demonstrating that temporal ordering awareness improves downstream prediction beyond standard masked modeling [19].

### 2.3 ICU Time-Series Analysis

ICU data present unique analytical challenges due to their irregular sampling, high dimensionality, and clinical context dependencies. The PhysioNet/CinC Challenge series has driven methodological advances, with the 2012 challenge focusing on mortality prediction from multi-parameter monitoring data [8] and the 2019 challenge on early sepsis prediction [9]. Huang et al. (2025) introduced MIMIC-Sepsis, a curated benchmark incorporating treatment variables (vasopressors, fluids, ventilation, antibiotics) that substantially improve model performance [20].

---

## 3. Methods

### 3.1 System Architecture

The system implements a three-stage framework (Figure 1):

```
Stage 1: Static Phenotyping Baseline
  Data Loading → Preprocessing → Feature Extraction → PCA → K-Means → Evaluation

Stage 2: Self-Supervised Representation Learning
  Hourly Time Series + Masks → Transformer Encoder → Masked Prediction + Contrastive Loss → Patient Embeddings

Stage 3: Descriptive Temporal Phenotype Trajectories
  Rolling-Window Extraction → Per-Window Embedding → Temporal Clustering → Transition Analysis
```

Each stage builds on the previous, with modules designed as independent components enabling systematic comparison across representation methods.

### 3.2 Data Sources and Outcome Labels

#### 3.2.1 PhysioNet 2012 Multi-parameter ICU Database

The primary dataset comes from the PhysioNet 2012 Challenge [8], containing records from 12,000 ICU patients across four hospitals. Each record includes time-stamped measurements of vital signs (heart rate, blood pressure, respiratory rate, temperature, SpO2, GCS) and laboratory values (creatinine, BUN, glucose, electrolytes, hematology, blood gas).

After quality filtering, 11,986 patients were retained. The data were organized into two experimental centers:
- **Center A** (set-a + set-b): 7,989 patients from the original hospital network
- **Center B** (set-c): 3,997 patients from a separate hospital within the PhysioNet 2012 cohort

**Outcome labels.** Ground-truth in-hospital mortality was obtained from PhysioNet Outcomes files (Outcomes-a.txt, Outcomes-b.txt, Outcomes-c.txt), providing verified labels for all 11,986 patients (mortality rate: 14.2%). An earlier version of this analysis used proxy mortality derived from physiological thresholds (GCS ≤ 5 or sustained MAP < 55 mmHg). Outcome audit revealed that this proxy had a positive predictive value of only 17.7%, with a false-positive mortality rate of 44.8%. All results in this paper use verified outcome labels.

**Data schema.** Following CLIF-inspired design principles [21], the data layer separates continuous measurements (21 variables), observation masks, and proxy-derived indicators into distinct tensors. Proxy indicators (MAP < 65 for vasopressor use, FiO2 > 0.21 for mechanical ventilation) are explicitly labeled as proxies and stored separately from true treatment records, which are unavailable in PhysioNet 2012.

#### 3.2.2 Supplementary Datasets

A configurable synthetic data generator with known ground-truth subtype labels enables quantitative evaluation of clustering recovery (ARI, NMI). A MIMIC-IV mock database (15 patients, 31 features) validates pipeline compatibility with the MIMIC-IV schema.

### 3.3 Preprocessing

Clinical measurements are resampled to a uniform hourly grid over 48 hours post-ICU admission using Last Observation Carried Forward (LOCF). Outliers exceeding 4 standard deviations are clipped. Missing values are imputed via a three-stage strategy: forward fill (up to 6 hours), global median imputation, and zero-fill as fallback. Observation masks recording which values were truly measured versus imputed are preserved as first-class inputs throughout all subsequent stages. Overall missing rate is 73.3%.

### 3.4 Static Feature Engineering (Stage 1 Baseline)

From the preprocessed 3D tensor (n_patients × 48 × 21), we extract a 2D feature matrix using six statistical functions (mean, std, min, max, trend, last value) applied across three temporal windows (12h, 24h, 48h), yielding 378 features plus derived clinical indicators (shock index, lactate clearance). PCA reduces this to 32 dimensions (62.5% variance explained).

### 3.5 Self-Supervised Representation Learning (Stage 2)

#### 3.5.1 Encoder Architecture

A Transformer-based encoder processes the hourly time series with observation masks as explicit input channels. The input is formed by concatenating value and mask vectors: `input = concat([x, mask])` where x ∈ ℝ^(T×21) and mask ∈ ℝ^(T×21), yielding a 42-dimensional input per timestep. A linear projection maps this to d_model = 128 dimensions. Learnable positional encodings provide temporal position awareness. A 2-layer, 4-head Transformer encoder with pre-layer normalization and GELU activation produces per-timestep representations, which are aggregated via observation-density-weighted mean pooling to produce a 128-dimensional patient embedding.

#### 3.5.2 Pretraining Objectives

**Masked value prediction.** For each training sample, 15% of observed values (mask = 1) are randomly zeroed in both the value and mask channels. The encoder must predict the original values at masked positions. The loss is MSE computed only on masked observed positions.

**Temporal contrastive window objective.** Two stochastic 30-hour views are extracted from each patient's 48-hour trajectory with constrained overlap (12–24 hours). Both views are encoded by the shared encoder, then projected through a 2-layer MLP projection head (128 → 128 → 64 with BatchNorm). NT-Xent loss (temperature τ = 0.1) is computed on projections with in-batch negative pairs. The projection head output is used exclusively for the contrastive loss and discarded after pretraining; the encoder output serves as the patient representation.

**Combined loss.** L_total = L_masked + λ(epoch) × L_contrastive, where λ warms up linearly from 0 to 0.5 over the first 10 epochs to allow the encoder to learn basic temporal patterns from reconstruction before contrastive reshaping.

#### 3.5.3 Representation Comparison

Four representations are systematically compared: (1) PCA baseline (32 dimensions), (2) masked reconstruction only (S1, 128 dimensions), (3) masked + contrastive (S1.5, λ=0.5, 128 dimensions), and (4) reduced contrastive ablation (S1.6, λ=0.2, 128 dimensions). All learned encoders have identical architecture (296K–321K parameters) and are trained for 50 epochs with cosine learning rate scheduling.

### 3.6 Descriptive Temporal Phenotype Trajectories (Stage 3)

Using the frozen S1.5 encoder, rolling-window embeddings are extracted for each patient: 5 windows of 24 hours with 6-hour stride, covering positions [0,24), [6,30), [12,36), [18,42), [24,48). K-Means (K=4) is fit on train-split window embeddings only (30,955 windows from 6,191 patients) and applied to all 59,930 windows. Per-patient phenotype trajectories are constructed as 5-element sequences of cluster assignments.

Patients are classified as: **stable** (all 5 labels identical), **single-transition** (exactly 1 label change), or **multi-transition** (2+ changes). Empirical transition matrices, prevalence shifts across window positions, and mortality descriptives by trajectory category are computed. All temporal analyses are descriptive; no latent-state models or causal claims are involved.

### 3.7 Evaluation and Robustness

**Clustering metrics**: Silhouette score, mortality separation (max − min mortality across clusters). **Representation diagnostics**: mortality linear probe (AUROC), center linear probe (AUROC; lower is better), ICU LOS probe (R²), observation density–embedding norm correlation. **Multi-seed evaluation**: All KMeans results reported as mean ± std over 5 random seeds. **Stride sensitivity**: The temporal analysis is repeated with stride=12h (3 windows, 50% overlap) to assess robustness of transition findings to overlap reduction.

---

## 4. Results

### 4.1 Static Phenotyping Baseline (Stage 1)

After quality filtering, 11,986 patients were retained (Center A: 7,989; Center B: 3,997). Overall missing rate was 73.3% across 21 continuous features. Ground-truth in-hospital mortality was 14.2%.

Static K=4 clustering on PCA-reduced statistical features identified phenotypes with mortality rates of 6.8%, 10.0%, 33.8%, and 36.0% (range 29.2 percentage points). Silhouette score was 0.061, consistent with the inherently fuzzy boundaries expected in real ICU data with high missingness.

### 4.2 Representation Learning Comparison (Stage 2)

**Table 1. Representation comparison (K=4, mean ± std over 5 seeds)**

| Method | Silhouette | Mortality Range | Center L1 ↓ | Mort. Probe AUROC | Density |r| ↓ |
|--------|-----------|----------------|------------|-------------------|---------|
| PCA baseline (32d) | 0.061 ± 0.000 | 29.2% | 0.027 | 0.825 | 0.231 |
| S1: masked only (128d) | 0.087 ± 0.000 | 17.6% | 0.024 | 0.825 | 0.247 |
| S1.5: masked + contrastive (128d) | 0.080 ± 0.001 | 24.6% | **0.016** | **0.830** | **0.148** |
| S1.6: λ=0.2 ablation (128d) | 0.079 ± 0.000 | 25.1% | 0.021 | 0.825 | 0.148 |

Center probe AUROC was 0.50–0.52 for all methods (evaluated on a random stratified split mixing both centers), confirming no representation encodes center identity. S1.5 was selected for temporal analysis based on its superior center stability (L1 = 0.016), reduced missingness sensitivity (|r| = 0.148), and highest mortality probe AUROC (0.830). PCA's stronger static mortality separation (29.2% vs 24.6%) is noted; however, PCA requires re-extraction of summary statistics per window and cannot natively process variable-length sub-sequences, making it unsuitable for rolling-window temporal analysis.

### 4.3 Descriptive Temporal Phenotype Trajectories (Stage 3)

#### 4.3.1 Rolling-Window Extraction

Using the frozen S1.5 encoder, 59,930 window embeddings (11,986 patients × 5 windows) were extracted. Per-window observation density decreased from 27.9% (earliest window) to 25.4% (latest window), reflecting declining measurement frequency during ICU stays. Per-window silhouette ranged from 0.072 to 0.082.

#### 4.3.2 Temporal Phenotype Stability and Transitions

Of 11,986 patients, 7,764 (64.8%) maintained a stable phenotype across all five windows, 3,509 (29.3%) exhibited exactly one transition, and 713 (5.9%) exhibited multiple transitions.

**Table 2. Stable phenotype mortality stratification**

| Stable Phenotype | N | In-Hospital Mortality |
|-----------------|---|----------------------|
| Phenotype 0 | 2,216 | 4.0% |
| Phenotype 3 | 1,891 | 9.7% |
| Phenotype 1 | 2,547 | 22.5% |
| Phenotype 2 | 1,110 | 31.7% |

The 27.7 percentage-point mortality range across temporally stable phenotypes exceeded the static S1.5 clustering result (24.6%), suggesting that restricting to temporally stable patients yields purer phenotype subgroups.

#### 4.3.3 Transition Pathways and Mortality

The most frequent non-self transitions were Phenotype 1→0 (956 events, 6.1% of Phenotype 1 exits), Phenotype 1→3 (728 events), and Phenotype 3→0 (713 events), collectively representing movement toward lower-acuity phenotypes. Single-transition patients had lower in-hospital mortality (11.4%) than stable patients (15.4%) or multi-transition patients (15.2%). This descriptive association suggests that phenotype transitions—particularly from higher-risk to lower-risk phenotypes—may reflect favorable clinical trajectory change, though causal interpretation requires controlled study designs.

#### 4.3.4 Temporal Prevalence Shift

Phenotype 0 (lowest mortality) increased in prevalence from 24.6% at window [0,24h) to 33.1% at window [24,48h), while Phenotype 1 decreased from 35.7% to 28.1% (Figure 3). This population-level shift is consistent with clinical stabilization during the first 48 ICU hours, though it may also reflect survivor bias or changes in measurement patterns.

### 4.4 Stride Sensitivity Analysis

To assess the influence of window overlap on temporal findings, we repeated the analysis with stride=12h (3 windows, 50% overlap). Cluster centroids showed near-perfect alignment with the primary analysis (cosine distance < 0.0001 after Hungarian matching). Results:

**Table 3. Stride sensitivity comparison**

| Metric | Stride=6h (primary) | Stride=12h (sensitivity) |
|--------|---------------------|--------------------------|
| Stable patient fraction | 64.8% | 65.6% |
| Non-self transition proportion | 10.4% | 19.1% |
| Mortality ordering | [P0, P3, P1, P2] | [P0, P3, P1, P2] |
| Highest-risk phenotype | P2 (31.7%) | P2 (31.9%) |
| Stable phenotype mortality range | 27.7pp | 28.0pp |

The mortality ordering of stable phenotypes was identical under both strides, the highest-risk phenotype remained Phenotype 2, and the mortality range was preserved (28.0pp vs 27.7pp). The proportion of non-self transitions increased from 10.4% to 19.1% under reduced overlap, indicating that the primary stride=6h analysis provides a conservative estimate of temporal phenotype change rather than an overlap-inflated one.

### 4.5 Cross-Center Temporal Validation

To assess whether the temporal phenotype trajectories generalize beyond the training center, we performed out-of-center validation: the KMeans model was trained on Center A rolling-window embeddings and evaluated on Center B (a separate hospital within the PhysioNet 2012 cohort). The S1.5 encoder was also trained exclusively on Center A data.

**Table 4. Cross-center temporal validation (Center A vs Center B)**

| Metric | Center A (train) | Center B (test) |
|--------|-----------------|-----------------|
| Patients | 7,989 | 3,997 |
| Stable fraction | 65.0% | 64.4% |
| Non-self transition proportion | 10.3% | 10.6% |
| Mortality ordering | [P0, P3, P1, P2] | [P0, P3, P1, P2] |
| Highest-risk phenotype | P2 (32.6%) | P2 (30.0%) |
| Stable P0 mortality | 3.9% | 4.2% |
| Stable P3 mortality | 9.7% | 9.8% |
| Stable P1 mortality | 21.8% | 24.0% |
| Stable P2 mortality | 32.6% | 30.0% |
| Mean prevalence L1 | — | 0.022 |

All six validation criteria were satisfied: stable fraction similarity (diff = 0.6pp), transition proportion similarity (diff = 0.3pp), identical mortality ordering, same highest-risk phenotype, clinically meaningful Center B mortality range (25.8pp), and low per-window prevalence divergence (L1 = 0.022). Per-window silhouette scores were within 0.001 across centers at all five window positions.

These results demonstrate cross-center temporal validation within the PhysioNet 2012 multi-center cohort. Both centers derive from the same source database; full external validation requires independently collected ICU cohorts (e.g., eICU-CRD, MIMIC-IV).

### 4.6 Simulated Data Validation

The synthetic data generator was used for pipeline validation. K-Means on PCA-reduced simulated features achieved ARI = 0.245 and NMI = 0.495 against known ground-truth labels. *Note: Clustering quality metrics from the PhysioNet/CinC 2019 Sepsis dataset require independent rerun and are not reported here.*

---

## 5. Discussion

### 5.1 Three-Level Phenotyping Framework

This work demonstrates a progression from static clustering (Stage 1) through self-supervised representation learning (Stage 2) to descriptive temporal phenotype trajectory analysis (Stage 3). Each stage addresses a limitation of the previous: Stage 2 replaces hand-crafted summary statistics with learned representations that encode temporal patterns and are robust to missingness; Stage 3 reveals within-stay dynamics that static methods cannot capture.

The temporal trajectory analysis shows that approximately one-third of ICU sepsis patients undergo phenotype transitions within the first 48 hours. The clinical significance of these transitions is supported by three findings: (1) stable phenotypes show strongly stratified mortality (4.0%–31.7%), (2) the most common transitions move from higher-risk to lower-risk phenotypes, and (3) single-transition patients have lower mortality than stable patients, consistent with the interpretation that phenotype change may reflect clinical improvement.

### 5.2 Representation Learning Trade-offs

The self-supervised encoder (S1.5) offered improved center stability and reduced missingness sensitivity compared to PCA, though PCA retained stronger static mortality separation. This trade-off is characteristic of learned representations: they capture more generalizable patterns at the cost of discriminative specificity on the training distribution. The selection of S1.5 for temporal analysis was motivated primarily by its ability to process variable-length sub-sequences directly, a capability PCA lacks. The contrastive window objective specifically encourages temporal consistency, making S1.5 naturally suited for rolling-window analysis.

### 5.3 Limitations

1. **Descriptive, not model-based**: The temporal phenotype trajectories arise from per-window clustering rather than latent-state inference (e.g., HMMs). This limits the ability to model transition dynamics probabilistically.
2. **Overlap sensitivity**: Window overlap (75% at stride=6h) smooths embedding changes, potentially underestimating transition rates. Sensitivity analysis with 50% overlap confirmed mortality stratification but showed higher transition rates, suggesting the primary analysis is conservative.
3. **Missing treatment data**: PhysioNet 2012 lacks true treatment records. Proxy indicators (MAP < 65, FiO2 > 0.21) are physiological thresholds, not treatment decisions. Treatment-aware phenotyping requires datasets with documented interventions (e.g., MIMIC-IV with vasopressor, fluid, and antibiotic timing [20]).
4. **Observational associations**: Patients who transition between phenotypes may differ systematically from stable patients in ways not captured by the current feature set. The lower mortality observed in transitioning patients is a descriptive association, not a causal effect of transition.
5. **Single-dataset scope**: While cross-center temporal validation within the PhysioNet 2012 cohort shows that phenotypes transfer across centers (identical mortality ordering, 25.8pp range on held-out Center B), both centers derive from the same source database. External validation on independently collected ICU cohorts is needed to establish broader generalizability.
6. **Data sparsity**: The 73.3% missing rate limits the signal available for both representation learning and clustering. Key sepsis biomarkers (lactate, bilirubin) are extremely sparse in PhysioNet 2012.

### 5.4 Future Directions

1. **Treatment-aware temporal phenotyping**: Integration of treatment variables from MIMIC-IV or eICU would enable analysis of phenotype-treatment interactions and heterogeneous treatment effects
2. **Latent-state models**: Replacing per-window clustering with HMMs or switching state-space models would enable probabilistic transition modeling
3. **External database validation**: Validating temporal phenotypes on independently collected ICU databases (eICU-CRD, MIMIC-IV) would establish generalizability beyond the PhysioNet 2012 cohort
4. **Bedside implementation**: Real-time phenotype assignment from early (6h, 12h) data windows could support clinical decision-making

---

## 6. Conclusion

This study presents a three-stage framework for sepsis phenotyping that progresses from static clustering to self-supervised representation learning to descriptive temporal phenotype trajectory analysis. Applied to 11,986 multi-center ICU patients with ground-truth mortality outcomes, the framework identifies four phenotypes with strongly stratified mortality (4.0%–31.7% for temporally stable patients). Temporal analysis reveals that 35.2% of patients undergo phenotype transitions within 48 hours, with transitions predominantly toward lower-acuity phenotypes and conservative transition rate estimates confirmed by stride sensitivity analysis. Cross-center temporal validation within the PhysioNet 2012 cohort confirms that phenotype structure, mortality ordering, and transition patterns transfer from the training center to a held-out center, though both centers derive from the same source database. These descriptive findings demonstrate that temporal phenotype trajectories provide clinically meaningful and reproducible information beyond static subtyping, supporting further investigation with treatment-aware datasets, external database validation, and prospective study designs.

---

## References

[1] Rudd, K.E., et al. (2020). Global, regional, and national sepsis incidence and mortality, 1990-2017. *The Lancet*, 395(10219), 200-211.

[2] Singer, M., et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 801-810.

[3] Seymour, C.W., et al. (2019). Derivation, validation, and potential treatment implications of novel clinical phenotypes for sepsis. *JAMA*, 321(20), 2003-2017.

[4] Shankar-Hari, M., et al. (2020). Classification of patients with sepsis according to blood genomic endotype. *The Lancet Respiratory Medicine*, 5(10), 816-826.

[5] Davido, A., et al. (2021). Sepsis phenotypes in the ICU: A systematic review. *Journal of Critical Care*, 63, 42-49.

[6] Siirtola, P., et al. (2022). Deep temporal clustering of ICU patient time series. *IEEE Access*, 10, 72716-72728.

[7] Wang, Y., et al. (2023). Contrastive learning for sepsis patient phenotyping. *Scientific Reports*, 13, 12345.

[8] Silva, I., et al. (2012). Predicting in-hospital mortality of ICU patients: The PhysioNet/Computing in Cardiology Challenge 2012. *Computing in Cardiology*, 39, 245-248.

[9] Reyna, M.A., et al. (2019). Early prediction of sepsis from clinical data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 48(2), 210-217.

[10] McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for dimension reduction. *arXiv preprint arXiv:1802.03426*.

[11] Johnson, A.E.W., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10, 1.

[12] Pollard, T.J., et al. (2018). The eICU Collaborative Research Database. *Scientific Data*, 5, 180178.

[13] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.

[14] Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95.

[15] Antcliffe, D., et al. (2025). Sepsis subphenotypes, theragnostics and personalized sepsis care. *Intensive Care Medicine*, 51, 756-768.

[16] Zheng, Y., et al. (2025). A scoping review of self-supervised representation learning for clinical decision making using EHR categorical data. *npj Digital Medicine*, 8(1), 362.

[17] Feng, X., et al. (2025). Subphenotyping sepsis based on organ interaction trajectory using a deep temporal graph clustering model. *EClinicalMedicine*, 90, 103691.

[18] Zhang, et al. (2024). Identification and validation of sepsis subphenotypes using time-series data. *Heliyon*, 10(3), e24551.

[19] Amirahmadi, A., et al. (2025). Trajectory-ordered objectives for self-supervised representation learning of temporal healthcare data using Transformers. *JMIR Medical Informatics*, 13, e68138.

[20] Huang, Y., et al. (2025). MIMIC-Sepsis: A curated benchmark for modeling and learning from sepsis trajectories in the ICU. *arXiv preprint arXiv:2510.24500*.

[21] Rojas, J.C., et al. (2025). A common longitudinal intensive care unit data format (CLIF) for critical illness research. *Intensive Care Medicine*, 51, 556-569.
