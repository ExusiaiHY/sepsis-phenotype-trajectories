# Self-Supervised Temporal Phenotype Trajectory Analysis of ICU Sepsis Patients from Multi-Center Time-Series Data

**Wang Ruike**

Department of Computer Science, Advanced Programming Course

---

## Abstract

**Background.** Sepsis exhibits substantial clinical heterogeneity, yet most phenotyping approaches rely on static summary features that compress temporal dynamics into single-timepoint representations. Whether learned temporal representations can reveal clinically meaningful phenotype trajectories within the first 48 ICU hours remains underexplored.

**Methods.** We developed a three-stage computational framework applied to 11,986 ICU patients from the PhysioNet 2012 multi-center database (4 hospitals, 2 centers). First, we established a static phenotyping baseline using statistical features and PCA. Second, we trained a Transformer-based encoder via self-supervised learning (masked value prediction combined with a temporal contrastive window objective) and compared four representation methods across clustering quality, mortality stratification, center stability, and missingness robustness. Third, using the selected self-supervised representation, we extracted rolling-window embeddings (24-hour windows, 6-hour stride) and performed descriptive temporal phenotype trajectory analysis via K-Means clustering on per-window embeddings. As supplementary downstream validation, we added three supervised analyses: a logistic-regression classifier on frozen S1.5 embeddings, a local bridge aligning 40,331 PhysioNet/CinC 2019 sepsis stays to 18 shared channels for auxiliary supervision, and an end-to-end attention-pooled fine-tuning model followed by a 35-run downstream hyperparameter search. We additionally characterized cohort composition and structured missingness to interpret why explicit mask-aware modeling is necessary. All mortality outcomes were obtained from verified PhysioNet Outcomes files (in-hospital mortality rate: 14.2%).

**Results.** Data analysis showed a steep missingness gradient between core hemodynamic variables (9.8%-11.9% missing for heart rate and blood pressure channels) and laboratory variables (mean 93.8% missing; bilirubin 98.3%). Static K=4 clustering on PCA-reduced features identified phenotypes with mortality rates ranging from 6.8% to 36.0%. The self-supervised encoder achieved superior center stability (cluster distribution L1 distance = 0.016 vs 0.027 for PCA) and reduced missingness sensitivity (density-norm correlation |r| = 0.15 vs 0.23), while PCA retained stronger static mortality separation. A downstream mortality classifier trained on frozen S1.5 embeddings achieved AUROC 0.829 and balanced accuracy 0.745 on the held-out Center B test split. Auxiliary supervision on the bridged 2019 cohort followed by end-to-end fine-tuning raised held-out performance to accuracy 0.795, balanced accuracy 0.753, and AUROC 0.842. A 35-run accuracy-oriented downstream search further increased selected-model test accuracy to 0.871 and identified a validation-AUROC leader with test accuracy 0.874 and AUROC 0.867, but these higher-accuracy operating points reduced recall to 36.1%-41.7%, again showing that accuracy alone is misleading under class imbalance. Temporal trajectory analysis revealed that 64.8% of patients maintained a stable phenotype across all windows, while 35.2% exhibited at least one phenotype transition. Stable phenotypes showed strongly stratified mortality (4.0%, 9.7%, 22.5%, 31.7%; range 27.7 percentage points). The three most frequent non-self transitions accounted for 48.1% of all transition events and predominantly moved toward lower-acuity phenotypes. Sensitivity analysis with reduced overlap (stride=12h) confirmed identical mortality ordering and range (28.0pp), with transition rates increasing under reduced overlap, indicating the primary analysis provides a conservative estimate. Cross-center temporal validation within the PhysioNet 2012 multi-center cohort (train on Center A, evaluate on Center B) confirmed identical mortality ordering, the same highest-risk phenotype, and a 25.8 percentage-point mortality range on the held-out center.

**Conclusions.** Self-supervised temporal representations enable descriptive phenotype trajectory analysis that reveals clinically meaningful within-stay dynamics beyond static clustering. The same learned backbone also supports both frozen and end-to-end supervised mortality modeling, and auxiliary transfer from a bridged second ICU dataset can improve held-out discrimination. However, accuracy-optimized operating points still trade away clinically important recall, so downstream evaluation must remain imbalance-aware. The data audit in this study shows that explicit treatment of missingness is not optional but foundational for ICU time-series modeling. The identified phenotypes show strong mortality stratification with ground-truth outcomes and cross-center temporal validation within the PhysioNet 2012 multi-center cohort. Both centers derive from the same source database; full external validation requires independently collected ICU cohorts. These findings support further investigation of temporal phenotyping for sepsis precision medicine, though causal treatment-response claims require additional study designs.

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
- A **cross-dataset bridge** aligning 40,331 PhysioNet/CinC 2019 sepsis stays to the S0 tensor interface, enabling auxiliary supervision with 18 shared physiologic channels
- A **supplementary downstream validation suite** spanning frozen probes, end-to-end attention-pooled fine-tuning, and systematic downstream hyperparameter search, clarifying why raw accuracy is misleading at 14% event prevalence
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

A configurable synthetic data generator with known ground-truth subtype labels enables quantitative evaluation of clustering recovery (ARI, NMI). In addition, we built a local bridge from the PhysioNet/CinC 2019 Sepsis Challenge stubs [9], yielding 40,331 ICU stays in the same on-disk format as `data/s0`. Eighteen of the 21 continuous PhysioNet 2012 channels map directly (`gcs`, `sodium`, and `pao2` are unavailable in the 2019 source), and PhysioNet 2012 preprocessing statistics are reused so the auxiliary source is numerically compatible with the pretrained encoder. A MIMIC-IV mock database (15 patients, 31 features) continues to serve as a schema-compatibility smoke test only.

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

#### 3.5.4 Principle Interpretation of the Representation Design

The Stage 2 design separates four complementary roles. First, concatenating values and observation masks preserves whether a measurement is truly observed or merely imputed, preventing the model from conflating absence with physiologic normality. Second, masked reconstruction forces the encoder to learn local temporal and cross-variable dependencies from real observations rather than memorizing complete trajectories. Third, the contrastive window objective aligns two partially overlapping 30-hour views from the same patient stay, encouraging patient-level consistency under local time shifts and sampling variation. Fourth, observation-density-weighted pooling gives more influence to hours carrying more measured signal without discarding sparse hours entirely. Together, these choices explain why the learned representation can be reused on arbitrary sub-windows in Stage 3 while remaining less sensitive to center mix and raw observation density than simpler baselines.

### 3.6 Descriptive Temporal Phenotype Trajectories (Stage 3)

Using the frozen S1.5 encoder, rolling-window embeddings are extracted for each patient: 5 windows of 24 hours with 6-hour stride, covering positions [0,24), [6,30), [12,36), [18,42), [24,48). K-Means (K=4) is fit on train-split window embeddings only (30,955 windows from 6,191 patients) and applied to all 59,930 windows. Per-patient phenotype trajectories are constructed as 5-element sequences of cluster assignments. Conceptually, each 24-hour embedding acts as a local clinical state summary, so Stage 3 converts patient stays into observable state sequences without imposing a parametric transition model.

Patients are classified as: **stable** (all 5 labels identical), **single-transition** (exactly 1 label change), or **multi-transition** (2+ changes). Empirical transition matrices, prevalence shifts across window positions, and mortality descriptives by trajectory category are computed. All temporal analyses are descriptive; no latent-state models or causal claims are involved.

### 3.7 Evaluation and Robustness

**Clustering metrics**: Silhouette score, mortality separation (max − min mortality across clusters). **Representation diagnostics**: mortality linear probe (AUROC), center linear probe (AUROC; lower is better), ICU LOS probe (R²), observation density–embedding norm correlation. **Supplementary downstream validation**: in addition to a frozen logistic-regression probe on S1.5 embeddings, we train an end-to-end attention-pooled classifier initialized from the S1.5 encoder and optionally warm-started with auxiliary supervision from the bridged PhysioNet 2019 cohort. We also run a 35-candidate downstream hyperparameter search across LogisticRegression, HistGradientBoosting, and HGB ensemble variants over multiple feature views. Because mortality prevalence is only ~14%, we report AUROC, balanced accuracy, precision, recall, F1, and plain accuracy, and compare accuracy against the majority-class baseline. **Multi-seed evaluation**: All KMeans results reported as mean ± std over 5 random seeds. **Stride sensitivity**: The temporal analysis is repeated with stride=12h (3 windows, 50% overlap) to assess robustness of transition findings to overlap reduction.

---

## 4. Results

### 4.1 Cohort Characterization and Missingness Pattern

After quality filtering, 11,986 patients were retained (Center A: 7,989; Center B: 3,997). Ground-truth in-hospital mortality was 14.2%, and the retained cohort covered nearly the full 48-hour analytic window (mean ICU LOS 47.3 hours, median 47.5 hours).

**Table 1. Cohort characterization and data completeness**

| Metric | Value |
|--------|-------|
| Patients | 11,986 |
| Age | 64.6 ± 17.2 years (median 67) |
| Sex | 43.9% female, 56.0% male, 0.1% unknown |
| Center split | 7,989 Center A / 3,997 Center B |
| In-hospital mortality | 14.2% overall; 14.0% in Center A, 14.6% in Center B |
| ICU LOS | 47.3h mean; 47.5h median |
| ICU type distribution | 14.7% / 21.1% / 35.8% / 28.4% for types 1-4 |
| Height / weight missing | 47.7% / 8.3% |
| Core hemodynamic missingness | 9.8%-11.9% for heart rate, SBP, DBP, MAP |
| Blood gas missingness | 87.3% mean across PaO2, FiO2, PaCO2, pH |
| Laboratory missingness | 93.8% mean; bilirubin 98.3%, lactate 95.9% |

The missingness pattern was strongly structured rather than random. Core hemodynamic channels were relatively dense, whereas respiratory, neurological, and laboratory variables were far sparser: respiratory rate was 75.9% missing, temperature 62.9%, GCS 67.9%, and most laboratory markers exceeded 92% missingness. This gradient explains why explicit observation masks are essential in this project: a model that sees only imputed values would incorrectly treat "not measured" as weak evidence of physiologic normality.

### 4.2 Static Phenotyping Baseline (Stage 1)

Static K=4 clustering on PCA-reduced statistical features identified phenotypes with mortality rates of 6.8%, 10.0%, 33.8%, and 36.0% (range 29.2 percentage points). Silhouette score was 0.061, consistent with the fuzzy but clinically stratified cluster boundaries expected in real ICU data with severe sparsity. This baseline established that clinically meaningful risk separation was present in the cohort before any self-supervised learning was introduced.

### 4.3 Representation Learning Comparison (Stage 2)

**Table 2. Representation comparison (K=4, mean ± std over 5 seeds)**

| Method | Silhouette | Mortality Range | Center L1 ↓ | Mort. Probe AUROC | Density |r| ↓ |
|--------|-----------|----------------|------------|-------------------|---------|
| PCA baseline (32d) | 0.061 ± 0.000 | 29.2% | 0.027 | 0.825 | 0.231 |
| S1: masked only (128d) | 0.087 ± 0.000 | 17.6% | 0.024 | 0.825 | 0.247 |
| S1.5: masked + contrastive (128d) | 0.080 ± 0.001 | 24.6% | **0.016** | **0.830** | **0.148** |
| S1.6: λ=0.2 ablation (128d) | 0.079 ± 0.000 | 25.1% | 0.021 | 0.825 | 0.148 |

Center probe AUROC was 0.50-0.52 for all methods (evaluated on a random stratified split mixing both centers), confirming that no representation encoded center identity in a directly separable way. Compared with S1, adding the contrastive window objective in S1.5 slightly reduced geometric compactness (silhouette 0.087 to 0.080) but restored 7.0 percentage points of mortality separation, reduced center prevalence divergence from 0.024 to 0.016, and lowered missingness sensitivity from 0.247 to 0.148. Relative to PCA, S1.5 traded some static mortality range for substantially better center stability and a roughly 36% lower density-norm correlation. This pattern supports the intended interpretation: contrastive regularization makes embeddings less tied to site mix and raw measurement density while preserving clinically relevant outcome information.

S1.5 was therefore selected for temporal analysis not because it was uniformly best on every metric, but because it offered the best balance between clinical signal, robustness to structured sparsity, and direct suitability for arbitrary rolling windows. PCA remained a useful static baseline, but it could not naturally process variable-length sub-sequences without re-engineering features for every window.

#### 4.3.1 Supplementary Downstream Mortality Classification

**Table 2b. Supplementary downstream mortality classification variants**

| Model / Operating Point | Test Accuracy | Test Balanced Accuracy | Test Precision | Test Recall | Test F1 | Test AUROC |
|-------------------------|--------------:|-----------------------:|---------------:|------------:|--------:|-----------:|
| Frozen S1.5 probe, balanced threshold (`thr=0.55`) | 0.784 | 0.745 | 0.372 | 0.691 | 0.484 | 0.829 |
| Frozen S1.5 probe, accuracy threshold (`thr=0.85`) | 0.865 | 0.623 | 0.582 | 0.280 | 0.378 | 0.829 |
| Feature-fusion HGB | 0.791 | 0.780 | 0.391 | 0.764 | 0.517 | 0.862 |
| End-to-end fine-tune + Sepsis2019 auxiliary supervision | 0.795 | 0.753 | 0.388 | 0.692 | 0.498 | 0.842 |
| Accuracy-search ensemble (`val acc` leader) | 0.871 | 0.660 | 0.601 | 0.361 | 0.451 | 0.863 |
| Majority-class baseline | 0.854 | 0.500 | - | 0.000 | - | - |

A logistic-regression classifier trained on frozen S1.5 embeddings established that the learned representation already carried usable prognostic information (AUROC 0.829). Adding richer tabular feature views via HistGradientBoosting improved the balance-aware operating point to balanced accuracy 0.780 and AUROC 0.862. Initializing an attention-pooled end-to-end classifier from the same S1.5 encoder, then warming it with auxiliary supervision on 40,331 bridged PhysioNet 2019 sepsis stays, raised held-out performance to accuracy 0.795, balanced accuracy 0.753, recall 69.2%, and AUROC 0.842. Finally, a 35-run accuracy-oriented hyperparameter search over HGB, HGB ensemble, and logistic variants pushed selected-model test accuracy to 0.871 with AUROC 0.863, while the validation-AUROC leader under the same search reached 0.874 test accuracy and 0.867 AUROC. However, both higher-accuracy search winners operated at much lower recall (36.1%-41.7%) than the balance-oriented models. Taken together, these results show that extra data and end-to-end supervision improve the downstream accuracy frontier, but that class imbalance still makes plain accuracy an incomplete summary.

### 4.4 Descriptive Temporal Phenotype Trajectories (Stage 3)

#### 4.4.1 Rolling-Window Extraction

Using the frozen S1.5 encoder, 59,930 window embeddings (11,986 patients × 5 windows) were extracted. Per-window observation density decreased from 27.9% in the earliest window to 25.4% in the latest window, a 2.5 percentage-point absolute drop that reflects declining measurement frequency during ICU stays. Despite this decline, per-window silhouette remained stable and even improved modestly from 0.072 to 0.082, suggesting that the learned representation remained usable as the observation stream became sparser.

#### 4.4.2 Temporal Phenotype Stability and Transitions

Of 11,986 patients, 7,764 (64.8%) maintained a stable phenotype across all five windows, 3,509 (29.3%) exhibited exactly one transition, and 713 (5.9%) exhibited multiple transitions. The four fully stable trajectories, [0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2], and [3,3,3,3,3], were also the four most common patient-level patterns, indicating that temporal structure was dominated by persistent low-dimensional states rather than noisy label oscillation.

**Table 3. Stable phenotype mortality stratification**

| Stable Phenotype | N | In-Hospital Mortality |
|-----------------|---|----------------------|
| Phenotype 0 | 2,216 | 4.0% |
| Phenotype 3 | 1,891 | 9.7% |
| Phenotype 1 | 2,547 | 22.5% |
| Phenotype 2 | 1,110 | 31.7% |

The 27.7 percentage-point mortality range across temporally stable phenotypes exceeded the static S1.5 clustering result (24.6%), suggesting that restricting attention to temporally consistent patients yields cleaner phenotype strata. Phenotype 2 was the smallest stable subgroup (9.3% of the cohort) but the highest-risk one, whereas Phenotype 0 combined the largest low-risk population with the lowest mortality.

#### 4.4.3 Transition Pathways and Mortality

Across 47,944 adjacent window pairs, 4,987 events (10.4%) were non-self transitions. The transition entropy ratio was 0.637, indicating that transitions were neither random nor trivially concentrated in a single pathway. The most frequent non-self transitions were Phenotype 1->0 (956 events), Phenotype 1->3 (728), and Phenotype 3->0 (713); together they accounted for 48.1% of all non-self transitions and primarily represented movement toward lower-risk states.

Patients with exactly one transition had lower mortality (11.4%) than stable patients (15.4%) or multi-transition patients (15.2%). This descriptive association is clinically interesting because it suggests that a subset of transitions may correspond to early stabilization rather than deterioration. At the same time, the similar mortality of stable and multi-transition patients indicates that transition count alone is not sufficient to characterize risk; transition direction matters.

#### 4.4.4 Temporal Prevalence Shift

Phenotype 0 (lowest mortality) increased in prevalence from 24.6% at window [0,24h) to 33.1% at window [24,48h), while Phenotype 1 decreased from 35.7% to 28.1%. This population-level shift is consistent with partial clinical stabilization during the first 48 ICU hours. Because observation density also decreased over time, the prevalence shift should be read as a joint effect of evolving physiology, survivor filtering, and measurement practice rather than as direct evidence of treatment response.

### 4.5 Stride Sensitivity Analysis

To assess the influence of window overlap on temporal findings, we repeated the analysis with stride=12h (3 windows, 50% overlap). Cluster centroids showed near-perfect alignment with the primary analysis (cosine distance < 0.0001 after Hungarian matching), indicating that the recovered phenotype geometry was effectively unchanged.

**Table 4. Stride sensitivity comparison**

| Metric | Stride=6h (primary) | Stride=12h (sensitivity) |
|--------|---------------------|--------------------------|
| Stable patient fraction | 64.8% | 65.6% |
| Non-self transition proportion | 10.4% | 19.1% |
| Mortality ordering | [P0, P3, P1, P2] | [P0, P3, P1, P2] |
| Highest-risk phenotype | P2 (31.7%) | P2 (31.9%) |
| Stable phenotype mortality range | 27.7pp | 28.0pp |

The mortality ordering of stable phenotypes was identical under both strides, the highest-risk phenotype remained Phenotype 2, and the mortality range was preserved (28.0pp vs 27.7pp). The proportion of non-self transitions increased from 10.4% to 19.1% under reduced overlap, indicating that the primary stride=6h analysis provides a conservative estimate of temporal phenotype change rather than an overlap-inflated one.

### 4.6 Cross-Center Temporal Validation

To assess whether the temporal phenotype trajectories generalized beyond the training center, we performed out-of-center validation: the KMeans model was trained on Center A rolling-window embeddings and evaluated on Center B (a separate hospital within the PhysioNet 2012 cohort). The S1.5 encoder was also trained exclusively on Center A data.

**Table 5. Cross-center temporal validation (Center A vs Center B)**

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
| Mean prevalence L1 | - | 0.022 |

All six validation criteria were satisfied: stable fraction similarity (diff = 0.6pp), transition proportion similarity (diff = 0.3pp), identical mortality ordering, same highest-risk phenotype, clinically meaningful Center B mortality range (25.8pp), and low per-window prevalence divergence (mean L1 = 0.0218; range 0.0155-0.0265 across windows). Per-window silhouette scores were within 0.001 across centers at all five window positions.

The detailed pattern was also consistent. Low-risk Phenotypes 0 and 3 showed nearly identical mortality in both centers, whereas higher-risk Phenotypes 1 and 2 shifted modestly in absolute rate but preserved the same ordering. This is the pattern expected from a useful temporal phenotyping system: absolute calibration can vary by site, but the relative state structure remains stable.

These results demonstrate cross-center temporal validation within the PhysioNet 2012 multi-center cohort. Both centers derive from the same source database; full external validation still requires independently collected ICU cohorts such as eICU-CRD or MIMIC-IV.

### 4.7 Simulated Data Validation

The synthetic data generator was used for pipeline validation. K-Means on PCA-reduced simulated features achieved ARI = 0.245 and NMI = 0.495 against known ground-truth labels. This result is modest rather than spectacular, but it is useful as an honesty check: the pipeline can recover non-random subtype structure in controlled data, yet clustering remains a genuinely difficult problem even under simpler conditions. *Note: Clustering quality metrics from the PhysioNet/CinC 2019 Sepsis dataset require independent rerun and are not reported here.*

---

## 5. Discussion

### 5.1 Three-Level Phenotyping Framework

This work demonstrates a progression from static clustering (Stage 1) through self-supervised representation learning (Stage 2) to descriptive temporal phenotype trajectory analysis (Stage 3). Each stage addresses a limitation of the previous one. Stage 1 shows that clinically meaningful heterogeneity exists in the cohort. Stage 2 replaces hand-crafted summaries with a learned representation that can directly ingest sub-sequences while accounting for structured missingness. Stage 3 then turns those embeddings into temporal phenotype sequences that make within-stay movement visible.

The temporal trajectory analysis shows that approximately one-third of ICU sepsis patients undergo at least one phenotype transition within the first 48 hours. The importance of those transitions is supported by three convergent findings: stable phenotypes show strong mortality stratification, the dominant transition directions move toward lower-risk states, and single-transition patients have lower mortality than the cohort of fully stable patients. Taken together, these results argue that early within-stay dynamics contain clinically meaningful information that static clustering alone cannot expose.

### 5.2 Principle Interpretation of the Framework

The central methodological idea is not simply "use a Transformer," but to align the representation design with the structure of ICU data. In this project, missingness is informative, temporally local patterns matter, and patient state should remain partially stable across nearby windows. The framework therefore combines explicit mask channels, reconstruction of withheld observed values, contrastive alignment of overlapping windows from the same patient, and window-level clustering of frozen embeddings.

These components play different roles. Mask channels preserve the distinction between measured and imputed values. Masked reconstruction encourages the model to learn short-range temporal and cross-variable regularities from observed clinical signals. Contrastive window alignment encourages patient-level consistency across local time shifts and sampling variation, which is especially important when later applying the encoder to rolling 24-hour windows. Density-weighted pooling reduces the influence of almost-empty hours without dropping them entirely. In combination, the model is encouraged to treat sparsity as context rather than noise while avoiding a trivial dependence on measurement count alone.

### 5.3 Data Interpretation and Clinical Meaning

The data analysis in Section 4 clarifies why the learned temporal phenotypes behave the way they do. The cohort contains dense hemodynamic streams but extremely sparse laboratory variables, so the recovered temporal states are likely driven most strongly by evolving vital-sign patterns, with laboratory information acting as occasional but clinically important corrections. This helps explain why prevalence shifts are smooth rather than abrupt and why robustness to observation density matters so much.

The transition structure is also clinically interpretable at a descriptive level. Nearly half of all non-self transitions come from three pathways, 1->0, 1->3, and 3->0, all of which move toward lower-risk states. Meanwhile, the highest-risk stable phenotype remains Phenotype 2 in both centers. A reasonable interpretation is that the framework is separating persistent high-risk patients from a larger population that gradually stabilizes during early ICU care. However, this interpretation remains phenomenological because the temporal phenotypes were not post-hoc labeled with specific biomarker signatures or treatment exposure profiles.

### 5.4 Representation Learning Trade-offs

The self-supervised encoder (S1.5) offered improved center stability and reduced missingness sensitivity compared with PCA, though PCA retained stronger static mortality separation. This trade-off is characteristic of learned representations: they often sacrifice some training-distribution sharpness in exchange for invariances that matter when the representation is reused under shifted conditions. In this project, that reuse condition was rolling-window temporal analysis.

S1 alone achieved the highest silhouette, but it also produced the weakest clinical separation among the learned models and the highest coupling to observation density. Adding the contrastive term partially reversed that behavior, suggesting that contrastive regularization did not merely smooth the space; it reshaped the space toward patient-level temporal consistency. That is the main reason S1.5, rather than PCA or S1, was the right representation for Stage 3.

The newer downstream experiments sharpen that interpretation. On the held-out Center B split, the frozen S1.5 space already supported AUROC 0.829; reusing the same encoder in an attention-pooled end-to-end model and warming it with the bridged 2019 cohort increased held-out accuracy to 0.795 and AUROC to 0.842 while keeping recall near 69%. A separate 35-run accuracy-oriented search pushed selected-model accuracy above 0.87, but only by operating at substantially lower recall. The more meaningful signal is therefore not that one operating point maximizes a single scalar metric, but that the same representation supports temporal trajectory analysis, auxiliary transfer, and out-of-center mortality discrimination when the trade-off between accuracy and sensitivity is reported explicitly.

### 5.5 Limitations

1. **Descriptive, not model-based**: The temporal phenotype trajectories arise from per-window clustering rather than latent-state inference such as hidden Markov models. This limits probabilistic transition modeling and uncertainty quantification.
2. **Overlap sensitivity**: Window overlap (75% at stride=6h) smooths embedding changes and likely underestimates transition rates. Sensitivity analysis confirmed that the main clinical ordering is robust, but exact transition frequency depends on the window design.
3. **Missing treatment data**: PhysioNet 2012 lacks true treatment records. Proxy indicators such as MAP < 65 or FiO2 > 0.21 are physiological thresholds, not interventions. Treatment-aware phenotyping requires datasets with documented vasopressor, fluid, ventilation, and antibiotic timing.
4. **Observational associations**: Patients who transition between phenotypes may differ systematically from stable patients in ways not captured here. The lower mortality observed in single-transition patients is a descriptive association, not a causal effect of transition.
5. **Single-dataset scope**: Cross-center temporal validation within the PhysioNet 2012 cohort is strong internal evidence, but both centers derive from the same source database. External validation on independently collected ICU cohorts is still required.
6. **Sparse phenotype semantics**: The current paper can rank temporal phenotypes by risk, but it does not yet assign them mechanistic labels based on treatments, organ-failure pathways, or multimodal evidence such as notes or waveforms.
7. **Data sparsity ceiling**: With an overall missing rate of 73.3% and key biomarkers such as bilirubin and lactate observed only rarely, there is an upper bound on how much physiologic detail any model can recover from this dataset alone.

### 5.6 Future Directions

1. **Treatment-aware temporal phenotyping**: Extending the same pipeline to MIMIC-IV or eICU with true intervention timestamps would allow direct analysis of phenotype-treatment interactions and heterogeneous treatment effects rather than descriptive risk stratification alone.
2. **State-space modeling and uncertainty**: Replacing per-window clustering with hidden Markov models, switching state-space models, or neural latent-state approaches would make transition probabilities and confidence intervals explicit.
3. **External database validation**: Repeating the full pipeline on independently collected ICU cohorts is the most important next scientific step for establishing generalizability beyond the PhysioNet 2012 ecosystem.
4. **Earlier bedside deployment**: The new end-to-end fine-tuning results show that the learned space is operationally usable both as a frozen representation and as a supervised backbone. A practical next step would infer phenotype membership or mortality risk from 6-hour or 12-hour windows and update predictions online as new measurements arrive.
5. **Richer phenotype interpretation**: Post-hoc attribution analysis, multimodal augmentation, and linkage to organ-support variables could turn the current risk-ranked phenotypes into clinically named subgroups with clearer bedside meaning.

---

## 6. Conclusion

This paper extends the project from a concise manuscript into a fuller research narrative by tying together cohort characterization, model design principles, quantitative comparison, temporal trajectory analysis, and future outlook. Applied to 11,986 multi-center ICU patients with verified in-hospital mortality labels, the framework identifies four temporally meaningful phenotypes with strong mortality stratification (4.0%-31.7% among stable patients), substantial within-stay movement (35.2% of patients transition at least once), and consistent cross-center behavior within the PhysioNet 2012 cohort.

The main scientific lesson is that explicit handling of structured missingness and temporal locality is essential for ICU time-series phenotyping. Static feature clustering can reveal coarse heterogeneity, but mask-aware self-supervised representation learning is what makes rolling-window analysis feasible. The new auxiliary-data bridge and end-to-end supervised results further show that the same S1.5 backbone can transfer across related ICU sources and improve held-out mortality prediction, though accuracy-optimized operating points still trade away recall under severe class imbalance. The present results are descriptive rather than causal, yet they show that temporal phenotype trajectories carry reproducible clinical information beyond static subtyping. That makes them a plausible foundation for future treatment-aware, externally validated, and eventually bedside-deployable sepsis phenotyping systems.

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
