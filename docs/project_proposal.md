# Project Proposal

# ICU Sepsis Dynamic Subtype Discovery via Self-Supervised Patient Trajectory Representation with Cross-Center Generalization Validation

---

## 1. Project Background

Sepsis is one of the most prevalent critical illnesses in the Intensive Care Unit (ICU), with approximately 48.9 million cases globally per year and a mortality rate ranging from 20% to 40%. Sepsis is not a single disease entity but rather a systemic inflammatory response syndrome triggered by infection, with pathophysiological processes involving immune dysregulation, coagulation disorders, and multi-organ dysfunction. Due to the high heterogeneity of sepsis, significant differences exist among patients in terms of clinical presentation, disease progression, and treatment response. Traditional "one-size-fits-all" treatment strategies have been shown to be of limited effectiveness, and the precise identification of sepsis subtypes for individualized treatment has become an important direction in critical care medicine research.

In recent years, with the widespread adoption of Electronic Health Record (EHR) systems in ICUs, large volumes of high-frequency, multi-dimensional temporal patient data have been accumulated. The MIMIC-IV v3.1 database contains complete clinical records of over 65,000 ICU patients and 200,000 emergency department patients; the eICU Collaborative Research Database covers more than 200,000 ICU admissions across over 200 hospitals in the United States. These publicly available datasets provide a valuable data foundation for data-driven sepsis subtype discovery research.

However, existing research still has the following limitations: First, most studies continue to adopt the "single time-point + static features" modeling paradigm, failing to fully exploit the temporal dynamic characteristics of ICU data; Second, the majority of work is confined to single-center validation, lacking cross-dataset generalizability testing; Third, clustering methods based on traditional statistical features have difficulty capturing complex temporal patterns, while the application of emerging self-supervised representation learning methods in the EHR domain remains in its infancy.

## 2. Research Significance

The research significance of this project is reflected at three levels:

**Theoretical Significance.** This project introduces self-supervised patient trajectory representation learning into sepsis dynamic subtype discovery, exploring the paradigm shift from static feature clustering to dynamic trajectory modeling. By incorporating outcome-constrained clustering methods, it addresses the problem of "statistically separable but clinically meaningless" pseudo-subtypes, providing a novel methodological framework for critical illness heterogeneity modeling.

**Practical Significance.** The discovered sepsis subtypes can provide decision support for precision clinical treatment. For example, subtypes primarily characterized by respiratory failure may be more suitable for early mechanical ventilation intervention, while high-risk subtypes characterized by multi-organ failure require more aggressive comprehensive treatment strategies.

**Engineering Significance.** The standardized temporal data processing pipeline and modular analysis framework constructed in this project possess good generalizability and reproducibility, and can be extended to subtype discovery research for other ICU critical illnesses (such as Acute Respiratory Distress Syndrome, Acute Kidney Injury, etc.).

## 3. Current State of Research

### 3.1 Sepsis Subtype Discovery

Sepsis phenotype classification research has established a relatively rich body of literature. Seymour et al. (2019) used K-Means clustering on MIMIC-III and eICU data to classify sepsis patients into four phenotypes: alpha, beta, gamma, and delta, where the delta type is characterized by multi-organ failure and high mortality. A 2025 review in Frontiers in Immunology explicitly identified sepsis subtype classification as a core direction for advancing precision treatment. In the same year, a study in EClinicalMedicine began employing deep temporal graph clustering methods to quantify multi-organ interaction trajectories within a 48-hour window. The GEMS method published in Nature Communications demonstrated that directly incorporating outcome consistency into the subtype discovery objective is an effective technical approach.

### 3.2 EHR Temporal Modeling and Foundation Models

Structured EHR data modeling is evolving toward Foundation Models. EHRMamba proposed a state-space model-based long-sequence EHR encoder, emphasizing linear complexity and multi-task fine-tuning capabilities. A 2025 study on ICU temporal foundation models demonstrated that self-supervised pretraining has significant advantages for mortality prediction in low-sample scenarios. Additionally, MEDS (Medical Event Data Standard), as a lightweight event standard for EHR machine learning, provides a foundation for cross-dataset, cross-model interoperability and reproducibility.

### 3.3 Evaluation Standardization

A 2023 study by YAIB (Yet Another ICU Benchmark) noted that in ICU machine learning, dataset selection, cohort definition, and preprocessing often have an impact on results no less significant than the model architecture itself. This underscores the importance of standardization and transparency in cohort definition and preprocessing in experimental design. The MIMIC-Sepsis benchmark dataset (2025) provides a standardized sepsis cohort comprising 35,239 ICU patients, establishing a foundation for reproducible evaluation.

## 4. Research Objectives

The research objectives of this project encompass three levels:

**Objective 1:** Construct a standardized temporal event stream data processing pipeline for ICU sepsis patients, uniformly handling vital signs, laboratory tests, therapeutic interventions, and outcome information, with support for both MIMIC-IV and eICU data sources.

**Objective 2:** Design and implement patient trajectory representation methods based on statistical feature extraction and self-supervised learning, encoding temporal data within a 48-hour time window around sepsis onset to obtain discriminative patient-level feature representations.

**Objective 3:** Perform dynamic subtype discovery based on the learned trajectory representations, and validate the stability, transferability, and outcome stratification capability (mortality, shock, and length of stay) of the subtypes on cross-center data.

## 5. Research Content

### 5.1 Data Engineering and Standardization

An analysis window is constructed around "48 hours before and after sepsis diagnosis." The extracted variables include:

- **Vital Signs** (7 items): Heart rate, systolic blood pressure, diastolic blood pressure, mean arterial pressure, respiratory rate, oxygen saturation, body temperature
- **Laboratory Indicators** (7 items): Lactate, creatinine, bilirubin, platelet count, white blood cell count, oxygenation index, INR
- **Therapeutic Interventions** (3 items): Vasopressor use, mechanical ventilation, renal replacement therapy
- **Outcome Variables** (3 items): 28-day mortality, ICU length of stay, shock occurrence

The preprocessing pipeline includes: temporal resampling (1-hour intervals), outlier clipping (4 sigma), missing value imputation (forward fill + median fallback), and data standardization (z-score).

### 5.2 Patient Trajectory Representation Learning

A "2+1" structure is adopted:

- **Baseline Method 1**: Statistical feature extraction (mean, standard deviation, extreme values, trend slope, last value) + PCA dimensionality reduction
- **Baseline Method 2**: XGBoost supervised feature importance ranking (if auxiliary prediction tasks are available)
- **Primary Method**: Self-supervised patient trajectory encoder with Masked Event Modeling as the pretraining task and Transformer as the encoder architecture

### 5.3 Dynamic Subtype Discovery

An outcome-constrained clustering method is employed, ensuring that clustering results not only group patients by feature similarity but also account for outcome consistency within each subtype. The specific implementation includes:

- Optimal cluster number search (K=2~8, based on comprehensive evaluation of silhouette coefficient, Calinski-Harabasz index, and Davies-Bouldin index)
- Multi-method comparison (K-Means, GMM, hierarchical clustering)
- Subtype clinical profiling and interpretability analysis

### 5.4 Cross-Center Generalization Validation

The model is trained and subtypes are discovered on MIMIC-IV, with validation performed on eICU. The validation dimensions include:

- Subtype reproducibility: Whether the same method produces similar subgroups on different datasets
- Outcome stratification stability: Whether the ranking of mortality and ICU length of stay across subtypes remains consistent
- Predictive transfer performance: AUROC, AUPRC, C-index, and other metrics

## 6. Technical Roadmap

```
Data Acquisition → Cohort Construction → Time Window Segmentation (48h)
    ↓
Outlier Processing → Missing Value Imputation → Standardization → 3D Temporal Tensor
    ↓
Statistical Feature Extraction / Self-Supervised Encoder Pretraining
    ↓
PCA/UMAP Dimensionality Reduction → K-Means/GMM Clustering → Optimal K Search
    ↓
Evaluation: Silhouette Coefficient + Survival Stratification + Subtype Profiling
    ↓
Cross-Center Validation (eICU) → Visualization Report
```

Core Technology Stack: Python 3.9+, pandas, numpy, scipy, scikit-learn, umap-learn, matplotlib, lifelines, PyTorch (V2).

## 7. System Design

### 7.1 System Architecture

The system adopts a modular design, with data transfer between modules facilitated through standardized data interfaces (3D numpy tensors + pandas DataFrames). A configuration file (YAML) centrally manages all hyperparameters and paths.

### 7.2 Module Partitioning

| Module | File | Input | Output |
|--------|------|-------|--------|
| Data Loading | data_loader.py | Configuration | 3D Tensor + Patient Information Table |
| Preprocessing | preprocess.py | Raw Tensor | Clean Tensor + Standardization Parameters |
| Feature Engineering | feature_engineering.py | Clean Tensor | Feature Matrix |
| Representation Learning | representation_model.py | Feature Matrix / Tensor | Representation Vectors |
| Clustering Analysis | clustering.py | Representation Vectors | Cluster Labels |
| Evaluation | evaluation.py | Labels + Patient Information | Evaluation Report |
| Visualization | visualization.py | Data from All Stages | Graphic Files |

### 7.3 Data Flow Design

All intermediate data can be cached to disk (processed directory), supporting checkpoint-based resumption. Standardization parameters can be exported for alignment of external validation data.

## 8. Implementation Plan and Schedule

| Phase | Timeline | Work Content | Deliverables |
|-------|----------|-------------|--------------|
| Phase 1 | Weeks 1-2 | Project framework setup, simulated data generation, MVP pipeline completion | Runnable baseline version |
| Phase 2 | Weeks 3-4 | Statistical feature optimization, multi-method clustering comparison, evaluation system refinement | Complete evaluation report |
| Phase 3 | Weeks 5-6 | Self-supervised representation model implementation and training | Representation learning module |
| Phase 4 | Weeks 7-8 | Cross-center validation, outcome-constrained clustering, interpretability analysis | Validation report |
| Phase 5 | Weeks 9-10 | Documentation writing, system testing, defense preparation | Complete documentation + demonstration |

## 9. Expected Outcomes

1. **Software System**: A complete ICU sepsis dynamic subtype discovery analysis system, encompassing the full pipeline of data processing, representation learning, clustering analysis, evaluation, and visualization
2. **Analysis Report**: Clinical profiles of sepsis subtypes, survival stratification results, and cross-center validation conclusions
3. **Project Documentation**: Project proposal, mid-term progress evaluation, software performance evaluation, and software user manual
4. **Visualization Outputs**: Clustering scatter plots, K-selection curves, heatmaps, survival curves, trajectory comparison plots, etc.

## 10. Risks and Mitigation Strategies

| Risk | Level | Mitigation Strategy |
|------|-------|-------------------|
| Delay in real data acquisition | Medium | Prioritize using clinically realistic simulated data to complete the full pipeline; interface design supports seamless switching at a later stage |
| Unstable clustering results | Medium | Multi-method comparison, Bootstrap resampling stability testing |
| Difficulty in self-supervised model training | Low | Retain the statistical feature baseline as a fallback to ensure project deliverability |
| Poor cross-center generalization performance | Medium | Analyze causes of data distribution shift, explore domain adaptation strategies |
| Insufficient computational resources | Low | The MVP phase does not require GPU; the V2 Transformer model can be trained at small scale on CPU |
