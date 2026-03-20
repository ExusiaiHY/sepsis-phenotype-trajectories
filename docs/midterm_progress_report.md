# Midterm Progress Evaluation

# Discovery of Dynamic ICU Sepsis Subtypes via Self-Supervised Patient Trajectory Representation with Cross-Center Generalization Validation

---

## 1. Overview of Progress to Date

As of the midterm checkpoint, the project has completed the overall framework construction and MVP (Minimum Viable Product) version development. The core data processing and analysis pipeline has been fully executed end-to-end. The completion status of each module is as follows:

| Module | Planned Content | Status | Completion |
|--------|----------------|--------|------------|
| Project Architecture | Directory structure, configuration management, utility functions | Completed | 100% |
| Data Loading | Simulated data generation, unified data interface | Completed | 100% |
| Preprocessing | Missing value imputation, outlier handling, normalization | Completed | 100% |
| Feature Engineering | Statistical feature extraction, clinical derived indicators | Completed | 100% |
| Clustering Analysis | K-Means/GMM, optimal K search | Completed | 100% |
| Evaluation | Internal metrics, external metrics, survival analysis | Completed | 100% |
| Visualization | Automated generation of 7 types of plots | Completed | 100% |
| Self-Supervised Representation | Transformer encoder | Interface reserved | 20% |
| Cross-Center Validation | eICU external validation | Interface reserved | 20% |
| Real Data Integration | MIMIC-IV / eICU | Interface reserved | 10% |

**Overall Progress Assessment: The project is on schedule, with the MVP version completed ahead of plan.**

## 2. Description of Completed Modules

### 2.1 Data Loading Module (data_loader.py)

A clinically plausible simulated data generator has been implemented with the following key features:

- Baseline parameters and deterioration trends are configured according to four sepsis phenotypes (alpha/beta/gamma/delta) documented in the literature
- Vital signs are generated using an AR(1) autoregressive model, preserving temporal autocorrelation
- Laboratory test values simulate realistic low-frequency sampling patterns (every 4-8 hours), with a missing rate of approximately 82%
- Treatment interventions follow an event-stream pattern (continuous once initiated at a given time point)
- Subtype distribution is imbalanced (35%:25%:25%:15%), more closely approximating real-world distributions

### 2.2 Preprocessing Module (preprocess.py)

A complete preprocessing pipeline has been implemented:

- Outlier handling: Clipping based on a 4-sigma threshold, retaining extreme but clinically plausible ICU data
- Missing value imputation: Forward fill followed by median backfill, consistent with clinical assumptions for ICU time-series data
- Normalization: Z-score standardization with exportable parameters for alignment with external validation data
- Missing pattern analysis functionality, outputting missing rate statistics for each variable

### 2.3 Feature Engineering Module (feature_engineering.py)

A total of 311 statistical features are extracted, including:

- 6 types of summary statistics (mean, standard deviation, minimum, maximum, trend slope, last value)
- 3 temporal sub-windows (12h, 24h, 48h)
- Clinical derived indicators: lactate clearance rate, shock index, renal-hepatic burden index, oxygenation deterioration rate

### 2.4 Clustering and Evaluation Modules

- Support for 4 clustering methods (K-Means, GMM, hierarchical clustering, spectral clustering)
- Automated search for the optimal K (based on silhouette coefficient / Calinski-Harabasz index / Davies-Bouldin index)
- Comprehensive evaluation framework: internal metrics + external metrics (ARI, NMI) + survival stratification (Kaplan-Meier + log-rank test)

## 3. Interim Experimental Results

The following MVP experimental results are based on 500 simulated patients:

### 3.1 Clustering Quality Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Coefficient | 0.4355 | Moderately high, indicating reasonable cluster separation |
| CH Index | 164.65 | Relatively high, supporting the existence of cluster structure |
| DB Index | 0.945 | Close to 1, room for improvement |
| ARI | 0.2448 | Some correspondence with true subtypes but not precise |
| NMI | 0.4999 | Moderate, clustering captures approximately 50% of true information |

### 3.2 Survival Stratification Results

| Subtype | Number of Patients | 28-Day Mortality | Mean ICU Duration |
|---------|--------------------|------------------|-------------------|
| Subtype 0 (High Risk) | 83 | 44.6% | 71.6h |
| Subtype 1 (Low Risk) | 417 | 12.9% | 82.5h |

The mortality difference between the two subtypes is significant (44.6% vs. 12.9%), indicating that the clustering results carry clinical relevance.

### 3.3 System Runtime Performance

| Step | Time for 300 Patients | Time for 500 Patients |
|------|-----------------------|-----------------------|
| Data Generation | 0.15s | 0.25s |
| Preprocessing | 0.13s | 0.20s |
| Feature Extraction | 0.01s | 0.02s |
| UMAP Dimensionality Reduction | 2.57s | 3.10s |
| Clustering Search | 0.40s | 0.60s |
| Evaluation | 0.09s | 0.10s |
| **Full Pipeline** | **3.36s** | **5.24s** |

## 4. Current Issues

### Issue 1: Automatically Selected Optimal K Tends to Be Too Small

The current system selects the optimal K based on the silhouette coefficient, which tends to favor K=2. This occurs because the alpha type (mild) and delta type (severe) exhibit the greatest divergence in the simulated data, while the beta and gamma types overlap with neighboring types in the statistical feature space.

### Issue 2: Statistical Features Have Limited Discriminative Power for Fine-Grained Subtypes

Although the 311 statistical features can distinguish between "high-risk" and "low-risk" groups, they struggle to further differentiate fine-grained distinctions such as "respiratory failure-dominant" versus "chronic comorbidity-dominant" subtypes. The ARI of only 0.2448 indicates that the combination of statistical features and K-Means has limited correspondence with the true four subtypes.

### Issue 3: Real Clinical Data Has Not Yet Been Integrated

The data loading interfaces for MIMIC-IV and eICU are currently in a placeholder state. Cohort construction, variable mapping, and quality verification for real clinical data have not yet been undertaken.

## 5. Root Cause Analysis

**Cause of Issue 1:** Statistical features (mean, standard deviation, trends, etc.) are inherently a lossy compression of time-series information. When two subtypes exhibit high overlap in these summary statistics, clustering algorithms cannot distinguish between them. This precisely demonstrates the necessity of self-supervised representation learning -- deep temporal encoders can capture trajectory morphological differences that summary statistics fail to express.

**Cause of Issue 2:** K-Means assumes spherical cluster distributions, whereas real sepsis subtypes may exhibit non-spherical structures in high-dimensional space. GMM, as an alternative, permits ellipsoidal distributions, but covariance estimation in high-dimensional space is equally challenging.

**Cause of Issue 3:** This is a planned delay. The project strategy follows a "run the pipeline end-to-end first, then integrate real data" approach, with real data integration scheduled for the third and fourth stages.

## 6. Plan for the Next Stage

### 6.1 Near-Term Priorities (Stage 3)

1. **Implement the self-supervised representation learning module**: Based on a Transformer encoder, with Masked Event Modeling as the pretraining task
2. **Introduce outcome-constrained clustering**: Referencing the GEMS methodology, incorporating a survival consistency term into the clustering objective
3. **Conduct systematic comparative experiments with GMM and hierarchical clustering**

### 6.2 Mid-Term Goals (Stage 4)

1. **Integrate MIMIC-IV data** (contingent upon completion of PhysioNet credentialing)
2. **Implement the cross-center validation pipeline**: Train on MIMIC-IV and validate on eICU
3. **Add SHAP-based interpretability analysis**

### 6.3 Final Phase (Stage 5)

1. Finalize all documentation
2. System testing and performance optimization
3. Preparation of defense materials

## 7. Assessment of Whether Progress Aligns with Expectations

**Overall Assessment: Progress aligns with expectations, with some tasks completed ahead of schedule.**

The MVP version has been fully executed end-to-end, encompassing the complete pipeline from data processing to feature extraction to clustering to evaluation to visualization. All 7 types of visualizations and the structured evaluation report are generated automatically. System runtime performance is satisfactory (5.24 seconds for the full pipeline with 500 patients).

The core project risk -- "failure to run the complete pipeline end-to-end" -- has been eliminated prior to the midterm checkpoint. Subsequent work focuses on model enhancement (self-supervised representation) and data enhancement (real MIMIC-IV data), which constitute incremental improvements rather than foundational work.

## 8. Items Requiring Adjustment

1. **Optimal K selection strategy**: It is recommended to transition from relying solely on the silhouette coefficient to a multi-metric ensemble voting mechanism to avoid the tendency toward excessively small K values
2. **Add bootstrap stability testing**: Perform resampling-based stability analysis on clustering results as additional evidence of subtype reliability
3. **V2 representation model scale control**: Given the potential absence of GPU resources, the Transformer model should be constrained to no more than 2 layers and 4 attention heads to ensure trainability on CPU
