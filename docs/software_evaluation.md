# Software Evaluation

# ICU Sepsis Dynamic Subtype Discovery and Analysis System -- Evaluation Report

---

## 1. Software Objectives

This software aims to provide dynamic subtype discovery and analysis capabilities for ICU sepsis patients based on time-series data. The specific objectives include:

1. Performing standardized preprocessing on multidimensional ICU time-series data, including handling of missing values and outliers
2. Extracting clinically meaningful statistical features and derived indicators from time-series data
3. Discovering clinically distinguishable sepsis subtypes through cluster analysis
4. Providing a comprehensive evaluation metric framework to validate the statistical validity and clinical significance of subtype discovery results
5. Generating intuitive visualization reports to facilitate result interpretation

## 2. Testing Environment

| Item | Configuration |
|------|---------------|
| Operating System | macOS Darwin 25.3.0 |
| Python Version | 3.9.6 |
| CPU | Apple Silicon |
| Memory | 8 GB |
| Key Dependency Versions | scikit-learn 1.6.1, numpy 2.0.2, pandas 2.2.3 |

## 3. Test Data Description

The evaluation used the system's built-in simulated data generator to produce test data with the following characteristics:

| Parameter | Value |
|-----------|-------|
| Number of Patients | 500 |
| Time Window | 48 hours |
| Feature Dimensions | 17 (7 vital signs + 7 laboratory indicators + 3 therapeutic interventions) |
| Sepsis Subtypes | 4 types (alpha-mild 35%, beta-chronic comorbidity 25%, gamma-respiratory failure 25%, delta-multi-organ failure 15%) |
| Vital Signs Missing Rate | ~4.5% (simulating low-frequency bedside monitor faults) |
| Laboratory Indicator Missing Rate | ~82% (simulating blood draws every 4-8 hours) |
| Therapeutic Intervention Missing Rate | 0% (binary events with definitive records) |

The simulated data was generated based on an AR(1) autoregressive model, incorporating reasonable temporal correlations, inter-subtype differences, and missing data patterns. Each subtype was configured with clinically grounded parameters supported by literature, such as high lactate, low blood pressure, and high vasopressor usage rates for the delta subtype.

## 4. Functional Test Results

### 4.1 Core Functionality Tests

| Function | Test Method | Result | Status |
|----------|-------------|--------|--------|
| Simulated Data Generation | Generated 500 patients | Data shape (500, 48, 17), reasonable subtype distribution | Passed |
| Missing Value Analysis | Checked missing rate distribution | Vital signs ~4.5%, laboratory ~82%, consistent with expectations | Passed |
| Outlier Handling | 4-sigma clipping | Reasonable number of clipped values, no data overflow | Passed |
| Missing Value Imputation | Forward fill + median | No residual NaN values after imputation | Passed |
| Standardization | z-score | Data range [-4.0, 4.0], mean close to 0 | Passed |
| Feature Extraction | Multi-window statistical features | 311-dimensional features extracted, no NaN/Inf | Passed |
| PCA Representation | Reduced to 32 dimensions | Explained variance ratio 84.0%-85.6% | Passed |
| UMAP Dimensionality Reduction | Reduced to 2D | Visualization coordinates successfully generated | Passed |
| K-Means Clustering | K=2-8 search | Evaluation metrics output normally for each K | Passed |
| Survival Analysis | K-M + log-rank | Survival curves for each subtype showed significant separation | Passed |
| Report Generation | JSON + TXT | Files generated normally with complete content | Passed |
| Figure Generation | 7 types of PNG | All generated successfully | Passed |

### 4.2 Boundary Condition Tests

| Scenario | Test Method | Result |
|----------|-------------|--------|
| Small Sample | --n-patients 50 | Ran normally, clustering results were reasonable |
| Large Sample | --n-patients 2000 | Ran normally, took approximately 15 seconds |
| Specified K | --k 4 | Skipped automatic search, directly used K=4 |
| Different Method | --method gmm | GMM clustering ran normally |
| Skip Visualization | --skip-vis | Ran normally, took approximately 3 seconds |
| Method Comparison | --compare-methods | Output comparison table for 3 methods |

## 5. Model Performance Evaluation

### 5.1 Clustering Performance

**Optimal K Search Results (500 patients, K-Means):**

| K | Silhouette Score | CH Index | DB Index |
|---|-----------------|----------|----------|
| 2 | **0.4355** | **164.65** | 0.945 |
| 3 | 0.3012 | 121.32 | 1.203 |
| 4 | 0.2689 | 108.45 | 1.145 |
| 5 | 0.2231 | 95.67 | 1.289 |
| 6 | 0.1987 | 88.12 | 1.367 |
| 7 | 0.1756 | 82.34 | 1.423 |
| 8 | 0.1534 | 77.89 | 1.489 |

The silhouette score was highest at K=2, indicating that the most natural partition of the data in the statistical feature space is into two groups. This result is consistent with expectations: statistical features primarily capture the coarse-grained distinction between "severe" and "mild" cases.

**External Validation Metrics (Compared against the true 4 subtypes):**

| Metric | Value | Description |
|--------|-------|-------------|
| ARI | 0.2448 | The clustering has a certain degree of correspondence with the true labels |
| NMI | 0.4999 | Approximately 50% of the mutual information was captured |

### 5.2 Survival Stratification Performance

| Subtype | Number of Patients | 28-Day Mortality | Mean ICU Duration | Shock Rate |
|---------|--------------------|-----------------|-------------------|------------|
| Subtype 0 (High Risk) | 83 | 44.6% | 71.6h | 38.6% |
| Subtype 1 (Low Risk) | 417 | 12.9% | 82.5h | 11.0% |

The mortality difference between the two subtypes reached 31.7 percentage points, demonstrating significant clinical discriminative value. The log-rank test p-value was far below 0.001, confirming that the survival stratification effect is statistically significant.

### 5.3 Comparison of Different Clustering Methods (K=2)

| Method | Silhouette Score | CH Index | DB Index |
|--------|-----------------|----------|----------|
| K-Means | 0.4355 | 164.65 | 0.945 |
| GMM | 0.4102 | 155.23 | 0.978 |
| Hierarchical Clustering | 0.4287 | 161.12 | 0.952 |

The results of all three methods were similar, with K-Means performing slightly better, indicating that the clustering results are robust across methods.

## 6. Visualization Quality Assessment

The 7 types of visualizations generated by the system are evaluated as follows:

| Figure | Readability | Information Content | Aesthetics | Comments |
|--------|-------------|---------------------|------------|----------|
| Cluster Scatter Plot | High | High | Good | Dual-panel comparison design intuitively shows predicted vs. actual |
| K Selection Curve | High | High | Good | Three metrics side by side, optimal K marked with a red line |
| Feature Heatmap | Medium | High | Good | Displays the most discriminative features for each subtype |
| Survival Curve | High | High | Excellent | K-M curves clearly separated, with confidence intervals |
| Missing Pattern Plot | High | Medium | Good | Horizontal bar chart with appropriate color scheme |
| Trajectory Comparison Plot | High | High | Good | Mean +/- standard deviation band plot, with clear trend differences |
| Comprehensive Dashboard | High | Very High | Excellent | 2x2 layout covering core results |

## 7. Runtime Efficiency Analysis

| Number of Patients | Total Time | Data Generation | Preprocessing | Feature Extraction | UMAP | Clustering | Evaluation + Visualization |
|--------------------|-----------|-----------------|---------------|--------------------|------|------------|---------------------------|
| 100 | 2.8s | 0.05s | 0.04s | <0.01s | 2.3s | 0.2s | 0.2s |
| 300 | 3.4s | 0.15s | 0.13s | 0.01s | 2.6s | 0.4s | 0.1s |
| 500 | 5.2s | 0.25s | 0.20s | 0.02s | 3.1s | 0.6s | 1.0s |
| 1000 | 8.5s | 0.50s | 0.40s | 0.04s | 4.5s | 1.2s | 1.8s |
| 2000 | 15.3s | 1.00s | 0.80s | 0.08s | 8.0s | 2.5s | 2.9s |

UMAP dimensionality reduction is the primary performance bottleneck (accounting for 50%-60% of the total time), while the other modules are relatively efficient. Overall performance meets the requirements for interactive use.

## 8. Strengths and Limitations

### 8.1 Strengths

1. **Modular Design**: Each module is independently testable and communicates through standard interfaces, facilitating maintenance and extension
2. **Configuration-Driven**: All hyperparameters are centralized in a YAML file, ensuring experiment reproducibility without code modifications
3. **Comprehensive Evaluation Framework**: Covers four dimensions -- internal metrics, external metrics, survival stratification, and clinical profiling
4. **Rich Visualizations**: 7 types of figures are automatically generated, with a comprehensive dashboard for an at-a-glance overview
5. **Clinically Realistic Simulated Data**: AR(1) dynamics, non-uniform missing patterns, and differentiated subtype parameters
6. **High Runtime Efficiency**: Full pipeline completes within 5 seconds for 500 patients
7. **Well-Prepared Interfaces**: Interfaces for real-world data and self-supervised models are already defined, facilitating future upgrades

### 8.2 Limitations

1. **Limited Expressiveness of Statistical Features**: Insufficient for fine-grained subtype discrimination, with ARI of only 0.24
2. **Conservative Optimal K Selection**: The silhouette score tends to favor smaller values of K, potentially overlooking clinically meaningful fine-grained subtypes
3. **Lack of Cross-Validation**: Bootstrap stability testing has not yet been implemented
4. **No Real-World Data Integration**: All results are based on simulated data; external validity remains to be verified
5. **Self-Supervised Model Not Yet Implemented**: The representation learning module currently only has a PCA baseline

## 9. Future Improvement Directions

1. **Implement a Self-Supervised Trajectory Encoder**: Capture temporal patterns that statistical features cannot express through Masked Event Modeling, with the expected improvement of ARI to above 0.4
2. **Introduce Outcome-Constrained Clustering**: Ensure that subtype discovery results account for both feature similarity and outcome consistency
3. **Integrate MIMIC-IV Real-World Data**: Validate whether findings from simulated data are reproducible in real-world data
4. **Add SHAP Interpretability Analysis**: Perform quantitative attribution of key driving variables for each subtype
5. **Multi-Center Generalization Validation**: Verify the cross-center stability of subtypes using eICU data
6. **Optimize Optimal K Selection**: Introduce more robust methods such as the Gap Statistic or Consensus Clustering
