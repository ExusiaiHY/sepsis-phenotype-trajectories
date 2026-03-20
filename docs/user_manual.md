# Software User Manual

# ICU Sepsis Dynamic Subtype Discovery and Analysis System

---

## 1. Software Overview

This software is a dynamic subtype discovery and analysis system designed for time-series data of ICU sepsis patients. Based on electronic health record (EHR) time-series data, the system performs standardized preprocessing, statistical feature extraction, dimensionality reduction, and cluster analysis to automatically discover dynamic subtypes among sepsis patients, providing comprehensive evaluation metrics and visualization reports.

Key features include:

- Loading and preprocessing of simulated or real ICU sepsis patient time-series data
- Feature extraction across multiple time windows with multiple statistical measures
- Automatic optimal cluster number search and multi-method clustering comparison
- Clustering quality evaluation, survival stratification analysis, and subtype clinical profiling
- Automatic generation of 7 types of visualization charts
- Structured evaluation report output (JSON + plain text)

## 2. System Requirements

| Item | Requirement |
|------|-------------|
| Operating System | macOS / Linux / Windows |
| Python Version | 3.9 or above |
| Memory | 4 GB minimum (8 GB recommended) |
| Disk Space | 500 MB minimum (including dependencies) |
| GPU | Not required (MVP version) |

## 3. Installation Steps

### 3.1 Obtain the Project Code

Copy the project directory to your local working directory, or obtain it as follows:

```bash
# Assuming the project is located at the following path
cd /path/to/project
```

### 3.2 Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:

| Package | Version Requirement | Purpose |
|---------|---------------------|---------|
| numpy | >=1.24.0 | Numerical computation |
| pandas | >=2.0.0 | Data processing |
| scipy | >=1.10.0 | Scientific computation |
| pyyaml | >=6.0 | Configuration file parsing |
| scikit-learn | >=1.3.0 | Clustering and evaluation |
| umap-learn | >=0.5.3 | Dimensionality reduction |
| matplotlib | >=3.7.0 | Visualization |
| lifelines | >=0.27.0 | Survival analysis (recommended) |

### 3.3 Verify Installation

```bash
cd src
python main.py --n-patients 100 --skip-vis
```

If the output displays "All processes completed!", the installation was successful.

## 4. Project Directory Structure

```
project/
├── config/
│   └── config.yaml          # Global configuration file (hyperparameters, paths, variable definitions)
├── data/
│   ├── raw/                  # Raw data storage directory
│   ├── processed/            # Preprocessed cached data
│   └── demo/                 # Small-scale demo data
├── docs/                     # Project documentation
├── outputs/
│   ├── figures/              # Auto-generated visualization charts (PNG)
│   ├── models/               # Saved model files
│   └── reports/              # Evaluation reports (JSON + TXT)
├── src/                      # Source code directory
│   ├── main.py               # Main entry program
│   ├── data_loader.py        # Data loading and simulated data generation
│   ├── preprocess.py         # Preprocessing module
│   ├── feature_engineering.py # Feature extraction module
│   ├── representation_model.py # Representation learning module
│   ├── clustering.py         # Cluster analysis module
│   ├── evaluation.py         # Evaluation module
│   ├── visualization.py      # Visualization module
│   └── utils.py              # Common utility module
├── requirements.txt          # Python dependency list
└── README.md                 # Project overview
```

## 5. Input Data Format

### 5.1 Simulated Data (Default)

The system includes a built-in simulated data generator, allowing it to run without any external data. The simulated data is generated based on clinically reasonable parameter settings, producing time-series data for patients with 4 sepsis subtypes.

### 5.2 Real Data Interface (Reserved)

When using MIMIC-IV or eICU real data, the data must be formatted as follows:

**Time-Series Data (3D Tensor):**
- Shape: `(n_patients, n_timesteps, n_features)`
- Data type: `float64`
- Missing values: represented by `NaN`

**Patient Information Table (DataFrame):**

| Column Name | Type | Description |
|-------------|------|-------------|
| patient_id | str | Unique patient identifier |
| mortality_28d | int | 28-day mortality (0/1) |
| icu_los | float | ICU length of stay (hours) |
| shock_onset | int | Whether shock occurred (0/1) |
| age | int | Age |
| gender | str | Gender (M/F) |

## 6. Configuration File

The configuration file is located at `config/config.yaml` and uses the YAML format. The main configuration items are as follows:

### 6.1 Data Source Configuration

```yaml
data:
  source: "simulated"  # Options: "simulated" | "mimic" | "eicu"
  simulated:
    n_patients: 500     # Number of simulated patients
    n_timesteps: 48     # Time window (hours)
    n_subtypes: 4       # Number of subtypes
    missing_rate: 0.15  # Overall missing rate
    random_seed: 42
```

### 6.2 Preprocessing Configuration

```yaml
preprocess:
  missing_strategy: "forward_fill_then_median"  # Missing value imputation strategy
  outlier_method: "clip"      # Outlier handling method
  outlier_sigma: 4.0          # Outlier threshold
  normalization: "standard"   # Normalization method
```

### 6.3 Clustering Configuration

```yaml
clustering:
  method: "kmeans"            # Clustering method
  k_range: [2, 8]            # K value search range
  optimal_k_criterion: "silhouette"  # Criterion for optimal K selection
```

### 6.4 Dimensionality Reduction Configuration

```yaml
reduction:
  method: "umap"              # Reduction method: "umap" | "tsne" | "pca"
  n_components: 2
  umap:
    n_neighbors: 15
    min_dist: 0.1
```

## 7. How to Run

### 7.1 Basic Execution

```bash
cd project/src
python main.py
```

Running with default configuration will generate a complete analysis for 500 simulated patients.

### 7.2 Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Custom configuration file path | config/config.yaml |
| `--n-patients` | Number of simulated patients | 500 |
| `--method` | Clustering method (kmeans/gmm/hierarchical/spectral) | kmeans |
| `--k` | Specify number of clusters (skips automatic search) | Auto |
| `--reduction` | Dimensionality reduction method (umap/tsne/pca) | umap |
| `--seed` | Random seed | 42 |
| `--skip-vis` | Skip visualization generation | No |
| `--compare-methods` | Run multi-method clustering comparison | No |

### 7.3 Usage Examples

```bash
# Generate 1000 patients using GMM clustering
python main.py --n-patients 1000 --method gmm

# Specify K=4 with t-SNE dimensionality reduction
python main.py --k 4 --reduction tsne

# Quick run (skip visualization)
python main.py --n-patients 200 --skip-vis

# Run clustering method comparison
python main.py --compare-methods
```

## 8. Output Description

### 8.1 Visualization Charts (outputs/figures/)

| File | Description |
|------|-------------|
| missing_pattern.png | Missing rate distribution for each variable |
| cluster_scatter.png | Cluster scatter plot after UMAP/t-SNE dimensionality reduction |
| k_selection.png | Optimal K selection curves (Silhouette Score / CH Index / DB Index) |
| subtype_heatmap.png | Subtype feature profile heatmap |
| survival_curves.png | Kaplan-Meier survival curves for each subtype |
| trajectory_comparison.png | Key variable trajectory comparison across subtypes |
| summary_dashboard.png | Comprehensive results dashboard (2x2 layout) |

### 8.2 Evaluation Reports (outputs/reports/)

| File | Format | Content |
|------|--------|---------|
| evaluation_report.json | JSON | Complete evaluation data (machine-parseable) |
| evaluation_summary.txt | Plain text | Human-readable evaluation summary |

### 8.3 Cached Data (data/processed/)

| File | Description |
|------|-------------|
| time_series_preprocessed.npy | Preprocessed 3D time-series tensor |
| patient_info_preprocessed.csv | Patient information table |

## 9. Frequently Asked Questions

**Q1: "ModuleNotFoundError" occurs at runtime**

A: Please verify that all dependencies are installed: `pip install -r requirements.txt`

**Q2: Chinese characters are displayed as squares**

A: The system automatically adapts to Chinese fonts on macOS (PingFang SC), Windows (Microsoft YaHei), and Linux (WenQuanYi Micro Hei). If the issue persists, install the corresponding font or modify the `_setup_style` function in `visualization.py`.

**Q3: UMAP dimensionality reduction is slow**

A: UMAP may be slow with large datasets (>5000 patients). You can use `--reduction pca` as an alternative, or reduce the number of patients for quick validation.

**Q4: How to use real MIMIC-IV data?**

A: You must first complete PhysioNet credentialing (https://physionet.org/content/mimiciv/), then download the data and change `data.source` in `config.yaml` to `"mimic"`, specifying the data path. The `load_mimic_data` function in `data_loader.py` will be implemented in version 2.

**Q5: Clustering results differ between runs**

A: Ensure that a random seed is set (default is 42). All random processes are managed uniformly through `set_global_seed()`.

## 10. Important Notes

1. For the first run, it is recommended to use the default configuration with a smaller number of patients (e.g., 200-300) for validation.
2. After modifying the configuration file, no reinstallation is needed; changes take effect immediately upon the next run.
3. Results in the outputs directory will be overwritten on subsequent runs; back up manually if you need to preserve them.
4. Simulated data is intended for development and validation only; real data should be used for publications or formal reports.
5. The survival analysis feature requires the lifelines library; if not installed, the system will automatically fall back to a simplified analysis.
