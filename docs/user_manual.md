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
│   ├── external/             # External datasets (PhysioNet / MIMIC / eICU)
│   ├── processed/            # Preprocessed cached data
│   ├── processed_mimic_demo/ # MIMIC demo analysis tables
│   └── processed_eicu_demo/  # eICU demo cached tensors
├── docs/                     # Project documentation
├── scripts/                  # Reproducible entry scripts
├── outputs/                  # Auto-generated outputs
├── src/                      # Source code directory
│   ├── main.py               # Main entry program
│   ├── data_loader.py        # Data loading and simulated data generation
│   ├── eicu_loader.py        # eICU raw/demo loader
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

### 5.2 Real Data Interface (Unified Entry)

The repository now supports a single public entry point for both credentialed external databases:

- `python src/main.py --source mimic ...`
- `python src/main.py --source eicu ...`

For MIMIC-IV, the entry automatically handles `raw CSV -> DuckDB -> concepts -> patient_static / patient_timeseries -> clustering/evaluation`.

For eICU, the entry automatically handles `raw CSV -> cached tensor / patient_info -> clustering/evaluation`.

Use `archive/mimic-iv-mock` as a local smoke-test substitute for the MIMIC path if you do not yet have credentialed files.

After automatic preparation, the data consumed by the legacy V1 pipeline has the following standard format:

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
  source: "simulated"  # Options: "simulated" | "mimic" | "eicu" | "physionet2012" | "sepsis2019"
  simulated:
    n_patients: 500     # Number of simulated patients
    n_timesteps: 48     # Time window (hours)
    n_subtypes: 4       # Number of subtypes
    missing_rate: 0.15  # Overall missing rate
    random_seed: 42
  mimic:
    raw_data_dir: data/external/mimic_iv_demo
    processed_dir: data/processed_mimic_demo
    db_path: db/mimic4_demo.db
    hours: 48
    output_format: parquet
  eicu:
    data_dir: data/external/eicu_demo
    processed_dir: data/processed_eicu_demo
    n_timesteps: 48
    tag: eicu_demo
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
| `--source` | Data source (simulated/mimic/eicu/physionet2012/sepsis2019) | simulated |
| `--data-dir` | Raw data directory override for the selected source | Config value |
| `--processed-dir` | Prepared artifact directory override for the selected source | Config value |
| `--db-path` | DuckDB path override for MIMIC | Config value |
| `--hours` | Override extracted hours / timesteps for MIMIC or eICU | Config value |
| `--max-patients` | Optional eICU preparation/loading cap | None |
| `--overwrite-db` | Rebuild the MIMIC DuckDB import from raw CSVs | No |
| `--tag` | Run tag for isolated outputs and tagged cache files | None |
| `--output-reports-dir` | Explicit report directory override | outputs/reports |
| `--output-figures-dir` | Explicit figure directory override | outputs/figures |
| `--skip-vis` | Skip visualization generation | No |
| `--compare-methods` | Run multi-method clustering comparison | No |

### 7.3 Usage Examples

```bash
cd project

# Generate 1000 patients using GMM clustering
python src/main.py --n-patients 1000 --method gmm

# Specify K=4 with t-SNE dimensionality reduction
python src/main.py --k 4 --reduction tsne

# Quick run (skip visualization)
python src/main.py --n-patients 200 --skip-vis

# Run clustering method comparison
python src/main.py --compare-methods

# Run full MIMIC-IV from raw CSVs with isolated outputs
python src/main.py --source mimic --data-dir /path/to/mimic-iv-3.1 --processed-dir data/processed_mimic_real --db-path db/mimic4_real.db --reduction pca --k 4 --skip-vis --tag mimic_real

# Rebuild the MIMIC DuckDB import when needed
python src/main.py --source mimic --data-dir /path/to/mimic-iv-3.1 --processed-dir data/processed_mimic_real --db-path db/mimic4_real.db --reduction pca --k 4 --skip-vis --tag mimic_real --overwrite-db

# Run full eICU from raw CSVs with isolated outputs
python src/main.py --source eicu --data-dir /path/to/eicu-2.0 --processed-dir data/processed_eicu_real --reduction pca --k 4 --skip-vis --tag eicu_real

# Run the frozen S1.5 + Stage 3 external temporal pipeline on both full external cohorts
python scripts/run_external_temporal_stage3.py --source all --device auto

# Or run one external source explicitly
python scripts/run_external_temporal_stage3.py --source mimic --device auto
```

## 8. Output Description

### 8.1 Visualization Charts (`outputs/figures/` or tagged subdirectories)

| File | Description |
|------|-------------|
| missing_pattern.png | Missing rate distribution for each variable |
| cluster_scatter.png | Cluster scatter plot after UMAP/t-SNE dimensionality reduction |
| k_selection.png | Optimal K selection curves (Silhouette Score / CH Index / DB Index) |
| subtype_heatmap.png | Subtype feature profile heatmap |
| survival_curves.png | Kaplan-Meier survival curves for each subtype |
| trajectory_comparison.png | Key variable trajectory comparison across subtypes |
| summary_dashboard.png | Comprehensive results dashboard (2x2 layout) |

When `--tag` is provided, figures are saved under `outputs/figures/<tag>/` and filenames are suffixed, for example `cluster_scatter_mimic_real.png`.

### 8.2 Evaluation Reports (`outputs/reports/` or tagged subdirectories)

| File | Format | Content |
|------|--------|---------|
| evaluation_report.json | JSON | Complete evaluation data (machine-parseable) |
| evaluation_summary.txt | Plain text | Human-readable evaluation summary |

When `--tag` is provided, reports are saved under `outputs/reports/<tag>/` and filenames are suffixed, for example `evaluation_report_mimic_real.json`.

### 8.3 Cached Data (data/processed/)

| File | Description |
|------|-------------|
| time_series_preprocessed.npy | Preprocessed 3D time-series tensor |
| patient_info_preprocessed.csv | Patient information table |

When `--tag` is provided, the preprocessed cache filenames are also suffixed, for example `time_series_preprocessed_mimic_real.npy`.

## 9. Frequently Asked Questions

**Q1: "ModuleNotFoundError" occurs at runtime**

A: Please verify that all dependencies are installed: `pip install -r requirements.txt`

**Q2: Chinese characters are displayed as squares**

A: The system automatically adapts to Chinese fonts on macOS (PingFang SC), Windows (Microsoft YaHei), and Linux (WenQuanYi Micro Hei). If the issue persists, install the corresponding font or modify the `_setup_style` function in `visualization.py`.

**Q3: UMAP dimensionality reduction is slow**

A: UMAP may be slow with large datasets (>5000 patients). You can use `--reduction pca` as an alternative, or reduce the number of patients for quick validation.

**Q4: How to use real MIMIC-IV data?**

A: Complete PhysioNet credentialing, place the full CSV files in a local directory, then run:

```bash
python src/main.py --source mimic --data-dir /path/to/mimic-iv-3.1 --processed-dir data/processed_mimic_real --db-path db/mimic4_real.db --reduction pca --k 4 --skip-vis --tag mimic_real
```

Add `--overwrite-db` if you need to rebuild the DuckDB database from raw CSVs. If you only want a smoke test of the integration path, use `--data-dir archive/mimic-iv-mock`.

**Q5: How to use eICU data?**

A: Place the eICU CSV files in a local directory, then run:

```bash
python src/main.py --source eicu --data-dir /path/to/eicu-2.0 --processed-dir data/processed_eicu_real --reduction pca --k 4 --skip-vis --tag eicu_real
```

The eICU loader reads raw tables directly, so no DuckDB build step is required.

**Q6: How to reproduce the full external Stage 3 temporal trajectories on MIMIC-IV / eICU?**

A: First prepare the source-specific real-data artifacts with `src/main.py` as above. Then run:

```bash
python scripts/run_external_temporal_stage3.py --source all --device auto
```

This command builds `data/external_temporal/<source>/s0`, reuses `data/s0/processed/preprocess_stats.json`, extracts frozen S1.5 embeddings, and writes Stage 3 artifacts including `trajectory_stats.json`, `sanity_checks.json`, figures, and the merged run index `data/external_temporal/external_temporal_runs.json`.

**Q7: Clustering results differ between runs**

A: Ensure that a random seed is set (default is 42). All random processes are managed uniformly through `set_global_seed()`.

## 10. Important Notes

1. For the first run, it is recommended to use the default configuration with a smaller number of patients (e.g., 200-300) for validation.
2. After modifying the configuration file, no reinstallation is needed; changes take effect immediately upon the next run.
3. Results in the default outputs directory will be overwritten on subsequent runs unless you use `--tag` or explicit output directory overrides.
4. Simulated data is intended for development and validation only; real data should be used for publications or formal reports.
5. The survival analysis feature requires the lifelines library; if not installed, the system will automatically fall back to a simplified analysis.
