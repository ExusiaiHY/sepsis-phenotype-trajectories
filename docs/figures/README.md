# ICU Sepsis Pipeline Visualizations

**Publication-ready figures for all pipeline stages (S0-S5-v2)**

Generated: 2024-04-01  
Author: Wang Ruike, School of Biomedical Engineering, ShanghaiTech University  
Font: **Times New Roman** (LaTeX-ready)

## Quick Start

```bash
cd project
python scripts/generate_all_visualizations.py
```

Output: `docs/figures/paper/*.png` (11 high-resolution figures, 300 DPI, Times New Roman)

---

## Figure Overview

### Stage S0: Data Pipeline

| Figure | Description |
|--------|-------------|
| `s0_data_pipeline.png` | Simplified preprocessing flow diagram |
| `s0_missingness_pattern.png` | Observation pattern heatmap + missingness rate over time |

**Notes**: Full pipeline diagram will be drawn in LaTeX using TikZ (not generated).

### Stage S1-S1.5: Self-Supervised Learning

| Figure | Description |
|--------|-------------|
| `s15_representation_comparison.png` | S1 vs S1.5 vs S1.6 comparison (clustering, stability, robustness) |
| `s15_training_convergence.png` | Pretraining loss curves (convergence at epoch 35) |

**Key Finding**: S1.5 selected for best cross-center stability (L1: 0.016) and missingness robustness (|r|: 0.148)

### Stage S2-S3: Temporal Trajectory Analysis

| Figure | Description |
|--------|-------------|
| `s2_temporal_trajectories.png` | Phenotype transitions across 5 rolling windows |
| `s3_mortality_stratification.png` | Mortality by phenotype (27.7pp range) and trajectory category |

**Key Finding**: 35.2% transition rate; stable phenotype shows 15.4% mortality

### Stage S3.5: Calibration

| Figure | Description |
|--------|-------------|
| `s35_calibration_comparison.png` | Reliability diagram + metrics before/after calibration |

**Key Finding**: 91% ECE reduction (0.222 → 0.020), AUROC preserved (0.873)

### Stage S4: Treatment-Aware Model

| Figure | Description |
|--------|-------------|
| `s4_treatment_effects_comparison.png` | CATE estimates across MIMIC-IV vs eICU-CRD |
| `s4_cross_database_performance.png` | AUROC across 3 databases (0.870-0.898) |

**Key Finding**: 0/6 treatment effects show cross-source consistency — effects not transportable

### Stage S5-S5-v2: Real-Time Student

| Figure | Description |
|--------|-------------|
| `s5_deployment_profile.png` | Parameters vs AUROC + latency comparison |
| `s5_validation_gates.png` | Engineering validation gates (all pass) |

**Key Finding**: Calibrated transformer selected (1.1ms, ECE 0.01); TCN exploratory only

---

## Typography

### Font
- **Primary**: Times New Roman
- **Math**: STIX (compatible with LaTeX)
- **Fallback**: DejaVu Serif, serif

### Sizes (Paper preset)
| Element | Size |
|---------|------|
| Title | 12pt |
| Subtitle | 12pt |
| Axis labels | 11pt |
| Tick labels | 10pt |
| Legend | 10pt |

---

## Color Palette

### Phenotypes
- **P0** (Low): `#3498db` (Blue)
- **P3** (Intermediate): `#1abc9c` (Teal)
- **P1** (Medium): `#9b59b6` (Purple)
- **P2** (High): `#e74c3c` (Red)

### Stages
- S0: Gray | S1: Blue | S2: Purple | S3: Orange | S4: Green | S5: Teal

---

## LaTeX Integration

Example usage in manuscript:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/paper/s35_calibration_comparison.png}
    \caption{Calibration improvement showing 91\% ECE reduction.}
    \label{fig:calibration}
\end{figure}
```

**Note**: Full pipeline diagram should be drawn using TikZ in LaTeX for better vector quality and styling consistency.

---

## Configuration

Edit `config/viz_config.py` to customize:

```python
STYLE_PRESETS = {
    'paper': {      # Default: 300 DPI, Times New Roman
    'presentation': # Larger fonts for slides
    'poster':       # Maximum size for posters
    'dashboard':    # Compact for bedside displays
}
```

---

## Dashboard

HTML dashboard (`s5/dashboard.py`) also uses **Times New Roman** for consistency:

```css
font-family: "Times New Roman", Times, serif;
```
