# Figure Checklist for LaTeX Paper

**Paper:** From Static Clusters to Temporal Trajectories: Self-Supervised Phenotyping of ICU Sepsis with Real-Time Deployment

**Font:** Times New Roman (all figures)
**Resolution:** 300 DPI
**Location:** `docs/figures/paper/`

---

## Figure 1: Pipeline Overview (TikZ)
- **Type:** TikZ diagram in LaTeX
- **Status:** ✅ Built into RESEARCH_PAPER.tex
- **Content:** Six-stage S0-S5-v2 workflow with metrics

---

## Figure 2: S0 Data Preprocessing
- **Files:** 
  - `s0_data_pipeline.png` (73 KB)
  - `s0_missingness_pattern.png` (175 KB)
- **LaTeX:** `\ref{fig:s0}`
- **Content:**
  - (a) Preprocessing flow from 11,986 stays
  - (b) Heatmap + missingness rate curve (73.3%)

---

## Figure 3: S1.5 Representation Learning
- **Files:**
  - `s15_representation_comparison.png` (171 KB)
  - `s15_training_convergence.png` (151 KB)
- **LaTeX:** `\ref{fig:s15}`
- **Content:**
  - (a) PCA vs S1 vs S1.5 comparison
  - (b) Training loss curves (epoch 35 convergence)
- **Table:** `\ref{tab:repr}` (representation metrics)

---

## Figure 4: S2-S3 Temporal Trajectories
- **Files:**
  - `s2_temporal_trajectories.png` (114 KB)
  - `s3_mortality_stratification.png` (172 KB)
- **LaTeX:** `\ref{fig:s23}`
- **Content:**
  - (a) Five-window phenotype transitions (35.2% transition rate)
  - (b) Mortality by phenotype (27.7 pp range) and trajectory

---

## Figure 5: S3.5 Calibration
- **File:** `s35_calibration_comparison.png` (192 KB)
- **LaTeX:** `\ref{fig:s35}`
- **Content:**
  - (a) Reliability diagrams (91% ECE reduction)
  - (b) Metrics before/after calibration
- **Table:** `\ref{tab:cal}` (calibration metrics)

---

## Figure 6: S4 Treatment Effects
- **Files:**
  - `s4_treatment_effects_comparison.png` (133 KB)
  - `s4_cross_database_performance.png` (93 KB)
- **LaTeX:** `\ref{fig:s4}`
- **Content:**
  - (a) CATE comparison (0/6 cross-source consistency)
  - (b) AUROC across databases (0.870-0.898)
- **Table:** `\ref{tab:s4}` (Stage 4 performance)

---

## Figure 7: S5-S5-v2 Real-Time Deployment
- **Files:**
  - `s5_deployment_profile.png` (179 KB)
  - `s5_validation_gates.png` (91 KB)
- **LaTeX:** `\ref{fig:s5}`
- **Content:**
  - (a) Efficiency vs performance + latency comparison
  - (b) Engineering validation gates (5/5 passed)
- **Table:** `\ref{tab:s5}` (deployment comparison)

---

## Summary

| Figure | Components | Key Finding |
|--------|-----------|-------------|
| 1 | TikZ pipeline | Six-stage workflow |
| 2 | S0 (2 panels) | 73.3% missingness |
| 3 | S1.5 (2 panels) | Best robustness balance |
| 4 | S2-S3 (2 panels) | 35.2% transition rate |
| 5 | S3.5 (1 panel) | 91% ECE reduction |
| 6 | S4 (2 panels) | 0/6 treatment consistency |
| 7 | S5 (2 panels) | 1.1ms, all gates passed |

**Total figures:** 7 (with 12 sub-panels)
**Total PNG files:** 11

---

## LaTeX Compilation Order

```bash
cd docs
pdflatex RESEARCH_PAPER.tex
pdflatex RESEARCH_PAPER.tex  # Run twice for references
```

## Inserting Figures

All figures use:
```latex
\includegraphics[width=0.95\textwidth]{filename.png}
```

No path prefix needed (graphicspath set to `{figures/paper/}`).
