# Paper Updates: S0-S5-v2 Final Version

**Date:** 2024-04-02  
**Author:** Wang Ruike, School of Biomedical Engineering, ShanghaiTech University

---

## Major Changes from Previous Version

### 1. New Title
**Old:** From Static Clusters to Temporal Trajectories: Self-Supervised Phenotyping of ICU Sepsis  
**New:** From Static Clusters to Temporal Trajectories: Self-Supervised Phenotyping of ICU Sepsis **with Real-Time Deployment**

### 2. New Stage Added: S5-v2 Real-Time Student
- Knowledge distillation from 321K to 90K parameters
- Inference latency: 1.1 ms (vs 50 ms teacher)
- All 5 engineering gates passed on 2/2 sources
- Calibrated transformer selected (TCN exploratory only)

### 3. New Stage Added: S3.5 Calibration
- 91% ECE reduction (0.222 → 0.020)
- Preserved AUROC (0.873) and recall (83.8%)
- Essential for clinical probability interpretation

### 4. Updated Abstract (Structured Format)
Now follows medical journal format:
- **Background:** Problem statement
- **Methods:** Six-stage pipeline, cohorts
- **Results:** Key metrics with numbers
- **Conclusions:** Clinical significance

### 5. Updated Key Metrics

| Metric | Old | New |
|--------|-----|-----|
| Pipeline stages | 4 | 6 (S0-S5-v2) |
| Total patients | ~300K | 307,126 |
| Latency | Not reported | 1.1 ms |
| ECE | Not reported | 0.020 (91% reduction) |
| Real-time ready | No | Yes (validated) |

### 6. Updated Figure Set (Times New Roman)

**Removed:**
- Old pipeline_diagram.png (replaced with TikZ)
- per_window_prevalence.png
- sankey_transitions.png
- mortality_by_trajectory.png

**Added:**
- s0_data_pipeline.png
- s0_missingness_pattern.png (heat + time series)
- s15_representation_comparison.png
- s15_training_convergence.png
- s2_temporal_trajectories.png
- s3_mortality_stratification.png
- s35_calibration_comparison.png
- s4_treatment_effects_comparison.png
- s4_cross_database_performance.png
- s5_deployment_profile.png
- s5_validation_gates.png

**Total:** 11 high-resolution figures (300 DPI, Times New Roman)

### 7. Updated Tables

**New tables:**
- S0-S5-v2 pipeline stages (Table 1)
- Representation comparison (Table 2)
- Calibration results (Table 3)
- Stage 4 performance (Table 4)
- Deployment comparison (Table 5)

### 8. New "At a Glance" Items
- Key metrics: AUROC, ECE, latency
- Real-time readiness status
- Expanded claim boundaries

### 9. Refined Conclusions
- Stronger descriptive claim: trajectories visible AND deployable
- Clearer causal boundary: treatment signals need RCT validation
- Clinical actionability: ready for risk stratification, not treatment recommendations

### 10. Dashboard Font Update
- CSS updated to Times New Roman for consistency

---

## File Locations

```
docs/
├── RESEARCH_PAPER.md          # Updated markdown version
├── RESEARCH_PAPER.tex         # Updated LaTeX version
├── PAPER_UPDATES.md           # This file
└── figures/
    ├── paper/                 # 11 new PNG figures
    ├── README.md              # Figure documentation
    └── figure_checklist.md    # LaTeX insertion guide
```

---

## Compilation Instructions

```bash
cd docs
# Compile LaTeX (run twice for references)
pdflatex RESEARCH_PAPER.tex
pdflatex RESEARCH_PAPER.tex

# Output: RESEARCH_PAPER.pdf
```

---

## Key Claims Summary

### What the paper CAN claim:
1. ✅ Temporal phenotypes are visible and clinically ordered
2. ✅ Representation transfers across centers and external cohorts
3. ✅ Model is real-time ready (1.1 ms, all gates passed)
4. ✅ Probabilities are well-calibrated (ECE 0.020)
5. ✅ Predictive performance is strong (AUROC 0.87-0.90)

### What the paper CANNOT claim:
1. ❌ Treatment effects are causal
2. ❌ Treatment recommendations are transportable
3. ❌ Phenotypes map to biological mechanisms
4. ❌ Model is validated in prospective clinical trials

---

## Submission Ready Status

- [x] Abstract (structured, 250 words)
- [x] Introduction (background + gap + aims)
- [x] Methods (six stages, reproducible)
- [x] Results (7 figures, 5 tables)
- [x] Discussion (contributions + limitations)
- [x] Conclusion (aligned with evidence)
- [x] References (6 key citations)
- [x] Figure legends (7 figures)
- [x] All figures (Times New Roman, 300 DPI)

**Recommended journals:** Nature Medicine, JAMA, Critical Care Medicine, Scientific Reports
