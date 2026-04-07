# LaTeX Paper Compilation Guide

## Quick Start

```bash
cd docs
chmod +x compile.sh
./compile.sh
```

## Manual Compilation

```bash
cd docs
pdflatex RESEARCH_PAPER.tex
pdflatex RESEARCH_PAPER.tex  # Run twice for references
```

## Requirements

### macOS
```bash
brew install --cask mactex
```

### Ubuntu/Debian
```bash
sudo apt-get install texlive-full
```

### Windows
Install MiKTeX or TeX Live from https://www.tug.org/texlive/

## Figure Setup

All figures must be in:
```
docs/figures/paper/
```

Required files (11 PNG files, 300 DPI, Times New Roman):
1. `s0_data_pipeline.png`
2. `s0_missingness_pattern.png`
3. `s15_representation_comparison.png`
4. `s15_training_convergence.png`
5. `s2_temporal_trajectories.png`
6. `s3_mortality_stratification.png`
7. `s35_calibration_comparison.png`
8. `s4_treatment_effects_comparison.png`
9. `s4_cross_database_performance.png`
10. `s5_deployment_profile.png`
11. `s5_validation_gates.png`

## Figure Structure in Paper

| Figure | Content | Files |
|--------|---------|-------|
| Fig 1 | Pipeline diagram (TikZ) | Built-in |
| Fig 2 | S0 Data preprocessing | s0_data_pipeline.png + s0_missingness_pattern.png |
| Fig 3 | S1.5 Representation | s15_representation_comparison.png + s15_training_convergence.png |
| Fig 4 | S2-S3 Trajectories | s2_temporal_trajectories.png + s3_mortality_stratification.png |
| Fig 5 | S3.5 Calibration | s35_calibration_comparison.png |
| Fig 6 | S4 Treatment effects | s4_treatment_effects_comparison.png + s4_cross_database_performance.png |
| Fig 7 | S5 Deployment | s5_deployment_profile.png + s5_validation_gates.png |

## Key Packages Used

- `graphicx`: Figure inclusion
- `subcaption`: Subfigures with (a), (b) labels
- `placeins`: Float barriers to control figure placement
- `tikz`: Pipeline diagram (Figure 1)
- `booktabs`: Professional tables
- `natbib`: Citations

## Troubleshooting

### Figures not showing
- Check `graphicspath` is set correctly: `{figures/paper/}`
- Ensure PNG files exist in that directory
- Check file permissions

### Compilation errors
- Run `pdflatex` twice for references
- Install missing packages with `tlmgr install <package>`

### Font issues
All figures use Times New Roman. If fonts look wrong:
- Re-run `python scripts/generate_all_visualizations.py`
- Check matplotlib font cache: `rm -rf ~/.matplotlib/fontlist-v3*.json`

## Journal-Specific Adjustments

### For Nature Medicine
- Remove page numbers: `\pagestyle{empty}`
- Add line numbers: `\usepackage{lineno}`
- Double-spaced: `\usepackage{setspace} \doublespacing`

### For JAMA
- Structured abstract (already included)
- Word limit: 3000 words main text
- Max 6 figures (may need to combine)

### For Scientific Reports
- No page limits
- Can keep all 7 figures
- Ensure high-res figures (300+ DPI)

## Output Checklist

After compilation, verify:
- [ ] PDF generated successfully
- [ ] All 7 figures appear
- [ ] Subfigures labeled (a), (b)
- [ ] Tables numbered correctly
- [ ] Citations resolved (no [?])
- [ ] Page numbers correct
- [ ] No overfull hbox warnings

## Contact

For issues with figure generation:
```bash
python scripts/generate_all_visualizations.py
```

For LaTeX issues, check:
- docs/RESEARCH_PAPER.tex (main file)
- docs/figures/paper/ (figure directory)
