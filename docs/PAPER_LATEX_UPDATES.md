# LaTeX Paper Updates - Figure Integration

**Date:** 2024-04-02  
**Status:** ✅ Ready for compilation

---

## Changes Made

### 1. Added Required Packages
```latex
\usepackage{caption}      % Better caption control
\usepackage{subcaption}   % Subfigures (a), (b)
\usepackage{placeins}     % Float barriers
```

### 2. Caption Setup
```latex
\captionsetup{font=small, labelfont=bf}
\captionsetup[subfigure]{font=small, labelfont=bf}
```

### 3. Proper Subfigure Structure

**Before (problematic):**
```latex
\includegraphics[width=0.95\textwidth]{s2_temporal_trajectories.png}
\\[6pt]
\includegraphics[width=0.95\textwidth]{s3_mortality_stratification.png}
\caption{(a) ... (b) ...}
```

**After (fixed):**
```latex
\begin{subfigure}[t]{0.95\textwidth}
  \centering
  \includegraphics[width=0.85\textwidth]{s2_temporal_trajectories.png}
  \caption{Phenotype transitions...}
\end{subfigure}

\vspace{0.3cm}

\begin{subfigure}[t]{0.95\textwidth}
  \centering
  \includegraphics[width=0.85\textwidth]{s3_mortality_stratification.png}
  \caption{Mortality stratification...}
\end{subfigure}
\caption{S2--S3 temporal trajectory analysis.}
\label{fig:s23}
```

### 4. Added Float Barriers
```latex
\FloatBarrier  % Forces all floats to appear before continuing
```
Added before each major section to control figure placement.

### 5. Updated Figure Positioning
Changed from `[H]` (force here) to `[htbp]` (prefer top, then here, then bottom, then page):
- Better for journal submission
- Allows LaTeX to optimize placement
- Reduces blank space issues

### 6. Figure Size Optimization
- Reduced subfigure widths from 0.95 to 0.85-0.9 for margins
- Added proper spacing with `\vspace{0.3cm}` and `\hfill`
- Balanced multi-panel figures

---

## Figure Layout Summary

| Figure | Panels | Layout |
|--------|--------|--------|
| 1 | 1 | TikZ pipeline (single row) |
| 2 | 2 | Vertical stack (a) pipeline, (b) missingness |
| 3 | 2 | Side-by-side 58% / 38% |
| 4 | 2 | Vertical stack (a) trajectories, (b) mortality |
| 5 | 1 | Single wide calibration plot |
| 6 | 2 | Side-by-side 68% / 28% |
| 7 | 2 | Vertical stack (a) profile, (b) gates |

---

## Compilation Instructions

### Option 1: Automatic (Recommended)
```bash
cd docs
./compile.sh
```

### Option 2: Manual
```bash
cd docs
pdflatex RESEARCH_PAPER.tex
pdflatex RESEARCH_PAPER.tex  # Twice for references
```

### Option 3: With LaTeX IDE
Open `RESEARCH_PAPER.tex` in:
- TeXShop (macOS)
- TeXworks (Windows)
- Overleaf (Web)
- VS Code with LaTeX Workshop

---

## Expected Output

```
RESEARCH_PAPER.pdf
├── 7 main figures
│   ├── Fig 1: Pipeline (TikZ)
│   ├── Fig 2: S0 Data (2 panels)
│   ├── Fig 3: S1.5 Repr (2 panels)
│   ├── Fig 4: S2-S3 Temporal (2 panels)
│   ├── Fig 5: S3.5 Calibration (1 panel)
│   ├── Fig 6: S4 Treatment (2 panels)
│   └── Fig 7: S5 Deployment (2 panels)
├── 5 tables
└── ~10-12 pages
```

---

## Troubleshooting

### Issue: "File not found"
**Solution:** Check figures are in `docs/figures/paper/`
```bash
ls docs/figures/paper/*.png | wc -l  # Should show 11
```

### Issue: "Undefined control sequence"
**Solution:** Update LaTeX distribution or install missing packages
```bash
tlmgr install caption subcaption placeins
```

### Issue: Figures too large
**Solution:** Adjust widths in `\includegraphics[width=X\textwidth]`

### Issue: Subfigures not labeled (a), (b)
**Solution:** Ensure `subcaption` package is loaded and used correctly

---

## Journal Submission Tips

1. **Figure Quality:** All PNGs are 300 DPI, suitable for print
2. **Font Consistency:** Times New Roman throughout
3. **Color Safe:** Use patterns + color for accessibility
4. **Line Widths:** Thick enough for print (1-2pt)

### For Double-Blind Review
```latex
% Remove author info
% \author{Wang Ruike...}
\author{Anonymous}

% Remove header
% \fancyhead[R]{\small Wang Ruike...}
```

### For Word Count
```bash
detex RESEARCH_PAPER.tex | wc -w
```

---

## Files Ready for Submission

- [x] `RESEARCH_PAPER.tex` - Main LaTeX file
- [x] `RESEARCH_PAPER.pdf` - Compiled output (after compilation)
- [x] `figures/paper/*.png` - 11 figure files
- [x] Tables embedded in LaTeX
- [x] References in BibTeX format (thebibliography)

## Next Steps

1. Run `./compile.sh` to generate PDF
2. Review PDF for figure quality
3. Adjust figure sizes if needed
4. Submit to journal!
