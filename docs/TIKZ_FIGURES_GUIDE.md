# All TikZ Figures Version

## Overview

All figures are now drawn using LaTeX TikZ (no external PNG files needed for charts/flows).

## Files

```
docs/
├── RESEARCH_PAPER.tex         # Main LaTeX with all TikZ figures
├── RESEARCH_PAPER.pdf         # Compiled output (398 KB, 10 pages)
└── TIKZ_FIGURES_GUIDE.md      # This guide
```

## Figures (All TikZ)

| Figure | Content | Type |
|--------|---------|------|
| **Fig 1** | S0-S5-v2 Pipeline | TikZ nodes & arrows |
| **Fig 2a** | S0 Data Flow | TikZ nodes & arrows |
| **Fig 2b** | Missingness Heatmap | TikZ grid & plot |
| **Fig 3** | S1.5 Representation | 3 TikZ bar charts + convergence curve |
| **Fig 4** | S2-S3 Temporal | TikZ flow diagram + bar charts |
| **Fig 5** | S3.5 Calibration | TikZ reliability diagram + bar chart |
| **Fig 6** | S4 Treatment | TikZ grouped bar chart |
| **Fig 7** | S5 Deployment | TikZ scatter + horizontal bars |

## Advantages

1. **Font Consistency** - All text uses document font (Times-compatible)
2. **Vector Graphics** - Infinite zoom, no pixelation
3. **Smaller File** - 398 KB (was 1.7 MB)
4. **Faster Compile** - No external image files
5. **Easy Edit** - Change colors/text directly in .tex

## Compilation

```bash
cd docs
pdflatex RESEARCH_PAPER.tex
pdflatex RESEARCH_PAPER.tex  # Twice for references
```

## Customization

### Change Colors
```latex
\fill[blue!60]     % Blue bar
\fill[red!60]      % Red bar  
\fill[green!60]    % Green bar
```

### Change Font Size
```latex
\node[font=\footnotesize]    % Small
\node[font=\small]           % Medium
\node[font=\normalsize]      % Normal
```

### Add Data Points
```latex
% In bar chart:
\fill[color] (x,0) rectangle (x+width, height);

% In line plot:
\draw[thick] plot[smooth] coordinates {(x1,y1) (x2,y2) ...};
```

## Data Values (for reference)

### Figure 3 - S1.5 Metrics
- Silhouette: PCA=0.061, S1=0.087, S1.5=0.080, S1.6=0.079
- Center L1: PCA=0.027, S1=0.024, **S1.5=0.016**, S1.6=0.021
- Density |r|: PCA=0.231, S1=0.247, **S1.5=0.148**, S1.6=0.148

### Figure 4 - Mortality
- P0: 4.0%, P3: 9.7%, P1: 22.5%, P2: 31.7%
- Stable: 15.4%, Single: 11.4%, Multi: 15.2%

### Figure 6 - Treatment Effects
- 0/6 treatments show cross-source consistency
- MIMIC vs eICU: opposite directions for all

## Journal Submission

This version is ready for:
- Nature Medicine
- JAMA  
- Critical Care Medicine
- Scientific Reports

All figures meet:
- ✅ Vector format (no raster images)
- ✅ Embedded fonts
- ✅ 300+ DPI equivalent
- ✅ Grayscale-safe colors
