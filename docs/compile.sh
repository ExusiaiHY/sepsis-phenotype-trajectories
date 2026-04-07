#!/bin/bash
# compile.sh - Compile LaTeX paper with all figures

echo "=========================================="
echo "Compiling ICU Sepsis Phenotyping Paper"
echo "=========================================="

cd "$(dirname "$0")"

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found. Please install LaTeX."
    echo "  macOS: brew install --cask mactex"
    echo "  Ubuntu: sudo apt-get install texlive-full"
    exit 1
fi

# Check if figures exist
echo "Checking figures..."
if [ ! -d "figures/paper" ]; then
    echo "ERROR: figures/paper/ directory not found!"
    exit 1
fi

for fig in s0_data_pipeline.png s0_missingness_pattern.png \
           s15_representation_comparison.png s15_training_convergence.png \
           s2_temporal_trajectories.png s3_mortality_stratification.png \
           s35_calibration_comparison.png \
           s4_treatment_effects_comparison.png s4_cross_database_performance.png \
           s5_deployment_profile.png s5_validation_gates.png; do
    if [ ! -f "figures/paper/$fig" ]; then
        echo "WARNING: figures/paper/$fig not found!"
    fi
done

echo "All figures found."
echo ""

# First compilation
echo "First compilation pass..."
pdflatex -interaction=nonstopmode RESEARCH_PAPER.tex 2>&1 | grep -E "(Error|Warning|Output)" || true

# Second compilation (for references)
echo ""
echo "Second compilation pass (references)..."
pdflatex -interaction=nonstopmode RESEARCH_PAPER.tex 2>&1 | grep -E "(Error|Warning|Output)" || true

# Check output
if [ -f "RESEARCH_PAPER.pdf" ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS: RESEARCH_PAPER.pdf created!"
    echo "=========================================="
    ls -lh RESEARCH_PAPER.pdf
    
    # Count pages
    pages=$(pdfinfo RESEARCH_PAPER.pdf 2>/dev/null | grep Pages | awk '{print $2}')
    echo "Pages: $pages"
    
    # Count figures
    figs=$(grep -c "includegraphics" RESEARCH_PAPER.tex)
    echo "Figures included: $figs"
else
    echo ""
    echo "ERROR: PDF not generated. Check errors above."
    exit 1
fi
