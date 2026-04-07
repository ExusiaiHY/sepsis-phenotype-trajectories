#!/bin/bash
# launch.sh - Start the ICU Sepsis Phenotype Visualization Dashboard

cd "$(dirname "$0")/.."

# Check if flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip install flask flask-cors
fi

echo "=========================================="
echo "ICU Sepsis Phenotype Visualization"
echo "=========================================="
echo ""
echo "Starting server at http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="

python3 webapp/app.py
