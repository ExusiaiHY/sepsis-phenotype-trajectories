"""
Unified visualization configuration for S0-S5-v2.
Consistent styling across all project stages.
"""

# Color palette - professional medical research style
COLORS = {
    # Primary palette
    'primary': '#1a5276',      # Deep blue
    'secondary': '#2874a6',    # Medium blue
    'accent': '#e74c3c',       # Alert red
    'success': '#27ae60',      # Success green
    'warning': '#f39c12',      # Warning orange
    
    # Phenotype colors (4 phenotypes)
    'p0': '#3498db',  # Blue - low risk
    'p1': '#9b59b6',  # Purple - medium risk
    'p2': '#e74c3c',  # Red - high risk
    'p3': '#1abc9c',  # Teal - intermediate
    
    # Stage colors
    's0': '#95a5a6',  # Gray - data
    's1': '#3498db',  # Blue - representation
    's2': '#9b59b6',  # Purple - temporal
    's3': '#e67e22',  # Orange - validation
    's4': '#27ae60',  # Green - treatment
    's5': '#1abc9c',  # Teal - realtime
    
    # Grayscale
    'dark': '#2c3e50',
    'medium': '#7f8c8d',
    'light': '#ecf0f1',
    'white': '#ffffff',
}

PHENOTYPE_COLORS = [COLORS['p0'], COLORS['p3'], COLORS['p1'], COLORS['p2']]
STAGE_COLORS = [COLORS['s0'], COLORS['s1'], COLORS['s2'], COLORS['s3'], COLORS['s4'], COLORS['s5']]

# Typography - Times New Roman for publication
FONT_FAMILY = 'serif'
FONT_SERIF = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
FONT_SIZES = {
    'title': 16,        # Increased from 14
    'subtitle': 14,     # Increased from 12
    'label': 13,        # Increased from 11
    'tick': 12,         # Increased from 10
    'legend': 12,       # Increased from 10
    'annotation': 11,   # Increased from 9
}

# Figure settings
FIGURE_SETTINGS = {
    'dpi': 300,
    'bbox_inches': 'tight',
    'facecolor': 'white',
    'edgecolor': 'none',
}

# Style presets - Times New Roman for all
STYLE_PRESETS = {
    'paper': {
        'figure.figsize': (10, 6),  # Larger figure size
        'font.size': 13,            # Increased from 11
        'font.family': 'serif',
        'font.serif': FONT_SERIF,
        'axes.titlesize': 15,       # Increased from 12
        'axes.labelsize': 13,       # Increased from 11
        'legend.fontsize': 12,      # Increased from 10
        'xtick.labelsize': 12,      # Increased from 10
        'ytick.labelsize': 12,      # Increased from 10
        'mathtext.fontset': 'stix',  # For math symbols
    },
    'presentation': {
        'figure.figsize': (10, 6),
        'font.size': 14,
        'font.family': 'serif',
        'font.serif': FONT_SERIF,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'mathtext.fontset': 'stix',
    },
    'poster': {
        'figure.figsize': (12, 8),
        'font.size': 16,
        'font.family': 'serif',
        'font.serif': FONT_SERIF,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'mathtext.fontset': 'stix',
    },
    'dashboard': {
        'figure.figsize': (8, 5),
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': FONT_SERIF,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'mathtext.fontset': 'stix',
    }
}

# Export paths
OUTPUT_DIRS = {
    'paper': 'docs/figures/paper/',
    'presentation': 'docs/figures/presentation/',
    'poster': 'docs/figures/poster/',
    'dashboard': 'outputs/dashboards/figures/',
}
