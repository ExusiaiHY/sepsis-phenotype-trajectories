"""
webapp/app.py - ICU Sepsis Phenotype Visualization Dashboard

A web-based visualization portal for exploring:
- Phenotype comparison across methods (PCA, S1, S1.5)
- Temporal trajectory analysis
- Transition matrices and Sankey diagrams
- Patient-level trajectory browser

Run: python webapp/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS

# sklearn import for PCA (used in spatio-temporal projection)
try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ============================================================
# Data Loading Helpers
# ============================================================

def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_numpy(path: Path) -> np.ndarray:
    """Load numpy file."""
    return np.load(path)


# ============================================================
# API Routes
# ============================================================

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/api/comparison")
def api_comparison():
    """Get comparison report (PCA vs S1 vs S1.5)."""
    path = PROJECT_ROOT / "data/s15/comparison_report.json"
    if not path.exists():
        return jsonify({"error": "Comparison report not found"}), 404
    return jsonify(load_json(path))


@app.route("/api/trajectory/stats")
def api_trajectory_stats():
    """Get trajectory statistics."""
    path = PROJECT_ROOT / "data/s2/trajectory_stats.json"
    if not path.exists():
        return jsonify({"error": "Trajectory stats not found"}), 404
    return jsonify(load_json(path))


@app.route("/api/trajectory/transitions")
def api_trajectory_transitions():
    """Get transition matrix."""
    path = PROJECT_ROOT / "data/s2/transition_matrix.json"
    if not path.exists():
        return jsonify({"error": "Transition matrix not found"}), 404
    return jsonify(load_json(path))


@app.route("/api/trajectory/window-labels")
def api_window_labels():
    """Get window labels for patient trajectories."""
    path = PROJECT_ROOT / "data/s2/window_labels.npy"
    if not path.exists():
        return jsonify({"error": "Window labels not found"}), 404
    labels = load_numpy(path)
    return jsonify({
        "shape": labels.shape,
        "labels": labels.tolist()
    })


@app.route("/api/embeddings/s15")
def api_embeddings_s15():
    """Get S1.5 embeddings (sample for visualization)."""
    path = PROJECT_ROOT / "data/s15/embeddings_s15.npy"
    if not path.exists():
        return jsonify({"error": "Embeddings not found"}), 404
    
    embeddings = load_numpy(path)
    # Sample 2000 points for visualization performance
    n_samples = min(2000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sampled = embeddings[indices]
    
    return jsonify({
        "shape": embeddings.shape,
        "sampled_indices": indices.tolist(),
        "embeddings": sampled.tolist()
    })


@app.route("/api/patients/sample")
def api_patients_sample():
    """Get sample patient trajectories."""
    window_labels_path = PROJECT_ROOT / "data/s2/window_labels.npy"
    
    if not window_labels_path.exists():
        return jsonify({"error": "Window labels not found"}), 404
    
    window_labels = load_numpy(window_labels_path)
    
    # Get diverse patient samples
    n_patients = len(window_labels)
    
    # Categorize patients
    stable_patients = []
    single_transition = []
    multi_transition = []
    
    for i in range(n_patients):
        unique_phenotypes = len(set(window_labels[i]))
        if unique_phenotypes == 1:
            stable_patients.append(i)
        elif unique_phenotypes == 2:
            single_transition.append(i)
        else:
            multi_transition.append(i)
    
    # Sample from each category
    samples = {
        "stable": np.random.choice(stable_patients, min(10, len(stable_patients)), replace=False).tolist(),
        "single_transition": np.random.choice(single_transition, min(10, len(single_transition)), replace=False).tolist(),
        "multi_transition": np.random.choice(multi_transition, min(10, len(multi_transition)), replace=False).tolist(),
    }
    
    return jsonify({
        "total_patients": n_patients,
        "categories": {
            "stable": len(stable_patients),
            "single_transition": len(single_transition),
            "multi_transition": len(multi_transition),
        },
        "sample_indices": samples,
    })


@app.route("/api/patient/<int:patient_id>")
def api_patient_detail(patient_id: int):
    """Get detailed trajectory for a specific patient."""
    window_labels_path = PROJECT_ROOT / "data/s2/window_labels.npy"
    
    if not window_labels_path.exists():
        return jsonify({"error": "Window labels not found"}), 404
    
    window_labels = load_numpy(window_labels_path)
    
    if patient_id >= len(window_labels):
        return jsonify({"error": "Patient ID out of range"}), 404
    
    labels = window_labels[patient_id].tolist()
    unique_phenotypes = list(set(labels))
    
    # Determine trajectory type
    if len(unique_phenotypes) == 1:
        trajectory_type = "stable"
    elif len(unique_phenotypes) == 2:
        trajectory_type = "single_transition"
    else:
        trajectory_type = "multi_transition"
    
    return jsonify({
        "patient_id": patient_id,
        "trajectory": labels,
        "window_starts": [0, 6, 12, 18, 24],
        "window_len": 24,
        "trajectory_type": trajectory_type,
        "unique_phenotypes": unique_phenotypes,
    })


@app.route("/api/spatiotemporal/embedding-trajectory/<int:patient_id>")
def api_embedding_trajectory(patient_id: int):
    """Get embedding trajectory for a specific patient (for spatio-temporal viz)."""
    rolling_emb_path = PROJECT_ROOT / "data/s2/rolling_embeddings.npy"
    window_labels_path = PROJECT_ROOT / "data/s2/window_labels.npy"
    
    if not rolling_emb_path.exists():
        return jsonify({"error": "Rolling embeddings not found"}), 404
    
    rolling_emb = load_numpy(rolling_emb_path)
    
    if patient_id >= len(rolling_emb):
        return jsonify({"error": "Patient ID out of range"}), 404
    
    # Get window labels
    window_labels = load_numpy(window_labels_path)[patient_id].tolist()
    
    # Return trajectory (5 windows × 128 dims)
    return jsonify({
        "patient_id": patient_id,
        "embedding_trajectory": rolling_emb[patient_id].tolist(),
        "window_labels": window_labels,
        "window_starts": [0, 6, 12, 18, 24],
        "d_model": 128,
    })


@app.route("/api/spatiotemporal/projection")
def api_spatiotemporal_projection():
    """Get 2D projection of embeddings for spatio-temporal visualization."""
    rolling_emb_path = PROJECT_ROOT / "data/s2/rolling_embeddings.npy"
    window_labels_path = PROJECT_ROOT / "data/s2/window_labels.npy"
    
    if not rolling_emb_path.exists():
        return jsonify({"error": "Rolling embeddings not found"}), 404
    
    if PCA is None:
        return jsonify({"error": "scikit-learn not installed. Run: pip install scikit-learn"}), 500
    
    rolling_emb = load_numpy(rolling_emb_path)  # (11986, 5, 128)
    window_labels = load_numpy(window_labels_path)  # (11986, 5)
    
    # Sample patients for visualization (performance)
    n_patients = len(rolling_emb)
    n_samples = min(500, n_patients)
    sample_indices = np.random.choice(n_patients, n_samples, replace=False)
    
    # Flatten: each window becomes a point (patient, window) -> point
    # Shape: (n_samples * 5, 128)
    sample_embeddings = rolling_emb[sample_indices]  # (500, 5, 128)
    sample_labels = window_labels[sample_indices]  # (500, 5)
    
    # Flatten to (2500, 128)
    flat_embeddings = sample_embeddings.reshape(-1, 128)
    flat_labels = sample_labels.reshape(-1)
    
    # Create metadata for each point
    patient_ids = np.repeat(sample_indices, 5)
    window_indices = np.tile(np.arange(5), n_samples)
    
    # PCA to 2D
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(flat_embeddings)
    
    return jsonify({
        "projection": "pca",
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "coords_2d": coords_2d.tolist(),
        "labels": flat_labels.tolist(),
        "patient_ids": patient_ids.tolist(),
        "window_indices": window_indices.tolist(),
        "n_points": len(coords_2d),
        "sample_patient_ids": sample_indices.tolist()[:20],  # First 20 for quick access
    })


@app.route("/api/spatiotemporal/window-evolution")
def api_window_evolution():
    """Get statistics about how embeddings evolve across windows."""
    rolling_emb_path = PROJECT_ROOT / "data/s2/rolling_embeddings.npy"
    window_labels_path = PROJECT_ROOT / "data/s2/window_labels.npy"
    
    if not rolling_emb_path.exists():
        return jsonify({"error": "Rolling embeddings not found"}), 404
    
    rolling_emb = load_numpy(rolling_emb_path)
    window_labels = load_numpy(window_labels_path)
    
    # Calculate average embedding movement between consecutive windows
    movements = []
    for w in range(4):  # 4 transitions between 5 windows
        diff = rolling_emb[:, w+1] - rolling_emb[:, w]  # (n_patients, 128)
        dist = np.linalg.norm(diff, axis=1)  # (n_patients,)
        movements.append({
            "from_window": w,
            "to_window": w + 1,
            "mean_distance": float(dist.mean()),
            "std_distance": float(dist.std()),
            "median_distance": float(np.median(dist)),
        })
    
    # Calculate embedding norm by phenotype
    phenotype_norms = {0: [], 1: [], 2: [], 3: []}
    for p in range(4):
        mask = window_labels == p
        norms = np.linalg.norm(rolling_emb[mask], axis=-1)
        phenotype_norms[p] = {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "median": float(np.median(norms)),
        }
    
    return jsonify({
        "embedding_movements": movements,
        "phenotype_embedding_norms": phenotype_norms,
        "window_obs_density": {
            "mean": [0.279, 0.273, 0.265, 0.260, 0.254],
            "std": [0.056, 0.056, 0.055, 0.055, 0.056],
        }
    })


@app.route("/api/summary")
def api_summary():
    """Get summary statistics for dashboard."""
    # Load trajectory stats
    traj_path = PROJECT_ROOT / "data/s2/trajectory_stats.json"
    comparison_path = PROJECT_ROOT / "data/s15/comparison_report.json"
    
    summary = {
        "n_patients": 11986,
        "n_windows": 5,
        "k": 4,
    }
    
    if traj_path.exists():
        traj = load_json(traj_path)
        summary["stable_fraction"] = traj["patient_level"]["stable_fraction"]
        summary["single_transition_fraction"] = traj["patient_level"]["single_transition_fraction"]
        summary["multi_transition_fraction"] = traj["patient_level"]["multi_transition_fraction"]
        summary["non_self_transition_fraction"] = traj["event_level"]["non_self_fraction"]
        summary["mortality_by_stable_phenotype"] = traj["mortality_by_stable_phenotype"]
    
    if comparison_path.exists():
        comp = load_json(comparison_path)
        # Get K=4 results
        summary["method_comparison"] = {
            "PCA": {
                "silhouette": comp["PCA_baseline"]["K=4"]["aggregated"]["silhouette"]["mean"],
                "mort_range": comp["PCA_baseline"]["K=4"]["aggregated"]["mort_range"]["mean"],
            },
            "S1_masked": {
                "silhouette": comp["S1_masked"]["K=4"]["aggregated"]["silhouette"]["mean"],
                "mort_range": comp["S1_masked"]["K=4"]["aggregated"]["mort_range"]["mean"],
            },
            "S15_contrastive": {
                "silhouette": comp["S15_contrastive"]["K=4"]["aggregated"]["silhouette"]["mean"],
                "mort_range": comp["S15_contrastive"]["K=4"]["aggregated"]["mort_range"]["mean"],
            },
        }
    
    return jsonify(summary)


# ============================================================
# Static Assets
# ============================================================

@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files."""
    return send_from_directory(app.static_folder, filename)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ICU Sepsis Phenotype Visualization Dashboard")
    print("=" * 60)
    print("\nOpen your browser at: http://localhost:5050")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5050, debug=True)
