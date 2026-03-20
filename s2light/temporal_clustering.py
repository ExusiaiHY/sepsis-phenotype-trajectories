"""
temporal_clustering.py - KMeans on rolling-window embeddings.

Fits KMeans on train-split windows only, then assigns all windows.
Reports per-window cluster distribution and silhouette.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger("s2light.clustering")


def fit_and_assign(
    rolling_embeddings: np.ndarray,
    splits: dict,
    k: int = 4,
    n_init: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, dict, KMeans]:
    """
    Fit KMeans on train-split rolling windows, assign all windows.

    Parameters
    ----------
    rolling_embeddings: (N, W, D)
    splits: dict with 'train', 'val', 'test' index lists

    Returns
    -------
    window_labels: (N, W) int array
    quality: dict with per-window stats
    km: fitted KMeans model
    """
    N, W, D = rolling_embeddings.shape
    train_idx = np.array(splits["train"])

    # Flatten train windows for fitting
    train_emb = rolling_embeddings[train_idx]  # (Ntrain, W, D)
    train_flat = train_emb.reshape(-1, D)       # (Ntrain*W, D)

    logger.info(f"Fitting KMeans K={k} on {train_flat.shape[0]} train windows (D={D})...")
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed, max_iter=300)
    km.fit(train_flat)

    # Assign all windows
    all_flat = rolling_embeddings.reshape(-1, D)  # (N*W, D)
    all_labels_flat = km.predict(all_flat)
    window_labels = all_labels_flat.reshape(N, W)  # (N, W)

    # Per-window quality metrics
    quality = {"k": k, "n_windows": W, "per_window": []}

    for wi in range(W):
        wi_emb = rolling_embeddings[:, wi, :]     # (N, D)
        wi_labels = window_labels[:, wi]           # (N,)

        n_unique = len(np.unique(wi_labels))
        if n_unique >= 2:
            sil = float(silhouette_score(wi_emb, wi_labels))
        else:
            sil = float("nan")

        counts = {int(c): int((wi_labels == c).sum()) for c in range(k)}
        fracs = {int(c): round(int((wi_labels == c).sum()) / N, 4) for c in range(k)}

        # Sparse window detection: windows where median obs density < 10%
        quality["per_window"].append({
            "window_idx": wi,
            "silhouette": round(sil, 4),
            "cluster_counts": counts,
            "cluster_fractions": fracs,
        })

        logger.info(f"  Window {wi}: sil={sil:.4f}, dist={fracs}")

    # Overall flat silhouette
    overall_sil = float(silhouette_score(all_flat, all_labels_flat))
    quality["overall_silhouette"] = round(overall_sil, 4)
    logger.info(f"Overall window-level silhouette: {overall_sil:.4f}")

    return window_labels, quality, km


def save_kmeans_model(km: KMeans, path: Path) -> None:
    """Save cluster centers for reproducibility."""
    data = {
        "n_clusters": km.n_clusters,
        "centers": km.cluster_centers_.tolist(),
        "inertia": float(km.inertia_),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
