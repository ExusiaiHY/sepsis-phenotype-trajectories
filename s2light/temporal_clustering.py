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
    fit_sample_size: int | None = None,
    silhouette_sample_size: int | None = None,
    overall_silhouette_sample_size: int | None = None,
    predict_batch_size: int | None = None,
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
    rng = np.random.default_rng(seed)

    # Flatten train windows for fitting
    train_emb = rolling_embeddings[train_idx]  # (Ntrain, W, D)
    train_flat = train_emb.reshape(-1, D)       # (Ntrain*W, D)
    fit_flat = train_flat
    fit_sample_n = None
    if fit_sample_size is not None and train_flat.shape[0] > int(fit_sample_size):
        fit_sample_n = int(fit_sample_size)
        sample_idx = rng.choice(train_flat.shape[0], size=fit_sample_n, replace=False)
        fit_flat = np.asarray(train_flat[sample_idx])
        logger.info(
            "Subsampling train windows for KMeans fit: %d -> %d",
            train_flat.shape[0],
            fit_sample_n,
        )

    logger.info(f"Fitting KMeans K={k} on {fit_flat.shape[0]} train windows (D={D})...")
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed, max_iter=300)
    km.fit(fit_flat)

    # Assign all windows
    all_flat = rolling_embeddings.reshape(-1, D)  # (N*W, D)
    if predict_batch_size is None or predict_batch_size <= 0:
        all_labels_flat = km.predict(all_flat)
    else:
        all_labels_flat = np.empty(all_flat.shape[0], dtype=np.int32)
        batch_n = int(predict_batch_size)
        for start in range(0, all_flat.shape[0], batch_n):
            end = min(start + batch_n, all_flat.shape[0])
            all_labels_flat[start:end] = km.predict(all_flat[start:end])
    window_labels = all_labels_flat.reshape(N, W)  # (N, W)

    # Per-window quality metrics
    quality = {
        "k": k,
        "n_windows": W,
        "fit_sample_size_used": fit_sample_n,
        "silhouette_sample_size": silhouette_sample_size,
        "overall_silhouette_sample_size": overall_silhouette_sample_size,
        "predict_batch_size": predict_batch_size,
        "per_window": [],
    }

    for wi in range(W):
        wi_emb = rolling_embeddings[:, wi, :]     # (N, D)
        wi_labels = window_labels[:, wi]           # (N,)

        n_unique = len(np.unique(wi_labels))
        if n_unique >= 2:
            sample_size = None
            if silhouette_sample_size is not None:
                sample_size = min(int(silhouette_sample_size), wi_emb.shape[0])
            sil = float(
                silhouette_score(
                    wi_emb,
                    wi_labels,
                    sample_size=sample_size,
                    random_state=seed,
                )
            )
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
    overall_sample_size = None
    if overall_silhouette_sample_size is not None:
        overall_sample_size = min(int(overall_silhouette_sample_size), all_flat.shape[0])
    overall_sil = float(
        silhouette_score(
            all_flat,
            all_labels_flat,
            sample_size=overall_sample_size,
            random_state=seed,
        )
    )
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
