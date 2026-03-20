"""
clustering.py - Clustering and subtype discovery module

Responsibilities:
1. Run multiple clustering algorithms on patient feature matrices
2. Automatically search for optimal cluster count K
3. Support dimensionality reduction for visualization
4. Return cluster labels and evaluation metrics
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import setup_logger, timer, set_global_seed

logger = setup_logger(__name__)


# ============================================================
# Optimal K Search
# ============================================================

@timer
def search_optimal_k(
    X: np.ndarray,
    config: dict,
) -> dict:
    """
    Search for the optimal number of clusters within a given range.

    Returns
    -------
    results : dict with optimal_k, k_scores DataFrame, and criterion used.
    """
    clust_cfg = config["clustering"]
    k_min, k_max = clust_cfg["k_range"]
    criterion = clust_cfg["optimal_k_criterion"]
    method = clust_cfg["method"]
    seed = clust_cfg["random_seed"]

    set_global_seed(seed)
    logger.info(f"Searching optimal K: range [{k_min}, {k_max}], method {method}, criterion {criterion}")

    records = []
    for k in range(k_min, k_max + 1):
        labels = _run_clustering(X, method=method, n_clusters=k, seed=seed)
        metrics = _compute_cluster_metrics(X, labels)
        metrics["k"] = k
        records.append(metrics)
        logger.info(f"  K={k}: silhouette={metrics['silhouette']:.3f}, "
                     f"CH={metrics['calinski_harabasz']:.1f}, "
                     f"DB={metrics['davies_bouldin']:.3f}")

    k_scores = pd.DataFrame(records)

    if criterion == "silhouette":
        optimal_k = k_scores.loc[k_scores["silhouette"].idxmax(), "k"]
    elif criterion == "calinski":
        optimal_k = k_scores.loc[k_scores["calinski_harabasz"].idxmax(), "k"]
    elif criterion == "davies_bouldin":
        optimal_k = k_scores.loc[k_scores["davies_bouldin"].idxmin(), "k"]
    elif criterion == "elbow":
        optimal_k = _elbow_method(k_scores)
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")

    optimal_k = int(optimal_k)
    logger.info(f"Optimal K = {optimal_k} (criterion: {criterion})")

    return {
        "optimal_k": optimal_k,
        "k_scores": k_scores,
        "criterion": criterion,
    }


# ============================================================
# Run Clustering
# ============================================================

@timer
def run_final_clustering(
    X: np.ndarray,
    config: dict,
    n_clusters: int | None = None,
) -> np.ndarray:
    """Run final clustering with the chosen K. Returns label array."""
    clust_cfg = config["clustering"]
    method = clust_cfg["method"]
    seed = clust_cfg["random_seed"]

    if n_clusters is None:
        search_result = search_optimal_k(X, config)
        n_clusters = search_result["optimal_k"]

    labels = _run_clustering(X, method=method, n_clusters=n_clusters, seed=seed)
    logger.info(f"Final clustering: {n_clusters} clusters, distribution {dict(zip(*np.unique(labels, return_counts=True)))}")

    return labels


def _run_clustering(
    X: np.ndarray,
    method: str,
    n_clusters: int,
    seed: int = 42,
) -> np.ndarray:
    """Execute a specific clustering algorithm, return labels."""
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed, max_iter=300)
        return model.fit_predict(X)
    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=seed, n_init=5)
        return model.fit_predict(X)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        return model.fit_predict(X)
    elif method == "spectral":
        model = SpectralClustering(n_clusters=n_clusters, random_state=seed, affinity="rbf")
        return model.fit_predict(X)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")


# ============================================================
# Cluster Metrics
# ============================================================

def _compute_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute standard internal clustering metrics."""
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        return {"silhouette": -1, "calinski_harabasz": 0, "davies_bouldin": float("inf")}

    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
    }


def _elbow_method(k_scores: pd.DataFrame) -> int:
    """Simple elbow method: find the point where CH index growth rate drops most."""
    ch = k_scores["calinski_harabasz"].values
    if len(ch) < 3:
        return k_scores.loc[k_scores["calinski_harabasz"].idxmax(), "k"]
    diffs = np.diff(ch)
    diffs2 = np.diff(diffs)
    elbow_idx = np.argmin(diffs2) + 2
    return k_scores.iloc[min(elbow_idx, len(k_scores) - 1)]["k"]


# ============================================================
# Dimensionality Reduction
# ============================================================

@timer
def reduce_dimensions(
    X: np.ndarray,
    config: dict,
) -> np.ndarray:
    """
    Reduce feature matrix dimensions for visualization.

    Returns
    -------
    coords : np.ndarray, shape (n_patients, n_components)
    """
    red_cfg = config["reduction"]
    method = red_cfg["method"]
    n_components = red_cfg["n_components"]
    seed = config["runtime"]["random_seed"]

    logger.info(f"Dimensionality reduction: {method}, target dims {n_components}")

    if method == "umap":
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=red_cfg["umap"]["n_neighbors"],
            min_dist=red_cfg["umap"]["min_dist"],
            metric=red_cfg["umap"]["metric"],
            random_state=seed,
        )
        coords = reducer.fit_transform(X)
    elif method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=red_cfg["tsne"]["perplexity"],
            learning_rate=red_cfg["tsne"]["learning_rate"],
            random_state=seed,
        )
        coords = reducer.fit_transform(X)
    elif method == "pca":
        reducer = PCA(n_components=n_components, random_state=seed)
        coords = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unsupported reduction method: {method}")

    logger.info(f"Reduction complete: {coords.shape}")
    return coords


# ============================================================
# Multi-Method Comparison
# ============================================================

def compare_methods(
    X: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare multiple clustering methods on the same data."""
    methods = ["kmeans", "gmm", "hierarchical"]
    records = []

    for method in methods:
        labels = _run_clustering(X, method=method, n_clusters=n_clusters, seed=seed)
        metrics = _compute_cluster_metrics(X, labels)
        metrics["method"] = method
        records.append(metrics)

    return pd.DataFrame(records)
