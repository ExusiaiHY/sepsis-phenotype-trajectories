"""
diagnostics.py - Representation diagnostic probes and analysis.

Probes:
  - Mortality linear probe (AUROC)
  - Center linear probe (AUROC — lower is better)
  - ICU LOS linear probe (R²)
  - Feature correlation analysis
  - Missingness sensitivity analysis
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

logger = logging.getLogger("s15.diagnostics")


def run_all_diagnostics(
    embeddings: np.ndarray,
    static_path: Path,
    splits_path: Path,
    masks_path: Path,
    output_path: Path,
    label: str = "unknown",
) -> dict:
    """
    Run all diagnostic probes on a set of embeddings.

    Parameters
    ----------
    embeddings: (N, D) patient embeddings
    static_path: path to static.csv
    splits_path: path to splits.json
    masks_path: path to masks_continuous.npy
    output_path: path to save diagnostics report
    label: identifier for this embedding set (e.g., "pca", "s1_masked", "s15_contrastive")

    Returns dict with all probe results.
    """
    static = pd.read_csv(static_path)
    masks = np.load(masks_path)

    with open(splits_path) as f:
        splits = json.load(f)

    train_idx = np.array(splits["train"])
    test_idx = np.array(splits["test"])

    results = {"label": label, "embedding_shape": list(embeddings.shape)}

    # === Probe 1: Mortality ===
    logger.info(f"  [{label}] Mortality probe...")
    mortality = static["mortality_inhospital"].fillna(0).values
    results["mortality_probe"] = _binary_probe(
        embeddings, mortality, train_idx, test_idx, "mortality"
    )

    # === Probe 2: Center ===
    # Uses a SEPARATE random split mixing both centers in train and test.
    # The cross-center phenotyping split has single-class center labels,
    # which makes logistic regression degenerate (D008).
    logger.info(f"  [{label}] Center probe (random split)...")
    center = (static["center_id"] == "center_a").astype(int).values
    results["center_probe"] = _binary_probe_random_split(
        embeddings, center, seed=42, test_ratio=0.3, name="center"
    )

    # === Probe 3: LOS ===
    logger.info(f"  [{label}] LOS probe...")
    los = static["icu_los_hours"].fillna(48).values
    results["los_probe"] = _regression_probe(
        embeddings, los, train_idx, test_idx, "los"
    )

    # === Probe 4: Feature correlations ===
    logger.info(f"  [{label}] Feature correlations...")
    results["feature_correlations"] = _feature_correlations(embeddings, masks)

    # === Probe 5: Missingness sensitivity ===
    logger.info(f"  [{label}] Missingness sensitivity...")
    results["missingness_sensitivity"] = _missingness_sensitivity(embeddings, masks)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"  [{label}] Diagnostics saved: {output_path}")
    return results


def _binary_probe_random_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    test_ratio: float = 0.3,
    name: str = "",
) -> dict:
    """
    Train logistic regression probe using a random split that mixes all classes.
    Designed for center probe where the phenotyping split is single-class per split.
    """
    from sklearn.model_selection import train_test_split

    n = len(labels)
    unique, counts = np.unique(labels, return_counts=True)

    if len(unique) < 2:
        return {"auroc": float("nan"), "note": "single class in full data"}

    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed, stratify=labels,
    )

    # Report class balance
    train_balance = {int(u): int((labels[train_idx] == u).sum()) for u in unique}
    test_balance = {int(u): int((labels[test_idx] == u).sum()) for u in unique}

    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings[train_idx])
    X_test = scaler.transform(embeddings[test_idx])
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probs)

    return {
        "auroc": round(float(auroc), 4),
        "split": "random_stratified",
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_class_balance": train_balance,
        "test_class_balance": test_balance,
    }


def _binary_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    name: str,
) -> dict:
    """Train logistic regression probe, report AUROC."""
    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Check if both classes present
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"auroc": float("nan"), "note": "single class in train or test"}

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probs)

    return {"auroc": round(float(auroc), 4), "n_train": len(train_idx), "n_test": len(test_idx)}


def _regression_probe(
    embeddings: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    name: str,
) -> dict:
    """Train Ridge regression probe, report R²."""
    X_train = embeddings[train_idx]
    y_train = targets[train_idx]
    X_test = embeddings[test_idx]
    y_test = targets[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)

    preds = reg.predict(X_test)
    r2 = r2_score(y_test, preds)

    return {"r2": round(float(r2), 4), "n_train": len(train_idx), "n_test": len(test_idx)}


def _feature_correlations(
    embeddings: np.ndarray,
    masks: np.ndarray,
    top_k: int = 5,
) -> dict:
    """
    Compute Pearson correlation between each embedding dim and
    patient-level mean of each clinical variable.
    """
    from s0.schema import CONTINUOUS_NAMES

    n_patients, n_hours, n_features = masks.shape
    n_dims = embeddings.shape[1]

    # Patient-level observation rate per feature (not the values, just coverage)
    obs_rate = masks.mean(axis=1)  # (N, F) — fraction of hours observed per feature

    correlations = {}
    for f_idx, f_name in enumerate(CONTINUOUS_NAMES):
        feature_obs = obs_rate[:, f_idx]
        if feature_obs.std() < 1e-8:
            correlations[f_name] = {"top_dims": [], "note": "no variance"}
            continue

        dim_corrs = []
        for d in range(n_dims):
            emb_dim = embeddings[:, d]
            if emb_dim.std() < 1e-8:
                continue
            r, p = pearsonr(feature_obs, emb_dim)
            dim_corrs.append({"dim": d, "r": round(float(r), 4), "p": round(float(p), 6)})

        dim_corrs.sort(key=lambda x: abs(x["r"]), reverse=True)
        correlations[f_name] = {"top_dims": dim_corrs[:top_k]}

    return correlations


def _missingness_sensitivity(
    embeddings: np.ndarray,
    masks: np.ndarray,
) -> dict:
    """
    Check whether embedding properties correlate with observation density.
    """
    # Per-patient observation density
    obs_density = masks.mean(axis=(1, 2))  # (N,)

    # Embedding L2 norms
    norms = np.linalg.norm(embeddings, axis=1)

    r_norm, p_norm = pearsonr(obs_density, norms)

    # Correlation with first 3 principal components of embedding
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=42)
    pcs = pca.fit_transform(embeddings)

    pc_corrs = []
    for i in range(3):
        r, p = pearsonr(obs_density, pcs[:, i])
        pc_corrs.append({"pc": i, "r": round(float(r), 4), "p": round(float(p), 6)})

    return {
        "density_vs_norm": {"r": round(float(r_norm), 4), "p": round(float(p_norm), 6)},
        "density_vs_pcs": pc_corrs,
        "obs_density_mean": round(float(obs_density.mean()), 4),
        "obs_density_std": round(float(obs_density.std()), 4),
    }


def print_diagnostics_comparison(reports: list[dict]) -> None:
    """Print formatted comparison of diagnostic probes across methods."""
    print(f"\n{'='*72}")
    print(f"  Representation Diagnostics Comparison")
    print(f"{'='*72}")

    headers = [r["label"] for r in reports]
    print(f"  {'Probe':<30s}", end="")
    for h in headers:
        print(f"  {h:>16s}", end="")
    print()
    print(f"  {'-'*70}")

    # Mortality AUROC
    print(f"  {'Mortality AUROC':<30s}", end="")
    for r in reports:
        v = r.get("mortality_probe", {}).get("auroc", float("nan"))
        print(f"  {v:>16.4f}", end="")
    print()

    # Center AUROC (lower is better)
    print(f"  {'Center AUROC (↓ better)':<30s}", end="")
    for r in reports:
        v = r.get("center_probe", {}).get("auroc", float("nan"))
        print(f"  {v:>16.4f}", end="")
    print()

    # LOS R²
    print(f"  {'LOS R²':<30s}", end="")
    for r in reports:
        v = r.get("los_probe", {}).get("r2", float("nan"))
        print(f"  {v:>16.4f}", end="")
    print()

    # Missingness correlation
    print(f"  {'Density-Norm corr |r|':<30s}", end="")
    for r in reports:
        v = r.get("missingness_sensitivity", {}).get("density_vs_norm", {}).get("r", float("nan"))
        print(f"  {abs(v):>16.4f}", end="")
    print()

    print(f"{'='*72}")
