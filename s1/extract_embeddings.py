"""
extract_embeddings.py - Extract patient embeddings from pretrained encoder.

Purpose:
  Load pretrained encoder checkpoint, run inference on all patients,
  save (N, d_model) embedding matrix. Also extract PCA baseline
  embeddings for fair comparison.

Connects to:
  - s1/encoder.py for model architecture
  - data/s1/checkpoints/pretrain_best.pt
  - s0/processed/ for data

Expected output artifacts:
  data/s1/embeddings_ss.npy   (N, 128)
  data/s1/embeddings_pca.npy  (N, 32)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from s1.encoder import ICUTransformerEncoder

logger = logging.getLogger("s1.extract")


def extract_ss_embeddings(
    s0_dir: Path,
    s1_dir: Path,
    device: str = "cpu",
    batch_size: int = 128,
) -> np.ndarray:
    """
    Extract self-supervised embeddings for all patients.

    Returns (N, d_model) array.
    """
    s0_dir = Path(s0_dir)
    s1_dir = Path(s1_dir)

    # Load checkpoint
    ckpt_path = s1_dir / "checkpoints" / "pretrain_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run pretraining first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = ckpt["config"]

    # Build encoder
    encoder = ICUTransformerEncoder(
        n_features=config["n_features"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        dropout=0.0,  # No dropout at inference
    ).to(device)

    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.train(False)

    # Load data
    continuous = np.load(s0_dir / "processed" / "continuous.npy")
    masks = np.load(s0_dir / "processed" / "masks_continuous.npy")
    n_patients = continuous.shape[0]

    logger.info(f"Extracting SS embeddings for {n_patients} patients (d_model={config['d_model']})...")

    all_embeddings = []
    with torch.no_grad():
        for start in range(0, n_patients, batch_size):
            end = min(start + batch_size, n_patients)
            x = torch.from_numpy(continuous[start:end]).float().to(device)
            mask = torch.from_numpy(masks[start:end]).float().to(device)
            emb = encoder(x, mask)  # (B, d_model)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, d_model)

    out_path = s1_dir / "embeddings_ss.npy"
    np.save(out_path, embeddings)
    logger.info(f"SS embeddings saved: {embeddings.shape} → {out_path}")

    return embeddings


def extract_pca_embeddings(
    s0_dir: Path,
    s1_dir: Path,
    n_components: int = 32,
) -> np.ndarray:
    """
    Extract PCA baseline embeddings using S0 processed data.

    Applies the same feature engineering as V1:
      processed continuous (N, 48, 21) → statistical features → PCA

    For fair comparison, uses the SAME data as the SS encoder.
    """
    s0_dir = Path(s0_dir)
    s1_dir = Path(s1_dir)

    continuous = np.load(s0_dir / "processed" / "continuous.npy")
    n_patients, n_hours, n_features = continuous.shape

    logger.info(f"Extracting PCA embeddings for {n_patients} patients...")

    # Statistical features (simplified version of V1 feature_engineering)
    features = []
    for win_hours in [12, 24, 48]:
        win_start = max(0, n_hours - win_hours)
        window = continuous[:, win_start:, :]  # (N, win, F)

        features.append(np.nanmean(window, axis=1))
        features.append(np.nanstd(window, axis=1))
        features.append(np.nanmin(window, axis=1))
        features.append(np.nanmax(window, axis=1))
        # Trend (slope)
        t = np.arange(window.shape[1], dtype=float)
        t_mean = t.mean()
        t_var = np.var(t) + 1e-8
        y_mean = np.nanmean(window, axis=1, keepdims=True)
        cov = np.nanmean((t[None, :, None] - t_mean) * (window - y_mean), axis=1)
        features.append(cov / t_var)
        # Last value
        features.append(window[:, -1, :])

    feature_matrix = np.concatenate(features, axis=1)  # (N, 6*3*21 = 378)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

    # Standardize + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    n_comp = min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    embeddings = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA: {n_comp} components, {explained:.1%} variance explained")

    out_path = s1_dir / "embeddings_pca.npy"
    np.save(out_path, embeddings)
    logger.info(f"PCA embeddings saved: {embeddings.shape} → {out_path}")

    return embeddings
1