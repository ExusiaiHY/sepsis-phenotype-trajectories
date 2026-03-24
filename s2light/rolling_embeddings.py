"""
rolling_embeddings.py - Extract per-window embeddings using frozen S1.5 encoder.

Produces (N, W, D) tensor where W = number of rolling windows per patient.
The encoder is loaded from checkpoint and never trained further.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from s1.encoder import ICUTransformerEncoder

logger = logging.getLogger("s2light.rolling")


def extract_rolling_embeddings(
    s0_dir: Path,
    encoder_ckpt: Path,
    output_path: Path,
    window_len: int = 24,
    stride: int = 6,
    seq_len: int = 48,
    device: str = "cpu",
    batch_size: int = 128,
) -> tuple[np.ndarray, dict]:
    """
    Extract rolling-window embeddings for all patients.

    Returns:
        embeddings: (N, n_windows, d_model) array
        meta: dict with shapes, window positions, quality stats
    """
    s0_dir = Path(s0_dir)

    # Compute window start positions
    starts = list(range(0, seq_len - window_len + 1, stride))
    n_windows = len(starts)
    logger.info(f"Window config: len={window_len}h, stride={stride}h, "
                f"positions={starts}, n_windows={n_windows}")

    # Load encoder
    ckpt = torch.load(encoder_ckpt, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    encoder = ICUTransformerEncoder(
        n_features=cfg["n_features"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"], dropout=0.0,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.train(False)

    d_model = cfg["d_model"]

    # Load data lazily; large external cohorts should not fully materialize here.
    continuous = np.load(s0_dir / "processed" / "continuous.npy", mmap_mode="r")
    masks = np.load(s0_dir / "processed" / "masks_continuous.npy", mmap_mode="r")
    n_patients = continuous.shape[0]

    # Per-window observation density (before encoding)
    window_obs_density = np.zeros((n_patients, n_windows), dtype=np.float32)
    for wi, start in enumerate(starts):
        window_mask = masks[:, start:start + window_len, :]
        window_obs_density[:, wi] = window_mask.mean(axis=(1, 2))

    # Extract embeddings directly into an on-disk array to keep memory bounded.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_embeddings = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_patients, n_windows, d_model),
    )

    with torch.no_grad():
        for wi, start in enumerate(starts):
            logger.info(f"  Window {wi} [{start}, {start + window_len})...")
            for batch_start in range(0, n_patients, batch_size):
                batch_end = min(batch_start + batch_size, n_patients)
                x_np = np.array(
                    continuous[batch_start:batch_end, start:start + window_len, :],
                    dtype=np.float32,
                    copy=True,
                )
                m_np = np.array(
                    masks[batch_start:batch_end, start:start + window_len, :],
                    dtype=np.float32,
                    copy=True,
                )
                x = torch.from_numpy(x_np).to(device)
                m = torch.from_numpy(m_np).to(device)
                emb = encoder(x, m)
                all_embeddings[batch_start:batch_end, wi, :] = emb.cpu().numpy()

    all_embeddings.flush()

    meta = {
        "n_patients": n_patients,
        "n_windows": n_windows,
        "d_model": d_model,
        "window_len": window_len,
        "stride": stride,
        "window_starts": starts,
        "shape": list(all_embeddings.shape),
        "window_obs_density_mean": window_obs_density.mean(axis=0).tolist(),
        "window_obs_density_std": window_obs_density.std(axis=0).tolist(),
    }

    logger.info(f"Rolling embeddings saved: {all_embeddings.shape} -> {output_path}")
    return np.load(output_path, mmap_mode="r"), meta
