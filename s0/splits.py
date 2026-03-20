"""
splits.py - Center-aware train/val/test splitting.

Purpose:
  Generate reproducible splits with center-aware logic:
  - random: stratified random split across all patients
  - cross_center: train+val on center_a, test on center_b

Connects to:
  - static.csv for center_id and mortality labels
  - scripts/s0_prepare.py calls build_splits()

Expected output artifacts:
  data/s0/splits.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("s0.splits")


def build_splits(
    static_path: Path,
    output_path: Path,
    method: str = "cross_center",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    stratify_by: str = "mortality_inhospital",
) -> dict:
    """
    Generate train/val/test splits and save to JSON.

    Returns dict with split indices and metadata.
    """
    static = pd.read_csv(static_path)
    n = len(static)

    if method == "cross_center":
        splits = _cross_center_split(static, val_ratio, seed, stratify_by)
    elif method == "random":
        splits = _random_split(static, train_ratio, val_ratio, seed, stratify_by)
    else:
        raise ValueError(f"Unknown split method: {method}")

    # Log statistics
    for name, indices in splits.items():
        subset = static.iloc[indices]
        mort_col = stratify_by
        if mort_col in subset.columns:
            mort_rate = subset[mort_col].mean()
        else:
            mort_rate = float("nan")
        logger.info(f"  {name}: {len(indices)} patients, mortality={mort_rate:.1%}")

    # Save
    output = {
        "method": method,
        "seed": seed,
        "train": [int(i) for i in splits["train"]],
        "val": [int(i) for i in splits["val"]],
        "test": [int(i) for i in splits["test"]],
        "metadata": {
            "n_total": n,
            "sizes": {k: len(v) for k, v in splits.items()},
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Splits saved to {output_path}")
    return output


def _cross_center_split(static, val_ratio, seed, stratify_by):
    """Center A → train+val, Center B → test."""
    center_a = np.where(static["center_id"] == "center_a")[0]
    center_b = np.where(static["center_id"] == "center_b")[0]

    if len(center_b) == 0:
        logger.warning("No center_b patients found; falling back to random split")
        return _random_split(static, 0.7, 0.15, seed, stratify_by)

    # Split center_a into train/val
    rel_val = val_ratio / (1.0 - (1.0 - len(center_a) / len(static)))
    rel_val = min(rel_val, 0.3)

    stratify = None
    if stratify_by in static.columns:
        s = static.iloc[center_a][stratify_by]
        if s.nunique() > 1 and not s.isna().all():
            stratify = s.values

    train_idx, val_idx = train_test_split(
        center_a, test_size=rel_val, random_state=seed, stratify=stratify,
    )

    logger.info("Cross-center split: center_a → train+val, center_b → test")
    return {"train": train_idx, "val": val_idx, "test": center_b}


def _random_split(static, train_ratio, val_ratio, seed, stratify_by):
    """Stratified random split."""
    indices = np.arange(len(static))
    test_ratio = 1.0 - train_ratio - val_ratio

    stratify = None
    if stratify_by in static.columns:
        s = static[stratify_by]
        if s.nunique() > 1 and not s.isna().all():
            stratify = s.values

    train_val, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed, stratify=stratify,
    )

    rel_val = val_ratio / (train_ratio + val_ratio)
    strat_tv = stratify[train_val] if stratify is not None else None
    train_idx, val_idx = train_test_split(
        train_val, test_size=rel_val, random_state=seed, stratify=strat_tv,
    )

    return {"train": train_idx, "val": val_idx, "test": test_idx}
