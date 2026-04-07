"""
saits_imputation.py - Integrate PyPOTS SAITS into the S6 optimization pipeline.

This module fits a lightweight SAITS imputer on processed physiology tensors and
derives patient-level imputation features that can be appended to causal
covariates. It keeps runtime bounded by allowing a configurable fit subset.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("s6.saits")


def _prepare_pypots_home(cache_root: Path | None = None) -> Path:
    cache_root = (cache_root or (Path.cwd() / ".cache" / "pypots_home")).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(cache_root)
    xdg_cache = cache_root / "xdg_cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    return cache_root


def _masked_nan_tensor(continuous: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.where(masks > 0.5, continuous, np.nan).astype(np.float32, copy=False)


def _safe_nanmean(arr: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    valid = np.isfinite(arr)
    counts = valid.sum(axis=axis)
    totals = np.nansum(arr, axis=axis)
    return np.divide(
        totals,
        counts,
        out=np.zeros_like(totals, dtype=np.float32),
        where=counts > 0,
    )


def run_saits_imputation(
    *,
    continuous: np.ndarray,
    masks: np.ndarray,
    output_dir: Path,
    config: dict | None = None,
) -> dict:
    """
    Fit SAITS on processed physiology and return imputed tensor plus summaries.
    """
    cfg = {
        "enabled": True,
        "fit_patients": 2048,
        "epochs": 2,
        "batch_size": 64,
        "n_layers": 2,
        "d_model": 64,
        "n_heads": 4,
        "d_ffn": 128,
        "patience": 1,
        "device": "cpu",
    }
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    if not cfg.get("enabled", True):
        return {
            "enabled": False,
            "imputed": continuous.astype(np.float32, copy=False),
            "features_df": pd.DataFrame(index=np.arange(len(continuous))),
            "summary": {"enabled": False},
        }

    _prepare_pypots_home(output_dir / ".pypots_home")
    from pypots.imputation import SAITS

    output_dir = Path(output_dir)
    cache_dir = output_dir / "saits"
    cache_dir.mkdir(parents=True, exist_ok=True)

    X_full = _masked_nan_tensor(continuous, masks)
    fit_n = min(int(cfg["fit_patients"]), len(X_full))
    X_fit = X_full[:fit_n]

    logger.info(
        "Running SAITS imputation on processed physiology: fit_n=%d/%d, shape=%s",
        fit_n,
        len(X_full),
        X_full.shape,
    )
    model = SAITS(
        n_steps=int(X_full.shape[1]),
        n_features=int(X_full.shape[2]),
        n_layers=int(cfg["n_layers"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        d_k=max(1, int(cfg["d_model"]) // int(cfg["n_heads"])),
        d_v=max(1, int(cfg["d_model"]) // int(cfg["n_heads"])),
        d_ffn=int(cfg["d_ffn"]),
        batch_size=int(cfg["batch_size"]),
        epochs=int(cfg["epochs"]),
        patience=int(cfg["patience"]),
        num_workers=0,
        device=cfg["device"],
        saving_path=None,
        verbose=False,
    )
    model.fit({"X": X_fit})
    pred = model.predict({"X": X_full})

    imputed = pred["imputation"].astype(np.float32, copy=False)
    combining = pred.get("combining_weights")
    reconstruction = pred.get("reconstruction", imputed)

    missing_mask = (masks < 0.5)
    observed_mask = ~missing_mask
    missing_fraction = missing_mask.mean(axis=(1, 2)).astype(np.float32)
    missing_abs_impute = np.where(missing_mask, np.abs(imputed), np.nan)
    observed_recon_error = np.where(observed_mask, np.abs(reconstruction - continuous), np.nan)

    features = {
        "saits_missing_fraction": missing_fraction,
        "saits_imputed_abs_mean": _safe_nanmean(missing_abs_impute, axis=(1, 2)),
        "saits_observed_recon_error": _safe_nanmean(observed_recon_error, axis=(1, 2)),
    }
    if combining is not None:
        features["saits_combining_weight_mean"] = combining.mean(axis=(1, 2))
        features["saits_combining_weight_std"] = combining.std(axis=(1, 2))

    features_df = pd.DataFrame(features).fillna(0.0)
    imputed_path = cache_dir / "processed_imputed.npy"
    features_path = cache_dir / "patient_features.csv"
    summary_path = cache_dir / "summary.json"

    np.save(imputed_path, imputed)
    features_df.to_csv(features_path, index=False)

    summary = {
        "enabled": True,
        "fit_patients": fit_n,
        "n_patients": int(len(X_full)),
        "epochs": int(cfg["epochs"]),
        "batch_size": int(cfg["batch_size"]),
        "mean_missing_fraction": round(float(missing_fraction.mean()), 4),
        "mean_imputed_abs_mean": round(float(features_df["saits_imputed_abs_mean"].mean()), 4),
        "mean_observed_recon_error": round(float(features_df["saits_observed_recon_error"].mean()), 4),
        "artifacts": {
            "imputed": str(imputed_path),
            "patient_features": str(features_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["artifacts"]["summary"] = str(summary_path)

    return {
        "enabled": True,
        "imputed": imputed,
        "features_df": features_df,
        "summary": summary,
    }
