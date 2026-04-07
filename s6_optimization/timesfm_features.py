"""
timesfm_features.py - Optional TimesFM-driven dynamic feature extraction for S6.

This module keeps TimesFM integration non-blocking:
  - if disabled, the pipeline behaves exactly as before
  - if the package/checkpoint is unavailable, the pipeline records a skip reason
  - if TimesFM is available, it appends forecast-derived patient features

The extracted features are intentionally compact and patient-level so they can be
concatenated onto the existing causal covariate matrix without a large refactor.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("s6.timesfm")


def _safe_nanmean(arr: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    valid = np.isfinite(arr)
    counts = valid.sum(axis=axis)
    totals = np.nansum(arr, axis=axis)
    return np.divide(
        totals,
        counts,
        out=np.zeros_like(totals, dtype=np.float32),
        where=counts > 0,
    )


def _filter_kwargs(factory, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}


def _call_with_supported_kwargs(factory, kwargs: dict):
    return factory(**_filter_kwargs(factory, kwargs))


def _candidate_repo_ids(repo_id: str | None) -> list[str | None]:
    if not repo_id:
        return [None]
    candidates = [repo_id]
    if not repo_id.endswith(("-pytorch", "-jax", "-flax", "-transformers")):
        candidates.append(f"{repo_id}-pytorch")
    deduped = []
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _maybe_load_checkpoint(model, cfg: dict) -> None:
    if not hasattr(model, "load_from_checkpoint"):
        return

    repo_id = cfg.get("repo_id")
    checkpoint_path = cfg.get("checkpoint_path")
    candidates = [
        {"repo_id": repo_id},
        {"huggingface_repo_id": repo_id},
        {"path": checkpoint_path},
        {"checkpoint_path": checkpoint_path},
        {"repo_id": repo_id, "checkpoint_path": checkpoint_path},
        {"huggingface_repo_id": repo_id, "path": checkpoint_path},
    ]
    last_error = None
    for kwargs in candidates:
        kwargs = {k: v for k, v in kwargs.items() if v}
        if not kwargs:
            continue
        try:
            model.load_from_checkpoint(**_filter_kwargs(model.load_from_checkpoint, kwargs))
            return
        except TypeError as exc:
            last_error = exc
            continue
    if repo_id or checkpoint_path:
        raise RuntimeError(f"Unable to load TimesFM checkpoint with provided config: {last_error}")


def _load_timesfm_model(cfg: dict):
    timesfm_mod = importlib.import_module("timesfm")

    hparams_cls = getattr(timesfm_mod, "TimesFmHparams", None)
    checkpoint_cls = getattr(timesfm_mod, "TimesFmCheckpoint", None)
    model_cls = getattr(timesfm_mod, "TimesFm", None) or getattr(timesfm_mod, "TimesFM", None)

    if model_cls is None:
        raise RuntimeError("timesfm package imported but no TimesFm/TimesFM class was found")

    if hparams_cls is not None and checkpoint_cls is not None:
        hparams = _call_with_supported_kwargs(
            hparams_cls,
            {
                "backend": cfg.get("backend"),
                "horizon_len": int(cfg["horizon_len"]),
                "context_len": int(cfg["context_len"]),
                "per_core_batch_size": int(cfg["batch_size"]),
            },
        )
        checkpoint_path = cfg.get("checkpoint_path")
        last_error = None
        for repo_id in _candidate_repo_ids(cfg.get("repo_id")):
            checkpoint = _call_with_supported_kwargs(
                checkpoint_cls,
                {
                    "huggingface_repo_id": repo_id,
                    "path": checkpoint_path,
                    "checkpoint_path": checkpoint_path,
                },
            )
            try:
                try:
                    return model_cls(hparams=hparams, checkpoint=checkpoint)
                except TypeError:
                    return model_cls(hparams, checkpoint)
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                last_error = exc
                logger.warning("TimesFM checkpoint candidate failed (%s): %s", repo_id, exc)
                continue
        raise RuntimeError(f"Unable to initialize TimesFM model: {last_error}")

    model = _call_with_supported_kwargs(
        model_cls,
        {
            "context_len": int(cfg["context_len"]),
            "horizon_len": int(cfg["horizon_len"]),
            "backend": cfg.get("backend"),
            "per_core_batch_size": int(cfg["batch_size"]),
        },
    )
    _maybe_load_checkpoint(model, cfg)
    return model


def _normalize_forecast_array(forecast_output, n_rows: int, horizon_len: int) -> np.ndarray:
    if isinstance(forecast_output, dict):
        for key in ("point_forecast", "forecast", "mean"):
            if key in forecast_output:
                forecast_output = forecast_output[key]
                break
    elif isinstance(forecast_output, (tuple, list)) and forecast_output:
        forecast_output = forecast_output[0]

    forecast = np.asarray(forecast_output, dtype=np.float32)
    if forecast.ndim == 1:
        forecast = forecast[:, None]
    if forecast.ndim != 2:
        raise RuntimeError(f"Unexpected TimesFM forecast shape: {forecast.shape}")
    if forecast.shape[0] != n_rows:
        raise RuntimeError(
            f"TimesFM returned {forecast.shape[0]} series for {n_rows} inputs"
        )
    if forecast.shape[1] < horizon_len:
        pad = np.repeat(forecast[:, -1:], horizon_len - forecast.shape[1], axis=1)
        forecast = np.concatenate([forecast, pad], axis=1)
    return forecast[:, :horizon_len]


def _forecast_in_batches(
    model,
    context: np.ndarray,
    horizon_len: int,
    batch_size: int,
    freq: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(context), batch_size):
        batch = context[start:start + batch_size]
        inputs = [row.astype(np.float32, copy=False) for row in batch]
        try:
            result = model.forecast(inputs=inputs, freq=[freq] * len(inputs))
        except TypeError:
            result = model.forecast(inputs, [freq] * len(inputs))
        outputs.append(_normalize_forecast_array(result, len(batch), horizon_len))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, horizon_len), dtype=np.float32)


def _fill_context(series: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    filled = np.asarray(series, dtype=np.float32).copy()
    observed_mask = np.asarray(observed_mask, dtype=bool)
    for i in range(filled.shape[0]):
        row = filled[i]
        row_mask = observed_mask[i] & np.isfinite(row)
        if not row_mask.any():
            filled[i] = 0.0
            continue
        first_idx = int(np.flatnonzero(row_mask)[0])
        row[:first_idx] = row[first_idx]
        last_val = row[first_idx]
        for t in range(first_idx, len(row)):
            if row_mask[t]:
                last_val = row[t]
            else:
                row[t] = last_val
        row[~np.isfinite(row)] = last_val
        filled[i] = row
    return filled


def run_timesfm_feature_extraction(
    *,
    continuous: np.ndarray,
    masks: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    config: dict | None = None,
    imputed_continuous: np.ndarray | None = None,
) -> dict:
    cfg = {
        "enabled": False,
        "context_len": 32,
        "horizon_len": 6,
        "batch_size": 128,
        "backend": "cpu",
        "freq": 0,
        "input_source": "imputed",
        "selected_features": ["map", "heart_rate", "lactate", "creatinine"],
        "repo_id": "google/timesfm-1.0-200m",
        "checkpoint_path": None,
    }
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})

    output_dir = Path(output_dir)
    artifact_dir = output_dir / "timesfm"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_path = artifact_dir / "patient_features.csv"
    summary_path = artifact_dir / "summary.json"

    summary = {
        "requested": bool(cfg.get("enabled", False)),
        "enabled": False,
        "context_len": int(cfg["context_len"]),
        "horizon_len": int(cfg["horizon_len"]),
        "batch_size": int(cfg["batch_size"]),
        "backend": cfg.get("backend"),
        "repo_id": cfg.get("repo_id"),
        "artifacts": {
            "patient_features": str(feature_path),
            "summary": str(summary_path),
        },
    }

    empty_df = pd.DataFrame(index=np.arange(len(continuous)))
    if not cfg.get("enabled", False):
        summary["reason"] = "disabled_by_config"
        empty_df.to_csv(feature_path, index=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return {
            "enabled": False,
            "features_df": empty_df,
            "summary": summary,
        }

    if importlib.util.find_spec("timesfm") is None:
        summary["reason"] = "timesfm_not_installed"
        empty_df.to_csv(feature_path, index=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return {
            "enabled": False,
            "features_df": empty_df,
            "summary": summary,
        }

    try:
        idx = {name: i for i, name in enumerate(feature_names)}
        selected_features = [name for name in cfg["selected_features"] if name in idx]
        if not selected_features:
            raise ValueError("No requested TimesFM features exist in feature_names")

        data_source = continuous
        source_name = "processed"
        if cfg.get("input_source", "imputed") == "imputed" and imputed_continuous is not None:
            data_source = imputed_continuous
            source_name = "processed_imputed"

        total_steps = data_source.shape[1]
        context_len = min(int(cfg["context_len"]), total_steps)
        horizon_len = min(int(cfg["horizon_len"]), max(1, total_steps - context_len))
        if total_steps <= 1:
            raise ValueError("TimesFM requires at least 2 time steps")

        model = _load_timesfm_model(
            {
                **cfg,
                "context_len": context_len,
                "horizon_len": horizon_len,
            }
        )

        feature_dict: dict[str, np.ndarray] = {}
        for feature_name in selected_features:
            fi = idx[feature_name]
            series = np.asarray(data_source[:, :context_len + horizon_len, fi], dtype=np.float32)
            context = series[:, :context_len]
            context_mask = masks[:, :context_len, fi] > 0.5
            filled_context = _fill_context(context, context_mask)
            forecast = _forecast_in_batches(
                model=model,
                context=filled_context,
                horizon_len=horizon_len,
                batch_size=int(cfg["batch_size"]),
                freq=int(cfg.get("freq", 0)),
            )

            last_context = filled_context[:, -1]
            feature_dict[f"timesfm_{feature_name}_forecast_mean"] = forecast.mean(axis=1)
            feature_dict[f"timesfm_{feature_name}_forecast_std"] = forecast.std(axis=1)
            feature_dict[f"timesfm_{feature_name}_forecast_delta"] = forecast.mean(axis=1) - last_context
            feature_dict[f"timesfm_{feature_name}_forecast_slope"] = forecast[:, -1] - forecast[:, 0]

            target = np.asarray(
                continuous[:, context_len:context_len + horizon_len, fi],
                dtype=np.float32,
            )
            target_mask = masks[:, context_len:context_len + horizon_len, fi] > 0.5
            target_observed = np.where(target_mask, target, np.nan)
            feature_dict[f"timesfm_{feature_name}_target_coverage"] = target_mask.mean(axis=1).astype(np.float32)
            feature_dict[f"timesfm_{feature_name}_forecast_mae"] = _safe_nanmean(
                np.abs(target_observed - forecast),
                axis=1,
            )
            feature_dict[f"timesfm_{feature_name}_forecast_bias"] = _safe_nanmean(
                target_observed - forecast,
                axis=1,
            )

        features_df = pd.DataFrame(feature_dict).fillna(0.0)
        features_df.to_csv(feature_path, index=False)

        summary.update(
            {
                "enabled": True,
                "source": source_name,
                "n_patients": int(len(data_source)),
                "n_features_used": int(len(selected_features)),
                "feature_names": selected_features,
                "n_output_columns": int(features_df.shape[1]),
                "mean_abs_feature_value": round(float(np.abs(features_df.to_numpy()).mean()), 4),
            }
        )
    except Exception as exc:
        logger.warning("TimesFM feature extraction skipped after failure: %s", exc)
        features_df = empty_df
        features_df.to_csv(feature_path, index=False)
        summary.update(
            {
                "reason": "timesfm_runtime_failure",
                "failed": True,
                "error": str(exc),
            }
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "enabled": bool(summary.get("enabled", False)),
        "features_df": features_df,
        "summary": summary,
    }
