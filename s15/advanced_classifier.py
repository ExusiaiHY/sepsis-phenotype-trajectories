"""
advanced_classifier.py - Stronger downstream mortality models using multi-view features.

Purpose:
  Build richer feature views from the existing S0 tensors and S1.5 embeddings,
  then train stronger downstream classifiers than the embedding-only linear probe.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("s15.advanced_classifier")

STATIC_NUMERIC_COLUMNS = ["age", "height_cm", "weight_kg"]
STATIC_BINARY_COLUMNS = ["sex_male"]
WINDOW_HOURS = (12, 24, 48)


def build_feature_views(
    s0_dir: Path,
    *,
    embeddings_path: Path | None = None,
    label_col: str = "mortality_inhospital",
) -> dict:
    """
    Build reusable downstream feature views from the full S0 data bundle.

    Returns a dict with:
      - labels
      - views: {view_name: array}
      - feature_names: {view_name: list[str]}
    """
    s0_dir = Path(s0_dir)
    static = pd.read_csv(s0_dir / "static.csv")
    labels = static[label_col].fillna(0).astype(int).to_numpy()

    continuous = np.load(s0_dir / "processed" / "continuous.npy")
    masks = np.load(s0_dir / "processed" / "masks_continuous.npy")
    proxy = np.load(s0_dir / "processed" / "proxy_indicators.npy")

    stats, stat_names = _build_statistical_features(continuous)
    mask_summary, mask_names = _build_mask_features(masks)
    proxy_summary, proxy_names = _build_proxy_features(proxy)
    static_feats, static_names = _build_static_features(static)

    views = {
        "stats_mask_proxy_static": np.concatenate(
            [stats, mask_summary, proxy_summary, static_feats],
            axis=1,
        ),
    }
    feature_names = {
        "stats_mask_proxy_static": stat_names + mask_names + proxy_names + static_names,
    }

    if embeddings_path is not None:
        embeddings = np.load(embeddings_path)
        emb_names = [f"emb_{i}" for i in range(embeddings.shape[1])]
        views["embeddings"] = embeddings
        feature_names["embeddings"] = emb_names

        views["embeddings_static"] = np.concatenate([embeddings, static_feats], axis=1)
        feature_names["embeddings_static"] = emb_names + static_names

        views["fused_all"] = np.concatenate(
            [embeddings, stats, mask_summary, proxy_summary, static_feats],
            axis=1,
        )
        feature_names["fused_all"] = emb_names + stat_names + mask_names + proxy_names + static_names

    return {
        "labels": labels,
        "views": views,
        "feature_names": feature_names,
    }


def train_advanced_mortality_classifier(
    *,
    s0_dir: Path,
    splits_path: Path,
    output_dir: Path,
    embeddings_path: Path | None = None,
    label_col: str = "mortality_inhospital",
    model_type: str = "hgb",
    feature_set: str = "stats_mask_proxy_static",
    threshold_metric: str = "balanced_accuracy",
    seed: int = 42,
    hgb_max_depth: int = 5,
    hgb_learning_rate: float = 0.05,
    hgb_max_iter: int = 300,
    ensemble_weight_step: float = 0.05,
) -> dict:
    """
    Train an advanced downstream mortality classifier using richer feature views.

    Supported model types:
      - logreg
      - hgb
      - hgb_ensemble
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_bundle = build_feature_views(
        s0_dir=Path(s0_dir),
        embeddings_path=embeddings_path,
        label_col=label_col,
    )
    labels = feature_bundle["labels"]

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    split_arrays = {name: np.asarray(splits[name], dtype=int) for name in ("train", "val", "test")}

    if model_type == "hgb_ensemble":
        required = ("fused_all", "stats_mask_proxy_static")
        for name in required:
            if name not in feature_bundle["views"]:
                raise ValueError(f"Feature view '{name}' is required for hgb_ensemble")
        report, model_bundle = _train_hgb_ensemble(
            labels=labels,
            split_arrays=split_arrays,
            feature_views={name: feature_bundle["views"][name] for name in required},
            threshold_metric=threshold_metric,
            max_depth=hgb_max_depth,
            learning_rate=hgb_learning_rate,
            max_iter=hgb_max_iter,
            weight_step=ensemble_weight_step,
            seed=seed,
        )
    else:
        if feature_set not in feature_bundle["views"]:
            raise ValueError(
                f"Unknown feature_set '{feature_set}'. Available: {sorted(feature_bundle['views'])}"
            )
        report, model_bundle = _train_single_model(
            x=feature_bundle["views"][feature_set],
            labels=labels,
            split_arrays=split_arrays,
            model_type=model_type,
            threshold_metric=threshold_metric,
            max_depth=hgb_max_depth,
            learning_rate=hgb_learning_rate,
            max_iter=hgb_max_iter,
            seed=seed,
        )
        report["feature_set"] = feature_set
        report["n_features"] = int(feature_bundle["views"][feature_set].shape[1])

    report["label_col"] = label_col
    report["model_type"] = model_type

    model_path = output_dir / "advanced_mortality_classifier.pkl"
    report_path = output_dir / "advanced_mortality_classifier_report.json"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved advanced classifier: %s", model_path)
    logger.info("Saved advanced report: %s", report_path)
    return report


def _build_statistical_features(continuous: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n_patients, n_hours, n_features = continuous.shape
    features = []
    names = []

    for win_hours in WINDOW_HOURS:
        win_start = max(0, n_hours - win_hours)
        window = continuous[:, win_start:, :]

        reducers = {
            "mean": np.nanmean(window, axis=1),
            "std": np.nanstd(window, axis=1),
            "min": np.nanmin(window, axis=1),
            "max": np.nanmax(window, axis=1),
            "last": window[:, -1, :],
        }
        for key, arr in reducers.items():
            features.append(arr)
            names.extend([f"cont_w{win_hours}_{key}_{i}" for i in range(n_features)])

        t = np.arange(window.shape[1], dtype=float)
        t_mean = t.mean()
        t_var = np.var(t) + 1e-8
        y_mean = np.nanmean(window, axis=1, keepdims=True)
        cov = np.nanmean((t[None, :, None] - t_mean) * (window - y_mean), axis=1)
        slope = cov / t_var
        features.append(slope)
        names.extend([f"cont_w{win_hours}_slope_{i}" for i in range(n_features)])

    return np.concatenate(features, axis=1), names


def _build_mask_features(masks: np.ndarray) -> tuple[np.ndarray, list[str]]:
    feature_blocks = [
        ("mask_full", masks.mean(axis=1)),
        ("mask_first24", masks[:, :24, :].mean(axis=1)),
        ("mask_last24", masks[:, 24:, :].mean(axis=1)),
    ]

    arrays = []
    names = []
    for prefix, arr in feature_blocks:
        arrays.append(arr)
        names.extend([f"{prefix}_{i}" for i in range(arr.shape[1])])

    return np.concatenate(arrays, axis=1), names


def _build_proxy_features(proxy: np.ndarray) -> tuple[np.ndarray, list[str]]:
    feature_blocks = [
        ("proxy_mean_full", proxy.mean(axis=1)),
        ("proxy_max_full", proxy.max(axis=1)),
        ("proxy_mean_first24", proxy[:, :24, :].mean(axis=1)),
        ("proxy_mean_last24", proxy[:, 24:, :].mean(axis=1)),
    ]

    arrays = []
    names = []
    for prefix, arr in feature_blocks:
        arrays.append(arr)
        names.extend([f"{prefix}_{i}" for i in range(arr.shape[1])])

    return np.concatenate(arrays, axis=1), names


def _build_static_features(static: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    frame = static.copy()
    frame["sex_male"] = (frame["sex"] == "male").astype(float)

    icu_cols = []
    for icu_type in sorted(frame["icu_type"].dropna().unique()):
        col = f"icu_type_{int(icu_type)}"
        frame[col] = (frame["icu_type"] == icu_type).astype(float)
        icu_cols.append(col)

    names = STATIC_NUMERIC_COLUMNS + STATIC_BINARY_COLUMNS + icu_cols
    return frame[names].to_numpy(dtype=float), names


def _make_model(
    model_type: str,
    *,
    seed: int,
    max_depth: int,
    learning_rate: float,
    max_iter: int,
) -> Pipeline:
    if model_type == "logreg":
        return Pipeline([
            ("imp", SimpleImputer()),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
                random_state=seed,
            )),
        ])
    if model_type == "hgb":
        return Pipeline([
            ("imp", SimpleImputer()),
            ("clf", HistGradientBoostingClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=seed,
            )),
        ])
    raise ValueError(f"Unsupported model_type: {model_type}")


def _train_single_model(
    *,
    x: np.ndarray,
    labels: np.ndarray,
    split_arrays: dict[str, np.ndarray],
    model_type: str,
    threshold_metric: str,
    max_depth: int,
    learning_rate: float,
    max_iter: int,
    seed: int,
) -> tuple[dict, dict]:
    model = _make_model(
        model_type,
        seed=seed,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
    )
    model.fit(x[split_arrays["train"]], labels[split_arrays["train"]])

    split_probs = {
        name: _predict_proba(model, x[idx])
        for name, idx in split_arrays.items()
    }
    threshold, search = _select_threshold(
        labels[split_arrays["val"]],
        split_probs["val"],
        metric_name=threshold_metric,
    )

    report = {
        "threshold_selection": {
            "metric": threshold_metric,
            "selected_threshold": round(float(threshold), 4),
            "search": search,
        },
        "splits": {},
        "baseline_accuracy": _baseline_accuracy(labels, split_arrays),
    }
    for name, idx in split_arrays.items():
        report["splits"][name] = _classification_metrics(
            labels[idx],
            split_probs[name],
            threshold,
        )

    model_bundle = {
        "kind": "single",
        "model_type": model_type,
        "model": model,
        "threshold": threshold,
    }
    return report, model_bundle


def _train_hgb_ensemble(
    *,
    labels: np.ndarray,
    split_arrays: dict[str, np.ndarray],
    feature_views: dict[str, np.ndarray],
    threshold_metric: str,
    max_depth: int,
    learning_rate: float,
    max_iter: int,
    weight_step: float,
    seed: int,
) -> tuple[dict, dict]:
    stats_model = _make_model(
        "hgb",
        seed=seed,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
    )
    fused_model = _make_model(
        "hgb",
        seed=seed,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
    )

    stats_model.fit(
        feature_views["stats_mask_proxy_static"][split_arrays["train"]],
        labels[split_arrays["train"]],
    )
    fused_model.fit(
        feature_views["fused_all"][split_arrays["train"]],
        labels[split_arrays["train"]],
    )

    stats_probs = {
        name: _predict_proba(stats_model, feature_views["stats_mask_proxy_static"][idx])
        for name, idx in split_arrays.items()
    }
    fused_probs = {
        name: _predict_proba(fused_model, feature_views["fused_all"][idx])
        for name, idx in split_arrays.items()
    }

    best = None
    weights = np.arange(0.0, 1.0 + 1e-9, weight_step)
    search = []
    for fused_weight in weights:
        blended_val = fused_weight * fused_probs["val"] + (1.0 - fused_weight) * stats_probs["val"]
        threshold, threshold_search = _select_threshold(
            labels[split_arrays["val"]],
            blended_val,
            metric_name=threshold_metric,
        )
        val_metrics = _classification_metrics(
            labels[split_arrays["val"]],
            blended_val,
            threshold,
        )
        metric_value = val_metrics[threshold_metric]
        search.append({
            "fused_weight": round(float(fused_weight), 4),
            "stats_weight": round(float(1.0 - fused_weight), 4),
            "threshold": round(float(threshold), 4),
            threshold_metric: metric_value,
        })
        if best is None or metric_value > best["metric_value"]:
            best = {
                "fused_weight": float(fused_weight),
                "stats_weight": float(1.0 - fused_weight),
                "threshold": float(threshold),
                "metric_value": float(metric_value),
                "threshold_search": threshold_search,
            }

    report = {
        "feature_set": "fused_all + stats_mask_proxy_static",
        "threshold_selection": {
            "metric": threshold_metric,
            "selected_threshold": round(best["threshold"], 4),
            "search": best["threshold_search"],
        },
        "ensemble_selection": {
            "metric": threshold_metric,
            "selected_fused_weight": round(best["fused_weight"], 4),
            "selected_stats_weight": round(best["stats_weight"], 4),
            "search": search,
        },
        "splits": {},
        "baseline_accuracy": _baseline_accuracy(labels, split_arrays),
    }

    for name, idx in split_arrays.items():
        blended = (
            best["fused_weight"] * fused_probs[name] +
            best["stats_weight"] * stats_probs[name]
        )
        report["splits"][name] = _classification_metrics(
            labels[idx],
            blended,
            best["threshold"],
        )

    model_bundle = {
        "kind": "ensemble",
        "model_type": "hgb_ensemble",
        "stats_model": stats_model,
        "fused_model": fused_model,
        "fused_weight": best["fused_weight"],
        "stats_weight": best["stats_weight"],
        "threshold": best["threshold"],
    }
    return report, model_bundle


def _predict_proba(model: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    raw = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-raw))


def _baseline_accuracy(labels: np.ndarray, split_arrays: dict[str, np.ndarray]) -> dict[str, float]:
    majority_label = int(round(float(labels[split_arrays["train"]].mean())))
    return {
        name: round(float(np.mean(labels[idx] == majority_label)), 4)
        for name, idx in split_arrays.items()
    }


def _select_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    metric_name: str,
) -> tuple[float, list[dict]]:
    if len(np.unique(y_true)) < 2:
        return 0.5, [{"threshold": 0.5, metric_name: None, "note": "single class in validation"}]

    candidates = np.linspace(0.05, 0.95, 37)
    best_threshold = 0.5
    best_score = -np.inf
    search = []

    for threshold in candidates:
        preds = (probs >= threshold).astype(int)
        score = _threshold_metric(y_true, preds, metric_name)
        search.append({
            "threshold": round(float(threshold), 4),
            metric_name: round(float(score), 4),
        })
        if score > best_score or (np.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, search


def _threshold_metric(y_true: np.ndarray, preds: np.ndarray, metric_name: str) -> float:
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, preds))
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, preds))
    if metric_name == "f1":
        return float(f1_score(y_true, preds, zero_division=0))
    raise ValueError(f"Unsupported metric: {metric_name}")


def _classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "n_samples": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "predicted_positive_rate": round(float(np.mean(preds)), 4),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "auroc": round(float(roc_auc_score(y_true, probs)), 4) if len(np.unique(y_true)) >= 2 else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
