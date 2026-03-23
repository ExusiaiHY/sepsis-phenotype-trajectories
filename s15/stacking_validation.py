"""
stacking_validation.py - Validation utilities for the OOF stacking classifier.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from s15.advanced_classifier import _classification_metrics, build_feature_views
from s15.stacking_classifier import predict_stacking_probabilities


def validate_stacking_classifier(
    *,
    model_path: Path,
    s0_dir: Path,
    splits_path: Path,
    output_dir: Path,
    embeddings_path: Path,
    label_col: str | None = None,
    n_bootstrap: int = 500,
    n_bins: int = 10,
    permutation_repeats: int = 20,
    seed: int = 42,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    resolved_label_col = label_col or model_bundle.get("label_col", "mortality_inhospital")
    feature_bundle = build_feature_views(
        s0_dir=Path(s0_dir),
        embeddings_path=Path(embeddings_path),
        label_col=resolved_label_col,
    )
    labels = feature_bundle["labels"]

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    split_arrays = {name: np.asarray(splits[name], dtype=int) for name in ("train", "val", "test")}

    threshold = float(model_bundle["threshold"])
    split_probs = {}
    split_meta = {}
    split_metrics = {}
    for split_name, idx in split_arrays.items():
        probs, meta_x = predict_stacking_probabilities(model_bundle, feature_bundle, idx)
        split_probs[split_name] = probs
        split_meta[split_name] = meta_x
        split_metrics[split_name] = _classification_metrics(labels[idx], probs, threshold)

    test_y = labels[split_arrays["test"]]
    test_probs = split_probs["test"]
    test_meta = split_meta["test"]

    report = {
        "label_col": resolved_label_col,
        "model_path": str(model_path),
        "threshold_metric": model_bundle["threshold_metric"],
        "threshold": round(float(threshold), 4),
        "splits": split_metrics,
        "test_calibration": _calibration_report(test_y, test_probs, n_bins=n_bins),
        "test_bootstrap": _bootstrap_metrics(
            y_true=test_y,
            probs=test_probs,
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "meta_feature_importance": _meta_feature_importance(
            meta_x=test_meta,
            y_true=test_y,
            meta_model=model_bundle["meta_model"],
            meta_feature_names=model_bundle["meta_feature_names"],
            permutation_repeats=permutation_repeats,
            seed=seed,
        ),
    }

    report_path = output_dir / "stacking_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _bootstrap_metrics(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc", "brier")}
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_sample = y_true[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        prob_sample = probs[idx]
        pred_sample = (prob_sample >= threshold).astype(int)

        samples["accuracy"].append(float(accuracy_score(y_sample, pred_sample)))
        samples["balanced_accuracy"].append(float(balanced_accuracy_score(y_sample, pred_sample)))
        samples["precision"].append(float(precision_score(y_sample, pred_sample, zero_division=0)))
        samples["recall"].append(float(recall_score(y_sample, pred_sample, zero_division=0)))
        samples["f1"].append(float(f1_score(y_sample, pred_sample, zero_division=0)))
        samples["auroc"].append(float(roc_auc_score(y_sample, prob_sample)))
        samples["brier"].append(float(brier_score_loss(y_sample, prob_sample)))

    point_preds = (probs >= threshold).astype(int)
    points = {
        "accuracy": float(accuracy_score(y_true, point_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, point_preds)),
        "precision": float(precision_score(y_true, point_preds, zero_division=0)),
        "recall": float(recall_score(y_true, point_preds, zero_division=0)),
        "f1": float(f1_score(y_true, point_preds, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
    }

    report = {}
    for name, point in points.items():
        arr = np.asarray(samples[name], dtype=float)
        report[name] = {
            "point": round(float(point), 4),
            "ci95_low": round(float(np.quantile(arr, 0.025)), 4),
            "ci95_high": round(float(np.quantile(arr, 0.975)), 4),
            "n_bootstrap": int(len(arr)),
        }
    return report


def _calibration_report(y_true: np.ndarray, probs: np.ndarray, *, n_bins: int) -> dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue

        frac = float(mask.mean())
        mean_prob = float(probs[mask].mean())
        positive_rate = float(y_true[mask].mean())
        ece += abs(positive_rate - mean_prob) * frac
        bin_rows.append(
            {
                "bin_start": round(float(lo), 4),
                "bin_end": round(float(hi), 4),
                "count": int(mask.sum()),
                "mean_predicted_prob": round(mean_prob, 4),
                "observed_positive_rate": round(positive_rate, 4),
            }
        )

    return {
        "brier": round(float(brier_score_loss(y_true, probs)), 4),
        "ece": round(float(ece), 4),
        "mean_predicted_probability": round(float(np.mean(probs)), 4),
        "observed_positive_rate": round(float(np.mean(y_true)), 4),
        "bins": bin_rows,
    }


def _meta_feature_importance(
    *,
    meta_x: np.ndarray,
    y_true: np.ndarray,
    meta_model,
    meta_feature_names: list[str],
    permutation_repeats: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    baseline_probs = meta_model.predict_proba(meta_x)[:, 1]
    baseline_auroc = float(roc_auc_score(y_true, baseline_probs))

    coef = meta_model.named_steps["clf"].coef_[0]
    importance = []
    for j, name in enumerate(meta_feature_names):
        drops = []
        for _ in range(permutation_repeats):
            shuffled = meta_x.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            perm_probs = meta_model.predict_proba(shuffled)[:, 1]
            perm_auroc = float(roc_auc_score(y_true, perm_probs))
            drops.append(baseline_auroc - perm_auroc)

        importance.append(
            {
                "feature": name,
                "meta_logistic_coef": round(float(coef[j]), 4),
                "mean_auroc_drop": round(float(np.mean(drops)), 4),
                "std_auroc_drop": round(float(np.std(drops)), 4),
            }
        )

    importance.sort(key=lambda row: row["mean_auroc_drop"], reverse=True)
    return {
        "baseline_auroc": round(float(baseline_auroc), 4),
        "permutation_repeats": int(permutation_repeats),
        "ranked_features": importance,
    }
