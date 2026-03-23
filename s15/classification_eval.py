"""
classification_eval.py - Supervised mortality classification on fixed embeddings.

Purpose:
  Fit a lightweight classifier on top of pretrained patient embeddings,
  select a validation-set decision threshold, and report classification metrics
  on train/val/test splits.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("s15.classification_eval")


def train_mortality_classifier(
    embeddings: np.ndarray,
    static_path: Path,
    splits_path: Path,
    output_dir: Path,
    *,
    label_col: str = "mortality_inhospital",
    class_weight: str | dict | None = "balanced",
    c_value: float = 1.0,
    max_iter: int = 1000,
    threshold_metric: str = "balanced_accuracy",
    seed: int = 42,
) -> dict:
    """
    Train a logistic-regression classifier on patient embeddings.

    Parameters
    ----------
    embeddings:
        Patient embedding matrix of shape (N, D).
    static_path:
        CSV containing supervised labels.
    splits_path:
        JSON with train/val/test patient indices.
    output_dir:
        Directory where classifier artifact and report are saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    static = pd.read_csv(static_path)
    labels = static[label_col].fillna(0).astype(int).to_numpy()

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)

    split_arrays = {
        name: np.asarray(splits[name], dtype=int)
        for name in ("train", "val", "test")
    }

    _validate_inputs(embeddings, labels, split_arrays)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(embeddings[split_arrays["train"]])
    y_train = labels[split_arrays["train"]]

    classifier = LogisticRegression(
        max_iter=max_iter,
        C=c_value,
        solver="lbfgs",
        class_weight=class_weight,
        random_state=seed,
    )
    classifier.fit(x_train, y_train)

    transformed = {
        name: scaler.transform(embeddings[idx])
        for name, idx in split_arrays.items()
    }

    val_probs = classifier.predict_proba(transformed["val"])[:, 1]
    threshold, threshold_search = _select_threshold(
        labels[split_arrays["val"]],
        val_probs,
        metric_name=threshold_metric,
    )

    report = {
        "label_col": label_col,
        "n_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "classifier": {
            "type": "logistic_regression",
            "class_weight": class_weight,
            "c_value": c_value,
            "max_iter": max_iter,
            "random_state": seed,
        },
        "threshold_selection": {
            "metric": threshold_metric,
            "selected_threshold": round(float(threshold), 4),
            "search": threshold_search,
        },
        "splits": {},
    }

    for name, idx in split_arrays.items():
        y_true = labels[idx]
        probs = classifier.predict_proba(transformed[name])[:, 1]
        report["splits"][name] = _classification_metrics(y_true, probs, threshold)

    majority_label = int(round(float(labels[split_arrays["train"]].mean())))
    report["baseline_accuracy"] = {
        name: round(
            float(np.mean(labels[idx] == majority_label)),
            4,
        )
        for name, idx in split_arrays.items()
    }

    model_path = output_dir / "mortality_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "classifier": classifier,
                "threshold": threshold,
                "label_col": label_col,
            },
            f,
        )

    report_path = output_dir / "mortality_classifier_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved mortality classifier: %s", model_path)
    logger.info("Saved mortality report: %s", report_path)
    return report


def _validate_inputs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D, got shape {embeddings.shape}")
    if len(labels) != len(embeddings):
        raise ValueError("labels length does not match embeddings")
    for name, idx in splits.items():
        if idx.size == 0:
            raise ValueError(f"Split '{name}' is empty")
        if np.any(idx < 0) or np.any(idx >= len(labels)):
            raise ValueError(f"Split '{name}' contains out-of-range indices")
        if len(np.unique(labels[idx])) < 2 and name in {"train", "val"}:
            raise ValueError(f"Split '{name}' must contain both classes for training/tuning")


def _select_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    metric_name: str = "balanced_accuracy",
) -> tuple[float, list[dict]]:
    if len(np.unique(y_true)) < 2:
        return 0.5, [{"threshold": 0.5, metric_name: None, "note": "single class in validation"}]

    candidates = np.linspace(0.05, 0.95, 37)
    search = []
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in candidates:
        preds = (probs >= threshold).astype(int)
        score = _threshold_metric(y_true, preds, metric_name)
        search.append(
            {
                "threshold": round(float(threshold), 4),
                metric_name: round(float(score), 4),
            }
        )
        if score > best_score or (np.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, search


def _threshold_metric(y_true: np.ndarray, preds: np.ndarray, metric_name: str) -> float:
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, preds))
    if metric_name == "f1":
        return float(f1_score(y_true, preds, zero_division=0))
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, preds))
    raise ValueError(f"Unsupported threshold metric: {metric_name}")


def _classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    metrics = {
        "n_samples": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "predicted_positive_rate": round(float(np.mean(preds)), 4),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if len(np.unique(y_true)) >= 2:
        metrics["auroc"] = round(float(roc_auc_score(y_true, probs)), 4)
    else:
        metrics["auroc"] = None

    return metrics
