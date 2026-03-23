"""
stacking_classifier.py - Cross-validated stacking for downstream mortality prediction.

Purpose:
  Build a leakage-aware committee of downstream models using out-of-fold (OOF)
  base predictions on the combined train+validation development split, then
  learn a meta-classifier on those OOF predictions.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from s15.advanced_classifier import (
    _classification_metrics,
    _make_model,
    _predict_proba,
    _select_threshold,
    build_feature_views,
)

logger = logging.getLogger("s15.stacking_classifier")

DEFAULT_BASE_SPECS = [
    {
        "name": "stats_hgb_d5",
        "view": "stats_mask_proxy_static",
        "model_type": "hgb",
        "hgb_max_depth": 5,
        "hgb_learning_rate": 0.05,
        "hgb_max_iter": 300,
    },
    {
        "name": "fused_hgb_d5",
        "view": "fused_all",
        "model_type": "hgb",
        "hgb_max_depth": 5,
        "hgb_learning_rate": 0.05,
        "hgb_max_iter": 200,
    },
    {
        "name": "fused_lr",
        "view": "fused_all",
        "model_type": "logreg",
        "hgb_max_depth": 5,
        "hgb_learning_rate": 0.05,
        "hgb_max_iter": 300,
    },
]


def train_stacking_mortality_classifier(
    *,
    s0_dir: Path,
    splits_path: Path,
    output_dir: Path,
    embeddings_path: Path | None = None,
    label_col: str = "mortality_inhospital",
    threshold_metric: str = "accuracy",
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    Train a leakage-aware stacking classifier on the development split.

    The model uses OOF predictions from three base learners:
      - stats-only HistGradientBoosting
      - fused feature HistGradientBoosting
      - fused feature LogisticRegression

    The meta-classifier is a logistic regression trained on OOF base
    predictions, while held-out test predictions are obtained from a committee
    of fold-specific base models averaged at inference time.
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

    dev_idx = np.concatenate([split_arrays["train"], split_arrays["val"]])
    y_dev = labels[dev_idx]
    y_test = labels[split_arrays["test"]]

    meta_feature_names = [spec["name"] for spec in DEFAULT_BASE_SPECS]
    oof_meta = np.zeros((len(dev_idx), len(DEFAULT_BASE_SPECS)), dtype=float)
    test_meta = np.zeros((len(split_arrays["test"]), len(DEFAULT_BASE_SPECS)), dtype=float)

    base_models: list[dict] = []
    base_reports = []
    base_views = feature_bundle["views"]
    base_feature_names = feature_bundle["feature_names"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for j, spec in enumerate(DEFAULT_BASE_SPECS):
        view_name = spec["view"]
        x_dev = base_views[view_name][dev_idx]
        x_test = base_views[view_name][split_arrays["test"]]

        fold_models = []
        fold_probs_test = []
        coef_stack = []
        fold_aurocs = []

        for fold_id, (train_rel, eval_rel) in enumerate(cv.split(x_dev, y_dev), start=1):
            train_idx = dev_idx[train_rel]
            eval_idx = dev_idx[eval_rel]

            model = _make_model(
                spec["model_type"],
                seed=seed + fold_id,
                max_depth=spec["hgb_max_depth"],
                learning_rate=spec["hgb_learning_rate"],
                max_iter=spec["hgb_max_iter"],
            )
            model.fit(base_views[view_name][train_idx], labels[train_idx])

            eval_probs = _predict_proba(model, base_views[view_name][eval_idx])
            oof_meta[eval_rel, j] = eval_probs
            fold_probs_test.append(_predict_proba(model, x_test))
            fold_models.append(model)

            fold_metrics = _classification_metrics(labels[eval_idx], eval_probs, threshold=0.5)
            fold_aurocs.append(fold_metrics["auroc"])

            if spec["model_type"] == "logreg":
                coef_stack.append(model.named_steps["clf"].coef_[0])

        test_meta[:, j] = np.mean(fold_probs_test, axis=0)

        base_report = {
            "name": spec["name"],
            "view": view_name,
            "model_type": spec["model_type"],
            "feature_count": int(base_views[view_name].shape[1]),
            "oof_auroc": round(float(_safe_auroc(y_dev, oof_meta[:, j])), 4),
            "fold_aurocs": [round(float(v), 4) if v is not None else None for v in fold_aurocs],
        }

        if coef_stack:
            coef_arr = np.stack(coef_stack, axis=0)
            mean_coef = coef_arr.mean(axis=0)
            mean_abs_coef = np.mean(np.abs(coef_arr), axis=0)
            top_idx = np.argsort(mean_abs_coef)[::-1][:10]
            base_report["top_logistic_coefficients"] = [
                {
                    "feature": base_feature_names[view_name][int(idx)],
                    "mean_coef": round(float(mean_coef[idx]), 4),
                    "mean_abs_coef": round(float(mean_abs_coef[idx]), 4),
                }
                for idx in top_idx
            ]

        base_reports.append(base_report)
        base_models.append(
            {
                "name": spec["name"],
                "view": view_name,
                "model_type": spec["model_type"],
                "models": fold_models,
            }
        )

    meta_model = Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    meta_model.fit(oof_meta, y_dev)

    dev_probs = meta_model.predict_proba(oof_meta)[:, 1]
    test_probs = meta_model.predict_proba(test_meta)[:, 1]

    operating_points = {}
    for metric_name in ("accuracy", "balanced_accuracy", "f1"):
        threshold, search = _select_threshold(y_dev, dev_probs, metric_name=metric_name)
        operating_points[metric_name] = {
            "threshold": round(float(threshold), 4),
            "threshold_search": search,
            "train": _classification_metrics(
                labels[split_arrays["train"]],
                dev_probs[: len(split_arrays["train"])],
                threshold,
            ),
            "val": _classification_metrics(
                labels[split_arrays["val"]],
                dev_probs[len(split_arrays["train"]) :],
                threshold,
            ),
            "dev": _classification_metrics(y_dev, dev_probs, threshold),
            "test": _classification_metrics(y_test, test_probs, threshold),
        }

    chosen = operating_points[threshold_metric]
    threshold = chosen["threshold"]
    meta_coef = meta_model.named_steps["clf"].coef_[0]
    meta_scaler = meta_model.named_steps["sc"]
    meta_coef_rescaled = meta_coef / np.maximum(meta_scaler.scale_, 1e-8)

    report = {
        "label_col": label_col,
        "model_type": "cv_stacking_committee",
        "n_splits": int(n_splits),
        "threshold_metric": threshold_metric,
        "threshold_selection": {
            "metric": threshold_metric,
            "selected_threshold": threshold,
            "search": chosen["threshold_search"],
        },
        "meta_feature_names": meta_feature_names,
        "meta_model": {
            "type": "logistic_regression",
            "coefficients": [
                {
                    "feature": name,
                    "coef_standardized": round(float(coef), 4),
                    "coef_rescaled": round(float(rescaled), 4),
                }
                for name, coef, rescaled in zip(meta_feature_names, meta_coef, meta_coef_rescaled)
            ],
        },
        "base_models": base_reports,
        "operating_points": operating_points,
        "splits": {
            "train": chosen["train"],
            "val": chosen["val"],
            "dev": chosen["dev"],
            "test": chosen["test"],
        },
        "baseline_accuracy": _baseline_accuracy(
            labels=labels,
            split_arrays={
                "train": split_arrays["train"],
                "val": split_arrays["val"],
                "dev": dev_idx,
                "test": split_arrays["test"],
            },
        ),
    }

    model_bundle = {
        "kind": "cv_stacking_committee",
        "label_col": label_col,
        "threshold": float(threshold),
        "threshold_metric": threshold_metric,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "meta_feature_names": meta_feature_names,
        "base_specs": DEFAULT_BASE_SPECS,
        "base_models": base_models,
        "meta_model": meta_model,
    }

    model_path = output_dir / "stacking_mortality_classifier.pkl"
    report_path = output_dir / "stacking_mortality_classifier_report.json"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved stacking classifier: %s", model_path)
    logger.info("Saved stacking report: %s", report_path)
    return report


def predict_stacking_probabilities(
    model_bundle: dict,
    feature_bundle: dict,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return final probabilities plus meta-features for a set of indices."""
    meta_x = np.zeros((len(indices), len(model_bundle["base_models"])), dtype=float)

    for j, base_model in enumerate(model_bundle["base_models"]):
        x = feature_bundle["views"][base_model["view"]][indices]
        fold_probs = [_predict_proba(model, x) for model in base_model["models"]]
        meta_x[:, j] = np.mean(fold_probs, axis=0)

    probs = model_bundle["meta_model"].predict_proba(meta_x)[:, 1]
    return probs, meta_x


def _baseline_accuracy(labels: np.ndarray, split_arrays: dict[str, np.ndarray]) -> dict[str, float]:
    majority_label = int(round(float(labels[split_arrays["dev"]].mean())))
    return {
        name: round(float(np.mean(labels[idx] == majority_label)), 4)
        for name, idx in split_arrays.items()
    }


def _safe_auroc(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, probs))
