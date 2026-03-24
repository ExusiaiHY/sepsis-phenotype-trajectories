"""
calibrated_stacking.py - Calibration-aware OOF stacking classifier.

Key improvements over the original stacking_classifier.py:
  1. Calibration-first meta-learner: no class_weight="balanced" inflation
  2. Shallower, regularized base learners to prevent overconfidence
  3. Integrated Platt scaling post-hoc calibration
  4. Calibration-aware threshold selection
  5. Comprehensive calibration reporting

Note: pickle is used for sklearn Pipeline compatibility (existing project pattern).
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
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
from s15.calibration import (
    PlattScaling,
    calibration_metrics,
    select_calibrated_threshold,
    threshold_metrics,
)

logger = logging.getLogger("s15.calibrated_stacking")

CALIBRATED_BASE_SPECS = [
    {
        "name": "stats_hgb_d3",
        "view": "stats_mask_proxy_static",
        "model_type": "hgb",
        "hgb_max_depth": 3,
        "hgb_learning_rate": 0.03,
        "hgb_max_iter": 200,
    },
    {
        "name": "fused_hgb_d3",
        "view": "fused_all",
        "model_type": "hgb",
        "hgb_max_depth": 3,
        "hgb_learning_rate": 0.03,
        "hgb_max_iter": 200,
    },
    {
        "name": "fused_lr",
        "view": "fused_all",
        "model_type": "logreg",
        "hgb_max_depth": 3,
        "hgb_learning_rate": 0.03,
        "hgb_max_iter": 200,
    },
]


def train_calibrated_stacking(
    *,
    s0_dir: Path,
    splits_path: Path,
    output_dir: Path,
    embeddings_path: Path | None = None,
    label_col: str = "mortality_inhospital",
    threshold_metric: str = "balanced_accuracy",
    n_splits: int = 5,
    seed: int = 42,
    prior_rate: float = 0.142,
    use_calibrated_specs: bool = True,
    apply_posthoc_calibration: bool = True,
) -> dict:
    """Train a calibration-aware stacking classifier."""
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
    y_val = labels[split_arrays["val"]]

    base_specs = CALIBRATED_BASE_SPECS if use_calibrated_specs else _get_original_specs()
    meta_feature_names = [spec["name"] for spec in base_specs]
    oof_meta = np.zeros((len(dev_idx), len(base_specs)), dtype=float)
    test_meta = np.zeros((len(split_arrays["test"]), len(base_specs)), dtype=float)

    base_models: list[dict] = []
    base_reports = []
    base_views = feature_bundle["views"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for j, spec in enumerate(base_specs):
        view_name = spec["view"]
        x_dev = base_views[view_name][dev_idx]
        x_test = base_views[view_name][split_arrays["test"]]

        fold_models = []
        fold_probs_test = []
        fold_briers = []
        fold_aurocs = []

        for fold_id, (train_rel, eval_rel) in enumerate(cv.split(x_dev, y_dev), start=1):
            model = _make_model(
                spec["model_type"],
                seed=seed + fold_id,
                max_depth=spec["hgb_max_depth"],
                learning_rate=spec["hgb_learning_rate"],
                max_iter=spec["hgb_max_iter"],
            )
            model.fit(base_views[view_name][dev_idx[train_rel]], labels[dev_idx[train_rel]])

            eval_probs = _predict_proba(model, base_views[view_name][dev_idx[eval_rel]])
            oof_meta[eval_rel, j] = eval_probs
            fold_probs_test.append(_predict_proba(model, x_test))
            fold_models.append(model)

            fold_brier = float(brier_score_loss(labels[dev_idx[eval_rel]], eval_probs))
            fold_briers.append(fold_brier)
            try:
                fold_auroc = float(roc_auc_score(labels[dev_idx[eval_rel]], eval_probs))
            except ValueError:
                fold_auroc = None
            fold_aurocs.append(fold_auroc)

        test_meta[:, j] = np.mean(fold_probs_test, axis=0)

        base_report = {
            "name": spec["name"],
            "view": view_name,
            "model_type": spec["model_type"],
            "oof_brier": round(float(np.mean(fold_briers)), 4),
            "fold_briers": [round(float(b), 4) for b in fold_briers],
            "fold_aurocs": [round(float(a), 4) if a else None for a in fold_aurocs],
        }
        base_reports.append(base_report)
        base_models.append({
            "name": spec["name"],
            "view": view_name,
            "model_type": spec["model_type"],
            "models": fold_models,
        })

    # Calibration-aware meta-learner: NO class_weight="balanced"
    meta_model = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=0.1,
            random_state=seed,
        )),
    ])
    meta_model.fit(oof_meta, y_dev)

    dev_probs = meta_model.predict_proba(oof_meta)[:, 1]
    test_probs = meta_model.predict_proba(test_meta)[:, 1]

    # Apply post-hoc calibration
    calibrator = None
    if apply_posthoc_calibration:
        val_start = len(split_arrays["train"])
        val_oof_probs = dev_probs[val_start:]
        calibrator = PlattScaling()
        calibrator.fit(val_oof_probs, y_val)
        dev_probs_cal = calibrator.predict(dev_probs)
        test_probs_cal = calibrator.predict(test_probs)
        logger.info("Post-hoc Platt scaling: a=%.4f, b=%.4f", calibrator.a, calibrator.b)
    else:
        dev_probs_cal = dev_probs
        test_probs_cal = test_probs

    # Threshold selection on calibrated probabilities
    threshold, search = select_calibrated_threshold(
        y_val,
        dev_probs_cal[len(split_arrays["train"]):],
        metric_name=threshold_metric,
    )

    # Evaluate
    test_cal_metrics = calibration_metrics(y_test, test_probs_cal, n_bins=10)
    test_cls_metrics = threshold_metrics(y_test, test_probs_cal, threshold)
    raw_cal_metrics = calibration_metrics(y_test, test_probs, n_bins=10)

    report = {
        "label_col": label_col,
        "model_type": "calibrated_cv_stacking",
        "n_splits": int(n_splits),
        "threshold_metric": threshold_metric,
        "threshold": round(float(threshold), 4),
        "use_calibrated_specs": use_calibrated_specs,
        "apply_posthoc_calibration": apply_posthoc_calibration,
        "prior_rate": prior_rate,
        "meta_feature_names": meta_feature_names,
        "base_models": base_reports,
        "raw_test_calibration": {
            "brier": round(raw_cal_metrics["brier"], 4),
            "ece": round(raw_cal_metrics["ece"], 4),
            "auroc": round(raw_cal_metrics["auroc"], 4) if raw_cal_metrics["auroc"] else None,
        },
        "calibrated_test_metrics": {
            "brier": round(test_cal_metrics["brier"], 4),
            "ece": round(test_cal_metrics["ece"], 4),
            "mce": round(test_cal_metrics["mce"], 4),
            "auroc": round(test_cal_metrics["auroc"], 4) if test_cal_metrics["auroc"] else None,
            "log_loss": round(test_cal_metrics["log_loss"], 4) if test_cal_metrics["log_loss"] else None,
            "mean_predicted_prob": round(test_cal_metrics["mean_predicted_prob"], 4),
            "observed_positive_rate": round(test_cal_metrics["observed_positive_rate"], 4),
        },
        "classification_at_threshold": test_cls_metrics,
        "stratified_calibration": test_cal_metrics["stratified"],
        "bins": test_cal_metrics["bins"],
    }

    # Save (pickle needed for sklearn Pipeline compatibility)
    model_bundle = {
        "kind": "calibrated_cv_stacking",
        "label_col": label_col,
        "threshold": float(threshold),
        "threshold_metric": threshold_metric,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "meta_feature_names": meta_feature_names,
        "base_specs": base_specs,
        "base_models": base_models,
        "meta_model": meta_model,
        "calibrator": calibrator,
    }

    model_path = output_dir / "calibrated_stacking_classifier.pkl"
    report_path = output_dir / "calibrated_stacking_report.json"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.save(output_dir / "test_probs_calibrated.npy", test_probs_cal)
    np.save(output_dir / "test_probs_raw.npy", test_probs)

    logger.info("Saved calibrated stacking classifier: %s", model_path)
    logger.info(
        "Test: Brier=%.4f ECE=%.4f AUROC=%.4f Recall=%.4f",
        test_cal_metrics["brier"],
        test_cal_metrics["ece"],
        test_cal_metrics["auroc"] or 0.0,
        test_cls_metrics["recall"],
    )
    return report


def _get_original_specs():
    from s15.stacking_classifier import DEFAULT_BASE_SPECS
    return DEFAULT_BASE_SPECS
