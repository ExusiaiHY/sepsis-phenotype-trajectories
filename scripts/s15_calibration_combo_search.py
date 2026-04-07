#!/usr/bin/env python3
"""
s15_calibration_combo_search.py - Wider calibrated-stacking combo search.

Searches branch-specific calibrated stacking configurations by:
  - tuning the stats HGB branch separately from the fused HGB branch
  - keeping the fused logistic branch fixed
  - tuning meta-learner C
  - comparing lightweight post-hoc calibrators

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    ./.venv/bin/python scripts/s15_calibration_combo_search.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from s15.advanced_classifier import _make_model, _predict_proba, build_feature_views
from s15.calibration import (
    BayesianPriorCalibration,
    CompositeCalibration,
    PlattScaling,
    TemperatureScaling,
    calibration_metrics,
    select_calibrated_threshold,
    threshold_metrics,
)


LOGGER = logging.getLogger("scripts.s15_calibration_combo_search")

STATS_CANDIDATES = [
    {"name": "stats_d3_lr0p03_i300", "max_depth": 3, "learning_rate": 0.03, "max_iter": 300},
    {"name": "stats_d3_lr0p05_i200", "max_depth": 3, "learning_rate": 0.05, "max_iter": 200},
    {"name": "stats_d3_lr0p05_i400", "max_depth": 3, "learning_rate": 0.05, "max_iter": 400},
    {"name": "stats_d5_lr0p03_i200", "max_depth": 5, "learning_rate": 0.03, "max_iter": 200},
    {"name": "stats_d7_lr0p03_i200", "max_depth": 7, "learning_rate": 0.03, "max_iter": 200},
    {"name": "stats_d7_lr0p03_i400", "max_depth": 7, "learning_rate": 0.03, "max_iter": 400},
]

FUSED_CANDIDATES = [
    {"name": "fused_d3_lr0p03_i300", "max_depth": 3, "learning_rate": 0.03, "max_iter": 300},
    {"name": "fused_d3_lr0p03_i400", "max_depth": 3, "learning_rate": 0.03, "max_iter": 400},
    {"name": "fused_d5_lr0p03_i200", "max_depth": 5, "learning_rate": 0.03, "max_iter": 200},
    {"name": "fused_d5_lr0p03_i400", "max_depth": 5, "learning_rate": 0.03, "max_iter": 400},
    {"name": "fused_d5_lr0p05_i200", "max_depth": 5, "learning_rate": 0.05, "max_iter": 200},
    {"name": "fused_d7_lr0p03_i200", "max_depth": 7, "learning_rate": 0.03, "max_iter": 200},
]

META_C_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]
CALIBRATORS = ["none", "platt", "temperature", "bayesian_prior", "composite"]


def _make_calibrator(name: str, *, prior_rate: float):
    if name == "none":
        return None
    if name == "platt":
        return PlattScaling()
    if name == "temperature":
        return TemperatureScaling()
    if name == "bayesian_prior":
        return BayesianPriorCalibration(prior_rate=prior_rate)
    if name == "composite":
        return CompositeCalibration(prior_rate=prior_rate)
    raise ValueError(f"Unsupported calibrator: {name}")


def _fit_branch_predictions(
    *,
    x: np.ndarray,
    y_dev: np.ndarray,
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    candidate: dict,
    seed: int,
    cv: StratifiedKFold,
) -> dict:
    oof = np.zeros(len(dev_idx), dtype=float)
    test_fold_probs = []

    for fold_id, (train_rel, eval_rel) in enumerate(cv.split(x[dev_idx], y_dev), start=1):
        model = _make_model(
            "hgb",
            seed=seed + fold_id,
            max_depth=candidate["max_depth"],
            learning_rate=candidate["learning_rate"],
            max_iter=candidate["max_iter"],
        )
        train_abs = dev_idx[train_rel]
        eval_abs = dev_idx[eval_rel]
        model.fit(x[train_abs], y_dev[train_rel])
        oof[eval_rel] = _predict_proba(model, x[eval_abs])
        test_fold_probs.append(_predict_proba(model, x[test_idx]))

    return {
        "candidate": candidate,
        "oof": oof,
        "test": np.mean(test_fold_probs, axis=0),
    }


def _fit_fixed_logreg_branch(
    *,
    x: np.ndarray,
    y_dev: np.ndarray,
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    cv: StratifiedKFold,
) -> dict:
    oof = np.zeros(len(dev_idx), dtype=float)
    test_fold_probs = []

    for fold_id, (train_rel, eval_rel) in enumerate(cv.split(x[dev_idx], y_dev), start=1):
        model = _make_model(
            "logreg",
            seed=seed + fold_id,
            max_depth=3,
            learning_rate=0.03,
            max_iter=300,
        )
        train_abs = dev_idx[train_rel]
        eval_abs = dev_idx[eval_rel]
        model.fit(x[train_abs], y_dev[train_rel])
        oof[eval_rel] = _predict_proba(model, x[eval_abs])
        test_fold_probs.append(_predict_proba(model, x[test_idx]))

    return {
        "candidate": {
            "name": "fused_lr_default",
            "model_type": "logreg",
            "view": "fused_all",
            "class_weight": "balanced",
            "C": 1.0,
        },
        "oof": oof,
        "test": np.mean(test_fold_probs, axis=0),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    with open(PROJECT_ROOT / "config" / "s15_trainval_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    s0_dir = PROJECT_ROOT / cfg["paths"]["s0_dir"]
    s15_dir = PROJECT_ROOT / cfg["paths"]["s15_dir"]
    output_dir = s15_dir / "calibration_combo_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_bundle = build_feature_views(
        s0_dir=s0_dir,
        embeddings_path=s15_dir / "embeddings_s15.npy",
        label_col="mortality_inhospital",
    )
    labels = feature_bundle["labels"]
    views = feature_bundle["views"]

    with open(s0_dir / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)
    split_arrays = {name: np.asarray(splits[name], dtype=int) for name in ("train", "val", "test")}
    dev_idx = np.concatenate([split_arrays["train"], split_arrays["val"]])
    y_dev = labels[dev_idx]
    y_val = labels[split_arrays["val"]]
    y_test = labels[split_arrays["test"]]

    seed = 42
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prior_rate = 0.142

    LOGGER.info("Precomputing branch predictions for %d stats configs, %d fused configs",
                len(STATS_CANDIDATES), len(FUSED_CANDIDATES))

    stats_cache = [
        _fit_branch_predictions(
            x=views["stats_mask_proxy_static"],
            y_dev=y_dev,
            dev_idx=dev_idx,
            test_idx=split_arrays["test"],
            candidate=candidate,
            seed=seed,
            cv=cv,
        )
        for candidate in STATS_CANDIDATES
    ]
    fused_cache = [
        _fit_branch_predictions(
            x=views["fused_all"],
            y_dev=y_dev,
            dev_idx=dev_idx,
            test_idx=split_arrays["test"],
            candidate=candidate,
            seed=seed,
            cv=cv,
        )
        for candidate in FUSED_CANDIDATES
    ]
    fused_lr_cache = _fit_fixed_logreg_branch(
        x=views["fused_all"],
        y_dev=y_dev,
        dev_idx=dev_idx,
        test_idx=split_arrays["test"],
        seed=seed,
        cv=cv,
    )

    total = len(stats_cache) * len(fused_cache) * len(META_C_VALUES) * len(CALIBRATORS)
    results = []
    run_id = 0
    val_start = len(split_arrays["train"])

    for stats_branch in stats_cache:
        for fused_branch in fused_cache:
            oof_meta = np.column_stack([
                stats_branch["oof"],
                fused_branch["oof"],
                fused_lr_cache["oof"],
            ])
            test_meta = np.column_stack([
                stats_branch["test"],
                fused_branch["test"],
                fused_lr_cache["test"],
            ])

            for meta_c in META_C_VALUES:
                meta_model = Pipeline([
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, C=meta_c, random_state=seed)),
                ])
                meta_model.fit(oof_meta, y_dev)
                dev_probs_raw = meta_model.predict_proba(oof_meta)[:, 1]
                test_probs_raw = meta_model.predict_proba(test_meta)[:, 1]
                val_probs_raw = dev_probs_raw[val_start:]

                for calibrator_name in CALIBRATORS:
                    run_id += 1
                    calibrator = _make_calibrator(calibrator_name, prior_rate=prior_rate)
                    if calibrator is None:
                        val_probs = val_probs_raw
                        test_probs = test_probs_raw
                    else:
                        calibrator.fit(val_probs_raw, y_val)
                        val_probs = calibrator.predict(val_probs_raw)
                        test_probs = calibrator.predict(test_probs_raw)

                    threshold, _ = select_calibrated_threshold(
                        y_val,
                        val_probs,
                        metric_name="balanced_accuracy",
                    )
                    cal_metrics = calibration_metrics(y_test, test_probs, n_bins=10)
                    cls_metrics = threshold_metrics(y_test, test_probs, threshold)

                    row = {
                        "run_id": run_id,
                        "stats_config": stats_branch["candidate"],
                        "fused_config": fused_branch["candidate"],
                        "fused_lr_config": fused_lr_cache["candidate"],
                        "meta_c": meta_c,
                        "calibrator": calibrator_name,
                        "brier": round(cal_metrics["brier"], 4),
                        "ece": round(cal_metrics["ece"], 4),
                        "mce": round(cal_metrics["mce"], 4),
                        "auroc": round(cal_metrics["auroc"], 4) if cal_metrics["auroc"] else None,
                        "threshold": round(float(threshold), 4),
                        "accuracy": cls_metrics["accuracy"],
                        "balanced_accuracy": cls_metrics["balanced_accuracy"],
                        "precision": cls_metrics["precision"],
                        "recall": cls_metrics["recall"],
                        "f1": cls_metrics["f1"],
                    }
                    results.append(row)
                    LOGGER.info(
                        "[%d/%d] %s + %s | C=%.3f | cal=%s => Brier=%.4f ECE=%.4f AUROC=%.4f BalAcc=%.4f",
                        run_id,
                        total,
                        stats_branch["candidate"]["name"],
                        fused_branch["candidate"]["name"],
                        meta_c,
                        calibrator_name,
                        row["brier"],
                        row["ece"],
                        row["auroc"] or 0.0,
                        row["balanced_accuracy"],
                    )

    valid = [row for row in results if row["auroc"] and row["auroc"] >= 0.85]
    valid.sort(
        key=lambda row: (
            row["brier"] + row["ece"],
            -row["balanced_accuracy"],
            -row["auroc"],
        )
    )
    best = valid[0] if valid else min(results, key=lambda row: row["brier"] + row["ece"])

    LOGGER.info("")
    LOGGER.info("=" * 120)
    LOGGER.info("TOP 10 COMBINATIONS (AUROC >= 0.85)")
    LOGGER.info("=" * 120)
    for rank, row in enumerate(valid[:10], start=1):
        LOGGER.info(
            "%2d. stats=%s fused=%s C=%.3f cal=%s | Brier=%.4f ECE=%.4f AUROC=%.4f BalAcc=%.4f",
            rank,
            row["stats_config"]["name"],
            row["fused_config"]["name"],
            row["meta_c"],
            row["calibrator"],
            row["brier"],
            row["ece"],
            row["auroc"] or 0.0,
            row["balanced_accuracy"],
        )

    report = {
        "search_type": "branch_specific_calibrated_stacking",
        "n_results": len(results),
        "objective": "minimize (Brier + ECE) subject to AUROC >= 0.85",
        "stats_candidates": STATS_CANDIDATES,
        "fused_candidates": FUSED_CANDIDATES,
        "meta_c_values": META_C_VALUES,
        "calibrators": CALIBRATORS,
        "best": best,
        "top10": valid[:10],
        "results": results,
    }
    report_path = output_dir / "search_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    LOGGER.info("")
    LOGGER.info("BEST CONFIGURATION")
    LOGGER.info("  stats=%s", best["stats_config"]["name"])
    LOGGER.info("  fused=%s", best["fused_config"]["name"])
    LOGGER.info("  meta_C=%.3f calibrator=%s", best["meta_c"], best["calibrator"])
    LOGGER.info("  Brier=%.4f ECE=%.4f AUROC=%.4f BalAcc=%.4f",
                best["brier"], best["ece"], best["auroc"] or 0.0, best["balanced_accuracy"])
    LOGGER.info("Saved report to %s", report_path)


if __name__ == "__main__":
    main()
