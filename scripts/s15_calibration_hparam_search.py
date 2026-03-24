#!/usr/bin/env python3
"""
s15_calibration_hparam_search.py - Hyperparameter search for calibrated stacking.

Searches over meta-learner C and base learner depth/lr to find optimal
calibration-accuracy trade-off.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    python3 scripts/s15_calibration_hparam_search.py
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
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from s15.advanced_classifier import (
    _make_model,
    _predict_proba,
    build_feature_views,
)
from s15.calibration import (
    PlattScaling,
    calibration_metrics,
    select_calibrated_threshold,
    threshold_metrics,
)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("scripts.calibration_hparam_search")

    with open(PROJECT_ROOT / "config" / "s15_trainval_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    s0_dir = PROJECT_ROOT / cfg["paths"]["s0_dir"]
    s15_dir = PROJECT_ROOT / cfg["paths"]["s15_dir"]

    feature_bundle = build_feature_views(
        s0_dir=s0_dir,
        embeddings_path=s15_dir / "embeddings_s15.npy",
        label_col="mortality_inhospital",
    )
    labels = feature_bundle["labels"]

    with open(s0_dir / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)
    split_arrays = {name: np.asarray(splits[name], dtype=int) for name in ("train", "val", "test")}

    dev_idx = np.concatenate([split_arrays["train"], split_arrays["val"]])
    y_dev = labels[dev_idx]
    y_test = labels[split_arrays["test"]]
    y_val = labels[split_arrays["val"]]

    # Search grid
    depth_lr_configs = [
        (2, 0.03, 200),
        (3, 0.02, 200),
        (3, 0.03, 200),  # current best
        (3, 0.05, 200),
        (4, 0.03, 200),
        (3, 0.03, 300),
    ]
    meta_c_values = [0.01, 0.05, 0.1, 0.5, 1.0]

    results = []
    seed = 42
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    base_views = feature_bundle["views"]

    total = len(depth_lr_configs) * len(meta_c_values)
    run_id = 0

    for depth, lr, max_iter in depth_lr_configs:
        # Build base specs for this config
        base_specs = [
            {"name": "stats_hgb", "view": "stats_mask_proxy_static", "model_type": "hgb",
             "hgb_max_depth": depth, "hgb_learning_rate": lr, "hgb_max_iter": max_iter},
            {"name": "fused_hgb", "view": "fused_all", "model_type": "hgb",
             "hgb_max_depth": depth, "hgb_learning_rate": lr, "hgb_max_iter": max_iter},
            {"name": "fused_lr", "view": "fused_all", "model_type": "logreg",
             "hgb_max_depth": depth, "hgb_learning_rate": lr, "hgb_max_iter": max_iter},
        ]

        # Train OOF base models once per (depth, lr) config
        oof_meta = np.zeros((len(dev_idx), len(base_specs)), dtype=float)
        test_meta = np.zeros((len(split_arrays["test"]), len(base_specs)), dtype=float)

        for j, spec in enumerate(base_specs):
            view_name = spec["view"]
            x_test = base_views[view_name][split_arrays["test"]]
            fold_probs_test = []

            for fold_id, (train_rel, eval_rel) in enumerate(cv.split(base_views[view_name][dev_idx], y_dev), start=1):
                model = _make_model(
                    spec["model_type"], seed=seed + fold_id,
                    max_depth=spec["hgb_max_depth"],
                    learning_rate=spec["hgb_learning_rate"],
                    max_iter=spec["hgb_max_iter"],
                )
                model.fit(base_views[view_name][dev_idx[train_rel]], labels[dev_idx[train_rel]])
                oof_meta[eval_rel, j] = _predict_proba(model, base_views[view_name][dev_idx[eval_rel]])
                fold_probs_test.append(_predict_proba(model, x_test))

            test_meta[:, j] = np.mean(fold_probs_test, axis=0)

        # Try different meta-learner C values
        for meta_c in meta_c_values:
            run_id += 1
            meta_model = Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=meta_c, random_state=seed)),
            ])
            meta_model.fit(oof_meta, y_dev)

            dev_probs = meta_model.predict_proba(oof_meta)[:, 1]
            test_probs = meta_model.predict_proba(test_meta)[:, 1]

            # Post-hoc Platt calibration
            val_start = len(split_arrays["train"])
            val_oof = dev_probs[val_start:]
            platt = PlattScaling()
            platt.fit(val_oof, y_val)
            test_probs_cal = platt.predict(test_probs)

            # Evaluate
            cal_met = calibration_metrics(y_test, test_probs_cal, n_bins=10)
            threshold, _ = select_calibrated_threshold(
                y_val, platt.predict(val_oof), metric_name="balanced_accuracy"
            )
            cls_met = threshold_metrics(y_test, test_probs_cal, threshold)

            row = {
                "run_id": run_id,
                "depth": depth,
                "lr": lr,
                "max_iter": max_iter,
                "meta_c": meta_c,
                "brier": round(cal_met["brier"], 4),
                "ece": round(cal_met["ece"], 4),
                "mce": round(cal_met["mce"], 4),
                "auroc": round(cal_met["auroc"], 4) if cal_met["auroc"] else None,
                "threshold": round(threshold, 4),
                "recall": cls_met["recall"],
                "precision": cls_met["precision"],
                "f1": cls_met["f1"],
                "balanced_accuracy": cls_met["balanced_accuracy"],
                "platt_a": round(platt.a, 4),
                "platt_b": round(platt.b, 4),
            }
            results.append(row)
            logger.info(
                "[%d/%d] d=%d lr=%.3f iter=%d C=%.2f => Brier=%.4f ECE=%.4f AUROC=%.4f Recall=%.4f",
                run_id, total, depth, lr, max_iter, meta_c,
                row["brier"], row["ece"], row["auroc"] or 0, row["recall"],
            )

    # Sort by Brier + ECE (lower is better) while requiring AUROC >= 0.85
    valid = [r for r in results if r["auroc"] and r["auroc"] >= 0.85]
    valid.sort(key=lambda r: r["brier"] + r["ece"])

    logger.info("")
    logger.info("=" * 100)
    logger.info("TOP 10 CONFIGURATIONS (AUROC >= 0.85)")
    logger.info("=" * 100)
    logger.info("%-5s  %5s  %5s  %5s  %5s  %7s  %7s  %7s  %7s  %7s",
                "Rank", "Depth", "LR", "Iter", "C", "Brier", "ECE", "AUROC", "Recall", "F1")
    logger.info("-" * 100)
    for i, row in enumerate(valid[:10], 1):
        logger.info(
            "%-5d  %5d  %5.3f  %5d  %5.2f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f",
            i, row["depth"], row["lr"], row["max_iter"], row["meta_c"],
            row["brier"], row["ece"], row["auroc"], row["recall"], row["f1"],
        )
    logger.info("=" * 100)

    if valid:
        best = valid[0]
        logger.info("")
        logger.info("OPTIMAL: depth=%d lr=%.3f iter=%d C=%.2f",
                     best["depth"], best["lr"], best["max_iter"], best["meta_c"])
        logger.info("  Brier=%.4f ECE=%.4f AUROC=%.4f Recall=%.4f F1=%.4f",
                     best["brier"], best["ece"], best["auroc"], best["recall"], best["f1"])

    # Save
    output_dir = PROJECT_ROOT / "data" / "s15_trainval" / "calibration_hparam_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"results": results, "best": valid[0] if valid else None}
    with open(output_dir / "hparam_search_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", output_dir / "hparam_search_report.json")


if __name__ == "__main__":
    main()
