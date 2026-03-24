"""
calibration.py - Post-hoc calibration methods for mortality prediction.

Implements three calibration strategies that preserve model rank-ordering
(AUROC) while correcting probability estimates:

  1. Temperature Scaling: learns a single parameter T on validation logits
  2. Platt Scaling: logistic regression on logits (affine transform a*z + b)
  3. Bayesian Prior Calibration: re-anchors predictions to a clinical prior

All methods are fitted on the development (train+val) split and evaluated
on the held-out Center B test set.

Note: pickle is used for compatibility with existing model artifacts
(stacking_mortality_classifier.pkl) already in the codebase.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger("s15.calibration")


# ---------------------------------------------------------------------------
# Core calibration classes
# ---------------------------------------------------------------------------

class TemperatureScaling:
    """Learn a single temperature T that minimizes NLL on held-out logits.

    After calibration: p_cal = sigmoid(logit(p_raw) / T)
    When T > 1 predictions become less confident (toward 0.5).
    When T < 1 predictions become more confident (toward 0/1).
    """

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        raw_logits = _safe_logit(probs)

        def nll(t):
            scaled = expit(raw_logits / t)
            return log_loss(y_true, scaled)

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        logger.info("Temperature scaling: T=%.4f (NLL %.4f -> %.4f)",
                     self.temperature, nll(1.0), result.fun)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        raw_logits = _safe_logit(probs)
        return expit(raw_logits / self.temperature)


class PlattScaling:
    """Platt scaling: fit a logistic regression on model logits.

    Learns parameters a, b such that p_cal = sigmoid(a * logit(p_raw) + b).
    This allows both rescaling (a) and shifting (b) the logit distribution.
    """

    def __init__(self):
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "PlattScaling":
        raw_logits = _safe_logit(probs).reshape(-1, 1)
        lr = LogisticRegression(max_iter=5000, solver="lbfgs", C=1e10)
        lr.fit(raw_logits, y_true)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        logger.info("Platt scaling: a=%.4f, b=%.4f", self.a, self.b)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        raw_logits = _safe_logit(probs)
        return expit(self.a * raw_logits + self.b)


class IsotonicCalibration:
    """Isotonic regression calibration (non-parametric, monotonic).

    Fits a non-decreasing step function mapping predicted probabilities
    to calibrated ones. More flexible than Platt but can overfit on
    small datasets.
    """

    def __init__(self):
        self.ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibration":
        self.ir.fit(probs, y_true)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self.ir.predict(probs)


class BayesianPriorCalibration:
    """Re-anchor predictions toward a clinical prior mortality rate.

    Uses a logit-space interpolation:
      cal_logit = (1 - strength) * raw_logit + strength * prior_logit
      p_cal = sigmoid(cal_logit)

    The 'strength' parameter controls how much the prior pulls predictions
    toward the population base rate. Optimized to minimize Brier score.
    """

    def __init__(self, prior_rate: float = 0.142):
        self.prior_rate = prior_rate
        self.strength: float = 0.0

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "BayesianPriorCalibration":
        raw_logits = _safe_logit(probs)
        prior_logit = logit(self.prior_rate)

        def brier(s):
            cal_logits = (1.0 - s) * raw_logits + s * prior_logit
            cal_probs = expit(cal_logits)
            return brier_score_loss(y_true, cal_probs)

        result = minimize_scalar(brier, bounds=(0.0, 1.0), method="bounded")
        self.strength = float(result.x)
        logger.info("Bayesian prior calibration: strength=%.4f, prior=%.3f",
                     self.strength, self.prior_rate)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        raw_logits = _safe_logit(probs)
        prior_logit = logit(self.prior_rate)
        cal_logits = (1.0 - self.strength) * raw_logits + self.strength * prior_logit
        return expit(cal_logits)


class CompositeCalibration:
    """Pipeline: Temperature Scaling followed by Bayesian Prior adjustment.

    Combines the strengths of both methods for maximum calibration
    improvement while preserving AUROC.
    """

    def __init__(self, prior_rate: float = 0.142):
        self.temp = TemperatureScaling()
        self.bayesian = BayesianPriorCalibration(prior_rate=prior_rate)

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "CompositeCalibration":
        self.temp.fit(probs, y_true)
        temp_calibrated = self.temp.predict(probs)
        self.bayesian.fit(temp_calibrated, y_true)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self.bayesian.predict(self.temp.predict(probs))


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def calibration_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """Compute comprehensive calibration and discrimination metrics.

    Parameters
    ----------
    strategy : "uniform" or "quantile"
        "uniform" uses equal-width bins (standard ECE).
        "quantile" uses equal-frequency bins (adaptive ECE).
    """
    metrics = {
        "brier": float(brier_score_loss(y_true, probs)),
        "mean_predicted_prob": float(np.mean(probs)),
        "observed_positive_rate": float(np.mean(y_true)),
        "n_samples": int(len(y_true)),
    }

    if len(np.unique(y_true)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y_true, probs))
        metrics["log_loss"] = float(log_loss(y_true, probs))
    else:
        metrics["auroc"] = None
        metrics["log_loss"] = None

    # ECE computation
    if strategy == "quantile":
        bin_edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_edges = np.unique(bin_edges)
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        if not mask.any():
            continue

        frac = float(mask.mean())
        mean_pred = float(probs[mask].mean())
        obs_rate = float(y_true[mask].mean())
        gap = abs(obs_rate - mean_pred)
        ece += gap * frac
        mce = max(mce, gap)

        bin_details.append({
            "bin_start": round(lo, 4),
            "bin_end": round(hi, 4),
            "count": int(mask.sum()),
            "fraction": round(frac, 4),
            "mean_predicted_prob": round(mean_pred, 4),
            "observed_positive_rate": round(obs_rate, 4),
            "calibration_gap": round(gap, 4),
        })

    metrics["ece"] = float(ece)
    metrics["mce"] = float(mce)
    metrics["bins"] = bin_details
    metrics["stratified"] = _stratified_calibration(y_true, probs)

    return metrics


def _stratified_calibration(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Compute calibration within clinically meaningful risk strata."""
    strata = {
        "low_risk": (0.0, 0.2),
        "moderate_risk": (0.2, 0.5),
        "high_risk": (0.5, 1.0),
    }
    result = {}
    for name, (lo, hi) in strata.items():
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        if not mask.any():
            result[name] = {"count": 0}
            continue

        result[name] = {
            "count": int(mask.sum()),
            "mean_predicted_prob": round(float(probs[mask].mean()), 4),
            "observed_positive_rate": round(float(y_true[mask].mean()), 4),
            "brier": round(float(brier_score_loss(y_true[mask], probs[mask])), 4),
        }
    return result


def threshold_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict:
    """Compute classification metrics at a specific threshold."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    result = {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    if len(np.unique(y_true)) >= 2:
        result["auroc"] = round(float(roc_auc_score(y_true, probs)), 4)
    else:
        result["auroc"] = None

    return result


def select_calibrated_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    metric_name: str = "balanced_accuracy",
) -> tuple[float, list[dict]]:
    """Select optimal threshold for calibrated probabilities."""
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

    candidates = np.linspace(0.05, 0.60, 56)
    best_threshold = 0.5
    best_score = -np.inf
    search = []

    for t in candidates:
        preds = (probs >= t).astype(int)
        if metric_name == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_true, preds))
        elif metric_name == "f1":
            score = float(f1_score(y_true, preds, zero_division=0))
        elif metric_name == "accuracy":
            score = float(accuracy_score(y_true, preds))
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

        search.append({"threshold": round(float(t), 4), metric_name: round(score, 4)})
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold, search


# ---------------------------------------------------------------------------
# Full calibration pipeline
# ---------------------------------------------------------------------------

def run_calibration_pipeline(
    *,
    model_path: Path,
    s0_dir: Path,
    splits_path: Path,
    embeddings_path: Path,
    output_dir: Path,
    prior_rate: float = 0.142,
    n_bins: int = 10,
    label_col: str | None = None,
) -> dict:
    """Run all calibration methods and produce a comparative report.

    Steps:
      1. Load model and produce raw predictions on val/test splits
      2. Fit each calibrator on the validation split
      3. Apply each calibrator to the test split
      4. Compare calibration metrics across all methods
      5. Select optimal threshold for each calibrated model
      6. Save calibrated probabilities and calibration parameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (pickle is required: existing model format)
    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    from s15.advanced_classifier import build_feature_views
    from s15.stacking_classifier import predict_stacking_probabilities

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

    # Get raw probabilities
    val_probs, _ = predict_stacking_probabilities(model_bundle, feature_bundle, split_arrays["val"])
    test_probs, _ = predict_stacking_probabilities(model_bundle, feature_bundle, split_arrays["test"])

    val_y = labels[split_arrays["val"]]
    test_y = labels[split_arrays["test"]]

    # Define calibrators
    calibrators = {
        "uncalibrated": None,
        "temperature_scaling": TemperatureScaling(),
        "platt_scaling": PlattScaling(),
        "isotonic": IsotonicCalibration(),
        "bayesian_prior": BayesianPriorCalibration(prior_rate=prior_rate),
        "composite_temp_bayesian": CompositeCalibration(prior_rate=prior_rate),
    }

    results = {}
    calibrated_test_probs = {}

    for name, calibrator in calibrators.items():
        if calibrator is None:
            cal_test = test_probs
            cal_val = val_probs
            params = {}
        else:
            calibrator.fit(val_probs, val_y)
            cal_test = calibrator.predict(test_probs)
            cal_val = calibrator.predict(val_probs)
            params = _extract_params(calibrator)

        # Select threshold on calibrated val probs
        threshold, _ = select_calibrated_threshold(val_y, cal_val, metric_name="balanced_accuracy")

        # Evaluate on test
        cal_metrics = calibration_metrics(test_y, cal_test, n_bins=n_bins)
        cls_metrics = threshold_metrics(test_y, cal_test, threshold)

        results[name] = {
            "calibration_params": params,
            "threshold": round(float(threshold), 4),
            "calibration_metrics": {
                "brier": round(cal_metrics["brier"], 4),
                "ece": round(cal_metrics["ece"], 4),
                "mce": round(cal_metrics["mce"], 4),
                "auroc": round(cal_metrics["auroc"], 4) if cal_metrics["auroc"] else None,
                "log_loss": round(cal_metrics["log_loss"], 4) if cal_metrics["log_loss"] else None,
                "mean_predicted_prob": round(cal_metrics["mean_predicted_prob"], 4),
                "observed_positive_rate": round(cal_metrics["observed_positive_rate"], 4),
            },
            "classification_metrics": cls_metrics,
            "stratified_calibration": cal_metrics["stratified"],
            "bins": cal_metrics["bins"],
        }

        calibrated_test_probs[name] = cal_test
        logger.info(
            "%s: Brier=%.4f ECE=%.4f AUROC=%.4f recall=%.4f",
            name,
            cal_metrics["brier"],
            cal_metrics["ece"],
            cal_metrics["auroc"] or 0.0,
            cls_metrics["recall"],
        )

    # Rank methods
    ranking = []
    for name, res in results.items():
        cm = res["calibration_metrics"]
        brier_val = cm["brier"]
        ece_val = cm["ece"]
        auroc_val = cm["auroc"] or 0.0
        recall_val = res["classification_metrics"]["recall"]
        composite = brier_val + ece_val - 0.1 * auroc_val
        ranking.append({
            "method": name,
            "brier": round(brier_val, 4),
            "ece": round(ece_val, 4),
            "auroc": round(auroc_val, 4),
            "recall": round(recall_val, 4),
            "composite_score": round(composite, 4),
        })
    ranking.sort(key=lambda x: x["composite_score"])

    report = {
        "label_col": resolved_label_col,
        "prior_rate": prior_rate,
        "n_bins": n_bins,
        "methods": results,
        "ranking": ranking,
        "recommendation": ranking[0]["method"] if ranking else None,
    }

    # Save outputs
    report_path = output_dir / "calibration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for name, cal_probs in calibrated_test_probs.items():
        np.save(output_dir / f"test_probs_{name}.npy", cal_probs)

    # Save calibrator objects (pickle needed for sklearn objects)
    cal_bundle = {}
    for name, calibrator in calibrators.items():
        if calibrator is not None:
            cal_bundle[name] = calibrator
    with open(output_dir / "calibrators.pkl", "wb") as f:
        pickle.dump(cal_bundle, f)

    logger.info("Calibration report saved to %s", report_path)
    logger.info("Recommended method: %s", report.get("recommendation"))
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_logit(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Clip probabilities and compute logits safely."""
    clipped = np.clip(probs, eps, 1.0 - eps)
    return logit(clipped)


def _extract_params(calibrator) -> dict:
    """Extract human-readable parameters from a calibrator."""
    if isinstance(calibrator, TemperatureScaling):
        return {"temperature": round(calibrator.temperature, 4)}
    if isinstance(calibrator, PlattScaling):
        return {"a": round(calibrator.a, 4), "b": round(calibrator.b, 4)}
    if isinstance(calibrator, IsotonicCalibration):
        return {"type": "isotonic_regression", "n_knots": len(calibrator.ir.X_thresholds_)}
    if isinstance(calibrator, BayesianPriorCalibration):
        return {"strength": round(calibrator.strength, 4), "prior_rate": round(calibrator.prior_rate, 4)}
    if isinstance(calibrator, CompositeCalibration):
        return {
            "temperature": round(calibrator.temp.temperature, 4),
            "bayesian_strength": round(calibrator.bayesian.strength, 4),
            "prior_rate": round(calibrator.bayesian.prior_rate, 4),
        }
    return {}
