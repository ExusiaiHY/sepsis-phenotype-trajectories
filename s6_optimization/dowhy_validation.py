"""
dowhy_validation.py - Add DoWhy-based causal identification and refutation to S6.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger("s6.dowhy")


def _ensure_networkx_d_separated() -> None:
    """
    Dowhy 0.8 expects nx.algorithms.d_separated, which was removed in newer
    networkx versions in favor of nx.algorithms.d_separation.is_d_separator.
    """
    if hasattr(nx.algorithms, "d_separated"):
        return
    if not hasattr(nx.algorithms, "d_separation"):
        return

    def _compat_d_separated(graph, x, y, z):
        return nx.algorithms.d_separation.is_d_separator(graph, x, y, z)

    nx.algorithms.d_separated = _compat_d_separated


def run_dowhy_validation(
    *,
    X_covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    output_dir: Path,
    config: dict | None = None,
) -> dict:
    cfg = {
        "enabled": True,
        "max_samples": 5000,
        "random_common_cause_simulations": 20,
        "placebo_treatment_simulations": 20,
    }
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    if not cfg.get("enabled", True):
        return {"enabled": False}

    _ensure_networkx_d_separated()
    from dowhy import CausalModel

    output_dir = Path(output_dir)
    max_samples = min(int(cfg["max_samples"]), len(X_covariates))
    covariate_cols = [f"x_{i:03d}" for i in range(X_covariates.shape[1])]
    data = pd.DataFrame(X_covariates[:max_samples], columns=covariate_cols)
    data["treatment"] = treatment[:max_samples].astype(int)
    data["outcome"] = outcome[:max_samples].astype(float)

    logger.info("Running DoWhy validation on %d samples and %d covariates", max_samples, len(covariate_cols))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model = CausalModel(
                data=data,
                treatment="treatment",
                outcome="outcome",
                common_causes=covariate_cols,
            )
            estimand = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                estimand,
                method_name="backdoor.linear_regression",
                test_significance=False,
            )
            random_refuter = model.refute_estimate(
                estimand,
                estimate,
                method_name="random_common_cause",
                num_simulations=int(cfg["random_common_cause_simulations"]),
            )
            placebo_refuter = model.refute_estimate(
                estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=int(cfg["placebo_treatment_simulations"]),
            )
        report = {
            "enabled": True,
            "n_samples": max_samples,
            "ate": round(float(estimate.value), 6),
            "identified_estimand": str(estimand),
            "refuters": {
                "random_common_cause": {
                    "new_effect": round(float(random_refuter.new_effect), 6),
                    "p_value": None if random_refuter.refutation_result is None else random_refuter.refutation_result.get("p_value"),
                },
                "placebo_treatment_refuter": {
                    "new_effect": round(float(placebo_refuter.new_effect), 6),
                    "p_value": None if placebo_refuter.refutation_result is None else placebo_refuter.refutation_result.get("p_value"),
                },
            },
        }
    except Exception as exc:
        logger.warning("DoWhy validation failed: %s", exc)
        report = {
            "enabled": True,
            "failed": True,
            "error": str(exc),
            "n_samples": max_samples,
        }

    path = output_dir / "dowhy_validation.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(path)
    return report
