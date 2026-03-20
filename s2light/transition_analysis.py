"""
transition_analysis.py - Transition matrix, stability analysis, mortality descriptives.

All analyses are descriptive. No causal claims. No treatment-response language.
Reports patient-level and event-level summaries separately.
Reports by split (train/val/test) and all-cohort.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger("s2light.transitions")


def compute_transitions(
    window_labels: np.ndarray,
    static: pd.DataFrame,
    splits: dict,
    k: int = 4,
) -> dict:
    """
    Compute all transition statistics.

    Parameters
    ----------
    window_labels: (N, W) int
    static: DataFrame with mortality_inhospital, center_id
    splits: dict with train/val/test indices

    Returns comprehensive transition report.
    """
    N, W = window_labels.shape
    mortality = static["mortality_inhospital"].fillna(0).values

    report = {
        "n_patients": N,
        "n_windows": W,
        "n_adjacent_pairs": W - 1,
        "k": k,
    }

    # === Event-level: transition matrix ===
    trans_counts = np.zeros((k, k), dtype=int)
    for t in range(W - 1):
        for i in range(N):
            src = window_labels[i, t]
            dst = window_labels[i, t + 1]
            trans_counts[src, dst] += 1

    total_events = trans_counts.sum()
    self_events = np.trace(trans_counts)
    non_self_events = total_events - self_events

    # Normalized probabilities (row-wise)
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    trans_probs = trans_counts / row_sums

    report["event_level"] = {
        "transition_count_matrix": trans_counts.tolist(),
        "transition_prob_matrix": np.round(trans_probs, 4).tolist(),
        "total_transition_events": int(total_events),
        "self_transition_events": int(self_events),
        "non_self_transition_events": int(non_self_events),
        "non_self_fraction": round(non_self_events / max(total_events, 1), 4),
    }

    # Top non-self transitions
    top_non_self = []
    for i in range(k):
        for j in range(k):
            if i != j and trans_counts[i, j] > 0:
                top_non_self.append({
                    "from": int(i), "to": int(j),
                    "count": int(trans_counts[i, j]),
                    "prob": round(float(trans_probs[i, j]), 4),
                })
    top_non_self.sort(key=lambda x: x["count"], reverse=True)
    report["event_level"]["top_non_self_transitions"] = top_non_self[:10]

    # Transition entropy (concentration statistic)
    flat_probs = trans_probs.flatten()
    flat_probs = flat_probs[flat_probs > 0]
    entropy = -float(np.sum(flat_probs * np.log2(flat_probs + 1e-12)))
    max_entropy = np.log2(k * k)
    report["event_level"]["transition_entropy"] = round(entropy, 4)
    report["event_level"]["max_possible_entropy"] = round(max_entropy, 4)
    report["event_level"]["entropy_ratio"] = round(entropy / max_entropy, 4)

    # === Patient-level: trajectory patterns ===
    trajectories = [tuple(window_labels[i].tolist()) for i in range(N)]
    n_transitions_per_patient = np.array([
        sum(1 for t in range(W - 1) if window_labels[i, t] != window_labels[i, t + 1])
        for i in range(N)
    ])

    stable_mask = n_transitions_per_patient == 0
    single_mask = n_transitions_per_patient == 1
    multi_mask = n_transitions_per_patient >= 2

    report["patient_level"] = {
        "stable_count": int(stable_mask.sum()),
        "stable_fraction": round(float(stable_mask.mean()), 4),
        "single_transition_count": int(single_mask.sum()),
        "single_transition_fraction": round(float(single_mask.mean()), 4),
        "multi_transition_count": int(multi_mask.sum()),
        "multi_transition_fraction": round(float(multi_mask.mean()), 4),
        "mean_transitions_per_patient": round(float(n_transitions_per_patient.mean()), 4),
    }

    # Top trajectory patterns
    traj_counts = Counter(trajectories)
    top_patterns = traj_counts.most_common(15)
    report["patient_level"]["top_trajectory_patterns"] = [
        {"pattern": list(p), "count": c, "fraction": round(c / N, 4)}
        for p, c in top_patterns
    ]

    # === Mortality descriptives by trajectory category ===
    def _mort_summary(mask, label):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "mortality_rate": None, "label": label}
        rate = float(mortality[mask].mean())
        return {"n": n, "mortality_rate": round(rate, 4), "label": label}

    report["mortality_descriptives"] = {
        "all_cohort": {
            "stable": _mort_summary(stable_mask, "stable"),
            "single_transition": _mort_summary(single_mask, "single_transition"),
            "multi_transition": _mort_summary(multi_mask, "multi_transition"),
        },
    }

    # Per-split descriptives
    for split_name in ["train", "val", "test"]:
        idx = np.array(splits[split_name])
        split_stable = stable_mask[idx]
        split_single = single_mask[idx]
        split_multi = multi_mask[idx]
        split_mort = mortality[idx]

        def _split_mort(smask):
            n = int(smask.sum())
            return {"n": n, "mortality_rate": round(float(split_mort[smask].mean()), 4) if n > 0 else None}

        report["mortality_descriptives"][split_name] = {
            "stable": _split_mort(split_stable),
            "single_transition": _split_mort(split_single),
            "multi_transition": _split_mort(split_multi),
        }

    # Mortality by stable phenotype (which cluster do stable patients stay in?)
    report["mortality_by_stable_phenotype"] = {}
    for c in range(k):
        stable_in_c = stable_mask & (window_labels[:, 0] == c)
        n = int(stable_in_c.sum())
        rate = round(float(mortality[stable_in_c].mean()), 4) if n > 0 else None
        report["mortality_by_stable_phenotype"][f"cluster_{c}"] = {"n": n, "mortality_rate": rate}

    # === Overlap-induced trivial stability flag ===
    report["stability_assessment"] = {
        "stable_fraction": report["patient_level"]["stable_fraction"],
        "non_self_event_fraction": report["event_level"]["non_self_fraction"],
        "entropy_ratio": report["event_level"]["entropy_ratio"],
    }

    if report["patient_level"]["stable_fraction"] > 0.90:
        report["stability_assessment"]["flag"] = (
            "WARNING: >90% patients are stable. Heavy window overlap (75%) may suppress transitions. "
            "Recommend sensitivity analysis with stride=12h."
        )
        logger.warning(report["stability_assessment"]["flag"])
    elif report["patient_level"]["stable_fraction"] > 0.75:
        report["stability_assessment"]["flag"] = (
            "NOTE: >75% patients are stable. This may reflect genuine clinical stability "
            "or overlap-induced smoothing. Consider stride=12h sensitivity check."
        )
        logger.info(report["stability_assessment"]["flag"])
    else:
        report["stability_assessment"]["flag"] = "OK: Meaningful transition activity detected."
        logger.info(report["stability_assessment"]["flag"])

    return report
