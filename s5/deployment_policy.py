"""
deployment_policy.py - Bedside alert-policy simulation and optimization helpers.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_POLICY_GRID = {
    "enter_thresholds": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
    "min_history_hours": [6],
    "hysteresis_gaps": [0.0, 0.05, 0.1],
    "min_consecutive_hours": [1, 2, 3],
    "refractory_hours": [6, 12, 24],
    "max_alerts_per_stay": [1, 2],
}

TIGHT_POLICY_GRID = {
    "enter_thresholds": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95],
    "min_history_hours": [6, 8, 10, 12, 18, 24],
    "hysteresis_gaps": [0.0, 0.05, 0.1],
    "min_consecutive_hours": [1, 2, 3, 4, 5, 6],
    "refractory_hours": [6, 12, 24, 36, 48],
    "max_alerts_per_stay": [1],
}

POLICY_GRID_PRESETS = {
    "default": DEFAULT_POLICY_GRID,
    "tight": TIGHT_POLICY_GRID,
}


def simulate_alert_policy(
    *,
    risk_matrix: np.ndarray,
    active_hours: np.ndarray,
    enter_threshold: float,
    exit_threshold: float | None = None,
    min_history_hours: int = 6,
    min_consecutive_hours: int = 1,
    refractory_hours: int = 0,
    max_alerts_per_stay: int | None = None,
) -> dict:
    """Simulate a deployment alert policy on hourly risk trajectories."""
    risk_matrix = np.asarray(risk_matrix, dtype=np.float32)
    active_hours = np.asarray(active_hours, dtype=int)
    n_patients, seq_len = risk_matrix.shape
    enter_threshold = float(enter_threshold)
    exit_threshold = enter_threshold if exit_threshold is None else float(exit_threshold)
    if exit_threshold > enter_threshold:
        raise ValueError("exit_threshold must be <= enter_threshold for hysteresis semantics")

    event_matrix = np.zeros((n_patients, seq_len), dtype=bool)
    alert_state_matrix = np.zeros((n_patients, seq_len), dtype=bool)
    first_event_hour = np.full(n_patients, np.nan, dtype=np.float32)
    n_alert_events = np.zeros(n_patients, dtype=np.int16)

    min_history_hours = max(1, int(min_history_hours))
    min_consecutive_hours = max(1, int(min_consecutive_hours))
    refractory_hours = max(0, int(refractory_hours))
    max_alerts = None if max_alerts_per_stay is None else max(1, int(max_alerts_per_stay))
    valid_hour_mask = np.arange(seq_len)[None, :] < np.clip(active_hours, 1, seq_len)[:, None]
    in_alert_state = np.zeros(n_patients, dtype=bool)
    consecutive_high = np.zeros(n_patients, dtype=np.int16)
    next_allowed_hour = np.full(n_patients, min_history_hours, dtype=int)

    for hour_idx in range(seq_len):
        hour = hour_idx + 1
        risk = risk_matrix[:, hour_idx]
        observed = valid_hour_mask[:, hour_idx] & np.isfinite(risk)
        if not np.any(observed):
            continue

        state_before = in_alert_state.copy()
        active_state = observed & state_before
        alert_state_matrix[active_state, hour_idx] = True
        exit_mask = active_state & (risk < exit_threshold)
        in_alert_state[exit_mask] = False

        inactive = observed & ~state_before
        if not np.any(inactive):
            continue

        blocked = inactive & (hour < min_history_hours)
        if max_alerts is not None:
            blocked |= inactive & (n_alert_events >= max_alerts)
        blocked |= inactive & (hour < next_allowed_hour)
        consecutive_high[blocked] = 0

        eligible = inactive & ~blocked
        if not np.any(eligible):
            continue

        high = eligible & (risk >= enter_threshold)
        low = eligible & (risk < enter_threshold)
        consecutive_high[high] += 1
        consecutive_high[low] = 0

        trigger = high & (consecutive_high >= min_consecutive_hours)
        if not np.any(trigger):
            continue

        event_matrix[trigger, hour_idx] = True
        alert_state_matrix[trigger, hour_idx] = True
        in_alert_state[trigger] = True
        n_alert_events[trigger] += 1
        first_mask = trigger & ~np.isfinite(first_event_hour)
        first_event_hour[first_mask] = float(hour)
        next_allowed_hour[trigger] = hour + refractory_hours + 1
        consecutive_high[trigger] = 0

    return {
        "event_matrix": event_matrix,
        "alert_state_matrix": alert_state_matrix,
        "first_event_hour": first_event_hour,
        "n_alert_events": n_alert_events,
        "enter_threshold": round(enter_threshold, 4),
        "exit_threshold": round(exit_threshold, 4),
        "min_history_hours": min_history_hours,
        "min_consecutive_hours": min_consecutive_hours,
        "refractory_hours": refractory_hours,
        "max_alerts_per_stay": max_alerts,
    }


def evaluate_alert_policy(
    *,
    labels: np.ndarray,
    risk_matrix: np.ndarray,
    active_hours: np.ndarray,
    policy_name: str,
    enter_threshold: float,
    exit_threshold: float | None = None,
    min_history_hours: int = 6,
    min_consecutive_hours: int = 1,
    refractory_hours: int = 0,
    max_alerts_per_stay: int | None = None,
    landmark_hours: tuple[int, ...] | list[int] = (6, 12, 24, 36, 48),
) -> dict:
    """Evaluate an alert policy with event-based deployment metrics."""
    labels = np.asarray(labels, dtype=int)
    risk_matrix = np.asarray(risk_matrix, dtype=np.float32)
    active_hours = np.asarray(active_hours, dtype=int)
    policy = simulate_alert_policy(
        risk_matrix=risk_matrix,
        active_hours=active_hours,
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold,
        min_history_hours=min_history_hours,
        min_consecutive_hours=min_consecutive_hours,
        refractory_hours=refractory_hours,
        max_alerts_per_stay=max_alerts_per_stay,
    )
    event_matrix = policy["event_matrix"]
    alert_state_matrix = policy["alert_state_matrix"]
    first_event_hour = policy["first_event_hour"]
    positives = labels == 1
    negatives = ~positives
    patient_days = max(float(np.sum(active_hours) / 24.0), 1.0e-6)

    ever_event = event_matrix.any(axis=1)
    summary = {
        "policy_name": policy_name,
        "enter_threshold": policy["enter_threshold"],
        "exit_threshold": policy["exit_threshold"],
        "min_history_hours": policy["min_history_hours"],
        "min_consecutive_hours": policy["min_consecutive_hours"],
        "refractory_hours": policy["refractory_hours"],
        "max_alerts_per_stay": policy["max_alerts_per_stay"],
        "n_patients": int(len(labels)),
        "positive_rate": round(float(np.mean(labels)), 4),
        "patient_alert_rate": round(float(np.mean(ever_event)), 4),
        "positive_patient_alert_rate": round(float(np.mean(ever_event[positives])), 4) if np.any(positives) else None,
        "negative_patient_alert_rate": round(float(np.mean(ever_event[negatives])), 4) if np.any(negatives) else None,
        "alert_events_per_patient_day": round(float(event_matrix.sum() / patient_days), 4),
        "alert_state_hours_per_patient_day": round(float(alert_state_matrix.sum() / patient_days), 4),
        "median_first_alert_hour_positive": _median_or_none(first_event_hour[positives & np.isfinite(first_event_hour)]),
        "median_first_alert_hour_all": _median_or_none(first_event_hour[np.isfinite(first_event_hour)]),
        "mean_alert_events_per_alerted_patient": round(float(np.mean(policy["n_alert_events"][ever_event])), 4) if np.any(ever_event) else 0.0,
    }

    landmark_rows = []
    seq_len = int(risk_matrix.shape[1])
    for hour in sorted({min(seq_len, max(1, int(value))) for value in landmark_hours}):
        eligible = active_hours >= hour
        if not np.any(eligible):
            continue
        cumulative = event_matrix[:, :hour].any(axis=1)
        row = {
            "hour": int(hour),
            "n_patients": int(eligible.sum()),
            "patient_alert_rate": round(float(np.mean(cumulative[eligible])), 4),
            "positive_alert_rate": round(float(np.mean(cumulative[eligible & positives])), 4) if np.any(eligible & positives) else None,
            "negative_alert_rate": round(float(np.mean(cumulative[eligible & negatives])), 4) if np.any(eligible & negatives) else None,
        }
        landmark_rows.append(row)
        summary[f"positive_alert_rate_at_{hour}h"] = row["positive_alert_rate"]
        summary[f"negative_alert_rate_at_{hour}h"] = row["negative_alert_rate"]
        summary[f"patient_alert_rate_at_{hour}h"] = row["patient_alert_rate"]

    summary["cumulative_alert_metrics"] = landmark_rows
    return summary


def build_policy_grid(
    *,
    enter_thresholds: list[float],
    min_history_hours: int | list[int],
    hysteresis_gaps: list[float],
    min_consecutive_hours: list[int],
    refractory_hours: list[int],
    max_alerts_per_stay: list[int | None],
) -> list[dict]:
    """Build candidate deployment policies from a simple grid."""
    candidates = []
    counter = 0
    min_history_candidates = _normalize_int_grid(min_history_hours)
    for enter, history_hours, gap, consecutive, refractory, max_alerts in itertools.product(
        enter_thresholds,
        min_history_candidates,
        hysteresis_gaps,
        min_consecutive_hours,
        refractory_hours,
        max_alerts_per_stay,
    ):
        enter = float(enter)
        exit_threshold = max(0.0, round(enter - float(gap), 4))
        counter += 1
        candidates.append(
            {
                "policy_name": f"policy_{counter:04d}",
                "enter_threshold": enter,
                "exit_threshold": exit_threshold,
                "min_history_hours": int(history_hours),
                "min_consecutive_hours": int(consecutive),
                "refractory_hours": int(refractory),
                "max_alerts_per_stay": None if max_alerts is None else int(max_alerts),
            }
        )
    return candidates


def evaluate_policy_grid(
    *,
    labels: np.ndarray,
    risk_matrix: np.ndarray,
    active_hours: np.ndarray,
    candidates: list[dict],
    landmark_hours: tuple[int, ...] | list[int] = (6, 12, 24, 36, 48),
) -> pd.DataFrame:
    """Evaluate a list of policy candidates and return a sortable DataFrame."""
    rows = []
    for candidate in candidates:
        summary = evaluate_alert_policy(
            labels=labels,
            risk_matrix=risk_matrix,
            active_hours=active_hours,
            policy_name=str(candidate["policy_name"]),
            enter_threshold=float(candidate["enter_threshold"]),
            exit_threshold=float(candidate["exit_threshold"]),
            min_history_hours=int(candidate["min_history_hours"]),
            min_consecutive_hours=int(candidate["min_consecutive_hours"]),
            refractory_hours=int(candidate["refractory_hours"]),
            max_alerts_per_stay=candidate.get("max_alerts_per_stay"),
            landmark_hours=landmark_hours,
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def select_best_policy(
    policy_frame: pd.DataFrame,
    *,
    max_negative_patient_alert_rate: float = 0.25,
    max_alert_events_per_patient_day: float = 1.0,
    min_positive_patient_alert_rate: float = 0.6,
    min_positive_alert_rate_24h: float = 0.5,
    ranking_mode: str = "balanced",
) -> tuple[pd.Series, pd.DataFrame]:
    """Rank candidates by hard constraints then clinical utility."""
    if policy_frame.empty:
        raise ValueError("policy_frame must contain at least one candidate")
    frame = policy_frame.copy()
    if "positive_alert_rate_at_24h" not in frame.columns:
        frame["positive_alert_rate_at_24h"] = np.nan
    negative_patient_alert_rate = pd.to_numeric(frame["negative_patient_alert_rate"], errors="coerce").fillna(np.inf)
    alert_events_per_patient_day = pd.to_numeric(frame["alert_events_per_patient_day"], errors="coerce").fillna(np.inf)
    positive_patient_alert_rate = pd.to_numeric(frame["positive_patient_alert_rate"], errors="coerce").fillna(0.0)
    positive_alert_rate_at_24h = pd.to_numeric(frame["positive_alert_rate_at_24h"], errors="coerce").fillna(0.0)
    penalties = (
        np.maximum(negative_patient_alert_rate - float(max_negative_patient_alert_rate), 0.0)
        + np.maximum(alert_events_per_patient_day - float(max_alert_events_per_patient_day), 0.0)
        + np.maximum(float(min_positive_patient_alert_rate) - positive_patient_alert_rate, 0.0)
        + np.maximum(float(min_positive_alert_rate_24h) - positive_alert_rate_at_24h, 0.0)
    )
    frame["constraint_penalty"] = penalties.round(6)
    frame["feasible"] = penalties <= 1.0e-9
    ranking_mode = str(ranking_mode).strip().lower()
    if ranking_mode == "balanced":
        ranked = frame.sort_values(
            by=[
                "constraint_penalty",
                "positive_alert_rate_at_24h",
                "positive_patient_alert_rate",
                "negative_patient_alert_rate",
                "alert_events_per_patient_day",
                "median_first_alert_hour_positive",
            ],
            ascending=[True, False, False, True, True, True],
            na_position="last",
        ).reset_index(drop=True)
    elif ranking_mode == "burden_first":
        ranked = frame.sort_values(
            by=[
                "constraint_penalty",
                "negative_patient_alert_rate",
                "alert_events_per_patient_day",
                "alert_state_hours_per_patient_day",
                "patient_alert_rate",
                "positive_alert_rate_at_24h",
                "positive_patient_alert_rate",
                "median_first_alert_hour_positive",
                "enter_threshold",
                "min_consecutive_hours",
                "min_history_hours",
            ],
            ascending=[True, True, True, True, True, False, False, True, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported ranking_mode: {ranking_mode}")
    return ranked.iloc[0], ranked


def write_policy_artifacts(
    *,
    output_dir: Path,
    source_name: str,
    ranked_candidates: pd.DataFrame,
    best_policy: pd.Series,
    constraints: dict,
) -> dict:
    """Persist policy sweep results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{source_name}_policy_candidates.csv"
    ranked_candidates.to_csv(csv_path, index=False)

    best_summary = best_policy.to_dict()
    json_path = output_dir / f"{source_name}_best_policy.json"
    with open(json_path, "w", encoding="utf-8") as f:
        import json

        json.dump(
            {
                "source": source_name,
                "constraints": constraints,
                "best_policy": _json_safe(best_summary),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {
        "candidates_csv": str(csv_path),
        "best_policy_json": str(json_path),
    }


def load_policy_artifact(policy_path: Path) -> dict:
    """Load a persisted best-policy artifact and normalize its payload."""
    with open(Path(policy_path), encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("policy artifact must be a JSON object")
    raw_policy = payload.get("best_policy", payload)
    if not isinstance(raw_policy, dict):
        raise ValueError("policy artifact does not contain a valid best_policy object")
    return {
        "source": payload.get("source"),
        "constraints": payload.get("constraints", {}),
        "policy": _json_safe(raw_policy),
        "path": str(policy_path),
    }


def load_replay_bundle(bundle_path: Path) -> dict:
    """Load a saved silent-deployment replay bundle."""
    payload = np.load(Path(bundle_path), allow_pickle=False)
    out = {}
    for key in payload.files:
        value = payload[key]
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        out[key] = value
    if "patient_ids" in out:
        out["patient_ids"] = np.asarray(out["patient_ids"]).astype(str)
    return out


def _median_or_none(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return round(float(np.median(values)), 2)


def _json_safe(payload: dict) -> dict:
    clean = {}
    for key, value in payload.items():
        if isinstance(value, (np.bool_, bool)):
            clean[key] = bool(value)
        elif isinstance(value, (np.integer,)):
            clean[key] = int(value)
        elif isinstance(value, (np.floating,)):
            clean[key] = None if np.isnan(value) else float(value)
        else:
            clean[key] = value
    return clean


def _normalize_int_grid(values: int | list[int]) -> list[int]:
    if isinstance(values, int):
        candidates = [values]
    else:
        candidates = list(values)
    return sorted({max(1, int(value)) for value in candidates})
