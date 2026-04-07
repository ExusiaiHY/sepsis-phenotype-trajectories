"""
silent_deployment.py - Replay bedside silent deployment on frozen Stage 5 artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from s4.treatment_aware_model import _load_or_build_splits
from s5.dashboard import render_clinical_dashboard_html
from s5.deployment_policy import load_policy_artifact, simulate_alert_policy
from s5.realtime_model import _align_stage5_inputs, load_realtime_student_artifact

DEFAULT_LANDMARK_HOURS = (6, 12, 24, 36, 48)


def run_silent_deployment_replay(
    *,
    model_artifact_path: Path,
    s0_dir: Path,
    treatment_dir: Path,
    output_dir: Path,
    policy_path: Path | None = None,
    note_embeddings_path: Path | None = None,
    label_col: str = "mortality_inhospital",
    split: str = "test",
    min_history_hours: int = 6,
    landmark_hours: tuple[int, ...] | list[int] = DEFAULT_LANDMARK_HOURS,
    batch_size: int = 256,
    device: str = "cpu",
    max_patients: int | None = None,
    save_replay_bundle: bool = False,
    replay_bundle_name: str = "replay_bundle.npz",
) -> dict:
    """Replay a frozen realtime student in silent bedside mode."""
    s0_dir = Path(s0_dir)
    treatment_dir = Path(treatment_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    continuous = np.load(s0_dir / "processed" / "continuous.npy")
    masks_cont = np.load(s0_dir / "processed" / "masks_continuous.npy")
    treatments = np.load(treatment_dir / "treatments.npy")
    masks_treat = np.load(treatment_dir / "masks_treatments.npy")
    s0_static = pd.read_csv(s0_dir / "static.csv")
    treatment_static_path = treatment_dir / "cohort_static.csv"
    if treatment_static_path.exists():
        treatment_static = pd.read_csv(treatment_static_path)
    else:
        treatment_static = s0_static[["patient_id"]].copy()
    notes = None
    if note_embeddings_path is not None and Path(note_embeddings_path).exists():
        notes = np.load(note_embeddings_path)

    aligned = _align_stage5_inputs(
        s0_static=s0_static,
        treatment_static=treatment_static,
        continuous=continuous,
        masks_continuous=masks_cont,
        treatments=treatments,
        masks_treatments=masks_treat,
        note_embeddings=notes,
    )
    continuous = aligned["continuous"]
    masks_cont = aligned["masks_continuous"]
    treatments = aligned["treatments"]
    masks_treat = aligned["masks_treatments"]
    static = aligned["s0_static"]
    notes = aligned["note_embeddings"]

    labels = static[label_col].fillna(0).astype(int).to_numpy()
    splits = _load_or_build_splits(
        n_samples=len(labels),
        labels=labels,
        splits_path=s0_dir / "splits.json",
        seed=42,
    )
    indices = np.asarray(splits[split], dtype=int)
    if max_patients is not None:
        indices = indices[: max(1, int(max_patients))]

    continuous = continuous[indices].astype(np.float32, copy=False)
    masks_cont = masks_cont[indices].astype(np.float32, copy=False)
    treatments = treatments[indices].astype(np.float32, copy=False)
    masks_treat = masks_treat[indices].astype(np.float32, copy=False)
    static = static.iloc[indices].reset_index(drop=True)
    labels = labels[indices]
    if notes is not None:
        notes = np.asarray(notes[indices], dtype=np.float32)
        if notes.ndim == 2:
            notes = np.repeat(notes[:, None, :], continuous.shape[1], axis=1)

    artifact = load_realtime_student_artifact(model_artifact_path, device=device)
    model = artifact["model"]
    model_threshold = float(artifact["threshold"])
    temperature = max(float(artifact["temperature"]), 1.0e-3)
    seq_len = int(continuous.shape[1])
    landmark_hours = tuple(
        sorted(
            {
                min(seq_len, max(1, int(hour)))
                for hour in landmark_hours
            }
        )
    )
    min_history_hours = min(seq_len, max(1, int(min_history_hours)))

    risk_matrix = _replay_partial_probabilities(
        model=model,
        continuous=continuous,
        masks_continuous=masks_cont,
        treatments=treatments,
        masks_treatments=masks_treat,
        note_embeddings=notes,
        batch_size=batch_size,
        device=device,
        temperature=temperature,
    )

    active_hours = _derive_active_hours(
        static=static,
        masks_continuous=masks_cont,
        masks_treatments=masks_treat,
        note_embeddings=notes,
    )
    valid_hour_mask = np.arange(seq_len)[None, :] < active_hours[:, None]

    risk_matrix_masked = risk_matrix.astype(np.float32, copy=True)
    risk_matrix_masked[~valid_hour_mask] = np.nan
    deployment_policy = None if policy_path is None else load_policy_artifact(policy_path)
    policy_summary = None
    alert_threshold = float(model_threshold)
    landmark_threshold = float(model_threshold)
    effective_min_history_hours = int(min_history_hours)
    if deployment_policy is None:
        alert_event_matrix = risk_matrix >= model_threshold
        alert_event_matrix &= valid_hour_mask
        alert_event_matrix[:, : max(0, min_history_hours - 1)] = False
        alert_state_matrix = alert_event_matrix.copy()
        deployment_mode = "threshold"
    else:
        policy_cfg = deployment_policy["policy"]
        policy_result = simulate_alert_policy(
            risk_matrix=risk_matrix_masked,
            active_hours=active_hours,
            enter_threshold=float(policy_cfg["enter_threshold"]),
            exit_threshold=policy_cfg.get("exit_threshold"),
            min_history_hours=int(policy_cfg.get("min_history_hours", min_history_hours)),
            min_consecutive_hours=int(policy_cfg.get("min_consecutive_hours", 1)),
            refractory_hours=int(policy_cfg.get("refractory_hours", 0)),
            max_alerts_per_stay=policy_cfg.get("max_alerts_per_stay"),
        )
        alert_event_matrix = policy_result["event_matrix"]
        alert_state_matrix = policy_result["alert_state_matrix"]
        deployment_mode = "policy"
        alert_threshold = float(policy_result["enter_threshold"])
        landmark_threshold = float(policy_result["enter_threshold"])
        effective_min_history_hours = int(policy_result["min_history_hours"])
        policy_summary = {
            "source": deployment_policy.get("source"),
            "path": deployment_policy.get("path"),
            "constraints": deployment_policy.get("constraints", {}),
            "policy": {
                "policy_name": policy_result.get("policy_name", policy_cfg.get("policy_name")),
                "enter_threshold": float(policy_result["enter_threshold"]),
                "exit_threshold": float(policy_result["exit_threshold"]),
                "min_history_hours": int(policy_result["min_history_hours"]),
                "min_consecutive_hours": int(policy_result["min_consecutive_hours"]),
                "refractory_hours": int(policy_result["refractory_hours"]),
                "max_alerts_per_stay": policy_result.get("max_alerts_per_stay"),
            },
        }

    treatment_feature_names = _load_treatment_feature_names(treatment_dir, n_treat=treatments.shape[-1])
    patient_summary = _build_patient_summary(
        static=static,
        labels=labels,
        risk_matrix=risk_matrix_masked,
        alert_state_matrix=alert_state_matrix,
        alert_event_matrix=alert_event_matrix,
        active_hours=active_hours,
        landmark_hours=landmark_hours,
    )
    landmark_metrics = _build_landmark_metrics(
        labels=labels,
        risk_matrix=risk_matrix_masked,
        active_hours=active_hours,
        threshold=landmark_threshold,
        landmark_hours=landmark_hours,
    )
    cumulative_alert_metrics = _build_cumulative_alert_metrics(
        labels=labels,
        alert_event_matrix=alert_event_matrix,
        active_hours=active_hours,
    )

    sample_idx = _select_sample_patient(patient_summary)
    sample_patient_id = str(static.iloc[sample_idx]["patient_id"])
    snapshots = _build_patient_snapshots(
        risk_row=risk_matrix_masked[sample_idx],
        alert_row=alert_state_matrix[sample_idx],
        treatment_row=treatments[sample_idx],
        treatment_mask_row=masks_treat[sample_idx],
        treatment_feature_names=treatment_feature_names,
        active_hours=int(active_hours[sample_idx]),
    )
    snapshots_path = output_dir / "sample_patient_snapshots.json"
    with open(snapshots_path, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=2)

    dashboard_path = output_dir / "sample_patient_dashboard.html"
    render_clinical_dashboard_html(
        patient_id=sample_patient_id,
        snapshots=snapshots,
        output_path=dashboard_path,
        model_meta={
            "artifact_path": str(model_artifact_path),
            "threshold": round(alert_threshold, 4),
            "model_threshold": round(model_threshold, 4),
            "temperature": round(temperature, 4),
            "min_history_hours": int(effective_min_history_hours),
            "split": split,
            "deployment_mode": deployment_mode,
            "policy": policy_summary,
            "model": artifact["config"],
        },
    )

    patient_summary_path = output_dir / "patient_summary.csv"
    patient_summary.to_csv(patient_summary_path, index=False)
    landmark_path = output_dir / "landmark_metrics.csv"
    landmark_metrics.to_csv(landmark_path, index=False)
    cumulative_path = output_dir / "cumulative_alert_metrics.csv"
    cumulative_alert_metrics.to_csv(cumulative_path, index=False)

    summary = _build_summary(
        static=static,
        labels=labels,
        patient_summary=patient_summary,
        landmark_metrics=landmark_metrics,
        cumulative_alert_metrics=cumulative_alert_metrics,
        model_artifact_path=model_artifact_path,
        threshold=alert_threshold,
        model_threshold=model_threshold,
        temperature=temperature,
        split=split,
        min_history_hours=effective_min_history_hours,
        deployment_mode=deployment_mode,
        deployment_policy=policy_summary,
        sample_patient_id=sample_patient_id,
        dashboard_path=dashboard_path,
        snapshots_path=snapshots_path,
    )
    if save_replay_bundle:
        replay_bundle_path = output_dir / str(replay_bundle_name)
        _write_replay_bundle(
            bundle_path=replay_bundle_path,
            risk_matrix=risk_matrix_masked,
            active_hours=active_hours,
            labels=labels,
            patient_ids=_resolve_patient_ids(static),
            threshold=model_threshold,
            temperature=temperature,
            split=split,
            min_history_hours=effective_min_history_hours,
            landmark_hours=landmark_hours,
            model_artifact_path=model_artifact_path,
        )
        summary["artifacts"]["replay_bundle_npz"] = str(replay_bundle_path)
    summary_path = output_dir / "silent_deployment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _replay_partial_probabilities(
    *,
    model: torch.nn.Module,
    continuous: np.ndarray,
    masks_continuous: np.ndarray,
    treatments: np.ndarray,
    masks_treatments: np.ndarray,
    note_embeddings: np.ndarray | None,
    batch_size: int,
    device: str,
    temperature: float,
) -> np.ndarray:
    n_patients, seq_len, _ = continuous.shape
    risk_matrix = np.zeros((n_patients, seq_len), dtype=np.float32)
    for start in range(0, n_patients, batch_size):
        stop = min(start + batch_size, n_patients)
        batch_x = continuous[start:stop]
        batch_mask = masks_continuous[start:stop]
        batch_treat = treatments[start:stop]
        batch_treat_mask = masks_treatments[start:stop]
        batch_notes = None if note_embeddings is None else note_embeddings[start:stop]

        x_partial = np.zeros_like(batch_x, dtype=np.float32)
        mask_partial = np.zeros_like(batch_mask, dtype=np.float32)
        treat_partial = np.zeros_like(batch_treat, dtype=np.float32)
        treat_mask_partial = np.zeros_like(batch_treat_mask, dtype=np.float32)
        notes_partial = None if batch_notes is None else np.zeros_like(batch_notes, dtype=np.float32)

        for hour in range(seq_len):
            x_partial[:, hour] = batch_x[:, hour]
            mask_partial[:, hour] = batch_mask[:, hour]
            treat_partial[:, hour] = batch_treat[:, hour]
            treat_mask_partial[:, hour] = batch_treat_mask[:, hour]
            if notes_partial is not None:
                notes_partial[:, hour] = batch_notes[:, hour]
            risk_matrix[start:stop, hour] = _predict_probabilities(
                model=model,
                continuous=x_partial,
                masks_continuous=mask_partial,
                treatments=treat_partial,
                masks_treatments=treat_mask_partial,
                note_embeddings=notes_partial,
                device=device,
                temperature=temperature,
            )
    return risk_matrix


def _predict_probabilities(
    *,
    model: torch.nn.Module,
    continuous: np.ndarray,
    masks_continuous: np.ndarray,
    treatments: np.ndarray,
    masks_treatments: np.ndarray,
    note_embeddings: np.ndarray | None,
    device: str,
    temperature: float,
) -> np.ndarray:
    with torch.no_grad():
        out = model(
            torch.from_numpy(continuous).to(device),
            torch.from_numpy(masks_continuous).to(device),
            torch.from_numpy(treatments).to(device),
            torch.from_numpy(masks_treatments).to(device),
            note_embeddings=None if note_embeddings is None else torch.from_numpy(note_embeddings).to(device),
        )
        probs = torch.sigmoid(out["logits"] / temperature).cpu().numpy()
    return probs.astype(np.float32, copy=False)


def _derive_active_hours(
    *,
    static: pd.DataFrame,
    masks_continuous: np.ndarray,
    masks_treatments: np.ndarray,
    note_embeddings: np.ndarray | None,
) -> np.ndarray:
    seq_len = int(masks_continuous.shape[1])
    observed = (masks_continuous.sum(axis=-1) > 0) | (masks_treatments.sum(axis=-1) > 0)
    if note_embeddings is not None:
        observed |= (np.abs(note_embeddings).sum(axis=-1) > 0)
    active_from_masks = np.full(len(static), 1, dtype=int)
    any_obs = observed.any(axis=1)
    if np.any(any_obs):
        active_from_masks[any_obs] = seq_len - np.argmax(observed[any_obs, ::-1], axis=1)
    if "icu_los_hours" in static.columns:
        los_hours = pd.to_numeric(static["icu_los_hours"], errors="coerce").to_numpy(dtype=float)
        valid_los = np.isfinite(los_hours)
        los_cap = np.clip(np.ceil(np.nan_to_num(los_hours, nan=float(seq_len))).astype(int), 1, seq_len)
        active_from_masks = np.where(valid_los, los_cap, active_from_masks)
    return np.clip(active_from_masks.astype(int), 1, seq_len)


def _build_patient_summary(
    *,
    static: pd.DataFrame,
    labels: np.ndarray,
    risk_matrix: np.ndarray,
    alert_state_matrix: np.ndarray,
    alert_event_matrix: np.ndarray,
    active_hours: np.ndarray,
    landmark_hours: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for idx in range(len(static)):
        valid_hours = int(active_hours[idx])
        valid_risk = risk_matrix[idx, :valid_hours]
        valid_alert_state = alert_state_matrix[idx, :valid_hours]
        valid_alert_events = alert_event_matrix[idx, :valid_hours]
        first_alert_hour = int(np.argmax(valid_alert_events) + 1) if np.any(valid_alert_events) else None
        max_risk_hour = int(np.nanargmax(valid_risk) + 1)
        row = {
            "patient_id": str(static.iloc[idx]["patient_id"]),
            "label": int(labels[idx]),
            "active_hours": valid_hours,
            "ever_alert": bool(np.any(valid_alert_events)),
            "first_alert_hour": first_alert_hour,
            "n_alert_events": int(valid_alert_events.sum()),
            "n_alert_hours": int(valid_alert_state.sum()),
            "alert_fraction": round(float(valid_alert_state.mean()), 4),
            "max_risk_probability": round(float(np.nanmax(valid_risk)), 4),
            "max_risk_hour": max_risk_hour,
            "terminal_risk_probability": round(float(valid_risk[valid_hours - 1]), 4),
        }
        for column in ("center_id", "icu_type", "data_source", "set_name", "anchor_time_type"):
            if column in static.columns:
                value = static.iloc[idx][column]
                row[column] = None if pd.isna(value) else str(value)
        for hour in landmark_hours:
            row[f"risk_at_{hour}h"] = (
                round(float(risk_matrix[idx, hour - 1]), 4)
                if valid_hours >= hour
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_landmark_metrics(
    *,
    labels: np.ndarray,
    risk_matrix: np.ndarray,
    active_hours: np.ndarray,
    threshold: float,
    landmark_hours: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for hour in landmark_hours:
        eligible = active_hours >= hour
        if not np.any(eligible):
            continue
        probs = risk_matrix[eligible, hour - 1]
        y_true = labels[eligible]
        metrics = _classification_metrics_safe(y_true, probs, threshold)
        metrics.update(
            {
                "hour": int(hour),
                "n_patients": int(eligible.sum()),
                "threshold": round(float(threshold), 4),
                "mean_risk_probability": round(float(np.nanmean(probs)), 4),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _build_cumulative_alert_metrics(
    *,
    labels: np.ndarray,
    alert_event_matrix: np.ndarray,
    active_hours: np.ndarray,
) -> pd.DataFrame:
    rows = []
    positives = labels == 1
    negatives = labels == 0
    seq_len = int(alert_event_matrix.shape[1])
    for hour in range(1, seq_len + 1):
        eligible = active_hours >= hour
        if not np.any(eligible):
            continue
        alerted = alert_event_matrix[:, :hour].any(axis=1)
        row = {
            "hour": hour,
            "n_patients": int(eligible.sum()),
            "patient_alert_rate": round(float(np.mean(alerted[eligible])), 4),
            "positive_alert_rate": round(float(np.mean(alerted[eligible & positives])), 4) if np.any(eligible & positives) else None,
            "negative_alert_rate": round(float(np.mean(alerted[eligible & negatives])), 4) if np.any(eligible & negatives) else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _build_patient_snapshots(
    *,
    risk_row: np.ndarray,
    alert_row: np.ndarray,
    treatment_row: np.ndarray,
    treatment_mask_row: np.ndarray,
    treatment_feature_names: list[str],
    active_hours: int,
) -> list[dict]:
    snapshots = []
    for hour in range(active_hours):
        snapshots.append(
            {
                "risk_probability": round(float(risk_row[hour]), 4),
                "risk_alert": bool(alert_row[hour]),
                "phenotype": None,
                "hours_seen": int(hour + 1),
                "top_treatment_signal": _dominant_treatment_signal(
                    treatment_values=treatment_row[hour],
                    treatment_mask=treatment_mask_row[hour],
                    treatment_feature_names=treatment_feature_names,
                ),
            }
        )
    return snapshots


def _dominant_treatment_signal(
    *,
    treatment_values: np.ndarray,
    treatment_mask: np.ndarray,
    treatment_feature_names: list[str],
) -> str:
    observed = np.asarray(treatment_mask, dtype=float) > 0
    if not np.any(observed):
        return "none"
    for idx, name in enumerate(treatment_feature_names):
        if observed[idx] and name.endswith("_on") and float(treatment_values[idx]) > 0:
            return name
    masked_values = np.abs(np.asarray(treatment_values, dtype=float)) * observed.astype(float)
    best_idx = int(np.argmax(masked_values))
    if masked_values[best_idx] <= 0:
        return "none"
    return str(treatment_feature_names[best_idx])


def _load_treatment_feature_names(treatment_dir: Path, *, n_treat: int) -> list[str]:
    feature_names_path = Path(treatment_dir) / "treatment_feature_names.json"
    if feature_names_path.exists():
        with open(feature_names_path, encoding="utf-8") as f:
            names = json.load(f)
        if isinstance(names, list) and len(names) == n_treat:
            return [str(name) for name in names]
    return [f"treatment_{idx}" for idx in range(n_treat)]


def _resolve_patient_ids(static: pd.DataFrame) -> np.ndarray:
    if "patient_id" in static.columns:
        return static["patient_id"].fillna("unknown").astype(str).to_numpy()
    return np.asarray([str(idx) for idx in range(len(static))], dtype=str)


def _write_replay_bundle(
    *,
    bundle_path: Path,
    risk_matrix: np.ndarray,
    active_hours: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    threshold: float,
    temperature: float,
    split: str,
    min_history_hours: int,
    landmark_hours: tuple[int, ...],
    model_artifact_path: Path,
) -> None:
    np.savez_compressed(
        bundle_path,
        risk_matrix=np.asarray(risk_matrix, dtype=np.float32),
        active_hours=np.asarray(active_hours, dtype=np.int16),
        labels=np.asarray(labels, dtype=np.int8),
        patient_ids=np.asarray(patient_ids, dtype=str),
        threshold=np.asarray(float(threshold), dtype=np.float32),
        temperature=np.asarray(float(temperature), dtype=np.float32),
        split=np.asarray(str(split)),
        min_history_hours=np.asarray(int(min_history_hours), dtype=np.int16),
        landmark_hours=np.asarray(landmark_hours, dtype=np.int16),
        model_artifact_path=np.asarray(str(model_artifact_path)),
    )


def _select_sample_patient(patient_summary: pd.DataFrame) -> int:
    positives = patient_summary[patient_summary["label"] == 1].copy()
    alerted = positives[positives["ever_alert"]].copy()
    if not alerted.empty:
        alerted = alerted.sort_values(
            by=["first_alert_hour", "max_risk_probability"],
            ascending=[True, False],
            na_position="last",
        )
        return int(alerted.index[0])
    ranked = patient_summary.sort_values(by=["max_risk_probability", "label"], ascending=[False, False])
    return int(ranked.index[0])


def _build_summary(
    *,
    static: pd.DataFrame,
    labels: np.ndarray,
    patient_summary: pd.DataFrame,
    landmark_metrics: pd.DataFrame,
    cumulative_alert_metrics: pd.DataFrame,
    model_artifact_path: Path,
    threshold: float,
    model_threshold: float,
    temperature: float,
    split: str,
    min_history_hours: int,
    deployment_mode: str,
    deployment_policy: dict | None,
    sample_patient_id: str,
    dashboard_path: Path,
    snapshots_path: Path,
) -> dict:
    positives = patient_summary["label"].to_numpy(dtype=int) == 1
    negatives = ~positives
    first_alert_hours = pd.to_numeric(patient_summary["first_alert_hour"], errors="coerce").to_numpy(dtype=float)
    ever_alert = patient_summary["ever_alert"].to_numpy(dtype=bool)
    active_hours = pd.to_numeric(patient_summary["active_hours"], errors="coerce").to_numpy(dtype=float)
    total_patient_days = max(float(np.sum(active_hours) / 24.0), 1.0e-6)

    source_values = [str(v) for v in static.get("data_source", pd.Series(["unknown"])).dropna().unique().tolist()]
    center_values = [str(v) for v in static.get("center_id", pd.Series(["unknown"])).dropna().unique().tolist()]
    return {
        "artifact_path": str(model_artifact_path),
        "split": split,
        "n_patients": int(len(patient_summary)),
        "n_positive": int(np.sum(positives)),
        "positive_rate": round(float(np.mean(labels)), 4),
        "source_values": source_values,
        "center_values_sample": center_values[:10],
        "deployment_mode": str(deployment_mode),
        "threshold": round(float(threshold), 4),
        "model_threshold": round(float(model_threshold), 4),
        "temperature": round(float(temperature), 4),
        "min_history_hours": int(min_history_hours),
        "mean_active_hours": round(float(np.mean(active_hours)), 2),
        "median_active_hours": round(float(np.median(active_hours)), 2),
        "patient_alert_rate": round(float(np.mean(ever_alert)), 4),
        "positive_patient_alert_rate": round(float(np.mean(ever_alert[positives])), 4) if np.any(positives) else None,
        "negative_patient_alert_rate": round(float(np.mean(ever_alert[negatives])), 4) if np.any(negatives) else None,
        "alerts_per_patient_day": round(float(patient_summary["n_alert_hours"].sum() / total_patient_days), 4),
        "alert_events_per_patient_day": round(float(patient_summary["n_alert_events"].sum() / total_patient_days), 4),
        "alert_state_hours_per_patient_day": round(float(patient_summary["n_alert_hours"].sum() / total_patient_days), 4),
        "median_first_alert_hour_positive": _median_or_none(first_alert_hours[positives & np.isfinite(first_alert_hours)]),
        "median_first_alert_hour_all": _median_or_none(first_alert_hours[np.isfinite(first_alert_hours)]),
        "mean_terminal_risk_positive": round(float(patient_summary.loc[patient_summary["label"] == 1, "terminal_risk_probability"].mean()), 4),
        "mean_terminal_risk_negative": round(float(patient_summary.loc[patient_summary["label"] == 0, "terminal_risk_probability"].mean()), 4),
        "sample_patient_id": sample_patient_id,
        "policy": deployment_policy,
        "artifacts": {
            "patient_summary_csv": str(dashboard_path.parent / "patient_summary.csv"),
            "landmark_metrics_csv": str(dashboard_path.parent / "landmark_metrics.csv"),
            "cumulative_alert_metrics_csv": str(dashboard_path.parent / "cumulative_alert_metrics.csv"),
            "sample_patient_dashboard_html": str(dashboard_path),
            "sample_patient_snapshots_json": str(snapshots_path),
        },
        "landmark_metrics": _dataframe_records(landmark_metrics),
        "cumulative_alert_metrics": _dataframe_records(cumulative_alert_metrics),
    }


def _classification_metrics_safe(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (np.asarray(probs) >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    auroc = None
    if len(np.unique(y_true)) >= 2:
        auroc = round(float(roc_auc_score(y_true, probs)), 4)
    return {
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
        "auroc": auroc,
    }


def _dataframe_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for record in frame.to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            if pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records


def _median_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return round(float(np.median(values)), 2)
