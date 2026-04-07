"""
reporting.py - Reproducible Stage 5 validation summaries and figures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE_LABELS = {
    "mimic": "MIMIC-IV",
    "eicu": "eICU",
}

VALIDATION_GATES = {
    "min_auroc": 0.85,
    "min_balanced_accuracy": 0.78,
    "max_ece": 0.02,
    "max_latency_ms_per_sample": 2.0,
}


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_s5_validation_summary(
    *,
    mimic_report_path: Path,
    eicu_report_path: Path,
) -> dict:
    rows: list[dict] = []
    for source_key, report_path in {
        "mimic": Path(mimic_report_path),
        "eicu": Path(eicu_report_path),
    }.items():
        report = _load_json(report_path)
        test_metrics = report["splits"]["test"]
        calibration_test = report["calibration"]["test"]
        deployment = report["deployment"]
        row = {
            "source": source_key,
            "source_label": SOURCE_LABELS[source_key],
            "n_samples": int(test_metrics["n_samples"]),
            "auroc": float(test_metrics["auroc"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "accuracy": float(test_metrics["accuracy"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "f1": float(test_metrics["f1"]),
            "positive_rate": float(test_metrics["positive_rate"]),
            "predicted_positive_rate": float(test_metrics["predicted_positive_rate"]),
            "ece": float(calibration_test["ece"]),
            "brier": float(calibration_test["brier"]),
            "cpu_latency_ms_per_sample": float(deployment["cpu_latency_ms_per_sample"]),
            "float_n_parameters": int(deployment["float_n_parameters"]),
            "dynamic_quantization_ok": bool(deployment["dynamic_quantization_ok"]),
            "selected_threshold": float(report["threshold_selection"]["selected_threshold"]),
            "epochs_trained": int(report["training"]["epochs_trained"]),
        }
        row["passes_auroc_gate"] = row["auroc"] >= VALIDATION_GATES["min_auroc"]
        row["passes_balanced_accuracy_gate"] = row["balanced_accuracy"] >= VALIDATION_GATES["min_balanced_accuracy"]
        row["passes_ece_gate"] = row["ece"] <= VALIDATION_GATES["max_ece"]
        row["passes_latency_gate"] = row["cpu_latency_ms_per_sample"] <= VALIDATION_GATES["max_latency_ms_per_sample"]
        row["overall_pass"] = bool(
            row["passes_auroc_gate"]
            and row["passes_balanced_accuracy_gate"]
            and row["passes_ece_gate"]
            and row["passes_latency_gate"]
        )
        rows.append(row)

    return {
        "validation_gates": VALIDATION_GATES,
        "realtime_validation_metrics": rows,
        "summary": {
            "n_sources": len(rows),
            "sources_passing_all_gates": int(sum(1 for row in rows if row["overall_pass"])),
            "sources": [row["source_label"] for row in rows],
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _plot_s5_validation(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(metrics_df))
    labels = metrics_df["source_label"].tolist()

    performance_metrics = [
        ("auroc", "Test AUROC", "#2a6f97"),
        ("balanced_accuracy", "Balanced Accuracy", "#588157"),
        ("ece", "Test ECE", "#bc4749"),
    ]
    for axis, (metric, title, color) in zip(axes, performance_metrics):
        values = metrics_df[metric].to_numpy(dtype=float)
        axis.bar(x, values, color=color, width=0.55)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        if metric != "ece":
            axis.set_ylim(0.0, 1.0)
        for idx, value in enumerate(values):
            axis.text(idx, value + (0.015 if metric != "ece" else max(values) * 0.04 + 1.0e-6), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Stage 5 Realtime Student Validation")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_s5_deployment(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(metrics_df))
    labels = metrics_df["source_label"].tolist()

    latency = metrics_df["cpu_latency_ms_per_sample"].to_numpy(dtype=float)
    params = metrics_df["float_n_parameters"].to_numpy(dtype=float) / 1000.0

    axes[0].bar(x, latency, color="#6a4c93", width=0.55)
    axes[0].axhline(VALIDATION_GATES["max_latency_ms_per_sample"], color="#bc4749", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels)
    axes[0].set_title("CPU Latency (ms/sample)")
    for idx, value in enumerate(latency):
        axes[0].text(idx, value + max(latency) * 0.05 + 1.0e-6, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, params, color="#ffb703", width=0.55)
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Parameter Count (thousands)")
    for idx, value in enumerate(params):
        axes[1].text(idx, value + max(params) * 0.03 + 1.0e-6, f"{value:.1f}k", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Stage 5 Deployment Profile")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_s5_validation_artifacts(bundle: dict, *, reports_dir: Path, figures_dir: Path) -> dict:
    reports_dir = Path(reports_dir)
    figures_dir = Path(figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_json = reports_dir / "s5_validation_summary.json"
    metrics_csv = reports_dir / "s5_realtime_metrics.csv"
    validation_fig = figures_dir / "s5_realtime_validation_metrics.png"
    deployment_fig = figures_dir / "s5_realtime_deployment_profile.png"

    _write_json(summary_json, bundle)
    metrics_df = pd.DataFrame(bundle["realtime_validation_metrics"])
    metrics_df.to_csv(metrics_csv, index=False)
    _plot_s5_validation(metrics_df, validation_fig)
    _plot_s5_deployment(metrics_df, deployment_fig)

    return {
        "summary_json": str(summary_json),
        "metrics_csv": str(metrics_csv),
        "validation_fig": str(validation_fig),
        "deployment_fig": str(deployment_fig),
    }


def build_s5_adaptation_trigger_report(
    *,
    validation_summary_path: Path,
    policy_best_json_path: Path,
    policy_candidates_csv_path: Path,
    shadow_replay_summary_path: Path,
    trigger_config_path: Path,
    source_key: str,
) -> dict:
    validation_summary = _load_json(Path(validation_summary_path))
    policy_best_payload = _load_json(Path(policy_best_json_path))
    shadow_summary = _load_json(Path(shadow_replay_summary_path))
    trigger_cfg = _load_json(Path(trigger_config_path))
    candidates_df = pd.read_csv(Path(policy_candidates_csv_path))

    validation_row = _select_validation_row(validation_summary, source_key=source_key)
    best_policy = dict(policy_best_payload["best_policy"])
    policy_constraints = dict(policy_best_payload.get("constraints", {}))
    production_gates = dict(trigger_cfg["production_policy_gates"])
    shadow_gates = dict(trigger_cfg["shadow_policy_gates"])
    offline_gates = dict(trigger_cfg["offline_quality_gates"])
    search_gates = dict(trigger_cfg["search_exhaustion_gates"])
    trigger_policy = dict(trigger_cfg.get("trigger_policy", {}))

    feasible_count = _count_feasible_candidates(candidates_df)
    candidate_count = int(len(candidates_df))
    positive_alert_rate_24h = _cumulative_rate_at_hour(
        shadow_summary.get("cumulative_alert_metrics", []),
        hour=24,
        field="positive_alert_rate",
    )

    production_checks = {
        "negative_patient_alert_rate": float(best_policy["negative_patient_alert_rate"]) <= float(production_gates["max_negative_patient_alert_rate"]),
        "alert_events_per_patient_day": float(best_policy["alert_events_per_patient_day"]) <= float(production_gates["max_alert_events_per_patient_day"]),
        "positive_patient_alert_rate": float(best_policy["positive_patient_alert_rate"]) >= float(production_gates["min_positive_patient_alert_rate"]),
        "positive_alert_rate_at_24h": float(best_policy["positive_alert_rate_at_24h"]) >= float(production_gates["min_positive_alert_rate_24h"]),
    }
    shadow_checks = {
        "negative_patient_alert_rate": float(shadow_summary["negative_patient_alert_rate"]) <= float(shadow_gates["max_negative_patient_alert_rate"]),
        "alert_events_per_patient_day": float(shadow_summary["alert_events_per_patient_day"]) <= float(shadow_gates["max_alert_events_per_patient_day"]),
        "positive_patient_alert_rate": float(shadow_summary["positive_patient_alert_rate"]) >= float(shadow_gates["min_positive_patient_alert_rate"]),
        "positive_alert_rate_at_24h": float(positive_alert_rate_24h) >= float(shadow_gates["min_positive_alert_rate_24h"]),
        "median_first_alert_hour_positive": float(shadow_summary["median_first_alert_hour_positive"]) <= float(shadow_gates["max_median_first_alert_hour_positive"]),
    }
    offline_checks = {
        "auroc": float(validation_row["auroc"]) >= float(offline_gates["min_auroc"]),
        "balanced_accuracy": float(validation_row["balanced_accuracy"]) >= float(offline_gates["min_balanced_accuracy"]),
        "ece": float(validation_row["ece"]) <= float(offline_gates["max_ece"]),
        "latency_ms_per_sample": float(validation_row["cpu_latency_ms_per_sample"]) <= float(offline_gates["max_latency_ms_per_sample"]),
    }
    search_checks = {
        "candidate_count": candidate_count >= int(search_gates["min_candidate_count"]),
        "feasible_count": feasible_count <= int(search_gates["max_feasible_count"]),
        "grid_profile": str(policy_constraints.get("grid_profile")) == str(search_gates["required_grid_profile"]),
        "ranking_mode": str(policy_constraints.get("ranking_mode")) == str(search_gates["required_ranking_mode"]),
    }

    production_ready = all(production_checks.values()) and bool(best_policy.get("feasible", False))
    shadow_ready = all(shadow_checks.values())
    offline_ready = all(offline_checks.values())
    search_exhausted = all(search_checks.values())

    triggered = (
        (not bool(trigger_policy.get("require_shadow_ready", True)) or shadow_ready)
        and (not bool(trigger_policy.get("require_offline_quality_ready", True)) or offline_ready)
        and (not bool(trigger_policy.get("require_production_policy_failure", True)) or not production_ready)
        and (not bool(trigger_policy.get("require_search_exhausted", True)) or search_exhausted)
    )

    if production_ready:
        next_step = "keep_policy_only"
    elif triggered:
        next_step = "start_source_specific_full_finetune"
    elif shadow_ready:
        next_step = "continue_shadow_monitoring"
    else:
        next_step = "continue_policy_tightening"

    return {
        "source": str(trigger_cfg.get("source", source_key)),
        "source_key": str(source_key),
        "triggered": bool(triggered),
        "next_step": next_step,
        "production_policy_ready": bool(production_ready),
        "shadow_policy_ready": bool(shadow_ready),
        "offline_model_ready": bool(offline_ready),
        "search_exhausted": bool(search_exhausted),
        "production_policy_checks": _bool_summary(production_checks),
        "shadow_policy_checks": _bool_summary(shadow_checks),
        "offline_quality_checks": _bool_summary(offline_checks),
        "search_exhaustion_checks": _bool_summary(search_checks),
        "artifacts": {
            "validation_summary_json": str(validation_summary_path),
            "policy_best_json": str(policy_best_json_path),
            "policy_candidates_csv": str(policy_candidates_csv_path),
            "shadow_replay_summary_json": str(shadow_replay_summary_path),
            "trigger_config_json": str(trigger_config_path),
        },
        "metrics_snapshot": {
            "offline_validation": {
                "auroc": float(validation_row["auroc"]),
                "balanced_accuracy": float(validation_row["balanced_accuracy"]),
                "ece": float(validation_row["ece"]),
                "cpu_latency_ms_per_sample": float(validation_row["cpu_latency_ms_per_sample"]),
            },
            "production_policy_best": {
                "feasible": bool(best_policy.get("feasible", False)),
                "negative_patient_alert_rate": float(best_policy["negative_patient_alert_rate"]),
                "alert_events_per_patient_day": float(best_policy["alert_events_per_patient_day"]),
                "positive_patient_alert_rate": float(best_policy["positive_patient_alert_rate"]),
                "positive_alert_rate_at_24h": float(best_policy["positive_alert_rate_at_24h"]),
            },
            "shadow_policy_replay": {
                "patient_alert_rate": float(shadow_summary["patient_alert_rate"]),
                "negative_patient_alert_rate": float(shadow_summary["negative_patient_alert_rate"]),
                "positive_patient_alert_rate": float(shadow_summary["positive_patient_alert_rate"]),
                "alert_events_per_patient_day": float(shadow_summary["alert_events_per_patient_day"]),
                "alert_state_hours_per_patient_day": float(shadow_summary["alert_state_hours_per_patient_day"]),
                "median_first_alert_hour_positive": float(shadow_summary["median_first_alert_hour_positive"]),
                "positive_alert_rate_at_24h": float(positive_alert_rate_24h),
            },
            "policy_search": {
                "candidate_count": candidate_count,
                "feasible_count": feasible_count,
                "grid_profile": policy_constraints.get("grid_profile"),
                "ranking_mode": policy_constraints.get("ranking_mode"),
            },
        },
        "trigger_policy": trigger_policy,
    }


def write_s5_adaptation_trigger_artifacts(bundle: dict, *, reports_dir: Path) -> dict:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_json = reports_dir / "s5_adaptation_trigger_report.json"
    _write_json(summary_json, bundle)
    return {
        "summary_json": str(summary_json),
    }


def _select_validation_row(bundle: dict, *, source_key: str) -> dict:
    for row in bundle.get("realtime_validation_metrics", []):
        if str(row.get("source")) == str(source_key):
            return row
    raise ValueError(f"Source {source_key} not found in validation summary")


def _count_feasible_candidates(frame: pd.DataFrame) -> int:
    if "feasible" not in frame.columns:
        return 0
    feasible = frame["feasible"]
    if feasible.dtype == bool:
        return int(feasible.sum())
    normalized = feasible.astype(str).str.lower().isin({"true", "1", "yes"})
    return int(normalized.sum())


def _cumulative_rate_at_hour(rows: list[dict], *, hour: int, field: str) -> float:
    for row in rows:
        if int(row.get("hour", -1)) == int(hour):
            return float(row[field])
    raise ValueError(f"Hour {hour} not found in cumulative alert metrics")


def _bool_summary(checks: dict[str, bool]) -> dict[str, dict]:
    return {
        key: {
            "pass": bool(value),
        }
        for key, value in checks.items()
    }
