"""
reporting.py - Reproducible Stage 4 closeout summaries and figures.
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

TREATMENT_LABELS = {
    "vasopressor_on_any_6h": "Vasopressor 0-6h",
    "antibiotic_on_any_6h": "Antibiotic 0-6h",
    "mechanical_vent_on_any_6h": "Mechanical Vent 0-6h",
    "fluid_bolus_ml_any_6h": "Fluid Bolus 0-6h",
    "crystalloid_fluid_ml_any_6h": "Crystalloid 0-6h",
    "rrt_on_any_24h": "RRT 0-24h",
}

METRIC_ORDER = ["auroc", "balanced_accuracy", "ece"]


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_sign(value: float | None, eps: float = 1.0e-4) -> str:
    if value is None:
        return "na"
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "neutral"


def build_s4_closeout_summary(
    *,
    mimic_treatment_report_path: Path,
    eicu_treatment_report_path: Path,
    mimic_causal_report_path: Path,
    eicu_causal_report_path: Path,
) -> dict:
    sources = {
        "mimic": {
            "treatment": _load_json(Path(mimic_treatment_report_path)),
            "causal": _load_json(Path(mimic_causal_report_path)),
        },
        "eicu": {
            "treatment": _load_json(Path(eicu_treatment_report_path)),
            "causal": _load_json(Path(eicu_causal_report_path)),
        },
    }

    treatment_aware_metrics: list[dict] = []
    recommendations: list[dict] = []
    causal_effects: list[dict] = []

    for source_key, payload in sources.items():
        source_label = SOURCE_LABELS[source_key]
        treatment_report = payload["treatment"]
        causal_report = payload["causal"]
        test_metrics = treatment_report["splits"]["test"]
        test_calibration = treatment_report["calibration"]["test"]
        treatment_aware_metrics.append(
            {
                "source": source_key,
                "source_label": source_label,
                "n_samples": int(test_metrics["n_samples"]),
                "positive_rate": float(test_metrics["positive_rate"]),
                "auroc": float(test_metrics["auroc"]),
                "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
                "accuracy": float(test_metrics["accuracy"]),
                "precision": float(test_metrics["precision"]),
                "recall": float(test_metrics["recall"]),
                "f1": float(test_metrics["f1"]),
                "brier": float(test_calibration["brier"]),
                "ece": float(test_calibration["ece"]),
            }
        )

        for recommendation in causal_report.get("recommendations", []):
            recommendations.append(
                {
                    "source": source_key,
                    "source_label": source_label,
                    "treatment_col": recommendation["treatment_col"],
                    "treatment_label": TREATMENT_LABELS.get(
                        recommendation["treatment_col"], recommendation["treatment_col"]
                    ),
                    "direction": recommendation["direction"],
                    "clinical_note": recommendation.get("clinical_note", ""),
                }
            )

        for treatment_col, treatment_bundle in causal_report["treatments"].items():
            psm = treatment_bundle.get("psm", {})
            dml = treatment_bundle.get("causal_forest_dml", {})
            rdd = treatment_bundle.get("rdd", {})
            causal_effects.append(
                {
                    "source": source_key,
                    "source_label": source_label,
                    "treatment_col": treatment_col,
                    "treatment_label": TREATMENT_LABELS.get(treatment_col, treatment_col),
                    "psm_ate": psm.get("ate"),
                    "dml_cate_mean": dml.get("cate_mean"),
                    "dml_cate_std": dml.get("cate_std"),
                    "rdd_local_effect": rdd.get("local_effect"),
                    "psm_sign": _safe_sign(psm.get("ate")),
                    "dml_sign": _safe_sign(dml.get("cate_mean")),
                    "recommendation_direction": next(
                        (
                            item["direction"]
                            for item in recommendations
                            if item["source"] == source_key and item["treatment_col"] == treatment_col
                        ),
                        "uncertain",
                    ),
                }
            )

    causal_df = pd.DataFrame(causal_effects)
    comparison_rows: list[dict] = []
    for treatment_col in causal_df["treatment_col"].unique():
        mimic_row = causal_df[
            (causal_df["source"] == "mimic") & (causal_df["treatment_col"] == treatment_col)
        ].iloc[0]
        eicu_row = causal_df[
            (causal_df["source"] == "eicu") & (causal_df["treatment_col"] == treatment_col)
        ].iloc[0]
        psm_consistent = mimic_row["psm_sign"] == eicu_row["psm_sign"] and mimic_row["psm_sign"] not in {"na", "neutral"}
        dml_consistent = mimic_row["dml_sign"] == eicu_row["dml_sign"] and mimic_row["dml_sign"] not in {"na", "neutral"}
        recommendation_consistent = (
            mimic_row["recommendation_direction"] == eicu_row["recommendation_direction"]
            and mimic_row["recommendation_direction"] != "uncertain"
        )
        comparison_rows.append(
            {
                "treatment_col": treatment_col,
                "treatment_label": TREATMENT_LABELS.get(treatment_col, treatment_col),
                "mimic_psm_ate": mimic_row["psm_ate"],
                "eicu_psm_ate": eicu_row["psm_ate"],
                "mimic_dml_cate_mean": mimic_row["dml_cate_mean"],
                "eicu_dml_cate_mean": eicu_row["dml_cate_mean"],
                "mimic_recommendation_direction": mimic_row["recommendation_direction"],
                "eicu_recommendation_direction": eicu_row["recommendation_direction"],
                "psm_direction_consistent": bool(psm_consistent),
                "dml_direction_consistent": bool(dml_consistent),
                "recommendation_consistent": bool(recommendation_consistent),
            }
        )

    return {
        "summary": {
            "n_sources": 2,
            "sources": [SOURCE_LABELS["mimic"], SOURCE_LABELS["eicu"]],
            "cross_source_consistent_treatments": int(
                sum(1 for row in comparison_rows if row["recommendation_consistent"])
            ),
            "cross_source_discordant_treatments": int(
                sum(1 for row in comparison_rows if not row["recommendation_consistent"])
            ),
        },
        "treatment_aware_metrics": treatment_aware_metrics,
        "causal_effects": causal_effects,
        "cross_source_comparison": comparison_rows,
        "recommendations": recommendations,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _plot_s4_performance(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(metrics_df))
    labels = metrics_df["source_label"].tolist()
    plotting = {
        "auroc": ("Test AUROC", "#2a6f97"),
        "balanced_accuracy": ("Balanced Accuracy", "#588157"),
        "ece": ("Test ECE", "#bc4749"),
    }
    for axis, metric in zip(axes, METRIC_ORDER):
        title, color = plotting[metric]
        values = metrics_df[metric].to_numpy(dtype=float)
        axis.bar(x, values, color=color, width=0.55)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        if metric != "ece":
            axis.set_ylim(0.0, 1.0)
        for idx, value in enumerate(values):
            axis.text(idx, value + (0.015 if metric != "ece" else max(values) * 0.04 + 1.0e-6), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Stage 4 Treatment-Aware External Performance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _annotate_heatmap(axis: plt.Axes, values: np.ndarray, row_labels: list[str], col_labels: list[str], title: str) -> None:
    im = axis.imshow(values, cmap="coolwarm", aspect="auto")
    axis.set_xticks(np.arange(len(col_labels)), col_labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(title)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            label = "n/a" if np.isnan(value) else f"{value:.3f}"
            axis.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=axis, fraction=0.046, pad=0.04)


def _plot_s4_causal_effects(causal_df: pd.DataFrame, output_path: Path) -> None:
    treatment_order = [
        col for col in TREATMENT_LABELS
        if col in causal_df["treatment_col"].unique().tolist()
    ]
    sources = ["mimic", "eicu"]
    row_labels = [SOURCE_LABELS[source] for source in sources]
    col_labels = [TREATMENT_LABELS[treatment] for treatment in treatment_order]

    psm_values = np.full((len(sources), len(treatment_order)), np.nan, dtype=float)
    dml_values = np.full((len(sources), len(treatment_order)), np.nan, dtype=float)
    for row_idx, source in enumerate(sources):
        for col_idx, treatment in enumerate(treatment_order):
            row = causal_df[
                (causal_df["source"] == source) & (causal_df["treatment_col"] == treatment)
            ].iloc[0]
            psm_values[row_idx, col_idx] = row["psm_ate"] if row["psm_ate"] is not None else np.nan
            dml_values[row_idx, col_idx] = row["dml_cate_mean"] if row["dml_cate_mean"] is not None else np.nan

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    _annotate_heatmap(axes[0], psm_values, row_labels, col_labels, "PSM Average Treatment Effect")
    _annotate_heatmap(axes[1], dml_values, row_labels, col_labels, "DML Mean CATE")
    fig.suptitle("Stage 4 Observational Treatment-Effect Comparison")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_s4_closeout_artifacts(bundle: dict, *, reports_dir: Path, figures_dir: Path) -> dict:
    reports_dir = Path(reports_dir)
    figures_dir = Path(figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_json = reports_dir / "s4_closeout_summary.json"
    performance_csv = reports_dir / "s4_treatment_aware_metrics.csv"
    causal_csv = reports_dir / "s4_causal_effects.csv"
    comparison_csv = reports_dir / "s4_cross_source_comparison.csv"
    recommendations_csv = reports_dir / "s4_recommendations.csv"
    performance_fig = figures_dir / "s4_treatment_aware_metrics.png"
    causal_fig = figures_dir / "s4_causal_effects_heatmap.png"

    _write_json(summary_json, bundle)
    metrics_df = pd.DataFrame(bundle["treatment_aware_metrics"])
    causal_df = pd.DataFrame(bundle["causal_effects"])
    comparison_df = pd.DataFrame(bundle["cross_source_comparison"])
    recommendations_df = pd.DataFrame(bundle["recommendations"])
    metrics_df.to_csv(performance_csv, index=False)
    causal_df.to_csv(causal_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    recommendations_df.to_csv(recommendations_csv, index=False)

    _plot_s4_performance(metrics_df, performance_fig)
    _plot_s4_causal_effects(causal_df, causal_fig)

    return {
        "summary_json": str(summary_json),
        "performance_csv": str(performance_csv),
        "causal_csv": str(causal_csv),
        "comparison_csv": str(comparison_csv),
        "recommendations_csv": str(recommendations_csv),
        "performance_fig": str(performance_fig),
        "causal_fig": str(causal_fig),
    }
