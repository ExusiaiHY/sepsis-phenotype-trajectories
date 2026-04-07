from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import s6_optimization.causal_phenotyping as causal_phenotyping
from s6_optimization.baseline_comparison import generate_baseline_comparison_report
from s6_optimization.domain_adaptation import align_covariates_by_group
from s6_optimization.missingness_encoder import (
    build_patient_missingness_features,
    compute_gap_lengths_vectorized,
    compute_missingness_features,
    run_missingness_stage,
)
from s6_optimization.run_comparison import compare_s6_runs
from s6_optimization.severity_split_search import search_severity_split_targets
from s6_optimization.saits_imputation import _prepare_pypots_home
from s6_optimization.timesfm_features import run_timesfm_feature_extraction
from s6_optimization.phenotype_naming import (
    assign_phenotype_by_causality,
    apply_cluster_severity_modifier,
    _sofa_respiratory,
    _sofa_cardiovascular,
    _sofa_hepatic,
    _sofa_renal,
    _sofa_coagulation,
    _sofa_neurological,
    compute_organ_scores,
)
from s0.schema import CONTINUOUS_NAMES


def test_missingness_encoder_derives_gap_and_density_channels():
    masks = np.array(
        [
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )

    gaps = compute_gap_lengths_vectorized(masks)
    enhanced = compute_missingness_features(masks, gap_window=2)

    assert gaps.shape == masks.shape
    assert np.allclose(gaps[0, :, 0], [0.0, 0.25, 0.5, 0.0])
    assert np.allclose(gaps[0, :, 1], [0.25, 0.5, 0.0, 0.0])
    assert enhanced.shape == (1, 4, 6)
    assert np.allclose(enhanced[..., :2], masks)


def test_missingness_stage_emits_patient_covariates(tmp_path):
    masks = np.array(
        [
            [
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
        ],
        dtype=np.float32,
    )
    feature_names = ["map", "lactate", "creatinine"]

    stage = run_missingness_stage(
        masks=masks,
        output_dir=tmp_path,
        feature_names=feature_names,
        config={"gap_window": 2, "selected_features": ["map", "lactate"]},
    )

    covariates = stage["features_df"]
    assert len(covariates) == 2
    assert "miss_global_density" in covariates.columns
    assert "miss_map_density" in covariates.columns
    assert "miss_lactate_gap_max" in covariates.columns
    assert stage["feature_summary"]["enabled"] is True
    assert (tmp_path / "missingness_patient_features.csv").exists()
    assert (tmp_path / "missingness_covariate_summary.json").exists()


def test_assign_phenotype_by_causality_respects_threshold_overrides():
    organ_scores = {
        "sofa_total": 6,
        "dominant_organ": "cardiovascular",
        "cardiovascular": 1,
        "respiratory": 0,
        "hepatic": 0,
        "renal": 0,
        "coagulation": 0,
        "neurological": 0,
    }

    responsive = assign_phenotype_by_causality(
        cluster_id=1,
        cate_score=0.03,
        mortality_risk=0.25,
        organ_scores=organ_scores,
    )
    refractory = assign_phenotype_by_causality(
        cluster_id=1,
        cate_score=0.03,
        mortality_risk=0.25,
        organ_scores=organ_scores,
        thresholds={"cate_responsiveness_threshold": 0.05},
    )

    assert responsive == "hemodynamic_unstable_proxy_responsive"
    assert refractory == "hemodynamic_unstable_proxy_refractory"


def test_baseline_comparison_report_quantifies_improvement():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        static_path = root / "static.csv"
        window_labels_path = root / "window_labels.npy"
        phenotype_path = root / "phenotype_assignments.csv"
        output_path = root / "baseline_comparison.json"

        static = pd.DataFrame(
            {
                "mortality_inhospital": [0, 0, 1, 1, 0, 1],
                "center_id": ["a", "a", "a", "b", "b", "b"],
            }
        )
        static.to_csv(static_path, index=False)
        np.save(
            window_labels_path,
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                dtype=np.int64,
            ),
        )
        pd.DataFrame(
            {
                "phenotype_key": [
                    "mild",
                    "mild",
                    "severe",
                    "severe",
                    "mild",
                    "severe",
                ]
            }
        ).to_csv(phenotype_path, index=False)

        report = generate_baseline_comparison_report(
            static_path=static_path,
            window_labels_path=window_labels_path,
            phenotype_assignments_path=phenotype_path,
            output_path=output_path,
            min_group_size=2,
        )

        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved["baseline"]["supported_group_count"] == 2
        assert saved["optimized"]["supported_group_count"] == 2
        assert report["metric_deltas"]["supported_mortality_range"]["optimized"] == 1.0
        assert report["metric_deltas"]["supported_mortality_range"]["baseline"] == 0.3333
        assert report["metric_deltas"]["supported_mortality_range"]["improved"] is True


# ============================================================
# Codex-required tests (3 mandatory)
# ============================================================

def test_sofa_thresholds_use_raw_clinical_units():
    """Codex test 1: GCS/platelet/MAP SOFA thresholds match Sepsis-3 raw units."""
    # GCS: 15=0, 13=1, 10=2, 6=3, 3=4
    assert _sofa_neurological(15.0) == 0
    assert _sofa_neurological(14.0) == 1
    assert _sofa_neurological(10.0) == 2
    assert _sofa_neurological(6.0) == 3
    assert _sofa_neurological(3.0) == 4

    # MAP: >=70 → 0, <70 → 1
    assert _sofa_cardiovascular(80.0) == 0
    assert _sofa_cardiovascular(70.0) == 0
    assert _sofa_cardiovascular(60.0) == 1

    # Platelet: >=150→0, >=100→1, >=50→2, >=20→3, <20→4
    assert _sofa_coagulation(200.0) == 0
    assert _sofa_coagulation(120.0) == 1
    assert _sofa_coagulation(50.0) == 2
    assert _sofa_coagulation(19.0) == 4

    # Bilirubin: <1.2→0, <2→1, <6→2, <12→3, >=12→4
    assert _sofa_hepatic(0.5) == 0
    assert _sofa_hepatic(1.5) == 1
    assert _sofa_hepatic(8.0) == 3

    # Creatinine: <1.2→0, <2→1, <3.5→2, <5→3, >=5→4
    assert _sofa_renal(0.8) == 0
    assert _sofa_renal(2.5) == 2
    assert _sofa_renal(6.0) == 4

    # PaO2/FiO2: >=400→0, >=300→1, >=200→2, >=100→3, <100→4
    assert _sofa_respiratory(450.0) == 0
    assert _sofa_respiratory(250.0) == 2
    assert _sofa_respiratory(50.0) == 4

    # NaN → 0 for all
    for fn in [_sofa_neurological, _sofa_cardiovascular, _sofa_coagulation,
               _sofa_hepatic, _sofa_renal, _sofa_respiratory]:
        assert fn(float("nan")) == 0


def test_zscore_guard_rejects_standardized_input():
    """Codex test 2: compute_organ_scores rejects z-score input, accepts raw."""
    rng = np.random.default_rng(42)
    N, T, F = 200, 48, len(CONTINUOUS_NAMES)

    # z-scored data: mean≈0, std≈1 — must be rejected
    zscore_data = rng.standard_normal((N, T, F)).astype(np.float32)
    masks = np.ones((N, T, F), dtype=np.float32)

    import pytest
    with pytest.raises(ValueError, match="z-score"):
        compute_organ_scores(zscore_data, masks, CONTINUOUS_NAMES)

    # Raw clinical data — must be accepted
    idx = {name: i for i, name in enumerate(CONTINUOUS_NAMES)}
    raw = np.full((10, T, F), 0.0, dtype=np.float32)
    raw_masks = np.ones((10, T, F), dtype=np.float32)
    raw[:, :, idx["map"]] = 75.0
    raw[:, :, idx["gcs"]] = 14.0
    raw[:, :, idx["platelet"]] = 200.0
    raw[:, :, idx["creatinine"]] = 1.0
    raw[:, :, idx["bilirubin"]] = 0.5

    df = compute_organ_scores(raw, raw_masks, CONTINUOUS_NAMES)
    assert len(df) == 10
    assert df["sofa_total"].mean() < 10  # raw data should give reasonable scores


def test_artifact_csv_json_counts_match():
    """Codex test 3: phenotype_assignments.csv and JSON report counts are consistent."""
    project_root = Path(__file__).resolve().parent.parent

    for s6_dir_name in ["data/s6_rerun_20260401", "data/s6"]:
        s6_dir = project_root / s6_dir_name
        csv_path = s6_dir / "phenotype_assignments.csv"
        json_path = s6_dir / "causal_phenotyping_report.json"

        if not csv_path.exists() or not json_path.exists():
            continue

        csv_df = pd.read_csv(csv_path)
        with open(json_path) as f:
            report = json.load(f)

        validation = report.get("phenotype_validation", {})

        for phenotype_key, stats in validation.items():
            csv_count = int((csv_df["phenotype_key"] == phenotype_key).sum())
            json_count = stats["n"]
            assert csv_count == json_count, (
                f"Count mismatch for '{phenotype_key}': CSV={csv_count} vs JSON={json_count}"
            )

        assert len(csv_df) == report["n_patients"], (
            f"Total count mismatch: CSV={len(csv_df)} vs JSON={report['n_patients']}"
        )
        return

    import pytest
    pytest.skip("No S6 artifacts found")


def test_compare_s6_runs_reports_metric_deltas():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        previous = root / "previous"
        current = root / "current"
        previous.mkdir()
        current.mkdir()

        previous_baseline = {
            "optimized": {
                "supported_group_count": 5,
                "supported_mortality_range": 0.30,
                "weighted_mortality_std": 0.12,
                "center_distribution_l1": 0.01,
                "center_mortality_deviation": 0.006,
                "dominant_group_fraction": 0.50,
                "rare_group_fraction": 0.02,
                "group_count": 6,
                "group_stats": [{"label": "a", "n": 10}],
            }
        }
        current_baseline = {
            "optimized": {
                "supported_group_count": 7,
                "supported_mortality_range": 0.25,
                "weighted_mortality_std": 0.10,
                "center_distribution_l1": 0.02,
                "center_mortality_deviation": 0.004,
                "dominant_group_fraction": 0.40,
                "rare_group_fraction": 0.00,
                "group_count": 7,
                "group_stats": [{"label": "b", "n": 12}],
            }
        }
        previous_causal = {"cate_summary": {"std": 0.08}}
        current_causal = {"cate_summary": {"std": 0.05}}

        pd.DataFrame({"sofa_total": [8.0, 10.0]}).to_csv(previous / "organ_scores.csv", index=False)
        pd.DataFrame({"sofa_total": [5.0, 6.0]}).to_csv(current / "organ_scores.csv", index=False)
        (previous / "baseline_comparison.json").write_text(json.dumps(previous_baseline), encoding="utf-8")
        (current / "baseline_comparison.json").write_text(json.dumps(current_baseline), encoding="utf-8")
        (previous / "causal_phenotyping_report.json").write_text(json.dumps(previous_causal), encoding="utf-8")
        (current / "causal_phenotyping_report.json").write_text(json.dumps(current_causal), encoding="utf-8")

        report = compare_s6_runs(previous, current)
        assert report["metric_deltas"]["supported_group_count"]["improved"] is True
        assert report["metric_deltas"]["center_distribution_l1"]["improved"] is False
        assert report["metric_deltas"]["cate_std"]["improved"] is True
        assert report["metric_deltas"]["mean_sofa_total"]["improved"] is None


def test_cluster_severity_modifier_splits_selected_targets():
    cluster_mortality_order = {0: 0.05, 1: 0.20, 2: 0.31, 3: 0.10}

    assert apply_cluster_severity_modifier(
        phenotype_key="respiratory_failure",
        dominant_cluster=2,
        cluster_mortality_order=cluster_mortality_order,
    ) == "respiratory_failure_critical"
    assert apply_cluster_severity_modifier(
        phenotype_key="respiratory_failure",
        dominant_cluster=0,
        cluster_mortality_order=cluster_mortality_order,
    ) == "respiratory_failure_recovering"
    assert apply_cluster_severity_modifier(
        phenotype_key="hemodynamic_unstable_proxy_responsive",
        dominant_cluster=1,
        cluster_mortality_order=cluster_mortality_order,
    ) == "hemodynamic_unstable_proxy_responsive"
    assert apply_cluster_severity_modifier(
        phenotype_key="neurological_decline",
        dominant_cluster=2,
        cluster_mortality_order=cluster_mortality_order,
    ) == "neurological_decline"


def test_search_severity_split_targets_recovers_heterogeneous_label():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        run_dir = root / "run"
        run_dir.mkdir()

        phenotype_df = pd.DataFrame(
            {
                "patient_idx": list(range(12)),
                "dominant_cluster": [0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 3, 3],
                "trajectory_direction": ["stable"] * 12,
                "phenotype_key": ["respiratory_failure"] * 8 + ["neurological_decline"] * 4,
                "phenotype_name": ["x"] * 12,
                "cate_score": [0.0] * 12,
                "mortality_risk": [0.1] * 12,
                "sofa_total": [5] * 12,
                "dominant_organ": ["respiratory"] * 8 + ["neurological"] * 4,
            }
        )
        phenotype_df.to_csv(run_dir / "phenotype_assignments.csv", index=False)

        static = pd.DataFrame(
            {
                "mortality_inhospital": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                "center_id": ["a", "a", "b", "b"] * 3,
            }
        )
        static_path = root / "static.csv"
        static.to_csv(static_path, index=False)

        report = search_severity_split_targets(
            run_dir=run_dir,
            static_path=static_path,
            min_group_size=2,
            min_candidate_size=2,
            max_combination_size=2,
            top_k=3,
        )

        assert report["recommendation"] is not None
        assert "respiratory_failure" in report["recommendation"]["targets"]
        assert report["recommendation"]["score_delta_vs_current"] > 0


def test_prepare_pypots_home_uses_workspace_cache_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_root = Path(tmp_dir) / "pypots_home"
        original_home = os.environ.get("HOME")
        original_xdg = os.environ.get("XDG_CACHE_HOME")
        try:
            prepared = _prepare_pypots_home(cache_root)
            assert prepared == cache_root.resolve()
            assert prepared.exists()
            assert Path(os.environ["HOME"]) == prepared
            assert Path(os.environ["XDG_CACHE_HOME"]) == prepared / "xdg_cache"
            assert (prepared / "xdg_cache").exists()
        finally:
            if original_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = original_home
            if original_xdg is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = original_xdg


def test_causal_stability_gate_falls_back_to_cross_fitted_dml(monkeypatch):
    X = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
    treatment = np.array([0, 1] * 10, dtype=np.int64)
    outcome = np.array([0, 1, 0, 1] * 5, dtype=np.float32)
    unstable = np.linspace(-0.4, 0.4, len(X), dtype=np.float32)
    stable = np.full(len(X), 0.01, dtype=np.float32)

    monkeypatch.setattr(
        causal_phenotyping,
        "_estimate_causalml_on_clean_inputs",
        lambda X_clean, treatment, outcome, method: unstable,
    )
    monkeypatch.setattr(
        causal_phenotyping,
        "_fallback_dml_cate",
        lambda X, treatment, outcome, n_folds=5, random_state=42: stable,
    )

    bundle = causal_phenotyping.estimate_cate_with_causalml(
        X=X,
        treatment=treatment,
        outcome=outcome,
        method="t_learner",
        stability_gate={"enabled": True, "max_std": 0.05, "max_abs_q90": 0.05},
    )

    assert bundle["estimator_selected"] == "cross_fitted_dml"
    assert "max_std" in bundle["fallback_reason"]
    assert np.allclose(bundle["cate"], stable)
    assert bundle["candidate_summary"]["std"] > bundle["selected_summary"]["std"]


def test_timesfm_feature_extraction_skips_cleanly_when_unavailable(tmp_path, monkeypatch):
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def _fake_find_spec(name, package=None):
        if name == "timesfm":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    continuous = np.ones((4, 12, len(CONTINUOUS_NAMES)), dtype=np.float32)
    masks = np.ones_like(continuous, dtype=np.float32)
    bundle = run_timesfm_feature_extraction(
        continuous=continuous,
        masks=masks,
        feature_names=CONTINUOUS_NAMES,
        output_dir=tmp_path,
        config={"enabled": True, "context_len": 6, "horizon_len": 3},
    )

    assert bundle["enabled"] is False
    assert bundle["summary"]["reason"] == "timesfm_not_installed"
    assert bundle["features_df"].empty
    assert (tmp_path / "timesfm" / "summary.json").exists()


def test_timesfm_feature_extraction_emits_dynamic_features_with_fake_model(tmp_path, monkeypatch):
    class FakeTimesFm:
        def forecast(self, inputs=None, freq=None):
            batch = np.stack(inputs).astype(np.float32)
            last = batch[:, -1:]
            return np.repeat(last, 2, axis=1)

    import importlib.util

    original_find_spec = importlib.util.find_spec

    def _fake_find_spec(name, package=None):
        if name == "timesfm":
            return object()
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setattr(
        "s6_optimization.timesfm_features._load_timesfm_model",
        lambda cfg: FakeTimesFm(),
    )

    N, T, F = 3, 10, len(CONTINUOUS_NAMES)
    continuous = np.zeros((N, T, F), dtype=np.float32)
    masks = np.ones((N, T, F), dtype=np.float32)
    map_idx = CONTINUOUS_NAMES.index("map")
    for i in range(N):
        continuous[i, :, map_idx] = np.arange(T, dtype=np.float32) + i

    bundle = run_timesfm_feature_extraction(
        continuous=continuous,
        masks=masks,
        feature_names=CONTINUOUS_NAMES,
        output_dir=tmp_path,
        config={
            "enabled": True,
            "context_len": 6,
            "horizon_len": 2,
            "selected_features": ["map"],
            "batch_size": 2,
        },
    )

    assert bundle["enabled"] is True
    assert "timesfm_map_forecast_mean" in bundle["features_df"].columns
    assert "timesfm_map_forecast_mae" in bundle["features_df"].columns
    assert len(bundle["features_df"]) == N


def test_domain_adaptation_coral_reduces_group_mean_gap(tmp_path):
    rng = np.random.default_rng(0)
    group_a = rng.normal(loc=0.0, scale=1.0, size=(400, 6))
    group_b = rng.normal(loc=3.0, scale=2.0, size=(400, 6))
    X = np.vstack([group_a, group_b]).astype(np.float32)
    groups = np.array(["a"] * len(group_a) + ["b"] * len(group_b), dtype=object)

    bundle = align_covariates_by_group(
        X,
        groups,
        output_dir=tmp_path,
        config={"enabled": True, "method": "coral", "min_group_size": 100, "reg": 1e-3},
    )

    summary = bundle["summary"]
    assert summary["enabled"] is True
    assert summary["improved_mean_gap"] is True
    assert summary["weighted_group_mean_gap_after"] < summary["weighted_group_mean_gap_before"]
    assert (tmp_path / "domain_adaptation_summary.json").exists()


def test_domain_adaptation_alpha_blend_supports_partial_alignment():
    rng = np.random.default_rng(1)
    group_a = rng.normal(loc=0.0, scale=1.0, size=(300, 4))
    group_b = rng.normal(loc=2.5, scale=1.5, size=(300, 4))
    X = np.vstack([group_a, group_b]).astype(np.float32)
    groups = np.array(["a"] * len(group_a) + ["b"] * len(group_b), dtype=object)

    identity = align_covariates_by_group(
        X,
        groups,
        config={"enabled": True, "method": "coral", "min_group_size": 50, "reg": 1e-3, "alpha": 0.0},
    )
    partial = align_covariates_by_group(
        X,
        groups,
        config={"enabled": True, "method": "coral", "min_group_size": 50, "reg": 1e-3, "alpha": 0.5},
    )
    full = align_covariates_by_group(
        X,
        groups,
        config={"enabled": True, "method": "coral", "min_group_size": 50, "reg": 1e-3, "alpha": 1.0},
    )

    assert np.allclose(identity["X_aligned"], X)
    before_gap = partial["summary"]["weighted_group_mean_gap_before"]
    partial_gap = partial["summary"]["weighted_group_mean_gap_after"]
    full_gap = full["summary"]["weighted_group_mean_gap_after"]
    assert before_gap > partial_gap > full_gap
    assert partial["summary"]["alpha"] == 0.5


def test_domain_adaptation_dann_reduces_group_gap(tmp_path):
    rng = np.random.default_rng(7)
    shared = rng.normal(loc=0.0, scale=1.0, size=(500, 6))
    group_a = shared + rng.normal(loc=0.0, scale=0.2, size=(500, 6))
    group_b = shared + 2.5 + rng.normal(loc=0.0, scale=0.2, size=(500, 6))
    X = np.vstack([group_a, group_b]).astype(np.float32)
    groups = np.array(["a"] * len(group_a) + ["b"] * len(group_b), dtype=object)

    bundle = align_covariates_by_group(
        X,
        groups,
        output_dir=tmp_path,
        config={
            "enabled": True,
            "method": "dann",
            "min_group_size": 100,
            "alpha": 1.0,
            "epochs": 10,
            "patience": 3,
            "batch_size": 128,
            "hidden_dim": 32,
            "embedding_dim": 12,
            "lambda_domain": 0.4,
            "lambda_recon": 1.0,
            "dropout": 0.0,
            "random_state": 42,
            "device": "cpu",
        },
    )

    summary = bundle["summary"]
    assert bundle["X_aligned"].shape == X.shape
    assert np.isfinite(bundle["X_aligned"]).all()
    assert summary["method"] == "dann"
    assert summary["adapter_summary"]["applied"] is True
    assert summary["weighted_group_mean_gap_after"] < summary["weighted_group_mean_gap_before"]
    if summary["domain_probe_accuracy_before"] is not None and summary["domain_probe_accuracy_after"] is not None:
        assert summary["domain_probe_accuracy_after"] <= summary["domain_probe_accuracy_before"]
    assert (tmp_path / "domain_adaptation_summary.json").exists()


def test_domain_adaptation_dann_skips_when_supported_groups_are_insufficient():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(24, 5)).astype(np.float32)
    groups = np.array(["a"] * 20 + ["b"] * 4, dtype=object)

    bundle = align_covariates_by_group(
        X,
        groups,
        config={
            "enabled": True,
            "method": "dann",
            "min_group_size": 10,
            "epochs": 4,
            "device": "cpu",
        },
    )

    assert np.allclose(bundle["X_aligned"], X)
    assert bundle["summary"]["reason"] == "insufficient_supported_groups_for_dann"


def test_domain_adaptation_dann_supports_coral_prealignment():
    rng = np.random.default_rng(17)
    group_a = rng.normal(loc=0.0, scale=1.0, size=(400, 6))
    group_b = rng.normal(loc=2.8, scale=1.4, size=(400, 6))
    X = np.vstack([group_a, group_b]).astype(np.float32)
    groups = np.array(["a"] * len(group_a) + ["b"] * len(group_b), dtype=object)

    bundle = align_covariates_by_group(
        X,
        groups,
        config={
            "enabled": True,
            "method": "dann",
            "min_group_size": 100,
            "prealign_method": "coral",
            "prealign_alpha": 0.6,
            "alpha": 1.0,
            "epochs": 6,
            "patience": 2,
            "batch_size": 128,
            "hidden_dim": 32,
            "embedding_dim": 12,
            "lambda_domain": 0.4,
            "lambda_recon": 1.0,
            "dropout": 0.0,
            "random_state": 42,
            "device": "cpu",
        },
    )

    summary = bundle["summary"]
    assert summary["method"] == "dann"
    assert summary["prealign_method"] == "coral"
    assert summary["prealign_alpha"] == 0.6
    assert summary["weighted_group_mean_gap_input"] < summary["weighted_group_mean_gap_before"]
    assert "mean_shift_l2_input" in summary["groups"][0]
    assert np.isfinite(bundle["X_aligned"]).all()


def test_domain_adaptation_dann_geometry_regularizer_reports_losses():
    rng = np.random.default_rng(23)
    shared = rng.normal(loc=0.0, scale=1.0, size=(350, 6))
    group_a = shared + rng.normal(loc=0.0, scale=0.2, size=(350, 6))
    group_b = shared + 2.2 + rng.normal(loc=0.0, scale=0.2, size=(350, 6))
    X = np.vstack([group_a, group_b]).astype(np.float32)
    groups = np.array(["a"] * len(group_a) + ["b"] * len(group_b), dtype=object)

    bundle = align_covariates_by_group(
        X,
        groups,
        config={
            "enabled": True,
            "method": "dann",
            "min_group_size": 100,
            "alpha": 1.0,
            "epochs": 6,
            "patience": 2,
            "batch_size": 128,
            "hidden_dim": 32,
            "embedding_dim": 12,
            "lambda_domain": 0.4,
            "lambda_recon": 1.0,
            "lambda_geometry": 8.0,
            "dropout": 0.0,
            "random_state": 42,
            "device": "cpu",
        },
    )

    adapter_summary = bundle["summary"]["adapter_summary"]
    assert adapter_summary["applied"] is True
    assert adapter_summary["lambda_geometry"] == 8.0
    assert "train_geometry_loss" in adapter_summary["last_epoch"]
    assert adapter_summary["last_epoch"]["train_geometry_loss"] >= 0.0
    if "val_geometry_loss" in adapter_summary["last_epoch"]:
        assert adapter_summary["last_epoch"]["val_geometry_loss"] >= 0.0
    assert np.isfinite(bundle["X_aligned"]).all()
