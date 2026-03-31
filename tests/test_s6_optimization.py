from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s6_optimization.baseline_comparison import generate_baseline_comparison_report
from s6_optimization.missingness_encoder import (
    compute_gap_lengths_vectorized,
    compute_missingness_features,
)
from s6_optimization.phenotype_naming import (
    assign_phenotype_by_causality,
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
