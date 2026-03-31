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
from s6_optimization.phenotype_naming import assign_phenotype_by_causality


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

    assert responsive == "hemodynamic_unstable_responsive"
    assert refractory == "hemodynamic_unstable_refractory"


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
