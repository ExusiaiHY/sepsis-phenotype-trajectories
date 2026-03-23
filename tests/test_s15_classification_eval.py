"""
test_s15_classification_eval.py - Unit tests for supervised embedding evaluation.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s15.classification_eval import train_mortality_classifier


def test_train_mortality_classifier_roundtrip():
    rng = np.random.default_rng(42)
    n_samples = 60

    labels = np.array([0] * 30 + [1] * 30)
    embeddings = np.column_stack([
        labels * 3.0 + rng.normal(scale=0.3, size=n_samples),
        rng.normal(size=n_samples),
    ])

    splits = {
        "train": list(range(0, 30)) + list(range(30, 45)),
        "val": list(range(45, 52)) + list(range(15, 22)),
        "test": list(range(52, 60)) + list(range(22, 30)),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        static_path = tmp_path / "static.csv"
        splits_path = tmp_path / "splits.json"
        output_dir = tmp_path / "out"

        pd.DataFrame({
            "mortality_inhospital": labels,
            "center_id": ["center_a"] * n_samples,
        }).to_csv(static_path, index=False)

        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits, f)

        report = train_mortality_classifier(
            embeddings=embeddings,
            static_path=static_path,
            splits_path=splits_path,
            output_dir=output_dir,
        )

        assert report["splits"]["val"]["accuracy"] >= 0.85
        assert report["splits"]["test"]["accuracy"] >= 0.85
        assert 0.05 <= report["threshold_selection"]["selected_threshold"] <= 0.95
        assert (output_dir / "mortality_classifier.pkl").exists()
        assert (output_dir / "mortality_classifier_report.json").exists()


if __name__ == "__main__":
    test_train_mortality_classifier_roundtrip()
    print("1 passed, 0 failed")
