#!/usr/bin/env python3
"""
s0_smoke_test.py - Smoke test with synthetic mock data.

Purpose:
  Verify the S0 pipeline works end-to-end without requiring real PhysioNet data.
  Creates small mock .txt files in a temp directory, runs extraction + preprocessing,
  and validates output shapes and types.

How to run:
  cd project
  python3.14 scripts/s0_smoke_test.py

Expected output:
  PASS / FAIL for each check, no files written to data/s0/.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s0.schema import N_CONTINUOUS, N_INTERVENTIONS, N_PROXY, CONTINUOUS_NAMES
from s0.physionet2012_extractor import extract_physionet2012
from s0.preprocessor import preprocess_raw_aligned
from s0.splits import build_splits
from s0.manifest import generate_manifest
from s0.compat import to_v1_format


def create_mock_patient(pid: int, directory: Path, n_hours: int = 10):
    """Create a minimal PhysioNet 2012 .txt file."""
    lines = ["Time,Parameter,Value"]
    # Demographics at 00:00
    lines.append(f"00:00,RecordID,{pid}")
    lines.append(f"00:00,Age,{55 + pid % 30}")
    lines.append(f"00:00,Gender,{pid % 2}")
    lines.append(f"00:00,Height,{170 + pid % 20}")
    lines.append(f"00:00,Weight,{70 + pid % 30}")
    lines.append(f"00:00,ICUType,{(pid % 4) + 1}")

    # Vital signs every hour
    rng = np.random.RandomState(pid)
    for h in range(n_hours):
        hr = 80 + rng.randn() * 10
        sbp = 120 + rng.randn() * 15
        dbp = 70 + rng.randn() * 10
        map_val = 85 + rng.randn() * 10
        temp = 37.0 + rng.randn() * 0.5
        gcs = max(3, min(15, int(14 + rng.randn())))
        lines.append(f"{h:02d}:00,HR,{hr:.1f}")
        lines.append(f"{h:02d}:00,SysABP,{sbp:.1f}")
        lines.append(f"{h:02d}:00,DiasABP,{dbp:.1f}")
        lines.append(f"{h:02d}:00,MAP,{map_val:.1f}")
        lines.append(f"{h:02d}:00,Temp,{temp:.2f}")
        lines.append(f"{h:02d}:00,GCS,{gcs}")

    # Labs every 6 hours
    for h in range(0, n_hours, 6):
        lines.append(f"{h:02d}:00,Creatinine,{1.0 + rng.rand():.2f}")
        lines.append(f"{h:02d}:00,BUN,{15 + rng.rand() * 10:.1f}")
        lines.append(f"{h:02d}:00,Glucose,{100 + rng.rand() * 50:.0f}")
        lines.append(f"{h:02d}:00,WBC,{8 + rng.rand() * 5:.1f}")

    fpath = directory / f"{pid}.txt"
    fpath.write_text("\n".join(lines))


def main():
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            failed += 1

    tmpdir = Path(tempfile.mkdtemp(prefix="s0_smoke_"))
    print(f"Temp directory: {tmpdir}")

    try:
        # Create mock data
        n_patients_a = 8
        n_patients_c = 4
        n_hours = 12

        mock_data = tmpdir / "mock_data"
        set_a = mock_data / "set-a"
        set_c = mock_data / "set-c"
        set_a.mkdir(parents=True)
        set_c.mkdir(parents=True)

        for i in range(n_patients_a):
            create_mock_patient(100 + i, set_a, n_hours)
        for i in range(n_patients_c):
            create_mock_patient(200 + i, set_c, n_hours)

        output_dir = tmpdir / "s0_output"

        # ===== Test 1: Extraction =====
        print("\n[Test 1] Extraction")
        stats = extract_physionet2012(
            data_dir=mock_data,
            output_dir=output_dir,
            n_hours=n_hours,
            sets=["set-a", "set-c"],
            center_a_sets={"set-a"},
            center_b_sets={"set-c"},
        )
        check("patients extracted", stats["n_patients"] == n_patients_a + n_patients_c)
        check("n_continuous", stats["n_continuous"] == N_CONTINUOUS)
        check("n_interventions", stats["n_interventions"] == N_INTERVENTIONS)
        check("n_proxy", stats["n_proxy"] == N_PROXY)

        # Check raw_aligned files exist
        raw_dir = output_dir / "raw_aligned"
        cont = np.load(raw_dir / "continuous.npy")
        masks = np.load(raw_dir / "masks_continuous.npy")
        proxy = np.load(raw_dir / "proxy_indicators.npy")
        interv = np.load(raw_dir / "interventions.npy")

        check("continuous shape", cont.shape == (12, n_hours, N_CONTINUOUS))
        check("masks shape", masks.shape == cont.shape)
        check("proxy shape", proxy.shape == (12, n_hours, N_PROXY))
        check("interventions shape", interv.shape == (12, n_hours, N_INTERVENTIONS))
        check("interventions all NaN", np.all(np.isnan(interv)))
        check("masks_interventions all zero", np.all(np.load(raw_dir / "masks_interventions.npy") == 0))

        # Check static.csv
        import pandas as pd
        static = pd.read_csv(output_dir / "static.csv")
        check("static rows", len(static) == 12)
        check("center_a count", (static["center_id"] == "center_a").sum() == n_patients_a)
        check("center_b count", (static["center_id"] == "center_b").sum() == n_patients_c)
        check("sepsis_onset is NaN", static["sepsis_onset_hour"].isna().all())
        check("anchor_time is icu_admission", (static["anchor_time_type"] == "icu_admission").all())
        check("data_source", (static["data_source"] == "physionet2012").all())

        # ===== Test 2: Preprocessing =====
        print("\n[Test 2] Preprocessing")
        prep_stats = preprocess_raw_aligned(
            input_dir=raw_dir,
            output_dir=output_dir / "processed",
        )
        proc_cont = np.load(output_dir / "processed" / "continuous.npy")
        check("processed shape matches raw", proc_cont.shape == cont.shape)
        check("no NaN after preprocessing", not np.any(np.isnan(proc_cont)))
        check("preprocess_stats.json exists", (output_dir / "processed" / "preprocess_stats.json").exists())

        # ===== Test 3: Splits =====
        print("\n[Test 3] Splits")
        split_result = build_splits(
            static_path=output_dir / "static.csv",
            output_path=output_dir / "splits.json",
            method="cross_center",
        )
        check("test set = center_b", len(split_result["test"]) == n_patients_c)
        check("train+val = center_a", len(split_result["train"]) + len(split_result["val"]) == n_patients_a)
        check("no overlap", len(set(split_result["train"]) & set(split_result["test"])) == 0)

        # ===== Test 4: Manifest =====
        print("\n[Test 4] Manifest")
        manifest = generate_manifest(output_dir, stats, prep_stats, {"test": True})
        check("manifest file exists", (output_dir / "data_manifest.json").exists())
        check("manifest has schema", "schema" in manifest)
        check("manifest has checksums", len(manifest["file_checksums"]) > 0)
        # Verify no Sepsis 2019 silhouette reference
        manifest_text = json.dumps(manifest)
        check("no silhouette in manifest", "silhouette" not in manifest_text.lower())

        # ===== Test 5: V1 Compatibility =====
        print("\n[Test 5] V1 Compatibility")

        ts_v1, info_v1, names_v1 = to_v1_format(output_dir, mode="exact_v1")
        check("exact_v1 features = 24", ts_v1.shape[2] == 24)
        check("exact_v1 feature names length", len(names_v1) == 24)
        check("exact_v1 has vasopressor", "vasopressor" in names_v1)
        check("exact_v1 has pao2_fio2_ratio", "pao2_fio2_ratio" in names_v1)
        check("exact_v1 patient_info has mortality_28d", "mortality_28d" in info_v1.columns)

        ts_ext, info_ext, names_ext = to_v1_format(output_dir, mode="extended_v1")
        check("extended_v1 features = 26", ts_ext.shape[2] == 26)
        check("extended_v1 has vasopressor_proxy", "vasopressor_proxy" in names_ext)

        # ===== Summary =====
        print(f"\n{'=' * 50}")
        print(f"Results: {passed} PASSED, {failed} FAILED")
        print(f"{'=' * 50}")

    finally:
        shutil.rmtree(tmpdir)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
