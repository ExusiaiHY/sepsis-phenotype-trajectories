"""
test_eicu_loader.py - Unit tests for eICU raw/demo integration.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eicu_loader import load_eicu_dataset, prepare_eicu_demo_artifacts


FEATURE_NAMES = [
    "heart_rate",
    "sbp",
    "dbp",
    "map",
    "resp_rate",
    "spo2",
    "temperature",
    "lactate",
    "creatinine",
    "bilirubin",
    "platelet",
    "wbc",
    "pao2_fio2",
    "inr",
    "vasopressor",
    "mechanical_vent",
    "rrt",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_eicu_dataset_and_prepare_artifacts():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)

        _write_csv(
            root / "patient.csv",
            [
                {
                    "patientunitstayid": 1,
                    "uniquepid": "UP1",
                    "gender": "Male",
                    "age": "65",
                    "hospitalid": 10,
                    "unittype": "MICU",
                    "hospitaldischargestatus": "Expired",
                    "unitdischargeoffset": 2880,
                },
                {
                    "patientunitstayid": 2,
                    "uniquepid": "UP2",
                    "gender": "Female",
                    "age": "> 89",
                    "hospitalid": 20,
                    "unittype": "CCU",
                    "hospitaldischargestatus": "Alive",
                    "unitdischargeoffset": 1440,
                },
            ],
        )
        _write_csv(
            root / "vitalPeriodic.csv",
            [
                {
                    "patientunitstayid": 1,
                    "observationoffset": 0,
                    "heartrate": 110,
                    "systemicsystolic": 92,
                    "systemicdiastolic": 50,
                    "systemicmean": 63,
                    "respiration": 24,
                    "sao2": 95,
                    "temperature": 38.2,
                },
                {
                    "patientunitstayid": 1,
                    "observationoffset": 60,
                    "heartrate": 108,
                    "systemicsystolic": 94,
                    "systemicdiastolic": 52,
                    "systemicmean": 66,
                    "respiration": 22,
                    "sao2": 96,
                    "temperature": 38.0,
                },
                {
                    "patientunitstayid": 2,
                    "observationoffset": 0,
                    "heartrate": 88,
                    "systemicsystolic": "",
                    "systemicdiastolic": "",
                    "systemicmean": "",
                    "respiration": 18,
                    "sao2": 99,
                    "temperature": 36.8,
                },
            ],
        )
        _write_csv(
            root / "vitalAperiodic.csv",
            [
                {
                    "patientunitstayid": 2,
                    "observationoffset": 0,
                    "noninvasivesystolic": 122,
                    "noninvasivediastolic": 68,
                    "noninvasivemean": 86,
                }
            ],
        )
        _write_csv(
            root / "lab.csv",
            [
                {"patientunitstayid": 1, "labresultoffset": 0, "labname": "lactate", "labresult": 4.8},
                {"patientunitstayid": 1, "labresultoffset": 60, "labname": "creatinine", "labresult": 2.1},
                {"patientunitstayid": 1, "labresultoffset": 120, "labname": "PT - INR", "labresult": 1.7},
                {"patientunitstayid": 2, "labresultoffset": 0, "labname": "WBC x 1000", "labresult": 11.5},
            ],
        )
        _write_csv(
            root / "infusionDrug.csv",
            [
                {"patientunitstayid": 1, "infusionoffset": 120, "drugname": "Norepinephrine", "drugrate": "0.1"}
            ],
        )
        _write_csv(
            root / "respiratoryCare.csv",
            [
                {
                    "patientunitstayid": 1,
                    "respcarestatusoffset": 60,
                    "ventstartoffset": 60,
                    "ventendoffset": 180,
                    "airwaytype": "ET Tube",
                }
            ],
        )
        _write_csv(
            root / "intakeOutput.csv",
            [
                {"patientunitstayid": 1, "intakeoutputoffset": 180, "dialysistotal": 5.0, "celllabel": "Dialysis"}
            ],
        )
        _write_csv(
            root / "apacheApsVar.csv",
            [
                {
                    "patientunitstayid": 1,
                    "vent": 1,
                    "dialysis": 1,
                    "heartrate": 111,
                    "meanbp": 62,
                    "temperature": 38.1,
                    "respiratoryrate": 23,
                    "creatinine": 2.0,
                    "bilirubin": 1.2,
                    "wbc": 13.0,
                    "pao2": 90,
                    "fio2": 45,
                },
                {
                    "patientunitstayid": 2,
                    "vent": 0,
                    "dialysis": 0,
                    "heartrate": 89,
                    "meanbp": 86,
                    "temperature": 36.9,
                    "respiratoryrate": 17,
                    "creatinine": 0.9,
                    "bilirubin": 0.6,
                    "wbc": 11.0,
                    "pao2": 80,
                    "fio2": 40,
                },
            ],
        )

        tensor, patient_info = load_eicu_dataset(
            data_dir=root,
            feature_names=FEATURE_NAMES,
            n_timesteps=48,
        )

        feature_to_idx = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

        assert tensor.shape == (2, 48, len(FEATURE_NAMES))
        assert patient_info["mortality_28d"].tolist() == [1, 0]
        assert patient_info["shock_onset"].tolist() == [1, 0]
        assert patient_info["gender"].tolist() == ["M", "F"]
        assert patient_info["age"].tolist() == [65.0, 89.0]
        assert np.isclose(tensor[0, 0, feature_to_idx["heart_rate"]], 110.0)
        assert np.isclose(tensor[1, 0, feature_to_idx["sbp"]], 122.0)
        assert np.isclose(tensor[0, 0, feature_to_idx["lactate"]], 4.8)
        assert np.isclose(tensor[0, 1, feature_to_idx["creatinine"]], 2.1)
        assert np.isclose(tensor[1, 0, feature_to_idx["pao2_fio2"]], 200.0)
        assert np.all(tensor[0, 2:, feature_to_idx["vasopressor"]] == 1.0)
        assert np.all(tensor[0, 1:4, feature_to_idx["mechanical_vent"]] == 1.0)
        assert np.all(tensor[0, 3:, feature_to_idx["rrt"]] == 1.0)

        config = {
            "data": {
                "simulated": {"n_timesteps": 48},
                "eicu": {"data_dir": str(root.relative_to(Path(tmp_dir).parent)), "n_timesteps": 48},
            },
            "variables": {
                "vitals": FEATURE_NAMES[:7],
                "labs": FEATURE_NAMES[7:14],
                "treatments": FEATURE_NAMES[14:],
            },
        }
        output_dir = root / "prepared"
        config["data"]["eicu"]["data_dir"] = str(root)
        report = prepare_eicu_demo_artifacts(config, output_dir=output_dir)

        assert report["n_patients"] == 2
        assert (output_dir / "time_series_eicu_demo.npy").exists()
        assert (output_dir / "patient_info_eicu_demo.csv").exists()
        assert (output_dir / "eicu_demo_report.json").exists()

        saved_report = json.loads((output_dir / "eicu_demo_report.json").read_text(encoding="utf-8"))
        assert saved_report["tables_found"]["patient"] == "patient.csv"


if __name__ == "__main__":
    test_load_eicu_dataset_and_prepare_artifacts()
    print("1 passed, 0 failed")
