"""
schema.py - CLIF-inspired unified schema for longitudinal ICU data.

Purpose:
  Define all variable names, indices, per-variable metadata, and validation
  logic for the S0 data layer. This is the single source of truth for the
  variable dictionary across all data sources.

Connects to:
  - physionet2012_extractor.py reads CONTINUOUS_SCHEMA / INTERVENTION_SCHEMA / PROXY_SCHEMA
  - preprocessor.py reads normalization_group and imputation_allowed
  - dataset.py reads index mappings
  - compat.py reads V1_FEATURE_ORDER
  - manifest.py serializes schema to feature_dict.json

How to run:
  Not a standalone script. Imported by other S0 modules.
  python3.14 -c "from s0.schema import CONTINUOUS_SCHEMA; print(len(CONTINUOUS_SCHEMA))"

Expected output artifacts:
  None (pure definitions).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


# ============================================================
# Per-Variable Metadata
# ============================================================

@dataclass(frozen=True)
class VariableMeta:
    """Metadata for a single time-series variable."""
    name: str
    index: int
    unit: str
    availability: str      # observed | proxy | derived | unavailable
    reliability: str        # high | medium | low | unknown
    imputation_allowed: bool
    normalization_group: str  # vitals | labs | blood_gas | binary | none
    description: str = ""
    physionet2012_source: str = ""  # Raw variable name(s) in PhysioNet 2012

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Continuous Measurements (21 channels)
# ============================================================

CONTINUOUS_SCHEMA: list[VariableMeta] = [
    VariableMeta("heart_rate",   0,  "bpm",    "observed", "high",   True,  "vitals",    "Heart rate",            "HR"),
    VariableMeta("sbp",          1,  "mmHg",   "observed", "high",   True,  "vitals",    "Systolic blood pressure", "SysABP,NISysABP"),
    VariableMeta("dbp",          2,  "mmHg",   "observed", "high",   True,  "vitals",    "Diastolic blood pressure", "DiasABP,NIDiasABP"),
    VariableMeta("map",          3,  "mmHg",   "observed", "high",   True,  "vitals",    "Mean arterial pressure", "MAP,NIMAP"),
    VariableMeta("resp_rate",    4,  "/min",   "observed", "medium", True,  "vitals",    "Respiratory rate",       "RespRate"),
    VariableMeta("spo2",         5,  "%",      "observed", "medium", True,  "vitals",    "Oxygen saturation",      "SaO2"),
    VariableMeta("temperature",  6,  "°C",     "observed", "high",   True,  "vitals",    "Body temperature",       "Temp"),
    VariableMeta("gcs",          7,  "score",  "observed", "high",   True,  "vitals",    "Glasgow Coma Scale",     "GCS"),
    VariableMeta("creatinine",   8,  "mg/dL",  "observed", "medium", True,  "labs",      "Serum creatinine",       "Creatinine"),
    VariableMeta("bun",          9,  "mg/dL",  "observed", "medium", True,  "labs",      "Blood urea nitrogen",    "BUN"),
    VariableMeta("glucose",      10, "mg/dL",  "observed", "medium", True,  "labs",      "Blood glucose",          "Glucose"),
    VariableMeta("wbc",          11, "K/uL",   "observed", "medium", True,  "labs",      "White blood cell count",  "WBC"),
    VariableMeta("platelet",     12, "K/uL",   "observed", "medium", True,  "labs",      "Platelet count",          "Platelets"),
    VariableMeta("potassium",    13, "mEq/L",  "observed", "medium", True,  "labs",      "Serum potassium",        "K"),
    VariableMeta("sodium",       14, "mEq/L",  "observed", "medium", True,  "labs",      "Serum sodium",           "Na"),
    VariableMeta("lactate",      15, "mmol/L", "observed", "low",    True,  "labs",      "Blood lactate",          "Lactate"),
    VariableMeta("bilirubin",    16, "mg/dL",  "observed", "low",    True,  "labs",      "Total bilirubin",        "Bilirubin"),
    VariableMeta("pao2",         17, "mmHg",   "observed", "medium", True,  "blood_gas", "Partial pressure O2",    "PaO2"),
    VariableMeta("fio2",         18, "frac",   "observed", "medium", True,  "blood_gas", "Fraction inspired O2",   "FiO2"),
    VariableMeta("paco2",        19, "mmHg",   "observed", "medium", True,  "blood_gas", "Partial pressure CO2",   "PaCO2"),
    VariableMeta("ph",           20, "",       "observed", "medium", True,  "blood_gas", "Blood pH",               "pH"),
]

N_CONTINUOUS = len(CONTINUOUS_SCHEMA)
CONTINUOUS_NAMES = [v.name for v in CONTINUOUS_SCHEMA]
CONTINUOUS_INDEX = {v.name: v.index for v in CONTINUOUS_SCHEMA}


# ============================================================
# Observed Interventions (2 channels, structural placeholders for PhysioNet 2012)
# ============================================================

INTERVENTION_SCHEMA: list[VariableMeta] = [
    VariableMeta("antibiotics_on", 0, "binary", "unavailable", "unknown", False, "binary",
                 "Antibiotic administration", ""),
    VariableMeta("rrt_on",         1, "binary", "unavailable", "unknown", False, "binary",
                 "Renal replacement therapy", ""),
]

N_INTERVENTIONS = len(INTERVENTION_SCHEMA)
INTERVENTION_NAMES = [v.name for v in INTERVENTION_SCHEMA]


# ============================================================
# Proxy Indicators (2 channels, NOT true treatment records)
# ============================================================

PROXY_SCHEMA: list[VariableMeta] = [
    VariableMeta("vasopressor_proxy", 0, "binary", "proxy", "low", False, "binary",
                 "Proxy: MAP < 65 mmHg (NOT a treatment record)", "Derived from MAP"),
    VariableMeta("mechvent_proxy",    1, "binary", "proxy", "low", False, "binary",
                 "Proxy: FiO2 > 0.21 (NOT a treatment record)", "Derived from FiO2"),
]

N_PROXY = len(PROXY_SCHEMA)
PROXY_NAMES = [v.name for v in PROXY_SCHEMA]


# ============================================================
# Static Metadata Fields
# ============================================================

STATIC_FIELDS = [
    "patient_id", "age", "sex", "height_cm", "weight_kg",
    "icu_type", "icu_los_hours",
    "mortality_inhospital", "mortality_source",
    "center_id", "set_name", "data_source",
    "sepsis_onset_hour", "anchor_time_type",
]

# anchor_time_type values
ANCHOR_ICU_ADMISSION = "icu_admission"
ANCHOR_SEPSIS_ONSET = "sepsis_onset"
ANCHOR_UNKNOWN = "unknown"

# mortality_source values
MORTALITY_OUTCOMES_FILE = "outcomes_file"
MORTALITY_PROXY_GCS_MAP = "proxy_gcs_map"
MORTALITY_UNAVAILABLE = "unavailable"


# ============================================================
# V1 Compatibility Feature Orders
# ============================================================

# Exact V1 feature list (from load_physionet2012.py PROJECT_FEATURES)
V1_FEATURE_ORDER = [
    "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature", "gcs",
    "creatinine", "bun", "glucose", "wbc", "platelet", "potassium", "sodium",
    "lactate", "bilirubin", "pao2", "fio2", "paco2", "ph",
    "pao2_fio2_ratio", "vasopressor", "mechanical_vent",
]

# Extended V1 adds proxy indicators but does NOT relabel them as treatments
V1_EXTENDED_ADDITIONS = ["vasopressor_proxy", "mechvent_proxy"]


# ============================================================
# PhysioNet 2012 Variable Mapping
# ============================================================

PHYSIONET_VITAL_MAP = {
    "HR": "heart_rate",
    "SysABP": "sbp", "NISysABP": "sbp",
    "DiasABP": "dbp", "NIDiasABP": "dbp",
    "MAP": "map", "NIMAP": "map",
    "RespRate": "resp_rate",
    "SaO2": "spo2",
    "Temp": "temperature",
    "GCS": "gcs",
}

PHYSIONET_LAB_MAP = {
    "Creatinine": "creatinine",
    "BUN": "bun",
    "Glucose": "glucose",
    "WBC": "wbc",
    "Platelets": "platelet",
    "K": "potassium",
    "Na": "sodium",
    "Lactate": "lactate",
    "Bilirubin": "bilirubin",
    "PaO2": "pao2",
    "FiO2": "fio2",
    "PaCO2": "paco2",
    "pH": "ph",
}

PHYSIONET_ALL_MAP = {**PHYSIONET_VITAL_MAP, **PHYSIONET_LAB_MAP}

# Demographics fields at time 00:00
PHYSIONET_DEMO_FIELDS = {"RecordID", "Age", "Gender", "Height", "Weight", "ICUType"}


# ============================================================
# Schema Serialization
# ============================================================

def schema_to_feature_dict() -> dict:
    """Serialize full schema to JSON-compatible dict for feature_dict.json."""
    return {
        "continuous": [v.to_dict() for v in CONTINUOUS_SCHEMA],
        "interventions": [v.to_dict() for v in INTERVENTION_SCHEMA],
        "proxy_indicators": [v.to_dict() for v in PROXY_SCHEMA],
        "n_continuous": N_CONTINUOUS,
        "n_interventions": N_INTERVENTIONS,
        "n_proxy": N_PROXY,
        "static_fields": STATIC_FIELDS,
    }


# ============================================================
# Validation
# ============================================================

def validate_continuous_array(arr, n_patients: int, n_hours: int) -> None:
    """Validate continuous tensor shape."""
    expected = (n_patients, n_hours, N_CONTINUOUS)
    if arr.shape != expected:
        raise ValueError(f"Continuous array shape {arr.shape} != expected {expected}")


def validate_intervention_array(arr, n_patients: int, n_hours: int) -> None:
    expected = (n_patients, n_hours, N_INTERVENTIONS)
    if arr.shape != expected:
        raise ValueError(f"Intervention array shape {arr.shape} != expected {expected}")


def validate_proxy_array(arr, n_patients: int, n_hours: int) -> None:
    expected = (n_patients, n_hours, N_PROXY)
    if arr.shape != expected:
        raise ValueError(f"Proxy array shape {arr.shape} != expected {expected}")
