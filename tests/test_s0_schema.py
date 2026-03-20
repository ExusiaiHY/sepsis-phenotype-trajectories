"""
test_s0_schema.py - Unit tests for S0 schema and validation.

How to run:
  cd project
  python3.14 -m pytest tests/test_s0_schema.py -v
  # or without pytest:
  python3.14 tests/test_s0_schema.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s0.schema import (
    CONTINUOUS_SCHEMA, INTERVENTION_SCHEMA, PROXY_SCHEMA,
    N_CONTINUOUS, N_INTERVENTIONS, N_PROXY,
    CONTINUOUS_NAMES, CONTINUOUS_INDEX,
    INTERVENTION_NAMES, PROXY_NAMES,
    V1_FEATURE_ORDER, V1_EXTENDED_ADDITIONS,
    PHYSIONET_ALL_MAP, PHYSIONET_DEMO_FIELDS,
    schema_to_feature_dict,
    validate_continuous_array, validate_intervention_array, validate_proxy_array,
)


def test_schema_counts():
    assert N_CONTINUOUS == 21, f"Expected 21 continuous, got {N_CONTINUOUS}"
    assert N_INTERVENTIONS == 2, f"Expected 2 interventions, got {N_INTERVENTIONS}"
    assert N_PROXY == 2, f"Expected 2 proxy, got {N_PROXY}"


def test_indices_sequential():
    for i, v in enumerate(CONTINUOUS_SCHEMA):
        assert v.index == i, f"Continuous index mismatch: {v.name} has index {v.index}, expected {i}"
    for i, v in enumerate(INTERVENTION_SCHEMA):
        assert v.index == i, f"Intervention index mismatch: {v.name}"
    for i, v in enumerate(PROXY_SCHEMA):
        assert v.index == i, f"Proxy index mismatch: {v.name}"


def test_no_duplicate_names():
    all_names = CONTINUOUS_NAMES + INTERVENTION_NAMES + PROXY_NAMES
    assert len(all_names) == len(set(all_names)), "Duplicate variable names in schema"


def test_continuous_index_dict():
    for name in CONTINUOUS_NAMES:
        assert name in CONTINUOUS_INDEX, f"{name} not in CONTINUOUS_INDEX"
    assert CONTINUOUS_INDEX["heart_rate"] == 0
    assert CONTINUOUS_INDEX["ph"] == 20


def test_physionet_map_targets_exist():
    """Every PhysioNet mapping target must be a valid continuous variable."""
    for pn_name, std_name in PHYSIONET_ALL_MAP.items():
        assert std_name in CONTINUOUS_INDEX, \
            f"PhysioNet var '{pn_name}' maps to '{std_name}' which is not in CONTINUOUS_INDEX"


def test_v1_feature_order():
    assert len(V1_FEATURE_ORDER) == 24, f"V1 should have 24 features, got {len(V1_FEATURE_ORDER)}"
    # Check V1 contains expected derived and proxy fields
    assert "pao2_fio2_ratio" in V1_FEATURE_ORDER
    assert "vasopressor" in V1_FEATURE_ORDER
    assert "mechanical_vent" in V1_FEATURE_ORDER


def test_availability_values():
    valid = {"observed", "proxy", "derived", "unavailable"}
    for v in CONTINUOUS_SCHEMA + INTERVENTION_SCHEMA + PROXY_SCHEMA:
        assert v.availability in valid, f"{v.name} has invalid availability: {v.availability}"


def test_normalization_groups():
    valid = {"vitals", "labs", "blood_gas", "binary", "none"}
    for v in CONTINUOUS_SCHEMA + INTERVENTION_SCHEMA + PROXY_SCHEMA:
        assert v.normalization_group in valid, f"{v.name} has invalid norm_group: {v.normalization_group}"


def test_proxy_not_imputable():
    for v in PROXY_SCHEMA:
        assert not v.imputation_allowed, f"Proxy {v.name} should not be imputable"


def test_intervention_not_imputable():
    for v in INTERVENTION_SCHEMA:
        assert not v.imputation_allowed, f"Intervention {v.name} should not be imputable"


def test_proxy_availability_is_proxy():
    for v in PROXY_SCHEMA:
        assert v.availability == "proxy", f"Proxy {v.name} availability should be 'proxy'"


def test_schema_serialization():
    d = schema_to_feature_dict()
    assert "continuous" in d
    assert "interventions" in d
    assert "proxy_indicators" in d
    assert d["n_continuous"] == N_CONTINUOUS
    assert len(d["continuous"]) == N_CONTINUOUS


def test_validation_functions():
    # Valid
    validate_continuous_array(np.zeros((10, 48, N_CONTINUOUS)), 10, 48)
    validate_intervention_array(np.zeros((10, 48, N_INTERVENTIONS)), 10, 48)
    validate_proxy_array(np.zeros((10, 48, N_PROXY)), 10, 48)

    # Invalid
    try:
        validate_continuous_array(np.zeros((10, 48, 5)), 10, 48)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_schema_counts,
        test_indices_sequential,
        test_no_duplicate_names,
        test_continuous_index_dict,
        test_physionet_map_targets_exist,
        test_v1_feature_order,
        test_availability_values,
        test_normalization_groups,
        test_proxy_not_imputable,
        test_intervention_not_imputable,
        test_proxy_availability_is_proxy,
        test_schema_serialization,
        test_validation_functions,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
