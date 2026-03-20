# S0 Data Layer

Unified data extraction, preprocessing, and splitting for ICU sepsis phenotyping.

## Quick Start

```bash
cd project

# 1. Run unit tests (no data required)
python3.14 tests/test_s0_schema.py

# 2. Run smoke test with mock data (no real data required)
python3.14 scripts/s0_smoke_test.py

# 3. Run full pipeline on PhysioNet 2012 data
#    Requires: data/external/set-a/, set-b/, set-c/
python3.14 scripts/s0_prepare.py

# 4. Run with random split instead of cross-center
python3.14 scripts/s0_prepare.py --split-method random
```

## Output Structure

```
data/s0/
├── raw_aligned/              # After extraction + hourly alignment (NO imputation)
│   ├── continuous.npy        # (N, 48, 21) observed continuous measurements
│   ├── interventions.npy     # (N, 48, 2)  observed interventions (NaN if unavailable)
│   ├── proxy_indicators.npy  # (N, 48, 2)  proxy-derived indicators (NOT treatments)
│   ├── masks_continuous.npy  # 1 = measured at this hour, 0 = missing
│   ├── masks_interventions.npy
│   └── masks_proxy.npy
├── processed/                # After forward fill + median imputation + z-score
│   ├── continuous.npy        # Same shape, no NaN
│   ├── (other tensors copied from raw_aligned)
│   └── preprocess_stats.json
├── static.csv                # Patient-level metadata
├── feature_dict.json         # Variable dictionary with per-variable metadata
├── splits.json               # Train/val/test indices
└── data_manifest.json        # Full provenance record
```

## Key Design Decisions

1. **Proxy indicators are NOT treatment records.** MAP<65 and FiO2>0.21 are physiological proxies stored in a separate tensor. They are never mixed into the intervention tensor.

2. **raw_aligned and processed are separate.** Extraction/alignment and preprocessing are independent steps with independent outputs.

3. **sepsis_onset_hour is NaN for PhysioNet 2012.** The `anchor_time_type` field records that hour 0 means ICU admission, not sepsis onset.

4. **Masks survive preprocessing.** The observation masks record what was truly measured vs imputed. They are preserved unchanged from raw_aligned to processed.

5. **V1 backward compatibility** is provided via `s0/compat.py` with two modes:
   - `exact_v1`: produces the exact 24-feature format V1 expects
   - `extended_v1`: adds proxy indicators with correct names

## Using with V1 Pipeline

```python
from s0.compat import to_v1_format

# Load S0 data in V1 format
ts_3d, patient_info, feature_names = to_v1_format("data/s0", mode="exact_v1")

# Now use with existing V1 modules
from feature_engineering import extract_features
features = extract_features(ts_3d, config, feature_names)
```

## PhysioNet 2012 Variable Availability

| Category | Available | Unavailable |
|----------|-----------|-------------|
| Vitals (8) | All 8 observed | — |
| Labs (7) | All 7 (lactate/bilirubin sparse) | — |
| Blood gas (4) | All 4 observed | — |
| Derived (2) | pao2_fio2_ratio (in compat) | — |
| Interventions | — | antibiotics, RRT |
| Proxy | vasopressor_proxy, mechvent_proxy | — |
| Outcomes | In-hospital mortality (file or proxy) | 28-day mortality |
| Timing | — | sepsis_onset_hour |

## Notes

- No V1 performance metrics are carried into S0 outputs.
- Sepsis 2019 silhouette values are unresolved pending rerun and are not referenced.
- The MIMIC-IV and eICU extractors are not yet implemented; the schema is ready for them.
