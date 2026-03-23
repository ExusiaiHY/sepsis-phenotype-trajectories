# MIMIC DuckDB Profile

- Generated: `2026-03-23T16:41:55.445856+00:00`
- Database: `/Users/exusiaihy/Desktop/Python高阶程序设计/project/archive/db/mimic4.db`
- Schemas: `information_schema, main, mimiciv_derived, mimiciv_hosp, mimiciv_icu, pg_catalog`

## Key Table Counts

| Table | Rows |
|------|-----:|
| `mimiciv_hosp.patients` | 20 |
| `mimiciv_hosp.admissions` | 20 |
| `mimiciv_icu.icustays` | 15 |
| `mimiciv_derived.icustay_detail` | 15 |
| `mimiciv_derived.sepsis3` | 5 |
| `mimiciv_derived.sirs` | 15 |
| `mimiciv_derived.sofa` | 720 |
| `mimiciv_derived.vitalsign` | 720 |
| `mimiciv_derived.chemistry` | 120 |
| `mimiciv_derived.complete_blood_count` | 120 |
| `mimiciv_derived.first_day_vitalsign` | 15 |
| `mimiciv_derived.first_day_lab` | 15 |

## Cohort Summary

- ICU stays in `mimiciv_derived.icustay_detail`: `15`
- Distinct patients: `15`
- Mean age: `59.99`
- Male rate: `0.5333`
- Hospital mortality: `0.2`
- Sepsis-3 prevalence: `0.3333`

## First-Day Missingness

| Variable | Missing Rate |
|----------|-------------:|
| `fd_heart_rate_missing` | 0.0 |
| `fd_sbp_missing` | 0.0 |
| `fd_resp_rate_missing` | 0.0 |
| `fd_spo2_missing` | 0.0 |
| `fd_creatinine_missing` | 0.0 |
| `fd_bilirubin_missing` | 0.0 |
| `fd_platelets_missing` | 0.0 |
| `fd_wbc_missing` | 0.0 |

## Analysis Table Readiness

- Required tables present: `True`
- Present tables: `9` / `9`
