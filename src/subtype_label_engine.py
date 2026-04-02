"""
subtype_label_engine.py - Generate split sepsis subtype labels for multi-task learning.

This module distinguishes:
  - gold labels: trial-anchored or biomarker-anchored targets when available
  - proxy labels: clinically motivated surrogate subtype assignments
  - score targets: continuous subtype affinity / treatment-benefit scores
  - masks: which tasks are actually observable per patient

For backward compatibility, legacy columns such as `immune_subtype` and
`organ_subtype` are still exported as aliases to the new proxy task columns.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:  # pragma: no cover - optional fallback dependency
    duckdb = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

IMMUNE_PROXY_CLASS_TO_ID = {
    "Unclassified": 0,
    "immunoparalysis-like": 1,
    "MALS-like": 2,
}
CLINICAL_PROXY_CLASS_TO_ID = {
    "Unclassified": 0,
    "alpha-like": 1,
    "beta-like": 2,
    "gamma-like": 3,
    "delta-like": 4,
}
TRAJECTORY_PROXY_CLASS_TO_ID = {
    "Unclassified": 0,
    "group-a": 1,
    "group-b": 2,
    "group-c": 3,
    "group-d": 4,
}
FLUID_PROXY_CLASS_TO_ID = {
    "Unclassified": 0,
    "restrictive-fluid-benefit-like": 1,
    "resuscitation-fluid-benefit-like": 2,
}
LEGACY_IMMUNE_ALIASES = {
    "Unclassified": "Unclassified",
    "immunoparalysis-like": "EIL-like",
    "MALS-like": "MAS-like",
}
LEGACY_FLUID_ALIASES = {
    "Unclassified": "Unclassified",
    "restrictive-fluid-benefit-like": "low_benefit",
    "resuscitation-fluid-benefit-like": "high_benefit",
}
MHLA_DR_CANDIDATE_COLS = [
    "fd_mhla_dr_min",
    "fd_mhla_dr_first24h",
    "mhla_dr_min",
    "mhla_dr_first24h",
    "mhla_dr",
    "monocyte_hla_dr",
]

CLASSIFICATION_TASKS = [
    {
        "name": "gold_mals",
        "label_col": "gold_mals_label",
        "mask_col": "mask_gold_mals_label",
        "classes": ["negative", "positive"],
        "description": "Ferritin-anchored MALS binary target",
    },
    {
        "name": "gold_immunoparalysis",
        "label_col": "gold_immunoparalysis_label",
        "mask_col": "mask_gold_immunoparalysis_label",
        "classes": ["negative", "positive"],
        "description": "mHLA-DR-anchored immunoparalysis binary target",
    },
    {
        "name": "proxy_immune_state",
        "label_col": "proxy_immune_state_label",
        "mask_col": "mask_proxy_immune_state_label",
        "classes": list(IMMUNE_PROXY_CLASS_TO_ID.keys()),
        "description": "Proxy immune-state task when direct immunophenotyping is absent",
    },
    {
        "name": "proxy_clinical_phenotype",
        "label_col": "proxy_clinical_phenotype_label",
        "mask_col": "mask_proxy_clinical_phenotype_label",
        "classes": list(CLINICAL_PROXY_CLASS_TO_ID.keys()),
        "description": "Proxy alpha/beta/gamma/delta phenotype task",
    },
    {
        "name": "proxy_trajectory_phenotype",
        "label_col": "proxy_trajectory_phenotype_label",
        "mask_col": "mask_proxy_trajectory_phenotype_label",
        "classes": list(TRAJECTORY_PROXY_CLASS_TO_ID.keys()),
        "description": "Proxy Group A/B/C/D early-vital-trajectory task",
    },
    {
        "name": "proxy_fluid_strategy",
        "label_col": "proxy_fluid_strategy_label",
        "mask_col": "mask_proxy_fluid_strategy_label",
        "classes": list(FLUID_PROXY_CLASS_TO_ID.keys()),
        "description": "Proxy fluid-strategy benefit task",
    },
]

REGRESSION_TASKS = [
    ("score_mals", "mask_score_mals"),
    ("score_immunoparalysis", "mask_score_immunoparalysis"),
    ("score_alpha", "mask_score_alpha"),
    ("score_beta", "mask_score_beta"),
    ("score_gamma", "mask_score_gamma"),
    ("score_delta", "mask_score_delta"),
    ("score_trajectory_a", "mask_score_trajectory_a"),
    ("score_trajectory_b", "mask_score_trajectory_b"),
    ("score_trajectory_c", "mask_score_trajectory_c"),
    ("score_trajectory_d", "mask_score_trajectory_d"),
    ("score_restrictive_fluid_benefit", "mask_score_restrictive_fluid_benefit"),
    ("score_resuscitation_fluid_benefit", "mask_score_resuscitation_fluid_benefit"),
]


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            if duckdb is None:
                raise
            return duckdb.sql(
                "SELECT * FROM read_parquet(?)",
                params=[str(path)],
            ).df()
    return pd.read_csv(path)


def _safe_numeric(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float32)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_binary(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    series = pd.to_numeric(df[col], errors="coerce")
    return series.fillna(0.0).clip(lower=0.0, upper=1.0).astype(np.float32)


def _score_high(series: pd.Series, low: float, high: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    denom = max(float(high) - float(low), 1.0e-6)
    scaled = ((values - float(low)) / denom).clip(lower=0.0, upper=1.0)
    return scaled.astype(np.float32)


def _score_low(series: pd.Series, high: float, low: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    denom = max(float(high) - float(low), 1.0e-6)
    scaled = ((float(high) - values) / denom).clip(lower=0.0, upper=1.0)
    return scaled.astype(np.float32)


def _score_band(series: pd.Series, low: float, high: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.between(low, high, inclusive="both").astype(np.float32)


def _max_of(*series_list: pd.Series) -> pd.Series:
    return pd.concat(series_list, axis=1).max(axis=1).astype(np.float32)


def _weighted_average(components: list[pd.Series], weights: list[float]) -> pd.Series:
    if not components:
        return pd.Series(dtype=np.float32)
    frame = pd.concat(components, axis=1)
    value_array = frame.fillna(0.0).to_numpy(dtype=np.float32)
    avail_array = frame.notna().to_numpy(dtype=np.float32)
    weight_array = np.asarray(weights, dtype=np.float32).reshape(1, -1)
    numer = (value_array * weight_array).sum(axis=1)
    denom = (avail_array * weight_array).sum(axis=1)
    out = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    return pd.Series(out.astype(np.float32), index=frame.index)


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _label_from_scores(
    score_frame: pd.DataFrame,
    *,
    threshold: float,
    class_order: list[str],
    fallback: str = "Unclassified",
) -> pd.Series:
    labels = pd.Series(fallback, index=score_frame.index, dtype="object")
    if score_frame.empty:
        return labels
    best_idx = score_frame.to_numpy(dtype=np.float32).argmax(axis=1)
    best_score = score_frame.max(axis=1)
    valid = best_score >= float(threshold)
    if np.any(valid):
        selected = pd.Index(class_order).take(best_idx[valid.to_numpy()])
        labels.loc[valid] = selected.astype(object)
    return labels


def _availability_mask(parts: list[pd.Series], minimum_present: int) -> pd.Series:
    frame = pd.concat(parts, axis=1)
    counts = frame.notna().sum(axis=1)
    return (counts >= int(minimum_present)).astype(np.float32)


def _dominance_score(target: pd.Series, others: list[pd.Series]) -> pd.Series:
    other_max = pd.concat(others, axis=1).max(axis=1).replace(0, np.nan)
    dominance = (target - other_max).fillna(0.0)
    return _score_high(dominance, 0.0, 2.0)


def _build_immune_targets(df: pd.DataFrame) -> pd.DataFrame:
    ferritin = _safe_numeric(df, "fd_ferritin_max")
    crp = _safe_numeric(df, "fd_crp_max")
    platelets = _safe_numeric(df, "fd_platelet_min")
    bilirubin = _safe_numeric(df, "fd_bilirubin_max")
    alt = _safe_numeric(df, "fd_alt_max")
    ddimer = _safe_numeric(df, "fd_ddimer_max")
    lymph_abs = _safe_numeric(df, "fd_lymphocytes_min")
    lymph_pct = _safe_numeric(df, "fd_lymphocytes_pct_min")
    culture_pos = _max_of(
        _safe_binary(df, "blood_culture_positive"),
        _safe_binary(df, "resp_culture_positive"),
        _safe_binary(df, "any_culture_positive"),
    )

    mhla_col = _first_existing_column(df, MHLA_DR_CANDIDATE_COLS)
    mhla = _safe_numeric(df, mhla_col) if mhla_col is not None else pd.Series(np.nan, index=df.index)

    ferritin_score = _score_high(ferritin, 1000.0, 4420.0)
    crp_score = _score_high(crp, 75.0, 175.0)
    platelet_score = _score_low(platelets, 150.0, 50.0)
    liver_or_coag = _max_of(
        _score_high(bilirubin, 1.5, 5.0),
        _score_high(alt, 80.0, 250.0),
        _score_high(ddimer, 1000.0, 6000.0),
    )
    lymph_low = _max_of(
        _score_low(lymph_abs, 1.2, 0.3),
        _score_low(lymph_pct, 18.0, 6.0),
    )
    modest_crp = _score_low(crp, 180.0, 80.0)
    mhla_low = _score_low(mhla, 15000.0, 5000.0) if mhla_col is not None else pd.Series(np.nan, index=df.index)

    mals_score = _weighted_average(
        [ferritin_score, crp_score, platelet_score, liver_or_coag],
        [0.40, 0.20, 0.15, 0.25],
    )
    immuno_score = _weighted_average(
        [mhla_low, lymph_low, culture_pos, modest_crp],
        [0.45 if mhla_col is not None else 0.0, 0.25, 0.20, 0.10],
    )
    proxy_mask = _availability_mask(
        [ferritin, crp, platelets, bilirubin, ddimer, lymph_abs, lymph_pct, culture_pos],
        minimum_present=3,
    )

    proxy_label = pd.Series("Unclassified", index=df.index, dtype="object")
    mals_like = (mals_score >= 0.55) & (mals_score >= (immuno_score.fillna(0.0) + 0.05))
    immuno_like = (immuno_score >= 0.50) & ~mals_like
    proxy_label.loc[mals_like] = "MALS-like"
    proxy_label.loc[immuno_like] = "immunoparalysis-like"

    gold_mals_mask = ferritin.notna().astype(np.float32)
    gold_mals_label = (ferritin >= 4420.0).fillna(False).astype(np.int64)

    gold_immuno_mask = mhla.notna().astype(np.float32)
    gold_immuno_label = (mhla <= 5000.0).fillna(False).astype(np.int64)

    out = pd.DataFrame(index=df.index)
    out["proxy_immune_state"] = proxy_label
    out["proxy_immune_state_label"] = proxy_label.map(IMMUNE_PROXY_CLASS_TO_ID).astype(np.int64)
    out["mask_proxy_immune_state_label"] = proxy_mask
    out["gold_mals_label"] = gold_mals_label
    out["mask_gold_mals_label"] = gold_mals_mask
    out["gold_immunoparalysis_label"] = gold_immuno_label
    out["mask_gold_immunoparalysis_label"] = gold_immuno_mask
    out["score_mals"] = mals_score.fillna(0.0).astype(np.float32)
    out["mask_score_mals"] = proxy_mask
    out["score_immunoparalysis"] = immuno_score.fillna(0.0).astype(np.float32)
    out["mask_score_immunoparalysis"] = proxy_mask
    out["immune_biomarker_source"] = mhla_col if mhla_col is not None else "none"
    return out


def _build_clinical_targets(df: pd.DataFrame) -> pd.DataFrame:
    sofa = _safe_numeric(df, "first_day_sofa")
    resp = _safe_numeric(df, "sofa_resp")
    coag = _safe_numeric(df, "sofa_coag")
    liver = _safe_numeric(df, "sofa_liver")
    cardio = _safe_numeric(df, "sofa_cardio")
    cns = _safe_numeric(df, "sofa_cns")
    renal = _safe_numeric(df, "sofa_renal")
    lactate = _safe_numeric(df, "fd_lactate_max")
    d_dimer = _safe_numeric(df, "fd_ddimer_max")
    spo2_min = _safe_numeric(df, "fd_spo2_min")
    rr_min = _safe_numeric(df, "fd_rr_min")
    vent = _safe_binary(df, "mech_vent_first24h")

    n_organs_ge2 = (
        (resp >= 2).fillna(False).astype(np.int64)
        + (coag >= 2).fillna(False).astype(np.int64)
        + (liver >= 2).fillna(False).astype(np.int64)
        + (cardio >= 2).fillna(False).astype(np.int64)
        + (cns >= 2).fillna(False).astype(np.int64)
        + (renal >= 2).fillna(False).astype(np.int64)
    )
    delta_score = _weighted_average(
        [
            _score_high(sofa, 4.0, 12.0),
            _score_high(pd.Series(n_organs_ge2, index=df.index), 2.0, 4.0),
            _score_high(lactate, 2.0, 4.5),
            _score_high(d_dimer, 1000.0, 6000.0),
        ],
        [0.35, 0.30, 0.20, 0.15],
    )

    gamma_score = _weighted_average(
        [
            _score_high(resp, 1.0, 4.0),
            _dominance_score(resp, [liver, cardio, renal]),
            _max_of(_score_low(spo2_min, 94.0, 84.0), _score_high(rr_min, 20.0, 35.0)),
            vent,
        ],
        [0.35, 0.20, 0.25, 0.20],
    )

    alpha_score = _weighted_average(
        [
            _weighted_average([_score_high(liver, 1.0, 4.0), _score_high(renal, 1.0, 4.0)], [0.5, 0.5]),
            _dominance_score(liver + renal, [resp + cardio, cns + coag]),
            _score_high(sofa, 3.0, 9.0),
        ],
        [0.50, 0.25, 0.25],
    )

    beta_score = _weighted_average(
        [
            _weighted_average([_score_high(cardio, 1.0, 4.0), _score_high(renal, 1.0, 4.0)], [0.6, 0.4]),
            _dominance_score(cardio + renal, [resp + liver, cns + coag]),
            _score_high(sofa, 3.0, 9.0),
        ],
        [0.50, 0.25, 0.25],
    )

    score_frame = pd.DataFrame(
        {
            "alpha-like": alpha_score,
            "beta-like": beta_score,
            "gamma-like": gamma_score,
            "delta-like": delta_score,
        },
        index=df.index,
    )
    proxy_mask = _availability_mask([sofa, resp, liver, cardio, renal], minimum_present=3)
    proxy_label = _label_from_scores(
        score_frame,
        threshold=0.45,
        class_order=["alpha-like", "beta-like", "gamma-like", "delta-like"],
    )

    out = pd.DataFrame(index=df.index)
    out["proxy_clinical_phenotype"] = proxy_label
    out["proxy_clinical_phenotype_label"] = proxy_label.map(CLINICAL_PROXY_CLASS_TO_ID).astype(np.int64)
    out["mask_proxy_clinical_phenotype_label"] = proxy_mask
    out["score_alpha"] = alpha_score.fillna(0.0).astype(np.float32)
    out["score_beta"] = beta_score.fillna(0.0).astype(np.float32)
    out["score_gamma"] = gamma_score.fillna(0.0).astype(np.float32)
    out["score_delta"] = delta_score.fillna(0.0).astype(np.float32)
    for col in ("score_alpha", "score_beta", "score_gamma", "score_delta"):
        out[f"mask_{col}"] = proxy_mask
    return out


def _trajectory_defaults(index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["proxy_trajectory_phenotype"] = "Unclassified"
    out["proxy_trajectory_phenotype_label"] = 0
    out["mask_proxy_trajectory_phenotype_label"] = 0.0
    for suffix in ("a", "b", "c", "d"):
        out[f"score_trajectory_{suffix}"] = 0.0
        out[f"mask_score_trajectory_{suffix}"] = 0.0
    return out


def _build_trajectory_targets(df: pd.DataFrame, timeseries_path: Path | None) -> pd.DataFrame:
    out = _trajectory_defaults(df.index)
    if timeseries_path is None:
        return out
    timeseries_path = Path(timeseries_path)
    if not timeseries_path.exists():
        return out

    ts = _read_table(timeseries_path)
    if "hr" not in ts.columns:
        return out

    id_col = "stay_id" if "stay_id" in ts.columns and "stay_id" in df.columns else None
    if id_col is None and "patient_id" in ts.columns and "patient_id" in df.columns:
        id_col = "patient_id"
    if id_col is None:
        return out

    needed = {"heart_rate", "resp_rate", "temperature", "sbp"}
    if not needed.intersection(set(ts.columns)):
        return out

    ts = ts.copy()
    ts[id_col] = ts[id_col].astype(str)
    ts["hr"] = pd.to_numeric(ts["hr"], errors="coerce")
    ts = ts[(ts["hr"] >= 0) & (ts["hr"] < 8)]
    if ts.empty:
        return out

    ts["map_proxy"] = pd.to_numeric(ts["map"], errors="coerce") if "map" in ts.columns else np.nan
    ts["sbp_proxy"] = pd.to_numeric(ts["sbp"], errors="coerce") if "sbp" in ts.columns else np.nan
    ts["heart_rate"] = pd.to_numeric(ts["heart_rate"], errors="coerce")
    ts["resp_rate"] = pd.to_numeric(ts["resp_rate"], errors="coerce")
    ts["temperature"] = pd.to_numeric(ts["temperature"], errors="coerce")

    agg = ts.groupby(id_col).agg(
        hours_observed=("hr", "nunique"),
        hr_mean=("heart_rate", "mean"),
        rr_mean=("resp_rate", "mean"),
        temp_mean=("temperature", "mean"),
        map_mean=("map_proxy", "mean"),
        sbp_mean=("sbp_proxy", "mean"),
    )

    join = df.copy()
    join[id_col] = join[id_col].astype(str)
    merged = join[[id_col] + ([ "age" ] if "age" in join.columns else [])].merge(
        agg,
        on=id_col,
        how="left",
    )

    age = _safe_numeric(merged, "age")
    fever = _score_high(merged["temp_mean"], 37.8, 39.5)
    hypothermia = _score_low(merged["temp_mean"], 36.5, 35.0)
    tachy = _score_high(merged["hr_mean"], 95.0, 130.0)
    brady = _score_low(merged["hr_mean"], 70.0, 45.0)
    tachypnea = _score_high(merged["rr_mean"], 20.0, 35.0)
    bradypnea = _score_low(merged["rr_mean"], 14.0, 8.0)
    hypotension = _max_of(
        _score_low(merged["map_mean"], 70.0, 55.0),
        _score_low(merged["sbp_mean"], 100.0, 80.0),
    )
    hypertension = _max_of(
        _score_high(merged["map_mean"], 85.0, 100.0),
        _score_high(merged["sbp_mean"], 140.0, 175.0),
    )
    younger = _score_low(age, 55.0, 30.0)
    older = _score_high(age, 65.0, 85.0)

    score_a = _weighted_average([fever, tachy, tachypnea, hypotension, younger], [0.22, 0.22, 0.22, 0.24, 0.10])
    score_b = _weighted_average([fever, tachy, tachypnea, hypertension, older], [0.22, 0.18, 0.18, 0.22, 0.20])
    score_c = _weighted_average(
        [
            _score_band(merged["temp_mean"], 36.2, 37.8),
            _score_band(merged["hr_mean"], 60.0, 95.0),
            _score_band(merged["rr_mean"], 12.0, 22.0),
            _score_band(merged["map_mean"].fillna(merged["sbp_mean"]), 65.0, 95.0),
        ],
        [0.25, 0.25, 0.25, 0.25],
    )
    score_d = _weighted_average([hypothermia, brady, bradypnea, hypotension, older], [0.22, 0.18, 0.18, 0.24, 0.18])

    score_frame = pd.DataFrame(
        {
            "group-a": score_a,
            "group-b": score_b,
            "group-c": score_c,
            "group-d": score_d,
        },
        index=df.index,
    )
    proxy_mask = (
        (pd.to_numeric(merged["hours_observed"], errors="coerce").fillna(0) >= 4)
        & _availability_mask(
            [merged["temp_mean"], merged["hr_mean"], merged["rr_mean"], merged["map_mean"], merged["sbp_mean"]],
            minimum_present=3,
        ).astype(bool)
    ).astype(np.float32)

    proxy_label = _label_from_scores(
        score_frame,
        threshold=0.40,
        class_order=["group-a", "group-b", "group-c", "group-d"],
    )

    out["proxy_trajectory_phenotype"] = proxy_label
    out["proxy_trajectory_phenotype_label"] = proxy_label.map(TRAJECTORY_PROXY_CLASS_TO_ID).astype(np.int64)
    out["mask_proxy_trajectory_phenotype_label"] = proxy_mask
    out["score_trajectory_a"] = score_a.fillna(0.0).astype(np.float32)
    out["score_trajectory_b"] = score_b.fillna(0.0).astype(np.float32)
    out["score_trajectory_c"] = score_c.fillna(0.0).astype(np.float32)
    out["score_trajectory_d"] = score_d.fillna(0.0).astype(np.float32)
    for suffix in ("a", "b", "c", "d"):
        out[f"mask_score_trajectory_{suffix}"] = proxy_mask
    return out


def _build_fluid_targets(df: pd.DataFrame) -> pd.DataFrame:
    lactate = _safe_numeric(df, "fd_lactate_max")
    sbp = _safe_numeric(df, "fd_sbp_min")
    mbp = _safe_numeric(df, "fd_mbp_min")
    cardio = _safe_numeric(df, "sofa_cardio")
    renal = _safe_numeric(df, "sofa_renal")
    resp = _safe_numeric(df, "sofa_resp")
    vent = _safe_binary(df, "mech_vent_first24h")

    hypoperfusion = _weighted_average(
        [
            _score_high(lactate, 2.0, 4.5),
            _max_of(_score_low(sbp, 100.0, 80.0), _score_low(mbp, 70.0, 55.0)),
            1.0 - _score_high(cardio, 2.0, 4.0),
        ],
        [0.40, 0.40, 0.20],
    )
    restrictive = _weighted_average(
        [
            _weighted_average([_score_high(cardio, 2.0, 4.0), _score_high(renal, 1.0, 4.0)], [0.6, 0.4]),
            _weighted_average([_score_high(resp, 2.0, 4.0), vent], [0.7, 0.3]),
        ],
        [0.55, 0.45],
    )
    proxy_mask = _availability_mask([lactate, sbp, mbp, cardio, renal, resp], minimum_present=3)

    proxy_label = pd.Series("Unclassified", index=df.index, dtype="object")
    restrictive_like = (restrictive >= 0.55) & (restrictive >= (hypoperfusion + 0.05))
    resuscitation_like = (hypoperfusion >= 0.55) & ~restrictive_like
    proxy_label.loc[restrictive_like] = "restrictive-fluid-benefit-like"
    proxy_label.loc[resuscitation_like] = "resuscitation-fluid-benefit-like"

    out = pd.DataFrame(index=df.index)
    out["proxy_fluid_strategy"] = proxy_label
    out["proxy_fluid_strategy_label"] = proxy_label.map(FLUID_PROXY_CLASS_TO_ID).astype(np.int64)
    out["mask_proxy_fluid_strategy_label"] = proxy_mask
    out["score_restrictive_fluid_benefit"] = restrictive.fillna(0.0).astype(np.float32)
    out["mask_score_restrictive_fluid_benefit"] = proxy_mask
    out["score_resuscitation_fluid_benefit"] = hypoperfusion.fillna(0.0).astype(np.float32)
    out["mask_score_resuscitation_fluid_benefit"] = proxy_mask
    return out


def _legacy_alias_columns(df: pd.DataFrame) -> None:
    df["immune_subtype"] = df["proxy_immune_state"].map(LEGACY_IMMUNE_ALIASES)
    df["immune_subtype_label"] = df["proxy_immune_state_label"]
    df["organ_subtype"] = df["proxy_clinical_phenotype"]
    df["organ_subtype_label"] = df["proxy_clinical_phenotype_label"]
    df["fluid_benefit_proxy"] = df["proxy_fluid_strategy"].map(LEGACY_FLUID_ALIASES)
    df["fluid_benefit_label"] = df["proxy_fluid_strategy_label"]


def _build_multitask_bundle(df: pd.DataFrame) -> dict:
    classification_labels = np.column_stack(
        [df[task["label_col"]].to_numpy(dtype=np.int64) for task in CLASSIFICATION_TASKS]
    )
    classification_masks = np.column_stack(
        [df[task["mask_col"]].to_numpy(dtype=np.float32) for task in CLASSIFICATION_TASKS]
    )
    regression_targets = np.column_stack(
        [df[target_col].to_numpy(dtype=np.float32) for target_col, _ in REGRESSION_TASKS]
    )
    regression_masks = np.column_stack(
        [df[mask_col].to_numpy(dtype=np.float32) for _, mask_col in REGRESSION_TASKS]
    )
    schema = {
        "schema_version": "2.0.0",
        "classification_tasks": CLASSIFICATION_TASKS,
        "regression_tasks": [
            {"name": target_col, "target_col": target_col, "mask_col": mask_col}
            for target_col, mask_col in REGRESSION_TASKS
        ],
        "legacy_aliases": {
            "immune_subtype": "proxy_immune_state",
            "organ_subtype": "proxy_clinical_phenotype",
            "fluid_benefit_proxy": "proxy_fluid_strategy",
        },
    }
    return {
        "classification_labels": classification_labels,
        "classification_masks": classification_masks,
        "classification_task_names": np.asarray([task["name"] for task in CLASSIFICATION_TASKS], dtype=object),
        "classification_num_classes": np.asarray([len(task["classes"]) for task in CLASSIFICATION_TASKS], dtype=np.int64),
        "regression_targets": regression_targets,
        "regression_masks": regression_masks,
        "regression_task_names": np.asarray([target_col for target_col, _ in REGRESSION_TASKS], dtype=object),
        "schema": schema,
    }


def _save_multitask_bundle(bundle: dict, output_dir: Path) -> dict[str, str]:
    npz_path = output_dir / "sepsis_multitask_targets.npz"
    json_path = output_dir / "sepsis_multitask_schema.json"
    np.savez_compressed(
        npz_path,
        classification_labels=bundle["classification_labels"],
        classification_masks=bundle["classification_masks"],
        classification_task_names=bundle["classification_task_names"],
        classification_num_classes=bundle["classification_num_classes"],
        regression_targets=bundle["regression_targets"],
        regression_masks=bundle["regression_masks"],
        regression_task_names=bundle["regression_task_names"],
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle["schema"], f, indent=2)
    return {
        "targets_npz": str(npz_path),
        "schema_json": str(json_path),
    }


def _print_summary(df: pd.DataFrame) -> None:
    outcome_col = "mortality_28d" if "mortality_28d" in df.columns else None
    print("\n[Proxy Immune State]")
    print(df["proxy_immune_state"].value_counts(dropna=False).to_string())
    if outcome_col is not None:
        print("  Mortality by proxy immune state:")
        print(df.groupby("proxy_immune_state")[outcome_col].mean().round(3).to_string())

    print("\n[Proxy Clinical Phenotype]")
    print(df["proxy_clinical_phenotype"].value_counts(dropna=False).to_string())
    if outcome_col is not None:
        print("  Mortality by proxy clinical phenotype:")
        print(df.groupby("proxy_clinical_phenotype")[outcome_col].mean().round(3).to_string())

    print("\n[Proxy Trajectory Phenotype]")
    print(df["proxy_trajectory_phenotype"].value_counts(dropna=False).to_string())
    if outcome_col is not None and float(df["mask_proxy_trajectory_phenotype_label"].sum()) > 0:
        sub = df[df["mask_proxy_trajectory_phenotype_label"] > 0]
        print("  Mortality by proxy trajectory phenotype:")
        print(sub.groupby("proxy_trajectory_phenotype")[outcome_col].mean().round(3).to_string())

    print("\n[Proxy Fluid Strategy]")
    print(df["proxy_fluid_strategy"].value_counts(dropna=False).to_string())
    if outcome_col is not None:
        print("  Mortality by proxy fluid strategy:")
        print(df.groupby("proxy_fluid_strategy")[outcome_col].mean().round(3).to_string())

    print("\n[Gold Label Availability]")
    print(f"  gold_mals_label available: {int(df['mask_gold_mals_label'].sum())} / {len(df)}")
    print(f"  gold_immunoparalysis_label available: {int(df['mask_gold_immunoparalysis_label'].sum())} / {len(df)}")

    if outcome_col is not None:
        print("\n[Proxy Immune x Clinical Cross-tab (mortality rate)]")
        ctab = pd.crosstab(
            df["proxy_immune_state"],
            df["proxy_clinical_phenotype"],
            values=df[outcome_col],
            aggfunc="mean",
        ).round(3)
        print(ctab.to_string())


def build_subtype_labels(
    static_path: Path,
    timeseries_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_format: str = "parquet",
) -> pd.DataFrame:
    """
    Read enhanced static data, build split multi-task subtype labels, and export.

    Returns the enriched static DataFrame with gold/proxy/score/mask columns.
    """
    print("=" * 72)
    print("Generating Split Sepsis Subtype Labels")
    print(f"Static input: {static_path}")
    if timeseries_path is not None:
        print(f"Time-series input: {timeseries_path}")
    print("=" * 72)

    df = _read_table(Path(static_path))
    print(f"  Loaded {len(df)} patients")

    immune_df = _build_immune_targets(df)
    clinical_df = _build_clinical_targets(df)
    trajectory_df = _build_trajectory_targets(df, timeseries_path)
    fluid_df = _build_fluid_targets(df)

    df = pd.concat([df, immune_df, clinical_df, trajectory_df, fluid_df], axis=1)
    _legacy_alias_columns(df)
    multitask_bundle = _build_multitask_bundle(df)

    _print_summary(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "parquet":
        out_path = output_dir / "patient_static_with_subtypes.parquet"
        df.to_parquet(out_path, index=False)
    else:
        out_path = output_dir / "patient_static_with_subtypes.csv"
        df.to_csv(out_path, index=False)

    bundle_paths = _save_multitask_bundle(multitask_bundle, output_dir)

    print(f"\n  Saved subtype table: {out_path}")
    print(f"  Saved multitask targets: {bundle_paths['targets_npz']}")
    print(f"  Saved task schema: {bundle_paths['schema_json']}")
    print("=" * 72)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate split sepsis subtype labels")
    parser.add_argument("--static-path", type=str, default=str(DEFAULT_OUTPUT_DIR / "patient_static_enhanced.parquet"))
    parser.add_argument("--timeseries-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--format", type=str, default="parquet", choices=["csv", "parquet"])
    args = parser.parse_args()

    build_subtype_labels(
        static_path=Path(args.static_path),
        timeseries_path=None if args.timeseries_path is None else Path(args.timeseries_path),
        output_dir=Path(args.output_dir),
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
