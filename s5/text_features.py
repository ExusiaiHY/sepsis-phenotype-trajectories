"""
text_features.py - Lightweight clinical note embedding utilities for Stage 5.

The repo does not depend on heavyweight LLM inference for bedside deployment.
Instead, notes are embedded with a deterministic sparse pipeline:
  text -> hashing vectorizer -> truncated SVD -> hourly tensor

This is intentionally small enough to ship beside the structured real-time
pipeline while still allowing multimodal fusion experiments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer


def build_hourly_note_embeddings(
    *,
    notes: pd.DataFrame,
    patient_ids: list[int] | list[str],
    n_hours: int,
    patient_col: str,
    hour_col: str,
    text_col: str,
    n_features: int = 512,
    n_components: int = 16,
    output_path: Path | None = None,
) -> np.ndarray:
    """
    Convert note rows into an `(N, T, D)` hourly embedding tensor.

    Multiple notes in the same hour are averaged in the reduced space.
    """
    notes = notes.copy()
    notes = notes[[patient_col, hour_col, text_col]].dropna()
    notes[hour_col] = pd.to_numeric(notes[hour_col], errors="coerce")
    notes = notes.dropna(subset=[hour_col])
    notes[hour_col] = notes[hour_col].astype(int)
    notes = notes[(notes[hour_col] >= 0) & (notes[hour_col] < n_hours)]
    notes[text_col] = notes[text_col].astype(str).str.strip()
    notes = notes[notes[text_col] != ""]

    patient_ids = [str(pid) for pid in patient_ids]
    patient_index = {pid: idx for idx, pid in enumerate(patient_ids)}
    notes[patient_col] = notes[patient_col].astype(str)
    notes = notes[notes[patient_col].isin(patient_index)]

    tensor = np.zeros((len(patient_ids), n_hours, n_components), dtype=np.float32)
    if notes.empty:
        if output_path is not None:
            np.save(output_path, tensor)
        return tensor

    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        analyzer="word",
        ngram_range=(1, 2),
    )
    x_sparse = vectorizer.transform(notes[text_col].tolist())
    n_components = min(n_components, x_sparse.shape[1] - 1, max(1, x_sparse.shape[0] - 1))
    reducer = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = reducer.fit_transform(x_sparse).astype(np.float32, copy=False)
    if reduced.shape[1] < tensor.shape[2]:
        padded = np.zeros((reduced.shape[0], tensor.shape[2]), dtype=np.float32)
        padded[:, : reduced.shape[1]] = reduced
        reduced = padded

    accum = np.zeros_like(tensor)
    counts = np.zeros((len(patient_ids), n_hours, 1), dtype=np.float32)
    for row_idx, row in enumerate(notes.itertuples(index=False)):
        pid = str(getattr(row, patient_col))
        hour = int(getattr(row, hour_col))
        p_idx = patient_index[pid]
        accum[p_idx, hour, :] += reduced[row_idx]
        counts[p_idx, hour, 0] += 1.0

    valid = counts > 0
    tensor[valid.squeeze(-1)] = (accum / np.maximum(counts, 1.0))[valid.squeeze(-1)]
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, tensor)
    return tensor


def build_eicu_note_embedding_tensor(
    *,
    raw_dir: Path,
    cohort_static_path: Path,
    output_path: Path,
    n_hours: int = 48,
    n_features: int = 512,
    n_components: int = 16,
) -> dict:
    """Read eICU `note.csv*` and build hourly note embeddings for the cohort."""
    raw_dir = Path(raw_dir)
    cohort_static_path = Path(cohort_static_path)
    output_path = Path(output_path)

    note_path = _first_existing_optional(
        raw_dir / "note.csv.gz",
        raw_dir / "note.csv",
    )
    cohort = pd.read_csv(cohort_static_path)
    patient_ids = cohort["stay_id"].tolist() if "stay_id" in cohort.columns else cohort["patient_id"].tolist()

    if note_path is None:
        tensor = np.zeros((len(patient_ids), n_hours, n_components), dtype=np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, tensor)
        report = {
            "source": "eicu",
            "note_path": None,
            "n_patients": int(len(patient_ids)),
            "n_hours": n_hours,
            "n_components": n_components,
            "n_note_rows": 0,
            "output_path": str(output_path),
        }
        with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    notes = pd.read_csv(note_path, low_memory=False)
    notes.columns = [str(col).strip().lower() for col in notes.columns]
    notes["text"] = (
        notes.get("notetype", "").fillna("").astype(str)
        + " | "
        + notes.get("notepath", "").fillna("").astype(str)
        + " | "
        + notes.get("notevalue", "").fillna("").astype(str)
        + " | "
        + notes.get("notetext", "").fillna("").astype(str)
    )
    notes["hour"] = pd.to_numeric(notes.get("noteoffset"), errors="coerce") / 60.0
    notes["hour"] = np.floor(notes["hour"]).astype("Int64")
    tensor = build_hourly_note_embeddings(
        notes=notes,
        patient_ids=patient_ids,
        n_hours=n_hours,
        patient_col="patientunitstayid",
        hour_col="hour",
        text_col="text",
        n_features=n_features,
        n_components=n_components,
        output_path=output_path,
    )

    report = {
        "source": "eicu",
        "note_path": str(note_path),
        "n_patients": int(len(patient_ids)),
        "n_hours": n_hours,
        "n_components": int(tensor.shape[-1]),
        "n_note_rows": int(len(notes)),
        "nonzero_hour_fraction": round(float((np.abs(tensor).sum(axis=-1) > 0).mean()), 4),
        "output_path": str(output_path),
    }
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def _first_existing_optional(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
