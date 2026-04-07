#!/usr/bin/env python3
"""
s6/api_service.py - FastAPI service for masked-NPZ S6 subtype inference.

Endpoints:
  GET  /health      - runtime and model status
  GET  /metadata    - frontend-facing subtype family metadata
  POST /predict     - generic named multitask predictions
  POST /recommend   - multitask predictions plus treatment guidance
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from uvicorn import run as uvicorn_run

from s6.masked_npz_runtime import SepsisSubtypeInferenceEngine
from s6.treatment_recommender import SubtypeTreatmentRecommender


app = FastAPI(title="S6 Sepsis Subtype Diagnosis API", version="0.3.0")

ENGINE: SepsisSubtypeInferenceEngine | None = None
RECOMMENDER: SubtypeTreatmentRecommender | None = None


class PredictRequest(BaseModel):
    time_series: list[list[list[float]]] = Field(
        ...,
        description="Batch of patient time-series with shape (B, T, F). F must equal n_cont_features + n_treat_features.",
    )
    mortality_threshold: float | None = Field(
        None,
        description="Optional override for mortality decision threshold. Defaults to the selected threshold from report.",
    )


class RecommendRequest(PredictRequest):
    probability_threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum subtype probability to surface as active recommendation.",
    )
    top_k: int = Field(2, ge=1, le=5)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": ENGINE is not None,
        "recommender_loaded": RECOMMENDER is not None,
        "model": None if ENGINE is None else ENGINE.model_metadata(),
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    if ENGINE is None:
        raise RuntimeError("Model not loaded")
    return ENGINE.metadata()


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if ENGINE is None:
        raise RuntimeError("Model not loaded")
    batch = np.asarray(req.time_series, dtype=np.float32)
    return ENGINE.predict_batch(batch, mortality_threshold=req.mortality_threshold)


@app.post("/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    if ENGINE is None:
        raise RuntimeError("Model not loaded")
    if RECOMMENDER is None:
        raise RuntimeError("Recommender not loaded")
    batch = np.asarray(req.time_series, dtype=np.float32)
    prediction_payload = ENGINE.predict_batch(batch, mortality_threshold=req.mortality_threshold)
    recommendations = [
        RECOMMENDER.recommend_from_prediction(
            prediction,
            prob_threshold=req.probability_threshold,
            top_k=req.top_k,
        )
        for prediction in prediction_payload["predictions"]
    ]
    return {
        "model": prediction_payload["model"],
        "metadata": ENGINE.metadata(),
        "predictions": prediction_payload["predictions"],
        "recommendations": recommendations,
    }


def load_artifacts(
    *,
    model_dir: Path,
    schema_path: Path | None = None,
    rules_path: Path | None = None,
    device: str = "cpu",
) -> None:
    global ENGINE, RECOMMENDER
    ENGINE = SepsisSubtypeInferenceEngine(model_dir=model_dir, schema_path=schema_path, device=device)
    RECOMMENDER = SubtypeTreatmentRecommender(rules_path=rules_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S6 masked-NPZ FastAPI inference service")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing multitask_student.pt and normalization arrays")
    parser.add_argument("--schema-path", type=str, default=None, help="Optional path to sepsis_multitask_schema.json")
    parser.add_argument("--rules-path", type=str, default=None, help="Optional path to treatment_rules.json")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_artifacts(
        model_dir=Path(args.model_dir),
        schema_path=None if args.schema_path is None else Path(args.schema_path),
        rules_path=None if args.rules_path is None else Path(args.rules_path),
        device=args.device,
    )
    uvicorn_run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

