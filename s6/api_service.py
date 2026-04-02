#!/usr/bin/env python3
"""
s6/api_service.py - FastAPI service for S6 multi-task diagnosis and treatment recommendations.

Endpoints:
  POST /health          - Health check + model metadata
  POST /predict         - Run multi-task inference on a single patient
  POST /recommend       - Run inference + generate treatment recommendations
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel, Field
from uvicorn import run as uvicorn_run

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s6.multitask_model import MultitaskRealtimeStudentClassifier
from s6.treatment_recommender import SubtypeTreatmentRecommender


app = FastAPI(title="S6 Sepsis Subtype Diagnosis API", version="0.2.0")

# Global state populated on startup
MODEL: MultitaskRealtimeStudentClassifier | None = None
RECOMMENDER: SubtypeTreatmentRecommender | None = None
FEAT_MEAN: np.ndarray | None = None
FEAT_STD: np.ndarray | None = None
CONFIG: dict[str, Any] = {}
DEVICE: str = "cpu"


class PredictRequest(BaseModel):
    time_series: list[list[list[float]]] = Field(
        ..., description="Batch of patient time-series with shape (B, T, F). F must match model input dims (43 continuous + 1 treatment)."
    )


class PredictResponse(BaseModel):
    mortality_prob: list[float]
    immune_prob: list[list[float]]
    organ_prob: list[list[float]]
    fluid_prob: list[list[float]]
    model_cfg: dict


class RecommendRequest(BaseModel):
    time_series: list[list[list[float]]] = Field(
        ..., description="Batch of patient time-series with shape (B, T, F)."
    )
    probability_threshold: float = Field(0.3, ge=0.0, le=1.0)


class RecommendResponse(BaseModel):
    predictions: PredictResponse
    recommendations: list[dict]


@app.on_event("startup")
def startup_event() -> None:
    global MODEL, RECOMMENDER, FEAT_MEAN, FEAT_STD, CONFIG, DEVICE
    # This is a stub; actual loading happens in load_artifacts() called before uvicorn starts
    pass


def load_artifacts(*, model_dir: Path, rules_path: Path | None = None, device: str = "cpu") -> None:
    global MODEL, RECOMMENDER, FEAT_MEAN, FEAT_STD, CONFIG, DEVICE
    DEVICE = device
    model_dir = Path(model_dir)
    ckpt = torch.load(model_dir / "multitask_student.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    CONFIG = dict(cfg)

    MODEL = MultitaskRealtimeStudentClassifier(
        n_cont_features=int(cfg["n_cont_features"]),
        n_treat_features=int(cfg["n_treat_features"]),
        note_dim=int(cfg.get("note_dim", 0)),
        student_arch=cfg.get("student_arch", "transformer"),
        student_d_model=int(cfg["student_d_model"]),
        teacher_dim=int(cfg.get("teacher_dim", 0)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_layers=int(cfg.get("n_layers", 1)),
        d_ff=int(cfg.get("d_ff", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_seq_len=int(cfg["max_seq_len"]),
        treatment_layers=int(cfg.get("treatment_layers", 1)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        tcn_kernel_size=int(cfg.get("tcn_kernel_size", 3)),
        tcn_dilations=tuple(int(v) for v in cfg.get("tcn_dilations", [1, 2, 4, 8])),
    ).to(DEVICE)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()

    FEAT_MEAN = np.load(model_dir / "feat_mean.npy")
    FEAT_STD = np.load(model_dir / "feat_std.npy")

    if rules_path and Path(rules_path).exists():
        RECOMMENDER = SubtypeTreatmentRecommender(rules_path=Path(rules_path))


def _preprocess(batch: np.ndarray) -> dict[str, torch.Tensor]:
    """Normalize and split continuous/treatment features."""
    assert FEAT_MEAN is not None and FEAT_STD is not None
    n_cont = int(CONFIG["n_cont_features"])
    n_treat = int(CONFIG["n_treat_features"])
    total_f = n_cont + n_treat
    if batch.shape[-1] != total_f:
        raise ValueError(f"Expected last dim={total_f} ({n_cont} cont + {n_treat} treat), got {batch.shape[-1]}")
    continuous = batch[:, :, :n_cont].astype(np.float32)
    treatments = batch[:, :, n_cont:n_cont + n_treat].astype(np.float32)
    mask_cont = np.isfinite(continuous).astype(np.float32)
    mask_treat = np.isfinite(treatments).astype(np.float32)
    continuous = np.nan_to_num(continuous, nan=0.0)
    treatments = np.nan_to_num(treatments, nan=0.0)
    continuous = (continuous - FEAT_MEAN[None, None, :]) / FEAT_STD[None, None, :]
    return {
        "x": torch.from_numpy(continuous).to(DEVICE),
        "mask": torch.from_numpy(mask_cont).to(DEVICE),
        "treatments": torch.from_numpy(treatments).to(DEVICE),
        "treatment_mask": torch.from_numpy(mask_treat).to(DEVICE),
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "recommender_loaded": RECOMMENDER is not None,
        "device": DEVICE,
        "model_config": CONFIG,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise RuntimeError("Model not loaded")
    arr = np.array(req.time_series, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"time_series must be 3D (B, T, F), got shape {arr.shape}")
    inputs = _preprocess(arr)
    with torch.no_grad():
        out = MODEL(**inputs)
    mortality_prob = torch.sigmoid(out["logits_mortality"]).cpu().tolist()
    immune_prob = F.softmax(out["logits_immune"], dim=-1).cpu().tolist()
    organ_prob = F.softmax(out["logits_organ"], dim=-1).cpu().tolist()
    fluid_prob = F.softmax(out["logits_fluid"], dim=-1).cpu().tolist()
    return PredictResponse(
        mortality_prob=mortality_prob,
        immune_prob=immune_prob,
        organ_prob=organ_prob,
        fluid_prob=fluid_prob,
        model_cfg=CONFIG,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    pred = predict(PredictRequest(time_series=req.time_series))
    recs: list[dict] = []
    if RECOMMENDER is not None:
        for i in range(len(pred.mortality_prob)):
            rec = RECOMMENDER.recommend(
                immune_probs=np.array(pred.immune_prob[i]),
                organ_probs=np.array(pred.organ_prob[i]),
                fluid_probs=np.array(pred.fluid_prob[i]),
                mortality_prob=pred.mortality_prob[i],
                prob_threshold=req.probability_threshold,
            )
            recs.append(rec)
    else:
        recs = [{"error": "Recommender not loaded"}] * len(pred.mortality_prob)
    return RecommendResponse(predictions=pred, recommendations=recs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S6 FastAPI inference service")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing multitask_student.pt, feat_mean.npy, feat_std.npy")
    parser.add_argument("--rules-path", type=str, default=None, help="Path to treatment_rules.json")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_artifacts(model_dir=Path(args.model_dir), rules_path=args.rules_path, device=args.device)
    uvicorn_run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
