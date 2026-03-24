"""
realtime_model.py - Lightweight Stage 5 student model and streaming monitor.

The student model reuses the Stage 4 treatment-aware encoder shape but shrinks
its hidden size and transformer depth for bedside deployment. Knowledge
distillation is done against teacher embeddings from the larger model.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from s15.classification_eval import _classification_metrics, _select_threshold
from s4.treatment_aware_model import TreatmentAwareEncoder, _calibration_report, _load_or_build_splits


class RealtimeStudentDataset(Dataset):
    """Dataset for student distillation."""

    def __init__(
        self,
        *,
        continuous: np.ndarray,
        masks_continuous: np.ndarray,
        treatments: np.ndarray,
        masks_treatments: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        teacher_embeddings: np.ndarray | None = None,
        note_embeddings: np.ndarray | None = None,
    ):
        self.continuous = continuous[indices].astype(np.float32, copy=False)
        self.masks_continuous = masks_continuous[indices].astype(np.float32, copy=False)
        self.treatments = treatments[indices].astype(np.float32, copy=False)
        self.masks_treatments = masks_treatments[indices].astype(np.float32, copy=False)
        self.labels = labels[indices].astype(np.float32, copy=False)
        self.teacher_embeddings = None if teacher_embeddings is None else teacher_embeddings[indices].astype(np.float32, copy=False)
        self.note_embeddings = None
        if note_embeddings is not None:
            notes = note_embeddings[indices].astype(np.float32, copy=False)
            if notes.ndim == 2:
                notes = np.repeat(notes[:, None, :], self.continuous.shape[1], axis=1)
            self.note_embeddings = notes

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        out = {
            "x": torch.from_numpy(self.continuous[idx]),
            "mask": torch.from_numpy(self.masks_continuous[idx]),
            "treatments": torch.from_numpy(self.treatments[idx]),
            "treatment_mask": torch.from_numpy(self.masks_treatments[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        if self.teacher_embeddings is not None:
            out["teacher_embedding"] = torch.from_numpy(self.teacher_embeddings[idx])
        if self.note_embeddings is not None:
            out["notes"] = torch.from_numpy(self.note_embeddings[idx])
        return out


class RealtimeStudentClassifier(nn.Module):
    """Smaller treatment-aware student model with optional teacher-space projection."""

    def __init__(
        self,
        *,
        n_cont_features: int,
        n_treat_features: int,
        note_dim: int = 0,
        student_d_model: int = 64,
        teacher_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 48,
        treatment_layers: int = 1,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TreatmentAwareEncoder(
            n_cont_features=n_cont_features,
            n_treat_features=n_treat_features,
            note_dim=note_dim,
            d_model=student_d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            treatment_layers=treatment_layers,
        )
        self.teacher_projection = nn.Linear(student_d_model, teacher_dim) if teacher_dim > 0 else None
        self.head = nn.Sequential(
            nn.Linear(student_d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        treatments: torch.Tensor,
        treatment_mask: torch.Tensor,
        note_embeddings: torch.Tensor | None = None,
    ) -> dict:
        embedding = self.encoder(
            x,
            mask,
            treatments,
            treatment_mask,
            note_embeddings=note_embeddings,
            return_sequence=False,
        )
        logits = self.head(embedding).squeeze(-1)
        teacher_embedding = self.teacher_projection(embedding) if self.teacher_projection is not None else None
        return {
            "logits": logits,
            "student_embedding": embedding,
            "teacher_embedding": teacher_embedding,
        }


def distill_realtime_student(
    *,
    s0_dir: Path,
    treatment_dir: Path,
    output_dir: Path,
    teacher_embeddings_path: Path | None = None,
    note_embeddings_path: Path | None = None,
    splits_path: Path | None = None,
    label_col: str = "mortality_inhospital",
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    patience: int = 4,
    bce_weight: float = 1.0,
    distill_weight: float = 1.0,
    threshold_metric: str = "balanced_accuracy",
    seed: int = 42,
    device: str = "cpu",
    student_d_model: int = 64,
    teacher_dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 1,
    d_ff: int = 128,
    dropout: float = 0.1,
    treatment_layers: int = 1,
    head_hidden_dim: int = 64,
    head_dropout: float = 0.1,
) -> dict:
    """Train and evaluate a distilled student model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    continuous = np.load(Path(s0_dir) / "processed" / "continuous.npy")
    masks_cont = np.load(Path(s0_dir) / "processed" / "masks_continuous.npy")
    treatments = np.load(Path(treatment_dir) / "treatments.npy")
    masks_treat = np.load(Path(treatment_dir) / "masks_treatments.npy")
    s0_static = pd.read_csv(Path(s0_dir) / "static.csv")
    treatment_static_path = Path(treatment_dir) / "cohort_static.csv"
    if "patient_id" not in s0_static.columns:
        s0_static = s0_static.copy()
        s0_static["patient_id"] = np.arange(len(s0_static)).astype(str)
    if treatment_static_path.exists():
        treatment_static = pd.read_csv(treatment_static_path)
    else:
        treatment_static = s0_static[["patient_id"]].copy()
    teacher_embeddings = np.load(teacher_embeddings_path) if teacher_embeddings_path is not None else None
    notes = np.load(note_embeddings_path) if note_embeddings_path is not None else None

    aligned = _align_stage5_inputs(
        s0_static=s0_static,
        treatment_static=treatment_static,
        continuous=continuous,
        masks_continuous=masks_cont,
        treatments=treatments,
        masks_treatments=masks_treat,
        teacher_embeddings=teacher_embeddings,
        note_embeddings=notes,
    )
    continuous = aligned["continuous"]
    masks_cont = aligned["masks_continuous"]
    treatments = aligned["treatments"]
    masks_treat = aligned["masks_treatments"]
    teacher_embeddings = aligned["teacher_embeddings"]
    notes = aligned["note_embeddings"]
    label_frame = aligned["s0_static"] if label_col in aligned["s0_static"].columns else aligned["treatment_static"]
    labels = label_frame[label_col].fillna(0).astype(int).to_numpy()
    n_samples = len(labels)
    if teacher_embeddings is not None:
        teacher_dim = teacher_embeddings.shape[-1]

    splits = _load_or_build_splits(
        n_samples=n_samples,
        labels=labels,
        splits_path=splits_path or (Path(s0_dir) / "splits.json"),
        seed=seed,
    )

    train_loader = DataLoader(
        RealtimeStudentDataset(
            continuous=continuous,
            masks_continuous=masks_cont,
            treatments=treatments,
            masks_treatments=masks_treat,
            labels=labels,
            indices=splits["train"],
            teacher_embeddings=teacher_embeddings,
            note_embeddings=notes,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        RealtimeStudentDataset(
            continuous=continuous,
            masks_continuous=masks_cont,
            treatments=treatments,
            masks_treatments=masks_treat,
            labels=labels,
            indices=splits["val"],
            teacher_embeddings=teacher_embeddings,
            note_embeddings=notes,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        RealtimeStudentDataset(
            continuous=continuous,
            masks_continuous=masks_cont,
            treatments=treatments,
            masks_treatments=masks_treat,
            labels=labels,
            indices=splits["test"],
            teacher_embeddings=teacher_embeddings,
            note_embeddings=notes,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    note_dim = 0 if notes is None else (notes.shape[-1] if notes.ndim == 3 else notes.shape[-1])
    model = RealtimeStudentClassifier(
        n_cont_features=continuous.shape[-1],
        n_treat_features=treatments.shape[-1],
        note_dim=note_dim,
        student_d_model=student_d_model,
        teacher_dim=teacher_dim if teacher_embeddings is not None else 0,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=continuous.shape[1],
        treatment_layers=treatment_layers,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    best_state = None
    best_auc = -np.inf
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                batch["x"].to(device),
                batch["mask"].to(device),
                batch["treatments"].to(device),
                batch["treatment_mask"].to(device),
                note_embeddings=batch.get("notes").to(device) if "notes" in batch else None,
            )
            y = batch["y"].to(device)
            loss = bce_weight * bce(out["logits"], y)
            if "teacher_embedding" in batch and out["teacher_embedding"] is not None:
                loss = loss + distill_weight * mse(out["teacher_embedding"], batch["teacher_embedding"].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_probs, val_y = _student_probs(model, val_loader, device)
        val_auc = 0.5 if len(np.unique(val_y)) < 2 else float(roc_auc_score(val_y, val_probs))
        history.append(
            {
                "epoch": epoch,
                "time_s": round(time.time() - start_time, 2),
                "train_loss": round(float(np.mean(losses)), 4) if losses else 0.0,
                "val_auroc": round(float(val_auc), 4),
            }
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": _student_config_dict(
                        n_cont_features=continuous.shape[-1],
                        n_treat_features=treatments.shape[-1],
                        note_dim=note_dim,
                        student_d_model=student_d_model,
                        teacher_dim=teacher_dim if teacher_embeddings is not None else 0,
                        n_heads=n_heads,
                        n_layers=n_layers,
                        d_ff=d_ff,
                        dropout=dropout,
                        max_seq_len=continuous.shape[1],
                        treatment_layers=treatment_layers,
                        head_hidden_dim=head_hidden_dim,
                        head_dropout=head_dropout,
                    ),
                },
                ckpt_dir / "student_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    train_probs, train_y = _student_probs(model, train_loader, device)
    val_probs, val_y = _student_probs(model, val_loader, device)
    test_probs, test_y = _student_probs(model, test_loader, device)
    threshold, threshold_search = _select_threshold(val_y.astype(int), val_probs, metric_name=threshold_metric)

    sample_batch = {
        "x": torch.from_numpy(continuous[:1]).float(),
        "mask": torch.from_numpy(masks_cont[:1]).float(),
        "treatments": torch.from_numpy(treatments[:1]).float(),
        "treatment_mask": torch.from_numpy(masks_treat[:1]).float(),
        "notes": torch.from_numpy(notes[:1]).float() if notes is not None else None,
    }
    quantized_ok = True
    try:
        quantized = quantize_realtime_model(model)
        latency = estimate_cpu_latency_ms(quantized, sample_batch=sample_batch)
    except Exception:
        quantized_ok = False
        latency = estimate_cpu_latency_ms(model.cpu().eval(), sample_batch=sample_batch)

    report = {
        "model": _student_config_dict(
            n_cont_features=continuous.shape[-1],
            n_treat_features=treatments.shape[-1],
            note_dim=note_dim,
            student_d_model=student_d_model,
            teacher_dim=teacher_dim if teacher_embeddings is not None else 0,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=continuous.shape[1],
            treatment_layers=treatment_layers,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
        ),
        "training": {
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "epochs_trained": len(history),
            "lr": lr,
            "weight_decay": weight_decay,
            "bce_weight": bce_weight,
            "distill_weight": distill_weight,
            "threshold_metric": threshold_metric,
            "seed": seed,
            "device": device,
        },
        "history": history,
        "threshold_selection": {
            "metric": threshold_metric,
            "selected_threshold": round(float(threshold), 4),
            "search": threshold_search,
        },
        "splits": {
            "train": _classification_metrics(train_y.astype(int), train_probs, threshold),
            "val": _classification_metrics(val_y.astype(int), val_probs, threshold),
            "test": _classification_metrics(test_y.astype(int), test_probs, threshold),
        },
        "calibration": {
            "train": _calibration_report(train_y, train_probs),
            "val": _calibration_report(val_y, val_probs),
            "test": _calibration_report(test_y, test_probs),
        },
        "deployment": {
            "cpu_latency_ms_per_sample": round(float(latency), 3),
            "float_n_parameters": int(sum(p.numel() for p in model.parameters())),
            "dynamic_quantization_ok": quantized_ok,
        },
    }

    with open(output_dir / "realtime_student_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": report["model"],
            "threshold": threshold,
        },
        output_dir / "realtime_student.pt",
    )
    return report


def quantize_realtime_model(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization for CPU deployment."""
    model = copy.deepcopy(model).cpu().eval()
    try:
        if torch.backends.quantized.engine == "none":
            available = torch.backends.quantized.supported_engines
            if available:
                torch.backends.quantized.engine = available[0]
        return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    except Exception:
        return model


def estimate_cpu_latency_ms(model: nn.Module, sample_batch: dict, n_warmup: int = 5, n_runs: int = 20) -> float:
    """Measure per-sample CPU inference latency in milliseconds."""
    model = model.cpu().eval()
    inputs = {
        "x": sample_batch["x"].cpu(),
        "mask": sample_batch["mask"].cpu(),
        "treatments": sample_batch["treatments"].cpu(),
        "treatment_mask": sample_batch["treatment_mask"].cpu(),
    }
    if sample_batch.get("notes") is not None:
        inputs["note_embeddings"] = sample_batch["notes"].cpu()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(**inputs)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(**inputs)
        elapsed = time.perf_counter() - t0
    batch_size = int(inputs["x"].shape[0])
    return (elapsed / max(n_runs, 1)) * 1000.0 / max(batch_size, 1)


class RealtimePatientBuffer:
    """Fixed-length buffer for hourly bedside updates."""

    def __init__(
        self,
        *,
        seq_len: int,
        n_cont_features: int,
        n_treat_features: int,
        note_dim: int = 0,
    ):
        self.seq_len = seq_len
        self.values = np.zeros((seq_len, n_cont_features), dtype=np.float32)
        self.masks = np.zeros((seq_len, n_cont_features), dtype=np.float32)
        self.treatments = np.zeros((seq_len, n_treat_features), dtype=np.float32)
        self.treatment_mask = np.zeros((seq_len, n_treat_features), dtype=np.float32)
        self.notes = np.zeros((seq_len, note_dim), dtype=np.float32) if note_dim > 0 else None
        self.n_updates = 0

    def update(
        self,
        *,
        values: np.ndarray,
        mask: np.ndarray,
        treatments: np.ndarray,
        treatment_mask: np.ndarray,
        note_embedding: np.ndarray | None = None,
    ) -> None:
        self.values = np.roll(self.values, shift=-1, axis=0)
        self.masks = np.roll(self.masks, shift=-1, axis=0)
        self.treatments = np.roll(self.treatments, shift=-1, axis=0)
        self.treatment_mask = np.roll(self.treatment_mask, shift=-1, axis=0)
        self.values[-1] = np.asarray(values, dtype=np.float32)
        self.masks[-1] = np.asarray(mask, dtype=np.float32)
        self.treatments[-1] = np.asarray(treatments, dtype=np.float32)
        self.treatment_mask[-1] = np.asarray(treatment_mask, dtype=np.float32)
        if self.notes is not None:
            self.notes = np.roll(self.notes, shift=-1, axis=0)
            if note_embedding is None:
                self.notes[-1] = 0.0
            else:
                self.notes[-1] = np.asarray(note_embedding, dtype=np.float32)
        self.n_updates += 1

    @property
    def ready(self) -> bool:
        return self.n_updates >= self.seq_len


class RealtimePhenotypeMonitor:
    """Runtime wrapper for rolling bedside inference and phenotype assignment."""

    def __init__(
        self,
        *,
        model: nn.Module,
        threshold: float,
        phenotype_centroids: np.ndarray | None = None,
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.threshold = float(threshold)
        self.phenotype_centroids = phenotype_centroids
        self.device = device

    def predict(self, buffer: RealtimePatientBuffer) -> dict | None:
        if not buffer.ready:
            return None

        note_tensor = None
        if buffer.notes is not None:
            note_tensor = torch.from_numpy(buffer.notes[None, :, :]).to(self.device)
        with torch.no_grad():
            out = self.model(
                torch.from_numpy(buffer.values[None, :, :]).to(self.device),
                torch.from_numpy(buffer.masks[None, :, :]).to(self.device),
                torch.from_numpy(buffer.treatments[None, :, :]).to(self.device),
                torch.from_numpy(buffer.treatment_mask[None, :, :]).to(self.device),
                note_embeddings=note_tensor,
            )
            risk_prob = float(torch.sigmoid(out["logits"]).cpu().item())
            phenotype = None
            if self.phenotype_centroids is not None and out.get("teacher_embedding") is not None:
                emb = out["teacher_embedding"].cpu().numpy()[0]
                dists = np.linalg.norm(self.phenotype_centroids - emb[None, :], axis=1)
                phenotype = int(np.argmin(dists))
        return {
            "risk_probability": round(risk_prob, 4),
            "risk_alert": bool(risk_prob >= self.threshold),
            "phenotype": phenotype,
            "hours_seen": int(buffer.n_updates),
        }


def _align_stage5_inputs(
    *,
    s0_static: pd.DataFrame,
    treatment_static: pd.DataFrame,
    continuous: np.ndarray,
    masks_continuous: np.ndarray,
    treatments: np.ndarray,
    masks_treatments: np.ndarray,
    teacher_embeddings: np.ndarray | None = None,
    note_embeddings: np.ndarray | None = None,
) -> dict:
    s0_static = s0_static.copy().reset_index(drop=True)
    treatment_static = treatment_static.copy().reset_index(drop=True)
    if "patient_id" not in s0_static.columns:
        s0_static["patient_id"] = np.arange(len(s0_static)).astype(str)
    if "patient_id" not in treatment_static.columns:
        treatment_static["patient_id"] = np.arange(len(treatment_static)).astype(str)
    s0_static["patient_id"] = s0_static["patient_id"].astype(str)
    treatment_static["patient_id"] = treatment_static["patient_id"].astype(str)

    s0_index = {pid: idx for idx, pid in enumerate(s0_static["patient_id"].tolist())}
    s0_keep = []
    treat_keep = []
    for treat_idx, pid in enumerate(treatment_static["patient_id"].tolist()):
        s_idx = s0_index.get(pid)
        if s_idx is None:
            continue
        s0_keep.append(s_idx)
        treat_keep.append(treat_idx)

    if not s0_keep:
        raise ValueError("No overlapping patient_id values between S0 and treatment cohort")

    s0_keep_arr = np.asarray(s0_keep, dtype=int)
    treat_keep_arr = np.asarray(treat_keep, dtype=int)

    aligned_teacher = None
    if teacher_embeddings is not None:
        if len(teacher_embeddings) == len(s0_static):
            aligned_teacher = np.asarray(teacher_embeddings[s0_keep_arr])
        elif len(teacher_embeddings) == len(treatment_static):
            aligned_teacher = np.asarray(teacher_embeddings[treat_keep_arr])
        else:
            raise ValueError("teacher_embeddings length does not match S0 or treatment cohort")

    aligned_notes = None
    if note_embeddings is not None:
        if len(note_embeddings) == len(s0_static):
            aligned_notes = np.asarray(note_embeddings[s0_keep_arr])
        elif len(note_embeddings) == len(treatment_static):
            aligned_notes = np.asarray(note_embeddings[treat_keep_arr])
        else:
            raise ValueError("note_embeddings length does not match S0 or treatment cohort")

    return {
        "continuous": np.asarray(continuous[s0_keep_arr]),
        "masks_continuous": np.asarray(masks_continuous[s0_keep_arr]),
        "treatments": np.asarray(treatments[treat_keep_arr]),
        "masks_treatments": np.asarray(masks_treatments[treat_keep_arr]),
        "teacher_embeddings": aligned_teacher,
        "note_embeddings": aligned_notes,
        "s0_static": s0_static.iloc[s0_keep_arr].reset_index(drop=True),
        "treatment_static": treatment_static.iloc[treat_keep_arr].reset_index(drop=True),
    }


def _student_probs(model: RealtimeStudentClassifier, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["x"].to(device),
                batch["mask"].to(device),
                batch["treatments"].to(device),
                batch["treatment_mask"].to(device),
                note_embeddings=batch.get("notes").to(device) if "notes" in batch else None,
            )
            probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
            labels.append(batch["y"].cpu().numpy())
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def _student_config_dict(
    *,
    n_cont_features: int,
    n_treat_features: int,
    note_dim: int,
    student_d_model: int,
    teacher_dim: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    max_seq_len: int,
    treatment_layers: int,
    head_hidden_dim: int,
    head_dropout: float,
) -> dict:
    return {
        "type": "realtime_student_treatment_aware",
        "n_cont_features": n_cont_features,
        "n_treat_features": n_treat_features,
        "note_dim": note_dim,
        "student_d_model": student_d_model,
        "teacher_dim": teacher_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "dropout": dropout,
        "max_seq_len": max_seq_len,
        "treatment_layers": treatment_layers,
        "head_hidden_dim": head_hidden_dim,
        "head_dropout": head_dropout,
    }
