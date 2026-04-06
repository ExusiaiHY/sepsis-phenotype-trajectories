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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from s15.calibration import TemperatureScaling
from s15.classification_eval import _classification_metrics, _select_threshold
from s4.treatment_aware_model import TreatmentAwareEncoder, _calibration_report, _load_or_build_splits
from s5.deployment_policy import load_policy_artifact


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
        teacher_probabilities: np.ndarray | None = None,
        note_embeddings: np.ndarray | None = None,
    ):
        self.continuous = continuous[indices].astype(np.float32, copy=False)
        self.masks_continuous = masks_continuous[indices].astype(np.float32, copy=False)
        self.treatments = treatments[indices].astype(np.float32, copy=False)
        self.masks_treatments = masks_treatments[indices].astype(np.float32, copy=False)
        self.labels = labels[indices].astype(np.float32, copy=False)
        self.teacher_embeddings = None if teacher_embeddings is None else teacher_embeddings[indices].astype(np.float32, copy=False)
        self.teacher_probabilities = None
        if teacher_probabilities is not None:
            self.teacher_probabilities = teacher_probabilities[indices].astype(np.float32, copy=False).reshape(-1)
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
        if self.teacher_probabilities is not None:
            out["teacher_probability"] = torch.tensor(self.teacher_probabilities[idx], dtype=torch.float32)
        if self.note_embeddings is not None:
            out["notes"] = torch.from_numpy(self.note_embeddings[idx])
        return out


class CausalConv1d(nn.Conv1d):
    """1D convolution with left padding only, preserving causal semantics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel_size = int(self.kernel_size[0])
        dilation = int(self.dilation[0])
        self.causal_trim = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.causal_trim > 0:
            out = out[..., :-self.causal_trim]
        return out


class CausalDepthwiseSeparableBlock(nn.Module):
    """Residual depthwise-separable temporal block for fast bedside inference."""

    def __init__(
        self,
        *,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.depthwise = CausalConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.depthwise(x)
        out = F.gelu(out)
        out = self.pointwise(out)
        out = self.dropout(out)
        out = (residual + out).transpose(1, 2)
        out = self.norm(out)
        return out.transpose(1, 2)


class CausalTCNStudentEncoder(nn.Module):
    """Causal TCN encoder for S5-v2 with structured+treatment fusion."""

    def __init__(
        self,
        *,
        n_cont_features: int,
        n_treat_features: int,
        note_dim: int = 0,
        d_model: int = 64,
        dropout: float = 0.1,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        input_dim = (2 * n_cont_features) + (2 * n_treat_features) + note_dim
        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                CausalDepthwiseSeparableBlock(
                    channels=d_model,
                    kernel_size=max(2, int(kernel_size)),
                    dilation=max(1, int(dilation)),
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        treatments: torch.Tensor,
        treatment_mask: torch.Tensor,
        note_embeddings: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pieces = [x, mask, treatments, treatment_mask]
        if note_embeddings is not None:
            pieces.append(note_embeddings)
        sequence = torch.cat(pieces, dim=-1).transpose(1, 2)
        sequence = self.input_proj(sequence)
        for block in self.blocks:
            sequence = block(sequence)
        sequence = self.output_norm(sequence.transpose(1, 2))

        density = mask.mean(dim=-1, keepdim=True) + 0.5 * treatment_mask.mean(dim=-1, keepdim=True)
        if note_embeddings is not None:
            note_obs = (note_embeddings.abs().sum(dim=-1, keepdim=True) > 0).float()
            density = density + 0.25 * note_obs
        weights = density.clamp(min=1e-6)
        embedding = (sequence * weights).sum(dim=1) / weights.sum(dim=1)

        if return_sequence:
            return embedding, sequence
        return embedding


class RealtimeStudentClassifier(nn.Module):
    """Smaller treatment-aware student model with optional teacher-space projection."""

    def __init__(
        self,
        *,
        n_cont_features: int,
        n_treat_features: int,
        note_dim: int = 0,
        student_arch: str = "transformer",
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
        tcn_kernel_size: int = 3,
        tcn_dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.student_arch = str(student_arch).lower()
        self.tcn_dilations = tuple(int(value) for value in tcn_dilations)
        if self.student_arch == "transformer":
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
        elif self.student_arch == "tcn":
            self.encoder = CausalTCNStudentEncoder(
                n_cont_features=n_cont_features,
                n_treat_features=n_treat_features,
                note_dim=note_dim,
                d_model=student_d_model,
                dropout=dropout,
                kernel_size=tcn_kernel_size,
                dilations=self.tcn_dilations,
            )
        else:
            raise ValueError(f"Unsupported student_arch: {student_arch}")
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
    init_checkpoint_path: Path | None = None,
    teacher_embeddings_path: Path | None = None,
    teacher_probabilities_path: Path | None = None,
    note_embeddings_path: Path | None = None,
    splits_path: Path | None = None,
    label_col: str = "mortality_inhospital",
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    patience: int = 4,
    bce_weight: float = 1.0,
    pos_weight: float | None = None,
    horizon_augmentation_min_h: int = 0,
    distill_weight: float = 1.0,
    distill_cosine_weight: float = 0.0,
    distill_prob_weight: float = 0.0,
    distill_temperature: float = 1.0,
    apply_temperature_scaling: bool = False,
    init_checkpoint_strict: bool = True,
    threshold_metric: str = "balanced_accuracy",
    target_positive_rate: float | None = None,
    seed: int = 42,
    device: str = "cpu",
    student_arch: str = "transformer",
    student_d_model: int = 64,
    teacher_dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 1,
    d_ff: int = 128,
    dropout: float = 0.1,
    treatment_layers: int = 1,
    head_hidden_dim: int = 64,
    head_dropout: float = 0.1,
    tcn_kernel_size: int = 3,
    tcn_dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8),
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
    teacher_probabilities = np.load(teacher_probabilities_path) if teacher_probabilities_path is not None else None
    notes = np.load(note_embeddings_path) if note_embeddings_path is not None else None

    aligned = _align_stage5_inputs(
        s0_static=s0_static,
        treatment_static=treatment_static,
        continuous=continuous,
        masks_continuous=masks_cont,
        treatments=treatments,
        masks_treatments=masks_treat,
        teacher_embeddings=teacher_embeddings,
        teacher_probabilities=teacher_probabilities,
        note_embeddings=notes,
    )
    continuous = aligned["continuous"]
    masks_cont = aligned["masks_continuous"]
    treatments = aligned["treatments"]
    masks_treat = aligned["masks_treatments"]
    teacher_embeddings = aligned["teacher_embeddings"]
    teacher_probabilities = aligned["teacher_probabilities"]
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
            teacher_probabilities=teacher_probabilities,
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
            teacher_probabilities=teacher_probabilities,
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
            teacher_probabilities=teacher_probabilities,
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
        student_arch=student_arch,
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
        tcn_kernel_size=tcn_kernel_size,
        tcn_dilations=tcn_dilations,
    ).to(device)
    initialization_summary = _initialize_realtime_student_model(
        model,
        checkpoint_path=init_checkpoint_path,
        strict=init_checkpoint_strict,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    _pos_weight_tensor = None if pos_weight is None else torch.tensor([float(pos_weight)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=_pos_weight_tensor)
    mse = nn.MSELoss()
    seq_len = continuous.shape[1]
    _aug_min_h = max(1, int(horizon_augmentation_min_h)) if horizon_augmentation_min_h > 0 else 0

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
            bx = batch["x"].to(device)
            bm = batch["mask"].to(device)
            bt = batch["treatments"].to(device)
            btm = batch["treatment_mask"].to(device)
            bn = batch.get("notes")
            if bn is not None:
                bn = bn.to(device)
            if _aug_min_h > 0:
                h = int(torch.randint(_aug_min_h, seq_len + 1, (1,)).item())
                bx = bx.clone(); bx[:, h:] = 0.0
                bm = bm.clone(); bm[:, h:] = 0.0
                bt = bt.clone(); bt[:, h:] = 0.0
                btm = btm.clone(); btm[:, h:] = 0.0
                if bn is not None:
                    bn = bn.clone(); bn[:, h:] = 0.0
            out = model(bx, bm, bt, btm, note_embeddings=bn)
            y = batch["y"].to(device)
            loss = bce_weight * bce(out["logits"], y)
            if "teacher_embedding" in batch and out["teacher_embedding"] is not None:
                loss = loss + distill_weight * mse(out["teacher_embedding"], batch["teacher_embedding"].to(device))
                if distill_cosine_weight > 0:
                    cosine_loss = 1.0 - F.cosine_similarity(
                        out["teacher_embedding"],
                        batch["teacher_embedding"].to(device),
                        dim=-1,
                    ).mean()
                    loss = loss + (distill_cosine_weight * cosine_loss)
            if "teacher_probability" in batch and distill_prob_weight > 0:
                teacher_prob = batch["teacher_probability"].to(device)
                loss = loss + (distill_prob_weight * _bernoulli_kl_from_logits(
                    out["logits"],
                    teacher_prob,
                    temperature=distill_temperature,
                ))
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
                        student_arch=student_arch,
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
                        tcn_kernel_size=tcn_kernel_size,
                        tcn_dilations=tcn_dilations,
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

    train_probs_raw, train_y = _student_probs(model, train_loader, device)
    val_probs_raw, val_y = _student_probs(model, val_loader, device)
    test_probs_raw, test_y = _student_probs(model, test_loader, device)
    raw_calibration = {
        "train": _calibration_report(train_y, train_probs_raw),
        "val": _calibration_report(val_y, val_probs_raw),
        "test": _calibration_report(test_y, test_probs_raw),
    }
    calibration_summary = {
        "method": "none",
        "temperature": 1.0,
        "raw": raw_calibration,
    }
    train_probs = train_probs_raw
    val_probs = val_probs_raw
    test_probs = test_probs_raw
    temperature = 1.0
    if apply_temperature_scaling:
        calibrator = TemperatureScaling().fit(val_probs_raw, val_y.astype(int))
        temperature = max(float(calibrator.temperature), 1.0e-3)
        train_probs = calibrator.predict(train_probs_raw)
        val_probs = calibrator.predict(val_probs_raw)
        test_probs = calibrator.predict(test_probs_raw)
        calibration_summary = {
            "method": "temperature_scaling",
            "temperature": round(float(temperature), 4),
            "raw": raw_calibration,
        }
    threshold, threshold_search = _select_threshold(
        val_y.astype(int),
        val_probs,
        metric_name=threshold_metric,
        target_positive_rate=target_positive_rate,
    )

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
            student_arch=student_arch,
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
            tcn_kernel_size=tcn_kernel_size,
            tcn_dilations=tcn_dilations,
        ),
        "training": {
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "epochs_trained": len(history),
            "lr": lr,
            "weight_decay": weight_decay,
            "bce_weight": bce_weight,
            "distill_weight": distill_weight,
            "distill_cosine_weight": distill_cosine_weight,
            "distill_prob_weight": distill_prob_weight,
            "distill_temperature": distill_temperature,
            "apply_temperature_scaling": bool(apply_temperature_scaling),
            "initialization": initialization_summary,
            "threshold_metric": threshold_metric,
            "target_positive_rate": target_positive_rate,
            "seed": seed,
            "device": device,
        },
        "history": history,
        "threshold_selection": {
            "metric": threshold_metric,
            "selected_threshold": round(float(threshold), 4),
            "target_positive_rate": None if target_positive_rate is None else round(float(target_positive_rate), 4),
            "search": threshold_search,
        },
        "posthoc_calibration": calibration_summary,
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
            "temperature": temperature,
        },
        output_dir / "realtime_student.pt",
    )
    return report


def load_realtime_student_artifact(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
) -> dict:
    """Load a distilled realtime student artifact for deployment/replay."""
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = RealtimeStudentClassifier(
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
        tcn_dilations=tuple(int(value) for value in cfg.get("tcn_dilations", [1, 2, 4, 8])),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "config": cfg,
        "threshold": float(ckpt.get("threshold", 0.5)),
        "temperature": float(ckpt.get("temperature", 1.0)),
        "checkpoint_path": str(checkpoint_path),
    }


def _initialize_realtime_student_model(
    model: nn.Module,
    *,
    checkpoint_path: Path | None,
    strict: bool,
    device: str,
) -> dict:
    """Warm-start a realtime student from a compatible checkpoint when requested."""
    if checkpoint_path is None:
        return {
            "mode": "random",
            "checkpoint_path": None,
            "strict": bool(strict),
            "loaded_tensors": 0,
            "missing_tensors": 0,
            "unexpected_tensors": 0,
            "shape_mismatch_tensors": 0,
        }

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported initialization payload at {checkpoint_path}")

    if strict:
        load_result = model.load_state_dict(state_dict, strict=True)
        return {
            "mode": "checkpoint",
            "checkpoint_path": str(checkpoint_path),
            "strict": True,
            "loaded_tensors": int(len(state_dict)),
            "missing_tensors": int(len(load_result.missing_keys)),
            "unexpected_tensors": int(len(load_result.unexpected_keys)),
            "shape_mismatch_tensors": 0,
        }

    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    shape_mismatch_keys: list[str] = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            unexpected_keys.append(key)
            continue
        if tuple(target.shape) != tuple(value.shape):
            shape_mismatch_keys.append(key)
            continue
        compatible_state[key] = value

    missing_keys = [key for key in model_state.keys() if key not in compatible_state]
    load_result = model.load_state_dict(compatible_state, strict=False)
    if load_result.unexpected_keys:
        unexpected_keys.extend(load_result.unexpected_keys)
    if load_result.missing_keys:
        missing_keys = list(dict.fromkeys([*missing_keys, *load_result.missing_keys]))
    return {
        "mode": "checkpoint",
        "checkpoint_path": str(checkpoint_path),
        "strict": False,
        "loaded_tensors": int(len(compatible_state)),
        "missing_tensors": int(len(missing_keys)),
        "unexpected_tensors": int(len(unexpected_keys)),
        "shape_mismatch_tensors": int(len(shape_mismatch_keys)),
    }


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


class RealtimeAlertPolicy:
    """Stateful online alert policy with hysteresis and alert-budget controls."""

    def __init__(
        self,
        *,
        enter_threshold: float,
        exit_threshold: float | None = None,
        min_history_hours: int = 1,
        min_consecutive_hours: int = 1,
        refractory_hours: int = 0,
        max_alerts_per_stay: int | None = None,
        policy_name: str | None = None,
    ):
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = self.enter_threshold if exit_threshold is None else float(exit_threshold)
        if self.exit_threshold > self.enter_threshold:
            raise ValueError("exit_threshold must be <= enter_threshold")
        self.min_history_hours = max(1, int(min_history_hours))
        self.min_consecutive_hours = max(1, int(min_consecutive_hours))
        self.refractory_hours = max(0, int(refractory_hours))
        self.max_alerts_per_stay = None if max_alerts_per_stay is None else max(1, int(max_alerts_per_stay))
        self.policy_name = None if policy_name is None else str(policy_name)
        self.reset()

    @classmethod
    def from_dict(
        cls,
        policy: dict | None,
        *,
        default_threshold: float,
        default_min_history_hours: int = 1,
    ) -> "RealtimeAlertPolicy":
        if policy is None:
            cfg: dict = {}
        elif isinstance(policy, dict) and "policy" in policy and isinstance(policy["policy"], dict):
            cfg = dict(policy["policy"])
        elif isinstance(policy, dict) and "best_policy" in policy and isinstance(policy["best_policy"], dict):
            cfg = dict(policy["best_policy"])
        elif isinstance(policy, dict):
            cfg = dict(policy)
        else:
            raise ValueError("deployment policy must be a dict or None")
        enter_threshold = float(cfg.get("enter_threshold", cfg.get("threshold", default_threshold)))
        return cls(
            enter_threshold=enter_threshold,
            exit_threshold=cfg.get("exit_threshold", enter_threshold),
            min_history_hours=int(cfg.get("min_history_hours", default_min_history_hours)),
            min_consecutive_hours=int(cfg.get("min_consecutive_hours", 1)),
            refractory_hours=int(cfg.get("refractory_hours", 0)),
            max_alerts_per_stay=cfg.get("max_alerts_per_stay"),
            policy_name=cfg.get("policy_name"),
        )

    def reset(self) -> None:
        self.in_alert_state = False
        self.consecutive_high = 0
        self.next_allowed_hour = int(self.min_history_hours)
        self.n_alert_events = 0
        self.first_alert_hour: int | None = None

    def update(self, *, risk_probability: float, hours_seen: int) -> dict:
        risk_probability = float(risk_probability)
        hours_seen = max(0, int(hours_seen))
        deployment_ready = hours_seen >= self.min_history_hours
        alert_event = False
        risk_alert = False

        if not np.isfinite(risk_probability):
            return self._snapshot(hours_seen=hours_seen, risk_alert=False, alert_event=False, deployment_ready=deployment_ready)

        if self.in_alert_state:
            risk_alert = True
            if risk_probability < self.exit_threshold:
                self.in_alert_state = False
            return self._snapshot(
                hours_seen=hours_seen,
                risk_alert=risk_alert,
                alert_event=False,
                deployment_ready=deployment_ready,
            )

        if not deployment_ready:
            self.consecutive_high = 0
            return self._snapshot(hours_seen=hours_seen, risk_alert=False, alert_event=False, deployment_ready=False)

        if self.max_alerts_per_stay is not None and self.n_alert_events >= self.max_alerts_per_stay:
            self.consecutive_high = 0
            return self._snapshot(hours_seen=hours_seen, risk_alert=False, alert_event=False, deployment_ready=True)

        if hours_seen < self.next_allowed_hour:
            self.consecutive_high = 0
            return self._snapshot(hours_seen=hours_seen, risk_alert=False, alert_event=False, deployment_ready=True)

        if risk_probability >= self.enter_threshold:
            self.consecutive_high += 1
        else:
            self.consecutive_high = 0

        if self.consecutive_high >= self.min_consecutive_hours:
            alert_event = True
            risk_alert = True
            self.in_alert_state = True
            self.n_alert_events += 1
            if self.first_alert_hour is None:
                self.first_alert_hour = hours_seen
            self.next_allowed_hour = hours_seen + self.refractory_hours + 1
            self.consecutive_high = 0

        return self._snapshot(
            hours_seen=hours_seen,
            risk_alert=risk_alert,
            alert_event=alert_event,
            deployment_ready=True,
        )

    def _snapshot(
        self,
        *,
        hours_seen: int,
        risk_alert: bool,
        alert_event: bool,
        deployment_ready: bool,
    ) -> dict:
        return {
            "policy_name": self.policy_name,
            "risk_alert": bool(risk_alert),
            "alert_event": bool(alert_event),
            "alert_state_active": bool(risk_alert),
            "deployment_ready": bool(deployment_ready),
            "alerts_emitted": int(self.n_alert_events),
            "first_alert_hour": None if self.first_alert_hour is None else int(self.first_alert_hour),
            "next_alert_eligible_hour": int(self.next_allowed_hour),
            "enter_threshold": round(float(self.enter_threshold), 4),
            "exit_threshold": round(float(self.exit_threshold), 4),
            "min_history_hours": int(self.min_history_hours),
            "min_consecutive_hours": int(self.min_consecutive_hours),
            "refractory_hours": int(self.refractory_hours),
            "max_alerts_per_stay": self.max_alerts_per_stay,
            "hours_seen": int(hours_seen),
        }


class RealtimePhenotypeMonitor:
    """Runtime wrapper for rolling bedside inference and phenotype assignment."""

    def __init__(
        self,
        *,
        model: nn.Module,
        threshold: float,
        phenotype_centroids: np.ndarray | None = None,
        device: str = "cpu",
        temperature: float = 1.0,
        deployment_policy: dict | None = None,
    ):
        self.model = model.to(device).eval()
        self.threshold = float(threshold)
        self.phenotype_centroids = phenotype_centroids
        self.device = device
        self.temperature = max(float(temperature), 1.0e-3)
        self.alert_policy = RealtimeAlertPolicy.from_dict(
            deployment_policy,
            default_threshold=self.threshold,
            default_min_history_hours=1,
        )
        self._last_hours_seen = 0

    @classmethod
    def from_artifacts(
        cls,
        *,
        checkpoint_path: Path,
        policy_path: Path | None = None,
        phenotype_centroids: np.ndarray | None = None,
        device: str = "cpu",
    ) -> "RealtimePhenotypeMonitor":
        artifact = load_realtime_student_artifact(checkpoint_path, device=device)
        policy_artifact = None if policy_path is None else load_policy_artifact(policy_path)
        return cls(
            model=artifact["model"],
            threshold=float(artifact["threshold"]),
            phenotype_centroids=phenotype_centroids,
            device=device,
            temperature=float(artifact["temperature"]),
            deployment_policy=policy_artifact,
        )

    def reset_patient_state(self) -> None:
        self.alert_policy.reset()
        self._last_hours_seen = 0

    def predict(self, buffer: RealtimePatientBuffer) -> dict | None:
        if buffer.n_updates <= 0:
            return None
        if buffer.n_updates < self._last_hours_seen:
            self.reset_patient_state()

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
            risk_prob = float(torch.sigmoid(out["logits"] / self.temperature).cpu().item())
            phenotype = None
            if self.phenotype_centroids is not None and out.get("teacher_embedding") is not None:
                emb = out["teacher_embedding"].cpu().numpy()[0]
                dists = np.linalg.norm(self.phenotype_centroids - emb[None, :], axis=1)
                phenotype = int(np.argmin(dists))
        policy_snapshot = self.alert_policy.update(
            risk_probability=risk_prob,
            hours_seen=int(buffer.n_updates),
        )
        self._last_hours_seen = int(buffer.n_updates)
        return {
            "risk_probability": round(risk_prob, 4),
            "phenotype": phenotype,
            **policy_snapshot,
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
    teacher_probabilities: np.ndarray | None = None,
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

    aligned_teacher_probabilities = None
    if teacher_probabilities is not None:
        teacher_probabilities = np.asarray(teacher_probabilities, dtype=np.float32).reshape(-1)
        if len(teacher_probabilities) == len(s0_static):
            aligned_teacher_probabilities = np.asarray(teacher_probabilities[s0_keep_arr])
        elif len(teacher_probabilities) == len(treatment_static):
            aligned_teacher_probabilities = np.asarray(teacher_probabilities[treat_keep_arr])
        else:
            raise ValueError("teacher_probabilities length does not match S0 or treatment cohort")

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
        "teacher_probabilities": aligned_teacher_probabilities,
        "note_embeddings": aligned_notes,
        "s0_static": s0_static.iloc[s0_keep_arr].reset_index(drop=True),
        "treatment_static": treatment_static.iloc[treat_keep_arr].reset_index(drop=True),
    }


def _student_probs(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
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


def _bernoulli_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    temp = max(float(temperature), 1.0e-3)
    student_probs = torch.sigmoid(student_logits / temp).clamp(1.0e-6, 1.0 - 1.0e-6)
    teacher_probs = teacher_probabilities.clamp(1.0e-6, 1.0 - 1.0e-6)
    kl = (
        teacher_probs * (torch.log(teacher_probs) - torch.log(student_probs))
        + (1.0 - teacher_probs) * (torch.log(1.0 - teacher_probs) - torch.log(1.0 - student_probs))
    )
    return kl.mean() * (temp ** 2)


def _student_config_dict(
    *,
    n_cont_features: int,
    n_treat_features: int,
    note_dim: int,
    student_arch: str,
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
    tcn_kernel_size: int,
    tcn_dilations: tuple[int, ...] | list[int],
) -> dict:
    return {
        "type": f"realtime_student_{student_arch}",
        "n_cont_features": n_cont_features,
        "n_treat_features": n_treat_features,
        "note_dim": note_dim,
        "student_arch": student_arch,
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
        "tcn_kernel_size": int(tcn_kernel_size),
        "tcn_dilations": [int(value) for value in tcn_dilations],
    }
