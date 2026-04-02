"""
multitask_model.py - Multi-task realtime student for sepsis subtype diagnosis.

Extends S5 RealtimeStudentClassifier with additional task heads:
  - mortality (binary)
  - immune subtype (3-class: Unclassified, EIL-like, MAS-like)
  - organ subtype (5-class: Unclassified, alpha, beta, gamma, delta)
  - fluid benefit (3-class: Unclassified, low_benefit, high_benefit)
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
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from s4.treatment_aware_model import TreatmentAwareEncoder
from s5.realtime_model import (
    CausalTCNStudentEncoder,
    _initialize_realtime_student_model,
    _student_config_dict,
    estimate_cpu_latency_ms,
    quantize_realtime_model,
)
from s15.calibration import TemperatureScaling
from s15.classification_eval import _classification_metrics, _select_threshold



def _compute_class_weights(labels: np.ndarray, n_classes: int, method: str = "inv_sqrt") -> torch.Tensor | None:
    """Compute per-class weights to address imbalance."""
    counts = np.bincount(labels.astype(int).flatten(), minlength=n_classes)
    if method == "inv_sqrt":
        weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    elif method == "inverse":
        weights = 1.0 / np.maximum(counts, 1.0)
    elif method == "effective":
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - beta ** np.maximum(counts, 1.0))
    else:
        return None
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal loss for classification with optional per-class alpha weights."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(inputs, dim=-1)
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = torch.exp(logp_t)
        focal_weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        loss = -focal_weight * logp_t
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultitaskStudentDataset(Dataset):
    """Dataset for multi-task student training."""

    def __init__(
        self,
        *,
        continuous: np.ndarray,
        masks_continuous: np.ndarray,
        treatments: np.ndarray,
        masks_treatments: np.ndarray,
        labels_mortality: np.ndarray,
        labels_immune: np.ndarray,
        labels_organ: np.ndarray,
        labels_fluid: np.ndarray,
        indices: np.ndarray,
    ):
        self.continuous = continuous[indices].astype(np.float32, copy=False)
        self.masks_continuous = masks_continuous[indices].astype(np.float32, copy=False)
        self.treatments = treatments[indices].astype(np.float32, copy=False)
        self.masks_treatments = masks_treatments[indices].astype(np.float32, copy=False)
        self.labels_mortality = labels_mortality[indices].astype(np.float32, copy=False)
        self.labels_immune = labels_immune[indices].astype(np.int64, copy=False)
        self.labels_organ = labels_organ[indices].astype(np.int64, copy=False)
        self.labels_fluid = labels_fluid[indices].astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.labels_mortality)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": torch.from_numpy(self.continuous[idx]),
            "mask": torch.from_numpy(self.masks_continuous[idx]),
            "treatments": torch.from_numpy(self.treatments[idx]),
            "treatment_mask": torch.from_numpy(self.masks_treatments[idx]),
            "y_mortality": torch.tensor(self.labels_mortality[idx], dtype=torch.float32),
            "y_immune": torch.tensor(self.labels_immune[idx], dtype=torch.long),
            "y_organ": torch.tensor(self.labels_organ[idx], dtype=torch.long),
            "y_fluid": torch.tensor(self.labels_fluid[idx], dtype=torch.long),
        }


class MultitaskRealtimeStudentClassifier(nn.Module):
    """S5 realtime student with multi-task heads for subtype diagnosis."""

    def __init__(
        self,
        *,
        n_cont_features: int,
        n_treat_features: int,
        note_dim: int = 0,
        student_arch: str = "transformer",
        student_d_model: int = 64,
        teacher_dim: int = 0,
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
        n_immune_classes: int = 3,
        n_organ_classes: int = 5,
        n_fluid_classes: int = 3,
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

        def make_head(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(student_d_model, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, out_dim),
            )

        self.head_mortality = make_head(1)
        self.head_immune = make_head(n_immune_classes)
        self.head_organ = make_head(n_organ_classes)
        self.head_fluid = make_head(n_fluid_classes)

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
        teacher_embedding = self.teacher_projection(embedding) if self.teacher_projection is not None else None
        return {
            "logits_mortality": self.head_mortality(embedding).squeeze(-1),
            "logits_immune": self.head_immune(embedding),
            "logits_organ": self.head_organ(embedding),
            "logits_fluid": self.head_fluid(embedding),
            "student_embedding": embedding,
            "teacher_embedding": teacher_embedding,
        }


def _multitask_probs(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    probs = {k: [] for k in ["mortality", "immune", "organ", "fluid"]}
    labels = {k: [] for k in ["mortality", "immune", "organ", "fluid"]}
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["x"].to(device),
                batch["mask"].to(device),
                batch["treatments"].to(device),
                batch["treatment_mask"].to(device),
            )
            probs["mortality"].append(torch.sigmoid(out["logits_mortality"]).cpu().numpy())
            labels["mortality"].append(batch["y_mortality"].cpu().numpy())
            probs["immune"].append(F.softmax(out["logits_immune"], dim=-1).cpu().numpy())
            labels["immune"].append(batch["y_immune"].cpu().numpy())
            probs["organ"].append(F.softmax(out["logits_organ"], dim=-1).cpu().numpy())
            labels["organ"].append(batch["y_organ"].cpu().numpy())
            probs["fluid"].append(F.softmax(out["logits_fluid"], dim=-1).cpu().numpy())
            labels["fluid"].append(batch["y_fluid"].cpu().numpy())

    return {
        k: {
            "probs": np.concatenate(v, axis=0),
            "labels": np.concatenate(labels[k], axis=0),
        }
        for k, v in probs.items()
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _evaluate_multitask(preds: dict, threshold: float) -> dict:
    out = {}
    # Mortality
    y = preds["mortality"]["labels"].astype(int)
    p = preds["mortality"]["probs"]
    out["mortality"] = {
        **_classification_metrics(y, p, threshold),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5,
    }
    # Immune
    y = preds["immune"]["labels"].astype(int)
    p = preds["immune"]["probs"].argmax(axis=1)
    out["immune"] = {
        "macro_f1": _macro_f1(y, p),
        "accuracy": float((y == p).mean()),
    }
    # Organ
    y = preds["organ"]["labels"].astype(int)
    p = preds["organ"]["probs"].argmax(axis=1)
    out["organ"] = {
        "macro_f1": _macro_f1(y, p),
        "accuracy": float((y == p).mean()),
    }
    # Fluid
    y = preds["fluid"]["labels"].astype(int)
    p = preds["fluid"]["probs"].argmax(axis=1)
    out["fluid"] = {
        "macro_f1": _macro_f1(y, p),
        "accuracy": float((y == p).mean()),
    }
    return out


def train_multitask_student(
    *,
    data_dir: Path,
    output_dir: Path,
    init_checkpoint_path: Path | None = None,
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    patience: int = 4,
    lambda_mortality: float = 1.0,
    lambda_immune: float = 1.0,
    lambda_organ: float = 1.0,
    lambda_fluid: float = 1.0,
    apply_temperature_scaling: bool = False,
    init_checkpoint_strict: bool = False,
    seed: int = 42,
    device: str = "cpu",
    student_arch: str = "transformer",
    student_d_model: int = 64,
    teacher_dim: int = 0,
    n_heads: int = 4,
    n_layers: int = 1,
    d_ff: int = 128,
    dropout: float = 0.1,
    treatment_layers: int = 1,
    head_hidden_dim: int = 64,
    head_dropout: float = 0.1,
    tcn_kernel_size: int = 3,
    tcn_dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    immune_boost: float = 1.0,
    organ_boost: float = 1.0,
    fluid_boost: float = 1.0,
) -> dict:
    """Train multi-task realtime student on enhanced MIMIC data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load enhanced artifacts
    time_series = np.load(data_dir / "time_series_enhanced.npy")
    patient_info = pd.read_csv(data_dir / "patient_info_enhanced.csv")

    n_samples = time_series.shape[0]
    # Last feature is mech_vent -> use as treatment
    n_cont = time_series.shape[2] - 1
    continuous = np.nan_to_num(time_series[:, :, :n_cont], nan=0.0).astype(np.float32)
    masks_cont = np.isfinite(time_series[:, :, :n_cont]).astype(np.float32)

    treatments = np.nan_to_num(time_series[:, :, n_cont:], nan=0.0).astype(np.float32)
    masks_treat = np.isfinite(time_series[:, :, n_cont:]).astype(np.float32)

    # Simple per-feature standardization using training stats
    feat_mean = np.zeros(n_cont, dtype=np.float32)
    feat_std = np.ones(n_cont, dtype=np.float32)
    for f in range(n_cont):
        valid = continuous[:, :, f][masks_cont[:, :, f] > 0]
        if len(valid) > 0:
            feat_mean[f] = float(valid.mean())
            feat_std[f] = max(float(valid.std()), 1.0e-6)
    continuous = (continuous - feat_mean[None, None, :]) / feat_std[None, None, :]
    np.save(output_dir / "feat_mean.npy", feat_mean)
    np.save(output_dir / "feat_std.npy", feat_std)

    labels_mortality = patient_info["mortality_28d"].fillna(0).astype(int).to_numpy()
    labels_immune = patient_info["immune_subtype_label"].fillna(0).astype(int).to_numpy()
    labels_organ = patient_info["organ_subtype_label"].fillna(0).astype(int).to_numpy()
    labels_fluid = patient_info["fluid_benefit_label"].fillna(0).astype(int).to_numpy()

    # Stratified splits by mortality
    from sklearn.model_selection import train_test_split
    idx = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(
        idx, test_size=1 - train_ratio, random_state=seed, stratify=labels_mortality
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - train_ratio - val_ratio) / (1 - train_ratio),
        random_state=seed,
        stratify=labels_mortality[temp_idx],
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    with open(output_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in splits.items()}, f)

    train_loader = DataLoader(
        MultitaskStudentDataset(
            continuous=continuous,
            masks_continuous=masks_cont,
            treatments=treatments,
            masks_treatments=masks_treat,
            labels_mortality=labels_mortality,
            labels_immune=labels_immune,
            labels_organ=labels_organ,
            labels_fluid=labels_fluid,
            indices=train_idx,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        MultitaskStudentDataset(
            continuous=continuous,
            masks_continuous=masks_cont,
            treatments=treatments,
            masks_treatments=masks_treat,
            labels_mortality=labels_mortality,
            labels_immune=labels_immune,
            labels_organ=labels_organ,
            labels_fluid=labels_fluid,
            indices=val_idx,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        MultitaskStudentDataset(
            continuous=continuous,
            masks_continuous=masks_cont,
            treatments=treatments,
            masks_treatments=masks_treat,
            labels_mortality=labels_mortality,
            labels_immune=labels_immune,
            labels_organ=labels_organ,
            labels_fluid=labels_fluid,
            indices=test_idx,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    n_immune_classes = int(labels_immune.max()) + 1
    n_organ_classes = int(labels_organ.max()) + 1
    n_fluid_classes = int(labels_fluid.max()) + 1

    model = MultitaskRealtimeStudentClassifier(
        n_cont_features=n_cont,
        n_treat_features=treatments.shape[-1],
        note_dim=0,
        student_arch=student_arch,
        student_d_model=student_d_model,
        teacher_dim=teacher_dim,
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
        n_immune_classes=n_immune_classes,
        n_organ_classes=n_organ_classes,
        n_fluid_classes=n_fluid_classes,
    ).to(device)

    initialization_summary = _initialize_realtime_student_model(
        model,
        checkpoint_path=init_checkpoint_path,
        strict=init_checkpoint_strict,
        device=device,
    )

    # Custom mapping: S5-v2 single-task head -> multitask mortality head
    if init_checkpoint_path is not None:
        ckpt = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        mapped_head = {}
        for key, value in state_dict.items():
            if key.startswith("head."):
                new_key = key.replace("head.", "head_mortality.")
                if new_key in model.state_dict() and model.state_dict()[new_key].shape == value.shape:
                    mapped_head[new_key] = value
        if mapped_head:
            load_result = model.load_state_dict(mapped_head, strict=False)
            initialization_summary["mortality_head_mapped_tensors"] = int(len(mapped_head))
            initialization_summary["mortality_head_missing_tensors"] = int(len(load_result.missing_keys))
        else:
            initialization_summary["mortality_head_mapped_tensors"] = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    # Class weights for imbalance
    immune_weights = _compute_class_weights(labels_immune[train_idx], n_immune_classes, method="inv_sqrt")
    organ_weights = _compute_class_weights(labels_organ[train_idx], n_organ_classes, method="inv_sqrt")
    fluid_weights = _compute_class_weights(labels_fluid[train_idx], n_fluid_classes, method="inv_sqrt")
    if immune_weights is not None:
        immune_weights = immune_weights * immune_boost
        immune_weights = immune_weights / immune_weights.mean()
    if organ_weights is not None:
        organ_weights = organ_weights * organ_boost
        organ_weights = organ_weights / organ_weights.mean()
    if fluid_weights is not None:
        fluid_weights = fluid_weights * fluid_boost
        fluid_weights = fluid_weights / fluid_weights.mean()

    if use_focal_loss:
        ce_immune = FocalLoss(gamma=focal_gamma, alpha=immune_weights).to(device)
        ce_organ = FocalLoss(gamma=focal_gamma, alpha=organ_weights).to(device)
        ce_fluid = FocalLoss(gamma=focal_gamma, alpha=fluid_weights).to(device)
    else:
        ce_immune = nn.CrossEntropyLoss(weight=immune_weights.to(device) if immune_weights is not None else None, ignore_index=-1)
        ce_organ = nn.CrossEntropyLoss(weight=organ_weights.to(device) if organ_weights is not None else None, ignore_index=-1)
        ce_fluid = nn.CrossEntropyLoss(weight=fluid_weights.to(device) if fluid_weights is not None else None, ignore_index=-1)

    best_state = None
    best_score = -np.inf
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
            )
            loss = (
                lambda_mortality * bce(out["logits_mortality"], batch["y_mortality"].to(device))
                + lambda_immune * ce_immune(out["logits_immune"], batch["y_immune"].to(device))
                + lambda_organ * ce_organ(out["logits_organ"], batch["y_organ"].to(device))
                + lambda_fluid * ce_fluid(out["logits_fluid"], batch["y_fluid"].to(device))
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_preds = _multitask_probs(model, val_loader, device)
        val_metrics = _evaluate_multitask(val_preds, threshold=0.5)
        val_score = (
            val_metrics["mortality"]["auroc"]
            + val_metrics["immune"]["macro_f1"]
            + val_metrics["organ"]["macro_f1"]
            + val_metrics["fluid"]["macro_f1"]
        )
        history.append(
            {
                "epoch": epoch,
                "time_s": round(time.time() - start_time, 2),
                "train_loss": round(float(np.mean(losses)), 4) if losses else 0.0,
                "val_mortality_auroc": round(val_metrics["mortality"]["auroc"], 4),
                "val_immune_f1": round(val_metrics["immune"]["macro_f1"], 4),
                "val_organ_f1": round(val_metrics["organ"]["macro_f1"], 4),
                "val_fluid_f1": round(val_metrics["fluid"]["macro_f1"], 4),
                "val_score": round(float(val_score), 4),
            }
        )
        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": _student_config_dict(
                        n_cont_features=n_cont,
                        n_treat_features=treatments.shape[-1],
                        note_dim=0,
                        student_arch=student_arch,
                        student_d_model=student_d_model,
                        teacher_dim=teacher_dim,
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

    # Threshold selection on mortality val
    val_preds = _multitask_probs(model, val_loader, device)
    threshold, threshold_search = _select_threshold(
        val_preds["mortality"]["labels"].astype(int),
        val_preds["mortality"]["probs"],
        metric_name="balanced_accuracy",
    )

    train_metrics = _evaluate_multitask(_multitask_probs(model, train_loader, device), threshold)
    val_metrics = _evaluate_multitask(val_preds, threshold)
    test_metrics = _evaluate_multitask(_multitask_probs(model, test_loader, device), threshold)

    sample_batch = {
        "x": torch.from_numpy(continuous[:1]).float(),
        "mask": torch.from_numpy(masks_cont[:1]).float(),
        "treatments": torch.from_numpy(treatments[:1]).float(),
        "treatment_mask": torch.from_numpy(masks_treat[:1]).float(),
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
            n_cont_features=n_cont,
            n_treat_features=treatments.shape[-1],
            note_dim=0,
            student_arch=student_arch,
            student_d_model=student_d_model,
            teacher_dim=teacher_dim,
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
            "lambda_mortality": lambda_mortality,
            "lambda_immune": lambda_immune,
            "lambda_organ": lambda_organ,
            "lambda_fluid": lambda_fluid,
            "use_focal_loss": bool(use_focal_loss),
            "focal_gamma": focal_gamma,
            "immune_boost": immune_boost,
            "organ_boost": organ_boost,
            "fluid_boost": fluid_boost,
            "apply_temperature_scaling": bool(apply_temperature_scaling),
            "initialization": initialization_summary,
            "seed": seed,
            "device": device,
        },
        "history": history,
        "threshold_selection": {
            "metric": "balanced_accuracy",
            "selected_threshold": round(float(threshold), 4),
            "search": threshold_search,
        },
        "splits": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "deployment": {
            "cpu_latency_ms_per_sample": round(float(latency), 3),
            "float_n_parameters": int(sum(p.numel() for p in model.parameters())),
            "dynamic_quantization_ok": quantized_ok,
        },
    }

    with open(output_dir / "multitask_student_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": report["model"],
            "threshold": threshold,
        },
        output_dir / "multitask_student.pt",
    )
    return report
