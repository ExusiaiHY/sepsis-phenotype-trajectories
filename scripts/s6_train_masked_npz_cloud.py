#!/usr/bin/env python3
"""
Dedicated cloud entrypoint for masked NPZ-based S6 multitask training.

This script does not depend on the legacy `s6/multitask_model.py` module so
cloud runs can reliably consume the new `sepsis_multitask_targets.npz`
supervision bundle even if older local files remain in the workspace.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s4.treatment_aware_model import TreatmentAwareEncoder
from s5.realtime_model import (
    CausalTCNStudentEncoder,
    _initialize_realtime_student_model,
    _student_config_dict,
    estimate_cpu_latency_ms,
    quantize_realtime_model,
)
from s15.classification_eval import _classification_metrics, _select_threshold


def _compute_class_weights(labels: np.ndarray, n_classes: int, method: str = "inv_sqrt") -> torch.Tensor | None:
    counts = np.bincount(labels.astype(int).flatten(), minlength=n_classes)
    if counts.sum() <= 0:
        return None
    if method == "inv_sqrt":
        weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    elif method == "inverse":
        weights = 1.0 / np.maximum(counts, 1.0)
    elif method == "effective":
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - beta ** np.maximum(counts, 1.0))
    else:
        return None
    weights = weights / max(float(weights.mean()), 1.0e-6)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(inputs, dim=-1)
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = torch.exp(logp_t)
        focal_weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha.gather(0, targets)
        return (-focal_weight * logp_t).mean()


class MultitaskStudentDataset(Dataset):
    def __init__(
        self,
        *,
        continuous: np.ndarray,
        masks_continuous: np.ndarray,
        treatments: np.ndarray,
        masks_treatments: np.ndarray,
        labels_mortality: np.ndarray,
        classification_labels: np.ndarray,
        classification_masks: np.ndarray,
        regression_targets: np.ndarray,
        regression_masks: np.ndarray,
        indices: np.ndarray,
    ):
        self.continuous = continuous[indices].astype(np.float32, copy=False)
        self.masks_continuous = masks_continuous[indices].astype(np.float32, copy=False)
        self.treatments = treatments[indices].astype(np.float32, copy=False)
        self.masks_treatments = masks_treatments[indices].astype(np.float32, copy=False)
        self.labels_mortality = labels_mortality[indices].astype(np.float32, copy=False)
        self.classification_labels = classification_labels[indices].astype(np.int64, copy=False)
        self.classification_masks = classification_masks[indices].astype(np.float32, copy=False)
        self.regression_targets = regression_targets[indices].astype(np.float32, copy=False)
        self.regression_masks = regression_masks[indices].astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.labels_mortality)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": torch.from_numpy(self.continuous[idx]),
            "mask": torch.from_numpy(self.masks_continuous[idx]),
            "treatments": torch.from_numpy(self.treatments[idx]),
            "treatment_mask": torch.from_numpy(self.masks_treatments[idx]),
            "y_mortality": torch.tensor(self.labels_mortality[idx], dtype=torch.float32),
            "y_classification": torch.from_numpy(self.classification_labels[idx]),
            "mask_classification": torch.from_numpy(self.classification_masks[idx]),
            "y_regression": torch.from_numpy(self.regression_targets[idx]),
            "mask_regression": torch.from_numpy(self.regression_masks[idx]),
        }


class MultitaskRealtimeStudentClassifier(nn.Module):
    def __init__(
        self,
        *,
        n_cont_features: int,
        n_treat_features: int,
        classification_tasks: list[dict],
        regression_tasks: list[dict],
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
    ):
        super().__init__()
        self.student_arch = str(student_arch).lower()
        self.classification_tasks = [dict(task) for task in classification_tasks]
        self.regression_tasks = [dict(task) for task in regression_tasks]
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
        self.classification_heads = nn.ModuleDict(
            {str(task["name"]): make_head(int(task["n_classes"])) for task in self.classification_tasks}
        )
        self.regression_heads = nn.ModuleDict(
            {str(task["name"]): make_head(1) for task in self.regression_tasks}
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
        teacher_embedding = self.teacher_projection(embedding) if self.teacher_projection is not None else None
        return {
            "logits_mortality": self.head_mortality(embedding).squeeze(-1),
            "classification_logits": {name: head(embedding) for name, head in self.classification_heads.items()},
            "regression_outputs": {name: head(embedding).squeeze(-1) for name, head in self.regression_heads.items()},
            "student_embedding": embedding,
            "teacher_embedding": teacher_embedding,
        }


def _load_schema_bundle(patient_info: pd.DataFrame, data_dir: Path) -> dict:
    npz_path = data_dir / "sepsis_multitask_targets.npz"
    schema_path = data_dir / "sepsis_multitask_schema.json"
    if not npz_path.exists() or not schema_path.exists():
        raise FileNotFoundError(
            "Masked NPZ training requires sepsis_multitask_targets.npz and sepsis_multitask_schema.json"
        )

    bundle = np.load(npz_path, allow_pickle=True)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    classification_schema = {task["name"]: dict(task) for task in schema.get("classification_tasks", [])}
    regression_schema = {task["name"]: dict(task) for task in schema.get("regression_tasks", [])}

    classification_tasks = []
    for task_name, n_classes in zip(
        [str(x) for x in bundle["classification_task_names"].tolist()],
        [int(x) for x in bundle["classification_num_classes"].tolist()],
    ):
        task_meta = classification_schema.get(task_name, {})
        classification_tasks.append(
            {
                "name": task_name,
                "n_classes": int(n_classes),
                "label_col": str(task_meta.get("label_col", task_name)),
                "mask_col": str(task_meta.get("mask_col", f"mask_{task_name}")),
                "description": str(task_meta.get("description", "")),
            }
        )

    regression_tasks = []
    for task_name in [str(x) for x in bundle["regression_task_names"].tolist()]:
        task_meta = regression_schema.get(task_name, {})
        regression_tasks.append(
            {
                "name": task_name,
                "target_col": str(task_meta.get("target_col", task_name)),
                "mask_col": str(task_meta.get("mask_col", f"mask_{task_name}")),
            }
        )

    return {
        "classification_tasks": classification_tasks,
        "regression_tasks": regression_tasks,
        "classification_labels": bundle["classification_labels"].astype(np.int64),
        "classification_masks": bundle["classification_masks"].astype(np.float32),
        "regression_targets": bundle["regression_targets"].astype(np.float32),
        "regression_masks": bundle["regression_masks"].astype(np.float32),
        "source": "npz",
    }


def _masked_classification_metrics(labels: np.ndarray, probs: np.ndarray, masks: np.ndarray) -> dict:
    valid = masks > 0.5
    observed = int(valid.sum())
    out = {
        "available": observed > 0,
        "observed_samples": observed,
        "macro_f1": None,
        "accuracy": None,
        "auroc": None,
    }
    if observed <= 0:
        return out
    y = labels[valid].astype(int)
    pred = probs[valid].argmax(axis=1)
    out["macro_f1"] = round(float(f1_score(y, pred, average="macro", zero_division=0)), 4)
    out["accuracy"] = round(float((y == pred).mean()), 4)
    if probs.shape[1] == 2 and len(np.unique(y)) > 1:
        out["auroc"] = round(float(roc_auc_score(y, probs[valid][:, 1])), 4)
    return out


def _masked_regression_metrics(targets: np.ndarray, preds: np.ndarray, masks: np.ndarray) -> dict:
    valid = masks > 0.5
    observed = int(valid.sum())
    out = {
        "available": observed > 0,
        "observed_samples": observed,
        "mae": None,
        "rmse": None,
    }
    if observed <= 0:
        return out
    err = preds[valid] - targets[valid]
    out["mae"] = round(float(np.mean(np.abs(err))), 4)
    out["rmse"] = round(float(np.sqrt(np.mean(np.square(err)))), 4)
    return out


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    classification_tasks: list[dict],
    regression_tasks: list[dict],
) -> dict:
    model.eval()
    payload = {
        "mortality_probs": [],
        "mortality_labels": [],
        "classification_probs": {task["name"]: [] for task in classification_tasks},
        "classification_labels": {task["name"]: [] for task in classification_tasks},
        "classification_masks": {task["name"]: [] for task in classification_tasks},
        "regression_preds": {task["name"]: [] for task in regression_tasks},
        "regression_targets": {task["name"]: [] for task in regression_tasks},
        "regression_masks": {task["name"]: [] for task in regression_tasks},
    }
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["x"].to(device),
                batch["mask"].to(device),
                batch["treatments"].to(device),
                batch["treatment_mask"].to(device),
            )
            payload["mortality_probs"].append(torch.sigmoid(out["logits_mortality"]).cpu().numpy())
            payload["mortality_labels"].append(batch["y_mortality"].cpu().numpy())
            for idx, task in enumerate(classification_tasks):
                name = task["name"]
                payload["classification_probs"][name].append(F.softmax(out["classification_logits"][name], dim=-1).cpu().numpy())
                payload["classification_labels"][name].append(batch["y_classification"][:, idx].cpu().numpy())
                payload["classification_masks"][name].append(batch["mask_classification"][:, idx].cpu().numpy())
            for idx, task in enumerate(regression_tasks):
                name = task["name"]
                payload["regression_preds"][name].append(out["regression_outputs"][name].cpu().numpy())
                payload["regression_targets"][name].append(batch["y_regression"][:, idx].cpu().numpy())
                payload["regression_masks"][name].append(batch["mask_regression"][:, idx].cpu().numpy())

    result = {
        "mortality": {
            "probs": np.concatenate(payload["mortality_probs"], axis=0),
            "labels": np.concatenate(payload["mortality_labels"], axis=0),
        },
        "classification": {},
        "regression": {},
    }
    for task in classification_tasks:
        name = task["name"]
        result["classification"][name] = {
            "probs": np.concatenate(payload["classification_probs"][name], axis=0),
            "labels": np.concatenate(payload["classification_labels"][name], axis=0),
            "masks": np.concatenate(payload["classification_masks"][name], axis=0),
        }
    for task in regression_tasks:
        name = task["name"]
        result["regression"][name] = {
            "preds": np.concatenate(payload["regression_preds"][name], axis=0),
            "targets": np.concatenate(payload["regression_targets"][name], axis=0),
            "masks": np.concatenate(payload["regression_masks"][name], axis=0),
        }
    return result


def _evaluate_multitask(preds: dict, threshold: float, classification_tasks: list[dict], regression_tasks: list[dict]) -> dict:
    out = {}
    y = preds["mortality"]["labels"].astype(int)
    p = preds["mortality"]["probs"]
    out["mortality"] = {
        **_classification_metrics(y, p, threshold),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5,
    }
    out["classification"] = {}
    for task in classification_tasks:
        out["classification"][task["name"]] = _masked_classification_metrics(
            preds["classification"][task["name"]]["labels"],
            preds["classification"][task["name"]]["probs"],
            preds["classification"][task["name"]]["masks"],
        )
    out["regression"] = {}
    for task in regression_tasks:
        out["regression"][task["name"]] = _masked_regression_metrics(
            preds["regression"][task["name"]]["targets"],
            preds["regression"][task["name"]]["preds"],
            preds["regression"][task["name"]]["masks"],
        )
    return out


def _score_summary(metrics: dict, classification_tasks: list[dict], regression_tasks: list[dict]) -> float:
    proxy_scores = []
    gold_scores = []
    regression_scores = []
    for task in classification_tasks:
        task_metrics = metrics["classification"][task["name"]]
        if not task_metrics["available"]:
            continue
        if str(task["name"]).startswith("proxy_") and task_metrics["macro_f1"] is not None:
            proxy_scores.append(float(task_metrics["macro_f1"]))
        elif str(task["name"]).startswith("gold_"):
            if task_metrics["auroc"] is not None:
                gold_scores.append(float(task_metrics["auroc"]))
            elif task_metrics["accuracy"] is not None:
                gold_scores.append(float(task_metrics["accuracy"]))
    for task in regression_tasks:
        task_metrics = metrics["regression"][task["name"]]
        if task_metrics["available"] and task_metrics["rmse"] is not None:
            regression_scores.append(1.0 / (1.0 + float(task_metrics["rmse"])))

    total = float(metrics["mortality"]["auroc"])
    if proxy_scores:
        total += float(np.mean(proxy_scores))
    if gold_scores:
        total += 0.5 * float(np.mean(gold_scores))
    if regression_scores:
        total += 0.25 * float(np.mean(regression_scores))
    return total


def _make_task_weights(
    classification_tasks: list[dict],
    regression_tasks: list[dict],
    *,
    lambda_immune: float,
    lambda_organ: float,
    lambda_fluid: float,
    lambda_trajectory: float,
    lambda_gold: float,
    lambda_regression: float,
) -> tuple[dict[str, float], dict[str, float]]:
    classification_weights: dict[str, float] = {}
    for task in classification_tasks:
        name = str(task["name"])
        if name == "proxy_immune_state":
            classification_weights[name] = float(lambda_immune)
        elif name == "proxy_clinical_phenotype":
            classification_weights[name] = float(lambda_organ)
        elif name == "proxy_fluid_strategy":
            classification_weights[name] = float(lambda_fluid)
        elif name == "proxy_trajectory_phenotype":
            classification_weights[name] = float(lambda_trajectory)
        elif name.startswith("gold_"):
            classification_weights[name] = float(lambda_gold)
        else:
            classification_weights[name] = 1.0
    regression_weights = {str(task["name"]): float(lambda_regression) for task in regression_tasks}
    return classification_weights, regression_weights


def _class_criterion(labels: np.ndarray, n_classes: int, *, use_focal_loss: bool, focal_gamma: float, device: str) -> nn.Module:
    weights = _compute_class_weights(labels, n_classes, method="inv_sqrt")
    if use_focal_loss:
        return FocalLoss(gamma=focal_gamma, alpha=None if weights is None else weights.to(device)).to(device)
    return nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return pd.Series(default, index=df.index).astype(float).to_numpy()
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy()


def train_masked_multitask_student(
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
    lambda_trajectory: float = 1.0,
    lambda_gold: float = 1.0,
    lambda_regression: float = 0.25,
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
    phase1_epochs: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    time_series = np.load(data_dir / "time_series_enhanced.npy")
    patient_info = pd.read_csv(data_dir / "patient_info_enhanced.csv")
    n_samples = time_series.shape[0]
    if len(patient_info) != n_samples:
        raise ValueError("patient_info_enhanced.csv length does not match time_series_enhanced.npy")

    target_bundle = _load_schema_bundle(patient_info, data_dir)
    classification_tasks = target_bundle["classification_tasks"]
    regression_tasks = target_bundle["regression_tasks"]
    classification_labels = target_bundle["classification_labels"]
    classification_masks = target_bundle["classification_masks"]
    regression_targets = target_bundle["regression_targets"]
    regression_masks = target_bundle["regression_masks"]

    n_cont = time_series.shape[2] - 1
    continuous = np.nan_to_num(time_series[:, :, :n_cont], nan=0.0).astype(np.float32)
    masks_cont = np.isfinite(time_series[:, :, :n_cont]).astype(np.float32)
    treatments = np.nan_to_num(time_series[:, :, n_cont:], nan=0.0).astype(np.float32)
    masks_treat = np.isfinite(time_series[:, :, n_cont:]).astype(np.float32)

    feat_mean = np.zeros(n_cont, dtype=np.float32)
    feat_std = np.ones(n_cont, dtype=np.float32)
    for f_idx in range(n_cont):
        valid = continuous[:, :, f_idx][masks_cont[:, :, f_idx] > 0]
        if len(valid) > 0:
            feat_mean[f_idx] = float(valid.mean())
            feat_std[f_idx] = max(float(valid.std()), 1.0e-6)
    continuous = (continuous - feat_mean[None, None, :]) / feat_std[None, None, :]
    np.save(output_dir / "feat_mean.npy", feat_mean)
    np.save(output_dir / "feat_std.npy", feat_std)

    labels_mortality = _safe_series(patient_info, "mortality_28d").astype(int)
    from sklearn.model_selection import train_test_split

    idx = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=1 - train_ratio,
        random_state=seed,
        stratify=labels_mortality if len(np.unique(labels_mortality)) > 1 else None,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - train_ratio - val_ratio) / (1 - train_ratio),
        random_state=seed,
        stratify=labels_mortality[temp_idx] if len(np.unique(labels_mortality[temp_idx])) > 1 else None,
    )
    with open(output_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}, f)

    def make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            MultitaskStudentDataset(
                continuous=continuous,
                masks_continuous=masks_cont,
                treatments=treatments,
                masks_treatments=masks_treat,
                labels_mortality=labels_mortality,
                classification_labels=classification_labels,
                classification_masks=classification_masks,
                regression_targets=regression_targets,
                regression_masks=regression_masks,
                indices=indices,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)

    model = MultitaskRealtimeStudentClassifier(
        n_cont_features=n_cont,
        n_treat_features=treatments.shape[-1],
        classification_tasks=classification_tasks,
        regression_tasks=regression_tasks,
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
    ).to(device)

    initialization_summary = _initialize_realtime_student_model(
        model,
        checkpoint_path=init_checkpoint_path,
        strict=init_checkpoint_strict,
        device=device,
    )

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
    smooth_l1 = nn.SmoothL1Loss()

    class_criteria: dict[str, nn.Module] = {}
    for idx_task, task in enumerate(classification_tasks):
        valid = classification_masks[train_idx, idx_task] > 0.5
        valid_labels = classification_labels[train_idx, idx_task][valid]
        class_criteria[task["name"]] = _class_criterion(
            valid_labels if len(valid_labels) > 0 else np.zeros(1, dtype=np.int64),
            int(task["n_classes"]),
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            device=device,
        )

    classification_task_weights, regression_task_weights = _make_task_weights(
        classification_tasks,
        regression_tasks,
        lambda_immune=lambda_immune,
        lambda_organ=lambda_organ,
        lambda_fluid=lambda_fluid,
        lambda_trajectory=lambda_trajectory,
        lambda_gold=lambda_gold,
        lambda_regression=lambda_regression,
    )

    best_state = None
    best_score = -np.inf
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        in_phase1 = epoch <= phase1_epochs and phase1_epochs > 0
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                batch["x"].to(device),
                batch["mask"].to(device),
                batch["treatments"].to(device),
                batch["treatment_mask"].to(device),
            )
            loss = lambda_mortality * bce(out["logits_mortality"], batch["y_mortality"].to(device))

            if not in_phase1:
                for idx_task, task in enumerate(classification_tasks):
                    name = task["name"]
                    valid = batch["mask_classification"][:, idx_task].to(device) > 0.5
                    if torch.any(valid):
                        logits = out["classification_logits"][name][valid]
                        targets = batch["y_classification"][:, idx_task].to(device)[valid]
                        loss = loss + classification_task_weights[name] * class_criteria[name](logits, targets)

                for idx_task, task in enumerate(regression_tasks):
                    name = task["name"]
                    valid = batch["mask_regression"][:, idx_task].to(device) > 0.5
                    if torch.any(valid):
                        preds = out["regression_outputs"][name][valid]
                        targets = batch["y_regression"][:, idx_task].to(device)[valid]
                        loss = loss + regression_task_weights[name] * smooth_l1(preds, targets)

            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_preds = _collect_predictions(model, val_loader, device, classification_tasks, regression_tasks)
        val_metrics = _evaluate_multitask(val_preds, threshold=0.5, classification_tasks=classification_tasks, regression_tasks=regression_tasks)
        val_score = _score_summary(val_metrics, classification_tasks, regression_tasks)
        history_item = {
            "epoch": epoch,
            "time_s": round(time.time() - start_time, 2),
            "train_loss": round(float(np.mean(losses)), 4) if losses else 0.0,
            "val_mortality_auroc": round(float(val_metrics["mortality"]["auroc"]), 4),
            "val_score": round(float(val_score), 4),
        }
        history.append(history_item)

        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            base_cfg = _student_config_dict(
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
            )
            base_cfg["classification_tasks"] = classification_tasks
            base_cfg["regression_tasks"] = regression_tasks
            torch.save({"model_state_dict": best_state, "config": base_cfg}, ckpt_dir / "student_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    val_preds = _collect_predictions(model, val_loader, device, classification_tasks, regression_tasks)
    threshold, threshold_search = _select_threshold(
        val_preds["mortality"]["labels"].astype(int),
        val_preds["mortality"]["probs"],
        metric_name="balanced_accuracy",
    )
    train_metrics = _evaluate_multitask(_collect_predictions(model, train_loader, device, classification_tasks, regression_tasks), threshold, classification_tasks, regression_tasks)
    val_metrics = _evaluate_multitask(val_preds, threshold, classification_tasks, regression_tasks)
    test_metrics = _evaluate_multitask(_collect_predictions(model, test_loader, device, classification_tasks, regression_tasks), threshold, classification_tasks, regression_tasks)

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

    model_cfg = _student_config_dict(
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
    )
    model_cfg["classification_tasks"] = classification_tasks
    model_cfg["regression_tasks"] = regression_tasks

    report = {
        "model": model_cfg,
        "targets_source": target_bundle["source"],
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
            "lambda_trajectory": lambda_trajectory,
            "lambda_gold": lambda_gold,
            "lambda_regression": lambda_regression,
            "use_focal_loss": bool(use_focal_loss),
            "focal_gamma": focal_gamma,
            "phase1_epochs": phase1_epochs,
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
        {"model_state_dict": model.state_dict(), "config": model_cfg, "threshold": threshold},
        output_dir / "multitask_student.pt",
    )
    return report


def get_device(pref: str) -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train masked NPZ-based S6 multitask student")
    parser.add_argument("--data-dir", type=str, default="data/processed_mimic_enhanced")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--student-arch", type=str, default="tcn")
    parser.add_argument("--student-d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-immune", type=float, default=1.0)
    parser.add_argument("--lambda-organ", type=float, default=1.0)
    parser.add_argument("--lambda-fluid", type=float, default=1.0)
    parser.add_argument("--lambda-trajectory", type=float, default=1.0)
    parser.add_argument("--lambda-gold", type=float, default=1.0)
    parser.add_argument("--lambda-regression", type=float, default=0.25)
    parser.add_argument("--lambda-mortality", type=float, default=1.0)
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--phase1-epochs", type=int, default=0)
    parser.add_argument("--init-strict", action="store_true")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    init_checkpoint = PROJECT_ROOT / args.init_checkpoint if args.init_checkpoint else None
    report = train_masked_multitask_student(
        data_dir=data_dir,
        output_dir=output_dir,
        init_checkpoint_path=init_checkpoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        lambda_mortality=args.lambda_mortality,
        lambda_immune=args.lambda_immune,
        lambda_organ=args.lambda_organ,
        lambda_fluid=args.lambda_fluid,
        lambda_trajectory=args.lambda_trajectory,
        lambda_gold=args.lambda_gold,
        lambda_regression=args.lambda_regression,
        init_checkpoint_strict=args.init_strict,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        phase1_epochs=args.phase1_epochs,
        seed=args.seed,
        device=get_device(args.device),
        student_arch=args.student_arch,
        student_d_model=args.student_d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    logger = logging.getLogger("s6.masked.train")
    logger.info("Targets source: %s", report["targets_source"])
    logger.info("Parameter count: %s", report["deployment"]["float_n_parameters"])
    logger.info("Test mortality AUROC: %s", report["splits"]["test"]["mortality"]["auroc"])
    logger.info("Test classification tasks: %s", ", ".join(report["splits"]["test"]["classification"].keys()))
    logger.info("Test regression tasks: %s", ", ".join(report["splits"]["test"]["regression"].keys()))
    logger.info("Output directory: %s", output_dir)


if __name__ == "__main__":
    main()
