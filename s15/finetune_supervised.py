"""
finetune_supervised.py - End-to-end supervised fine-tuning for S1.5.

Purpose:
  Fine-tune the S1.5 encoder directly on a supervised target instead of using a
  frozen embedding followed by a separate shallow classifier. Supports an
  optional auxiliary supervised stage on the bridged Sepsis 2019 dataset before
  the final mortality task on PhysioNet 2012.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from s1.encoder import ICUTransformerEncoder

logger = logging.getLogger("s15.finetune_supervised")


class SupervisedTensorDataset(Dataset):
    """Simple tensor dataset backed by preprocessed S0-style arrays."""

    def __init__(
        self,
        continuous: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
    ):
        self.continuous = continuous[indices].astype(np.float32, copy=False)
        self.masks = masks[indices].astype(np.float32, copy=False)
        self.labels = labels[indices].astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": torch.from_numpy(self.continuous[idx]),
            "mask": torch.from_numpy(self.masks[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class TemporalAttentionPooling(nn.Module):
    """Learned timestep attention on top of encoder sequence outputs."""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = max(d_model // 2, 16)
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, sequence: torch.Tensor, timestep_mask: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(sequence).squeeze(-1)
        valid_steps = timestep_mask > 0.0

        # If a sample is entirely missing, keep the first step valid to avoid NaN softmax.
        no_valid = ~valid_steps.any(dim=1)
        if torch.any(no_valid):
            valid_steps = valid_steps.clone()
            valid_steps[no_valid, 0] = True

        scores = scores.masked_fill(~valid_steps, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class EndToEndICUClassifier(nn.Module):
    """Sequence classifier that fine-tunes the encoder and uses attention pooling."""

    def __init__(
        self,
        *,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.3,
        max_seq_len: int = 48,
    ):
        super().__init__()
        self.encoder = ICUTransformerEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.attention_pool = TemporalAttentionPooling(d_model)
        self.fusion_norm = nn.LayerNorm(2 * d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedding, sequence = self.encoder(x, mask, return_sequence=True)
        timestep_mask = mask.mean(dim=-1)
        attended = self.attention_pool(sequence, timestep_mask)
        fused = self.fusion_norm(torch.cat([embedding, attended], dim=-1))
        return self.head(fused).squeeze(-1)

    def reset_prediction_head(self) -> None:
        for module in (self.fusion_norm, self.head):
            _reset_module(module)


def train_end_to_end_classifier(
    *,
    s0_dir: Path,
    output_dir: Path,
    pretrained_checkpoint: Path | None = None,
    label_col: str = "mortality_inhospital",
    aux_data_dir: Path | None = None,
    aux_label_col: str = "sepsis_label",
    batch_size: int = 128,
    epochs: int = 16,
    aux_epochs: int = 4,
    lr_encoder: float = 2.0e-4,
    lr_head: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    patience: int = 5,
    freeze_encoder_epochs: int = 0,
    grad_clip: float = 1.0,
    threshold_metric: str = "balanced_accuracy",
    monitor_metric: str = "auroc",
    head_hidden_dim: int = 128,
    head_dropout: float = 0.3,
    device: str = "cpu",
    seed: int = 42,
    n_features: int = 21,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
    encoder_dropout: float = 0.2,
) -> dict:
    """
    Fine-tune an encoder end-to-end and evaluate on train/val/test splits.

    If `aux_data_dir` is provided, a short auxiliary supervised stage is run
    first using `aux_label_col`, then the prediction head is reset before the
    final mortality task.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if pretrained_checkpoint is not None:
        ckpt = torch.load(pretrained_checkpoint, map_location=device)
        enc_cfg = ckpt.get("config", {})
        n_features = enc_cfg.get("n_features", n_features)
        d_model = enc_cfg.get("d_model", d_model)
        n_heads = enc_cfg.get("n_heads", n_heads)
        n_layers = enc_cfg.get("n_layers", n_layers)
        d_ff = enc_cfg.get("d_ff", d_ff)
        encoder_dropout = enc_cfg.get("dropout", encoder_dropout)
    else:
        ckpt = None

    model = EndToEndICUClassifier(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=encoder_dropout,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
    ).to(device)

    if ckpt is not None:
        model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        logger.info("Loaded pretrained encoder from %s", pretrained_checkpoint)

    report = {
        "model": {
            "type": "end_to_end_attention_finetune",
            "n_features": n_features,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "encoder_dropout": encoder_dropout,
            "head_hidden_dim": head_hidden_dim,
            "head_dropout": head_dropout,
        },
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "aux_epochs": aux_epochs if aux_data_dir else 0,
            "lr_encoder": lr_encoder,
            "lr_head": lr_head,
            "weight_decay": weight_decay,
            "patience": patience,
            "freeze_encoder_epochs": freeze_encoder_epochs,
            "grad_clip": grad_clip,
            "threshold_metric": threshold_metric,
            "monitor_metric": monitor_metric,
            "seed": seed,
            "device": device,
        },
    }

    if aux_data_dir is not None:
        aux_bundle = _load_supervised_bundle(aux_data_dir, aux_label_col)
        aux_report = _fit_supervised_stage(
            model=model,
            data_bundle=aux_bundle,
            stage_name="auxiliary",
            output_dir=output_dir,
            device=device,
            epochs=aux_epochs,
            batch_size=batch_size,
            lr_encoder=lr_encoder,
            lr_head=lr_head,
            weight_decay=weight_decay,
            patience=max(2, min(patience, aux_epochs)),
            freeze_encoder_epochs=0,
            grad_clip=grad_clip,
            threshold_metric=threshold_metric,
            monitor_metric=monitor_metric,
            seed=seed,
        )
        report["auxiliary_stage"] = aux_report
        model.reset_prediction_head()
        logger.info("Reset prediction head after auxiliary supervision.")

    main_bundle = _load_supervised_bundle(s0_dir, label_col)
    main_report = _fit_supervised_stage(
        model=model,
        data_bundle=main_bundle,
        stage_name="main",
        output_dir=output_dir,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        weight_decay=weight_decay,
        patience=patience,
        freeze_encoder_epochs=freeze_encoder_epochs,
        grad_clip=grad_clip,
        threshold_metric=threshold_metric,
        monitor_metric=monitor_metric,
        seed=seed,
    )
    report["main_task"] = main_report

    report_path = output_dir / "finetune_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved fine-tune report to %s", report_path)
    return report


def _fit_supervised_stage(
    *,
    model: EndToEndICUClassifier,
    data_bundle: dict,
    stage_name: str,
    output_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr_encoder: float,
    lr_head: float,
    weight_decay: float,
    patience: int,
    freeze_encoder_epochs: int,
    grad_clip: float,
    threshold_metric: str,
    monitor_metric: str,
    seed: int,
) -> dict:
    split_arrays = data_bundle["splits"]
    labels = data_bundle["labels"]

    train_ds = SupervisedTensorDataset(
        data_bundle["continuous"],
        data_bundle["masks"],
        labels,
        split_arrays["train"],
    )
    val_ds = SupervisedTensorDataset(
        data_bundle["continuous"],
        data_bundle["masks"],
        labels,
        split_arrays["val"],
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    train_labels = labels[split_arrays["train"]]
    pos_weight = _compute_pos_weight(train_labels)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr_encoder},
            {
                "params": list(model.attention_pool.parameters())
                + list(model.fusion_norm.parameters())
                + list(model.head.parameters()),
                "lr": lr_head,
            },
        ],
        weight_decay=weight_decay,
    )

    best_metric = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_threshold = 0.5
    best_val_metrics = None
    best_val_probs = None
    history = []
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        _set_encoder_trainable(model, epoch > freeze_encoder_epochs)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        train_probs = _predict_probs(model, train_loader, device)
        val_probs = _predict_probs(model, val_loader, device)
        threshold, threshold_search = _select_threshold(
            labels[split_arrays["val"]],
            val_probs,
            metric_name=threshold_metric,
        )
        train_metrics = _classification_metrics(labels[split_arrays["train"]], train_probs, threshold)
        val_metrics = _classification_metrics(labels[split_arrays["val"]], val_probs, threshold)

        current_metric = val_metrics[monitor_metric]
        if current_metric is None:
            current_metric = -np.inf

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss / max(n_batches, 1), 4),
            "threshold": round(float(threshold), 4),
            "train": train_metrics,
            "val": val_metrics,
            "threshold_search": threshold_search,
            "time_s": round(time.time() - t0, 2),
        })

        logger.info(
            "%s epoch %d/%d | loss=%.4f | val acc=%.4f bal_acc=%.4f auroc=%s thr=%.2f",
            stage_name,
            epoch,
            epochs,
            epoch_loss / max(n_batches, 1),
            val_metrics["accuracy"],
            val_metrics["balanced_accuracy"],
            val_metrics["auroc"],
            threshold,
        )

        if current_metric > best_metric:
            best_metric = float(current_metric)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_threshold = float(threshold)
            best_val_metrics = val_metrics
            best_val_probs = val_probs
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("%s early stopping at epoch %d", stage_name, epoch)
                break

    model.load_state_dict(best_state)
    stage_ckpt = output_dir / "checkpoints" / f"supervised_{stage_name}_best.pt"
    torch.save(
        {
            "stage_name": stage_name,
            "best_epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "threshold": best_threshold,
            "val_metrics": best_val_metrics,
        },
        stage_ckpt,
    )

    split_reports = {}
    for split_name in ("train", "val", "test"):
        idx = split_arrays[split_name]
        loader = DataLoader(
            SupervisedTensorDataset(
                data_bundle["continuous"],
                data_bundle["masks"],
                labels,
                idx,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        split_probs = _predict_probs(model, loader, device)
        split_reports[split_name] = _classification_metrics(labels[idx], split_probs, best_threshold)

    history_path = output_dir / f"supervised_{stage_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    threshold_search = _select_threshold(
        labels[split_arrays["val"]],
        best_val_probs if best_val_probs is not None else np.full(len(split_arrays["val"]), 0.5),
        metric_name=threshold_metric,
    )[1]
    return {
        "stage_name": stage_name,
        "label_col": data_bundle["label_col"],
        "n_samples": int(len(labels)),
        "threshold_selection": {
            "metric": threshold_metric,
            "selected_threshold": round(float(best_threshold), 4),
            "search": threshold_search,
        },
        "best_epoch": int(best_epoch),
        "monitor_metric": monitor_metric,
        "best_val_metric": round(float(best_metric), 4),
        "best_checkpoint": str(stage_ckpt),
        "history_path": str(history_path),
        "splits": split_reports,
        "baseline_accuracy": _baseline_accuracy(labels, split_arrays),
    }


def _load_supervised_bundle(data_dir: Path, label_col: str) -> dict:
    data_dir = Path(data_dir)
    static = pd.read_csv(data_dir / "static.csv")
    labels = pd.to_numeric(static[label_col], errors="coerce").fillna(0).astype(int).to_numpy()

    with open(data_dir / "splits.json", encoding="utf-8") as f:
        splits_raw = json.load(f)
    splits = {
        name: np.asarray(splits_raw[name], dtype=int)
        for name in ("train", "val", "test")
    }

    bundle = {
        "continuous": np.load(data_dir / "processed" / "continuous.npy"),
        "masks": np.load(data_dir / "processed" / "masks_continuous.npy"),
        "labels": labels,
        "splits": splits,
        "label_col": label_col,
    }
    _validate_splits(bundle["labels"], bundle["splits"])
    return bundle


def _predict_probs(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.train(False)
    outputs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device), batch["mask"].to(device))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)


def _set_encoder_trainable(model: EndToEndICUClassifier, trainable: bool) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = trainable


def _compute_pos_weight(labels: np.ndarray) -> float:
    pos = float(np.sum(labels == 1))
    neg = float(np.sum(labels == 0))
    if pos <= 0:
        return 1.0
    return max(neg / pos, 1.0)


def _validate_splits(labels: np.ndarray, splits: dict[str, np.ndarray]) -> None:
    for name, idx in splits.items():
        if idx.size == 0:
            raise ValueError(f"Split '{name}' is empty")
        if np.any(idx < 0) or np.any(idx >= len(labels)):
            raise ValueError(f"Split '{name}' contains out-of-range indices")
        if name in {"train", "val"} and len(np.unique(labels[idx])) < 2:
            raise ValueError(f"Split '{name}' must contain both classes")


def _reset_module(module: nn.Module) -> None:
    for child in module.modules():
        if child is module:
            continue
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def _baseline_accuracy(labels: np.ndarray, split_arrays: dict[str, np.ndarray]) -> dict[str, float]:
    majority_label = int(round(float(labels[split_arrays["train"]].mean())))
    return {
        name: round(float(np.mean(labels[idx] == majority_label)), 4)
        for name, idx in split_arrays.items()
    }


def _select_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    metric_name: str,
) -> tuple[float, list[dict]]:
    if len(np.unique(y_true)) < 2:
        return 0.5, [{"threshold": 0.5, metric_name: None, "note": "single class in validation"}]

    candidates = np.linspace(0.05, 0.95, 37)
    best_threshold = 0.5
    best_score = -np.inf
    search = []

    for threshold in candidates:
        preds = (probs >= threshold).astype(int)
        score = _threshold_metric(y_true, preds, metric_name)
        search.append({
            "threshold": round(float(threshold), 4),
            metric_name: round(float(score), 4),
        })
        if score > best_score or (np.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, search


def _threshold_metric(y_true: np.ndarray, preds: np.ndarray, metric_name: str) -> float:
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, preds))
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, preds))
    if metric_name == "f1":
        return float(f1_score(y_true, preds, zero_division=0))
    raise ValueError(f"Unsupported threshold metric: {metric_name}")


def _classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    metrics = {
        "n_samples": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "predicted_positive_rate": round(float(np.mean(preds)), 4),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    if len(np.unique(y_true)) >= 2:
        metrics["auroc"] = round(float(roc_auc_score(y_true, probs)), 4)
    else:
        metrics["auroc"] = None
    return metrics
