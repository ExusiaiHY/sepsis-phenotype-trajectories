"""
treatment_aware_model.py - Stage 4 treatment-aware extension on top of S1.5.

Design goal:
  keep the original S1.5 self-supervised encoder intact and add a light-weight
  treatment branch that can be fused into the temporal representation without
  re-architecting the whole model.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from s1.encoder import ICUTransformerEncoder
from s15.classification_eval import _classification_metrics, _select_threshold

logger = logging.getLogger("s4.treatment_aware")


class TreatmentAwareTensorDataset(Dataset):
    """Aligned dataset for physiology, treatments, optional notes, and labels."""

    def __init__(
        self,
        *,
        continuous: np.ndarray,
        masks_continuous: np.ndarray,
        treatments: np.ndarray,
        masks_treatments: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        note_embeddings: np.ndarray | None = None,
    ):
        self.continuous = continuous[indices].astype(np.float32, copy=False)
        self.masks_continuous = masks_continuous[indices].astype(np.float32, copy=False)
        self.treatments = treatments[indices].astype(np.float32, copy=False)
        self.masks_treatments = masks_treatments[indices].astype(np.float32, copy=False)
        self.labels = labels[indices].astype(np.float32, copy=False)
        self.note_embeddings = None
        if note_embeddings is not None:
            notes = note_embeddings[indices].astype(np.float32, copy=False)
            if notes.ndim == 2:
                notes = np.repeat(notes[:, None, :], self.continuous.shape[1], axis=1)
            self.note_embeddings = notes

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        batch = {
            "x": torch.from_numpy(self.continuous[idx]),
            "mask": torch.from_numpy(self.masks_continuous[idx]),
            "treatments": torch.from_numpy(self.treatments[idx]),
            "treatment_mask": torch.from_numpy(self.masks_treatments[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        if self.note_embeddings is not None:
            batch["notes"] = torch.from_numpy(self.note_embeddings[idx])
        return batch


class TemporalAttentionPooling(nn.Module):
    """Learned timestep attention over fused sequence states."""

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
        no_valid = ~valid_steps.any(dim=1)
        if torch.any(no_valid):
            valid_steps = valid_steps.clone()
            valid_steps[no_valid, 0] = True
        scores = scores.masked_fill(~valid_steps, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class TreatmentAwareEncoder(nn.Module):
    """
    S1.5-compatible encoder with a parallel treatment adapter branch.

    The base physiology branch is the original ICUTransformerEncoder. Treatments
    and optional text embeddings are projected into the same latent dimension and
    fused with a learned gate.
    """

    def __init__(
        self,
        *,
        n_cont_features: int = 21,
        n_treat_features: int = 7,
        note_dim: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 48,
        treatment_layers: int = 1,
    ):
        super().__init__()
        self.note_dim = note_dim
        self.base_encoder = ICUTransformerEncoder(
            n_features=n_cont_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        adapter_input_dim = (2 * n_treat_features) + note_dim
        self.treatment_proj = nn.Linear(adapter_input_dim, d_model)
        treatment_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, min(n_heads, 4)),
            dim_feedforward=max(d_ff // 2, d_model),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.treatment_transformer = nn.TransformerEncoder(
            treatment_layer,
            num_layers=max(1, int(treatment_layers)),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def load_pretrained_base(self, checkpoint_path: Path, device: str = "cpu") -> dict:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.base_encoder.load_state_dict(ckpt["encoder_state_dict"])
        return ckpt.get("config", {})

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        treatments: torch.Tensor,
        treatment_mask: torch.Tensor,
        note_embeddings: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        base_embedding, base_sequence = self.base_encoder(x, mask, return_sequence=True)
        pieces = [treatments, treatment_mask]
        if note_embeddings is not None:
            pieces.append(note_embeddings)
        adapter_input = torch.cat(pieces, dim=-1)
        treatment_sequence = self.treatment_proj(adapter_input)
        treatment_sequence = treatment_sequence + self.base_encoder.pos_embedding[:, : x.shape[1], :]
        treatment_sequence = self.treatment_transformer(treatment_sequence)

        gate = self.gate(torch.cat([base_sequence, treatment_sequence], dim=-1))
        fused_sequence = self.output_norm(base_sequence + gate * treatment_sequence)

        density = mask.mean(dim=-1, keepdim=True) + 0.5 * treatment_mask.mean(dim=-1, keepdim=True)
        if note_embeddings is not None:
            note_obs = (note_embeddings.abs().sum(dim=-1, keepdim=True) > 0).float()
            density = density + 0.25 * note_obs
        weights = density.clamp(min=1e-6)
        embedding = (fused_sequence * weights).sum(dim=1) / weights.sum(dim=1)

        if return_sequence:
            return embedding, fused_sequence
        return embedding


class TreatmentAwareClassifier(nn.Module):
    """Treatment-aware mortality classifier with attention-pooled temporal head."""

    def __init__(
        self,
        *,
        n_cont_features: int = 21,
        n_treat_features: int = 7,
        note_dim: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 48,
        treatment_layers: int = 1,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = TreatmentAwareEncoder(
            n_cont_features=n_cont_features,
            n_treat_features=n_treat_features,
            note_dim=note_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            treatment_layers=treatment_layers,
        )
        self.attention_pool = TemporalAttentionPooling(d_model)
        self.fusion_norm = nn.LayerNorm(2 * d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, head_hidden_dim),
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
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding, sequence = self.encoder(
            x,
            mask,
            treatments,
            treatment_mask,
            note_embeddings=note_embeddings,
            return_sequence=True,
        )
        timestep_mask = mask.mean(dim=-1) + 0.5 * treatment_mask.mean(dim=-1)
        attended = self.attention_pool(sequence, timestep_mask)
        fused = self.fusion_norm(torch.cat([embedding, attended], dim=-1))
        logits = self.head(fused).squeeze(-1)
        if return_embedding:
            return logits, embedding
        return logits


def train_treatment_aware_classifier(
    *,
    s0_dir: Path,
    treatment_dir: Path,
    output_dir: Path,
    splits_path: Path | None = None,
    pretrained_checkpoint: Path | None = None,
    note_embeddings_path: Path | None = None,
    label_col: str = "mortality_inhospital",
    batch_size: int = 128,
    epochs: int = 12,
    lr_encoder: float = 2.0e-4,
    lr_head: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    patience: int = 4,
    freeze_base_epochs: int = 0,
    grad_clip: float = 1.0,
    threshold_metric: str = "balanced_accuracy",
    monitor_metric: str = "auroc",
    seed: int = 42,
    device: str = "cpu",
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.2,
    treatment_layers: int = 1,
    head_hidden_dim: int = 128,
    head_dropout: float = 0.2,
) -> dict:
    """Supervised Stage 4 training entry point."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    s0_dir = Path(s0_dir)
    treatment_dir = Path(treatment_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    continuous = np.load(s0_dir / "processed" / "continuous.npy")
    masks_cont = np.load(s0_dir / "processed" / "masks_continuous.npy")
    treatments = np.load(treatment_dir / "treatments.npy")
    masks_treat = np.load(treatment_dir / "masks_treatments.npy")
    note_embeddings = np.load(note_embeddings_path) if note_embeddings_path is not None else None

    s0_static = pd.read_csv(s0_dir / "static.csv")
    treatment_static = pd.read_csv(treatment_dir / "cohort_static.csv")
    aligned = _align_s0_and_treatment_arrays(
        s0_static=s0_static,
        treatment_static=treatment_static,
        continuous=continuous,
        masks_continuous=masks_cont,
        treatments=treatments,
        masks_treatments=masks_treat,
        note_embeddings=note_embeddings,
    )
    continuous = aligned["continuous"]
    masks_cont = aligned["masks_continuous"]
    treatments = aligned["treatments"]
    masks_treat = aligned["masks_treatments"]
    note_embeddings = aligned["note_embeddings"]
    aligned_s0_static = aligned["s0_static"]
    aligned_treatment_static = aligned["treatment_static"]

    label_frame = aligned_s0_static if label_col in aligned_s0_static.columns else aligned_treatment_static
    if label_col not in label_frame.columns:
        raise ValueError(f"Label column '{label_col}' not found in aligned S0 or treatment cohort")
    labels = label_frame[label_col].fillna(0).astype(int).to_numpy()
    n_samples = len(labels)

    splits = _load_or_build_splits(
        n_samples=n_samples,
        labels=labels,
        splits_path=splits_path or (s0_dir / "splits.json"),
        seed=seed,
    )

    train_ds = TreatmentAwareTensorDataset(
        continuous=continuous,
        masks_continuous=masks_cont,
        treatments=treatments,
        masks_treatments=masks_treat,
        labels=labels,
        indices=splits["train"],
        note_embeddings=note_embeddings,
    )
    val_ds = TreatmentAwareTensorDataset(
        continuous=continuous,
        masks_continuous=masks_cont,
        treatments=treatments,
        masks_treatments=masks_treat,
        labels=labels,
        indices=splits["val"],
        note_embeddings=note_embeddings,
    )
    test_ds = TreatmentAwareTensorDataset(
        continuous=continuous,
        masks_continuous=masks_cont,
        treatments=treatments,
        masks_treatments=masks_treat,
        labels=labels,
        indices=splits["test"],
        note_embeddings=note_embeddings,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    note_dim = 0
    if note_embeddings is not None:
        note_dim = note_embeddings.shape[-1] if note_embeddings.ndim == 3 else note_embeddings.shape[-1]

    model = TreatmentAwareClassifier(
        n_cont_features=continuous.shape[-1],
        n_treat_features=treatments.shape[-1],
        note_dim=note_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=continuous.shape[1],
        treatment_layers=treatment_layers,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
    ).to(device)

    pretrained_cfg = {}
    if pretrained_checkpoint is not None:
        pretrained_cfg = model.encoder.load_pretrained_base(pretrained_checkpoint, device=device)
        logger.info("Loaded Stage 1.5 base encoder from %s", pretrained_checkpoint)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.base_encoder.parameters(), "lr": lr_encoder},
            {"params": model.encoder.treatment_proj.parameters(), "lr": lr_head},
            {"params": model.encoder.treatment_transformer.parameters(), "lr": lr_head},
            {"params": model.encoder.gate.parameters(), "lr": lr_head},
            {"params": model.encoder.output_norm.parameters(), "lr": lr_head},
            {"params": model.attention_pool.parameters(), "lr": lr_head},
            {"params": model.fusion_norm.parameters(), "lr": lr_head},
            {"params": model.head.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )

    best_state = None
    best_score = -np.inf
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        if epoch <= freeze_base_epochs:
            for param in model.encoder.base_encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.encoder.base_encoder.parameters():
                param.requires_grad = True

        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["x"].to(device),
                batch["mask"].to(device),
                batch["treatments"].to(device),
                batch["treatment_mask"].to(device),
                note_embeddings=batch.get("notes").to(device) if "notes" in batch else None,
            )
            y = batch["y"].to(device)
            loss = loss_fn(logits, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        train_probs, train_y, train_loss = _predict_probs(model, train_loader, loss_fn, device)
        val_probs, val_y, val_loss = _predict_probs(model, val_loader, loss_fn, device)
        train_metric = _monitor_metric(train_y, train_probs, monitor_metric)
        val_metric = _monitor_metric(val_y, val_probs, monitor_metric)

        history.append(
            {
                "epoch": epoch,
                "time_s": round(time.time() - t0, 2),
                "train_loss": round(float(train_loss), 4),
                "val_loss": round(float(val_loss), 4),
                "train_metric": round(float(train_metric), 4),
                "val_metric": round(float(val_metric), 4),
            }
        )

        if val_metric > best_score:
            best_score = val_metric
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": _model_config_dict(
                        model=model,
                        n_cont_features=continuous.shape[-1],
                        n_treat_features=treatments.shape[-1],
                        note_dim=note_dim,
                        d_model=d_model,
                        n_heads=n_heads,
                        n_layers=n_layers,
                        d_ff=d_ff,
                        dropout=dropout,
                        treatment_layers=treatment_layers,
                        max_seq_len=continuous.shape[1],
                        pretrained_base_config=pretrained_cfg,
                    ),
                    "best_metric": best_score,
                    "monitor_metric": monitor_metric,
                },
                ckpt_dir / "treatment_aware_best.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    train_probs, train_y, _ = _predict_probs(model, train_loader, loss_fn, device)
    val_probs, val_y, _ = _predict_probs(model, val_loader, loss_fn, device)
    test_probs, test_y, _ = _predict_probs(model, test_loader, loss_fn, device)
    threshold, threshold_search = _select_threshold(val_y.astype(int), val_probs, metric_name=threshold_metric)

    report = {
        "model": _model_config_dict(
            model=model,
            n_cont_features=continuous.shape[-1],
            n_treat_features=treatments.shape[-1],
            note_dim=note_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            treatment_layers=treatment_layers,
            max_seq_len=continuous.shape[1],
            pretrained_base_config=pretrained_cfg,
        ),
        "training": {
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "epochs_trained": len(history),
            "lr_encoder": lr_encoder,
            "lr_head": lr_head,
            "weight_decay": weight_decay,
            "freeze_base_epochs": freeze_base_epochs,
            "patience": patience,
            "grad_clip": grad_clip,
            "monitor_metric": monitor_metric,
            "threshold_metric": threshold_metric,
            "device": device,
            "seed": seed,
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
    }

    with open(output_dir / "treatment_aware_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": report["model"],
            "threshold": threshold,
        },
        output_dir / "treatment_aware_classifier.pt",
    )
    logger.info("Saved Stage 4 treatment-aware report to %s", output_dir / "treatment_aware_report.json")
    return report


def extract_treatment_aware_embeddings(
    *,
    s0_dir: Path,
    treatment_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    note_embeddings_path: Path | None = None,
    device: str = "cpu",
    batch_size: int = 128,
) -> np.ndarray:
    """Extract patient-level treatment-aware embeddings from a trained checkpoint."""
    model = _load_classifier_checkpoint(checkpoint_path, device=device)
    model.eval()

    continuous = np.load(Path(s0_dir) / "processed" / "continuous.npy", mmap_mode="r")
    masks_cont = np.load(Path(s0_dir) / "processed" / "masks_continuous.npy", mmap_mode="r")
    treatments = np.load(Path(treatment_dir) / "treatments.npy", mmap_mode="r")
    masks_treat = np.load(Path(treatment_dir) / "masks_treatments.npy", mmap_mode="r")
    notes = np.load(note_embeddings_path, mmap_mode="r") if note_embeddings_path is not None else None

    aligned = _align_s0_and_treatment_arrays(
        s0_static=pd.read_csv(Path(s0_dir) / "static.csv"),
        treatment_static=pd.read_csv(Path(treatment_dir) / "cohort_static.csv"),
        continuous=continuous,
        masks_continuous=masks_cont,
        treatments=treatments,
        masks_treatments=masks_treat,
        note_embeddings=notes,
    )
    continuous = aligned["continuous"]
    masks_cont = aligned["masks_continuous"]
    treatments = aligned["treatments"]
    masks_treat = aligned["masks_treatments"]
    notes = aligned["note_embeddings"]
    n_samples = len(continuous)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.empty((n_samples, model.encoder.base_encoder.d_model), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            note_batch = None
            if notes is not None:
                note_batch = np.asarray(notes[start:end], dtype=np.float32)
                if note_batch.ndim == 2:
                    note_batch = np.repeat(note_batch[:, None, :], continuous.shape[1], axis=1)
                note_batch = torch.from_numpy(note_batch).to(device)

            _, emb = model(
                torch.from_numpy(np.asarray(continuous[start:end], dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(masks_cont[start:end], dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(treatments[start:end], dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(masks_treat[start:end], dtype=np.float32)).to(device),
                note_embeddings=note_batch,
                return_embedding=True,
            )
            embeddings[start:end] = emb.detach().cpu().numpy().astype(np.float32, copy=False)

    np.save(output_path, embeddings)
    return embeddings


def extract_treatment_aware_rolling_embeddings(
    *,
    s0_dir: Path,
    treatment_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    note_embeddings_path: Path | None = None,
    window_len: int = 24,
    stride: int = 6,
    seq_len: int = 48,
    device: str = "cpu",
    batch_size: int = 128,
) -> tuple[np.ndarray, dict]:
    """Extract rolling-window treatment-aware embeddings for trajectory analysis."""
    model = _load_classifier_checkpoint(checkpoint_path, device=device)
    model.eval()

    continuous = np.load(Path(s0_dir) / "processed" / "continuous.npy", mmap_mode="r")
    masks_cont = np.load(Path(s0_dir) / "processed" / "masks_continuous.npy", mmap_mode="r")
    treatments = np.load(Path(treatment_dir) / "treatments.npy", mmap_mode="r")
    masks_treat = np.load(Path(treatment_dir) / "masks_treatments.npy", mmap_mode="r")
    notes = np.load(note_embeddings_path, mmap_mode="r") if note_embeddings_path is not None else None

    aligned = _align_s0_and_treatment_arrays(
        s0_static=pd.read_csv(Path(s0_dir) / "static.csv"),
        treatment_static=pd.read_csv(Path(treatment_dir) / "cohort_static.csv"),
        continuous=continuous,
        masks_continuous=masks_cont,
        treatments=treatments,
        masks_treatments=masks_treat,
        note_embeddings=notes,
    )
    continuous = aligned["continuous"]
    masks_cont = aligned["masks_continuous"]
    treatments = aligned["treatments"]
    masks_treat = aligned["masks_treatments"]
    notes = aligned["note_embeddings"]
    n_samples = len(continuous)
    starts = list(range(0, seq_len - window_len + 1, stride))
    n_windows = len(starts)
    d_model = model.encoder.base_encoder.d_model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_embeddings = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, n_windows, d_model),
    )

    with torch.no_grad():
        for wi, start in enumerate(starts):
            end_window = start + window_len
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                note_batch = None
                if notes is not None:
                    note_batch = np.asarray(notes[batch_start:batch_end], dtype=np.float32)
                    if note_batch.ndim == 2:
                        note_batch = np.repeat(note_batch[:, None, :], window_len, axis=1)
                    else:
                        note_batch = note_batch[:, start:end_window, :]
                    note_batch = torch.from_numpy(note_batch).to(device)
                _, emb = model(
                    torch.from_numpy(np.asarray(continuous[batch_start:batch_end, start:end_window, :], dtype=np.float32)).to(device),
                    torch.from_numpy(np.asarray(masks_cont[batch_start:batch_end, start:end_window, :], dtype=np.float32)).to(device),
                    torch.from_numpy(np.asarray(treatments[batch_start:batch_end, start:end_window, :], dtype=np.float32)).to(device),
                    torch.from_numpy(np.asarray(masks_treat[batch_start:batch_end, start:end_window, :], dtype=np.float32)).to(device),
                    note_embeddings=note_batch,
                    return_embedding=True,
                )
                all_embeddings[batch_start:batch_end, wi, :] = emb.detach().cpu().numpy().astype(np.float32, copy=False)

    all_embeddings.flush()
    meta = {
        "n_samples": n_samples,
        "window_len": window_len,
        "stride": stride,
        "window_starts": starts,
        "shape": [int(n_samples), int(n_windows), int(d_model)],
    }
    return np.load(output_path, mmap_mode="r"), meta


def _predict_probs(
    model: TreatmentAwareClassifier,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["x"].to(device),
                batch["mask"].to(device),
                batch["treatments"].to(device),
                batch["treatment_mask"].to(device),
                note_embeddings=batch.get("notes").to(device) if "notes" in batch else None,
            )
            y = batch["y"].to(device)
            loss = loss_fn(logits, y)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy())
            losses.append(float(loss.item()))
    return (
        np.concatenate(probs, axis=0),
        np.concatenate(labels, axis=0),
        float(np.mean(losses)) if losses else 0.0,
    )


def _monitor_metric(y_true: np.ndarray, probs: np.ndarray, metric_name: str) -> float:
    y_pred = (probs >= 0.5).astype(int)
    if metric_name == "auroc":
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, probs))
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError(f"Unsupported monitor_metric: {metric_name}")


def _calibration_report(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true).astype(float)
    probs = np.asarray(probs).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lower) & (probs < upper)
        else:
            mask = (probs >= lower) & (probs <= upper)
        if not np.any(mask):
            continue
        frac = float(np.mean(mask))
        mean_prob = float(np.mean(probs[mask]))
        positive_rate = float(np.mean(y_true[mask]))
        ece += abs(positive_rate - mean_prob) * frac
        rows.append(
            {
                "bin": i,
                "n": int(np.sum(mask)),
                "mean_prob": round(mean_prob, 4),
                "positive_rate": round(positive_rate, 4),
            }
        )
    brier = float(np.mean((probs - y_true) ** 2))
    return {
        "brier": round(brier, 4),
        "ece": round(float(ece), 4),
        "bins": rows,
    }


def _load_or_build_splits(
    *,
    n_samples: int,
    labels: np.ndarray,
    splits_path: Path,
    seed: int,
) -> dict[str, np.ndarray]:
    splits_path = Path(splits_path)
    if splits_path.exists():
        with open(splits_path, encoding="utf-8") as f:
            raw = json.load(f)
        split_arrays = {}
        for name in ("train", "val", "test"):
            arr = np.asarray(raw[name], dtype=int)
            arr = arr[arr < n_samples]
            split_arrays[name] = arr
        if all(len(arr) > 0 for arr in split_arrays.values()):
            return split_arrays

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    positives = indices[labels == 1]
    negatives = indices[labels == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def _split_group(group: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(group)
        n_train = max(1, int(round(n * 0.7)))
        n_val = max(1, int(round(n * 0.15)))
        train = group[:n_train]
        val = group[n_train:n_train + n_val]
        test = group[n_train + n_val:]
        if len(test) == 0:
            test = group[-1:]
            val = group[n_train - 1:n_train] if len(val) == 0 else val
        return train, val, test

    pos_train, pos_val, pos_test = _split_group(positives)
    neg_train, neg_val, neg_test = _split_group(negatives)
    return {
        "train": np.concatenate([pos_train, neg_train]),
        "val": np.concatenate([pos_val, neg_val]),
        "test": np.concatenate([pos_test, neg_test]),
    }


def _load_classifier_checkpoint(checkpoint_path: Path, device: str) -> TreatmentAwareClassifier:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TreatmentAwareClassifier(
        n_cont_features=cfg["n_cont_features"],
        n_treat_features=cfg["n_treat_features"],
        note_dim=cfg["note_dim"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_seq_len=cfg["max_seq_len"],
        treatment_layers=cfg["treatment_layers"],
        head_hidden_dim=cfg["head_hidden_dim"],
        head_dropout=cfg["head_dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _align_s0_and_treatment_arrays(
    *,
    s0_static: pd.DataFrame,
    treatment_static: pd.DataFrame,
    continuous: np.ndarray,
    masks_continuous: np.ndarray,
    treatments: np.ndarray,
    masks_treatments: np.ndarray,
    note_embeddings: np.ndarray | None = None,
) -> dict:
    s0_static = s0_static.copy().reset_index(drop=True)
    treatment_static = treatment_static.copy().reset_index(drop=True)
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
        raise ValueError("No overlapping patient_id values between S0 cohort and treatment cohort")

    s0_keep_arr = np.asarray(s0_keep, dtype=int)
    treat_keep_arr = np.asarray(treat_keep, dtype=int)
    aligned_notes = None
    if note_embeddings is not None:
        if len(note_embeddings) == len(treatment_static):
            aligned_notes = np.asarray(note_embeddings[treat_keep_arr])
        elif len(note_embeddings) == len(s0_static):
            aligned_notes = np.asarray(note_embeddings[s0_keep_arr])
        else:
            raise ValueError("note_embeddings length does not match S0 or treatment cohort")

    return {
        "continuous": np.asarray(continuous[s0_keep_arr]),
        "masks_continuous": np.asarray(masks_continuous[s0_keep_arr]),
        "treatments": np.asarray(treatments[treat_keep_arr]),
        "masks_treatments": np.asarray(masks_treatments[treat_keep_arr]),
        "note_embeddings": aligned_notes,
        "s0_static": s0_static.iloc[s0_keep_arr].reset_index(drop=True),
        "treatment_static": treatment_static.iloc[treat_keep_arr].reset_index(drop=True),
    }


def _model_config_dict(
    *,
    model: TreatmentAwareClassifier,
    n_cont_features: int,
    n_treat_features: int,
    note_dim: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    treatment_layers: int,
    max_seq_len: int,
    pretrained_base_config: dict,
) -> dict:
    return {
        "type": "treatment_aware_s15_fusion",
        "n_cont_features": n_cont_features,
        "n_treat_features": n_treat_features,
        "note_dim": note_dim,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "dropout": dropout,
        "treatment_layers": treatment_layers,
        "max_seq_len": max_seq_len,
        "head_hidden_dim": model.head[0].out_features,
        "head_dropout": float(model.head[2].p),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "pretrained_base_config": pretrained_base_config,
    }
