"""
pretrain_contrastive.py - Pretraining loop with lambda warmup + monitoring.

Loss: L_total = L_masked + lambda(epoch) * L_contrastive
Lambda warmup: 0.0 → 0.5 over first 10 epochs, then constant 0.5.

Per-epoch monitoring:
  - train/val masked loss, contrastive loss, total loss
  - lambda value
  - cosine similarity (positive pairs, negative pairs)
  - embedding norm (mean, std)
  - batch composition (obs density, center fraction, mortality fraction)

Alignment/uniformity computed at checkpoints (every 5 epochs).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from s15.contrastive_encoder import ContrastivePretrainModel

logger = logging.getLogger("s15.pretrain")


class S15Dataset(Dataset):
    """Dataset with continuous, masks, and metadata for monitoring."""

    def __init__(self, continuous, masks, static_df, indices):
        self.continuous = continuous[indices]
        self.masks = masks[indices]
        self.mortality = static_df.iloc[indices]["mortality_inhospital"].fillna(0).values.astype(np.float32)
        self.center = (static_df.iloc[indices]["center_id"] == "center_a").values.astype(np.float32)
        self.obs_density = masks[indices].mean(axis=(1, 2)).astype(np.float32)

    def __len__(self):
        return len(self.continuous)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.continuous[idx]).float(),
            "mask": torch.from_numpy(self.masks[idx]).float(),
            "mortality": self.mortality[idx],
            "center_a": self.center[idx],
            "obs_density": self.obs_density[idx],
        }


def lambda_schedule(epoch: int, warmup_epochs: int = 10, max_lambda: float = 0.5) -> float:
    """Linear warmup from 0 to max_lambda over warmup_epochs."""
    if epoch <= 0:
        return 0.0
    return min(max_lambda, max_lambda * epoch / warmup_epochs)


def pretrain_contrastive(
    s0_dir: Path,
    output_dir: Path,
    n_features: int = 21,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.2,
    view_len: int = 30,
    mask_ratio: float = 0.15,
    temperature: float = 0.1,
    proj_dim: int = 64,
    max_lambda: float = 0.5,
    warmup_epochs: int = 10,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    grad_clip: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Run masked + contrastive pretraining. Returns log dict."""
    s0_dir = Path(s0_dir)
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    continuous = np.load(s0_dir / "processed" / "continuous.npy")
    masks = np.load(s0_dir / "processed" / "masks_continuous.npy")
    static = pd.read_csv(s0_dir / "static.csv")

    with open(s0_dir / "splits.json") as f:
        splits = json.load(f)

    train_ds = S15Dataset(continuous, masks, static, np.array(splits["train"]))
    val_ds = S15Dataset(continuous, masks, static, np.array(splits["val"]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = ContrastivePretrainModel(
        n_features=n_features, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, dropout=dropout,
        view_len=view_len, mask_ratio=mask_ratio,
        temperature=temperature, proj_dim=proj_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lam = lambda_schedule(epoch, warmup_epochs, max_lambda)

        # === Train ===
        model.train()
        epoch_masked, epoch_contrast, epoch_total = 0.0, 0.0, 0.0
        epoch_stats = {"cos_pos": [], "cos_neg": [], "norm_mean": [], "norm_std": []}
        batch_obs, batch_center, batch_mort = [], [], []
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            out = model(x, mask)

            loss_total = out["loss_masked"] + lam * out["loss_contrastive"]
            loss_total.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_masked += out["loss_masked"].item()
            epoch_contrast += out["loss_contrastive"].item()
            epoch_total += loss_total.item()
            n_batches += 1

            s = out["stats"]
            epoch_stats["cos_pos"].append(s["cos_pos"])
            epoch_stats["cos_neg"].append(s["cos_neg"])
            epoch_stats["norm_mean"].append(s["embedding_norm_mean"])
            epoch_stats["norm_std"].append(s["embedding_norm_std"])

            # Batch composition monitoring
            batch_obs.append(batch["obs_density"].mean().item())
            batch_center.append(batch["center_a"].mean().item())
            batch_mort.append(batch["mortality"].mean().item())

        train_metrics = {
            "loss_masked": epoch_masked / n_batches,
            "loss_contrastive": epoch_contrast / n_batches,
            "loss_total": epoch_total / n_batches,
        }

        # === Validate ===
        model.train(False)
        val_masked, val_contrast, val_total, n_val = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                mask = batch["mask"].to(device)
                out = model(x, mask)
                val_masked += out["loss_masked"].item()
                val_contrast += out["loss_contrastive"].item()
                val_total += (out["loss_masked"] + lam * out["loss_contrastive"]).item()
                n_val += 1

        val_metrics = {
            "loss_masked": val_masked / max(n_val, 1),
            "loss_contrastive": val_contrast / max(n_val, 1),
            "loss_total": val_total / max(n_val, 1),
        }

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # === Alignment / Uniformity (at checkpoints) ===
        align_uniform = {}
        if epoch % 5 == 0 or epoch == 1:
            align_uniform = _compute_alignment_uniformity(model.encoder, continuous, masks,
                                                          np.array(splits["val"]), device)

        record = {
            "epoch": epoch,
            "lambda": round(lam, 4),
            "lr": current_lr,
            "time_s": round(elapsed, 1),
            "train": train_metrics,
            "val": val_metrics,
            "cos_pos": round(float(np.mean(epoch_stats["cos_pos"])), 4),
            "cos_neg": round(float(np.mean(epoch_stats["cos_neg"])), 4),
            "embedding_norm_mean": round(float(np.mean(epoch_stats["norm_mean"])), 4),
            "embedding_norm_std": round(float(np.mean(epoch_stats["norm_std"])), 4),
            "batch_obs_density_mean": round(float(np.mean(batch_obs)), 4),
            "batch_center_a_frac": round(float(np.mean(batch_center)), 4),
            "batch_mortality_frac": round(float(np.mean(batch_mort)), 4),
            **align_uniform,
        }
        history.append(record)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"L_m={train_metrics['loss_masked']:.4f} L_c={train_metrics['loss_contrastive']:.4f} "
            f"λ={lam:.2f} | val={val_metrics['loss_total']:.4f} | "
            f"cos+={record['cos_pos']:.3f} cos-={record['cos_neg']:.3f} | "
            f"norm={record['embedding_norm_mean']:.1f}±{record['embedding_norm_std']:.2f} | "
            f"{elapsed:.1f}s"
        )

        # === Checkpointing ===
        val_key = val_metrics["loss_total"]
        if val_key < best_val_loss:
            best_val_loss = val_key
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": model.encoder.state_dict(),
                "full_model_state_dict": model.state_dict(),
                "val_loss": val_key,
                "config": {
                    "n_features": n_features, "d_model": d_model, "n_heads": n_heads,
                    "n_layers": n_layers, "d_ff": d_ff, "dropout": dropout,
                    "view_len": view_len, "proj_dim": proj_dim,
                },
            }, ckpt_dir / "pretrain_best.pt")
        else:
            patience_counter += 1

        torch.save({
            "epoch": epoch,
            "encoder_state_dict": model.encoder.state_dict(),
            "full_model_state_dict": model.state_dict(),
            "val_loss": val_key,
        }, ckpt_dir / "pretrain_last.pt")

        scheduler.step()

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}. Best val_loss: {best_val_loss:.5f}")
            break

    log = {
        "best_val_loss": best_val_loss,
        "best_epoch": min(history, key=lambda h: h["val"]["loss_total"])["epoch"],
        "total_epochs": len(history),
        "n_params": n_params,
        "history": history,
    }
    with open(output_dir / "pretrain_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    logger.info(f"S1.5 pretraining complete. Best val_loss={best_val_loss:.5f}")
    return log


def _compute_alignment_uniformity(
    encoder, continuous, masks, val_indices, device,
    n_samples: int = 2000,
) -> dict:
    """
    Compute alignment and uniformity proxy metrics.
    Alignment: mean L2 distance between positive pair embeddings (from two halves of sequence).
    Uniformity: log-average pairwise Gaussian potential.
    """
    encoder_was_training = encoder.training
    encoder.train(False)

    n = min(n_samples, len(val_indices))
    idx = val_indices[:n]

    x_full = torch.from_numpy(continuous[idx]).float().to(device)
    m_full = torch.from_numpy(masks[idx]).float().to(device)

    with torch.no_grad():
        # Two halves as proxy positive pairs
        x_a = x_full[:, :24, :]
        m_a = m_full[:, :24, :]
        x_b = x_full[:, 24:, :]
        m_b = m_full[:, 24:, :]

        z_a = encoder(x_a, m_a)
        z_b = encoder(x_b, m_b)

        # Normalize
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        # Alignment: mean squared L2 of positive pairs
        alignment = ((z_a - z_b) ** 2).sum(dim=-1).mean().item()

        # Uniformity: log-average pairwise Gaussian potential
        # Use only z_a for computational reasons
        sq_dist = torch.cdist(z_a, z_a, p=2).pow(2)
        n_z = z_a.shape[0]
        mask_diag = ~torch.eye(n_z, device=z_a.device, dtype=torch.bool)
        uniformity = torch.log(torch.exp(-2 * sq_dist[mask_diag]).mean() + 1e-10).item()

    if encoder_was_training:
        encoder.train(True)

    return {"alignment": round(alignment, 4), "uniformity": round(uniformity, 4)}
