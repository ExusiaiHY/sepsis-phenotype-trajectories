"""
pretrain.py - Self-supervised pretraining loop for masked value prediction.

Purpose:
  Train the ICUTransformerEncoder via masked value prediction on
  S0 processed continuous data. Uses train split only for training,
  val split for monitoring, never touches test split.

Connects to:
  - s0/processed/continuous.npy and masks_continuous.npy
  - s0/splits.json for train/val indices
  - s1/encoder.py for model architecture

Expected output artifacts:
  data/s1/checkpoints/pretrain_best.pt
  data/s1/checkpoints/pretrain_last.pt
  data/s1/pretrain_log.json
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from s1.encoder import MaskedPretrainModel

logger = logging.getLogger("s1.pretrain")


class ContinuousDataset(Dataset):
    """Simple dataset wrapping continuous + mask arrays."""

    def __init__(self, continuous: np.ndarray, masks: np.ndarray, indices: np.ndarray):
        self.continuous = continuous[indices]
        self.masks = masks[indices]

    def __len__(self):
        return len(self.continuous)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.continuous[idx]).float(),
            "mask": torch.from_numpy(self.masks[idx]).float(),
        }


def pretrain(
    s0_dir: Path,
    output_dir: Path,
    n_features: int = 21,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.2,
    mask_ratio: float = 0.15,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    grad_clip: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Run pretraining. Returns log dict.
    """
    s0_dir = Path(s0_dir)
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    continuous = np.load(s0_dir / "processed" / "continuous.npy")
    masks = np.load(s0_dir / "processed" / "masks_continuous.npy")

    with open(s0_dir / "splits.json") as f:
        splits = json.load(f)

    train_ds = ContinuousDataset(continuous, masks, np.array(splits["train"]))
    val_ds = ContinuousDataset(continuous, masks, np.array(splits["val"]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    logger.info(f"Model: d_model={d_model}, layers={n_layers}, heads={n_heads}, "
                f"mask_ratio={mask_ratio}, device={device}")

    # Model
    model = MaskedPretrainModel(
        n_features=n_features, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, dropout=dropout, mask_ratio=mask_ratio,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss, _, _ = model(x, mask)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)

        # Validate
        model.train(False)
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                mask = batch["mask"].to(device)
                loss, _, _ = model(x, mask)
                val_loss_sum += loss.item()
                n_val += 1

        val_loss = val_loss_sum / max(n_val, 1)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": current_lr,
            "time_s": round(elapsed, 1),
        }
        history.append(record)

        logger.info(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.5f} | "
                    f"val_loss={val_loss:.5f} | lr={current_lr:.2e} | {elapsed:.1f}s")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.encoder.state_dict(),
                "full_model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "n_features": n_features, "d_model": d_model, "n_heads": n_heads,
                    "n_layers": n_layers, "d_ff": d_ff, "dropout": dropout,
                },
            }, ckpt_dir / "pretrain_best.pt")
        else:
            patience_counter += 1

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.encoder.state_dict(),
            "full_model_state_dict": model.state_dict(),
            "val_loss": val_loss,
        }, ckpt_dir / "pretrain_last.pt")

        scheduler.step()

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}. Best val_loss: {best_val_loss:.5f}")
            break

    # Save log
    log = {
        "best_val_loss": best_val_loss,
        "best_epoch": min(history, key=lambda h: h["val_loss"])["epoch"],
        "total_epochs": len(history),
        "n_params": n_params,
        "history": history,
    }
    with open(output_dir / "pretrain_log.json", "w") as f:
        json.dump(log, f, indent=2)

    logger.info(f"Pretraining complete. Best val_loss={best_val_loss:.5f}")
    return log
