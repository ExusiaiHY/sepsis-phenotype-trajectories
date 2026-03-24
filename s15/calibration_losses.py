"""
calibration_losses.py - Calibration-aware loss functions for PyTorch training.

Provides:
  1. FocalLoss: down-weights easy examples, handles class imbalance
  2. BrierLoss: differentiable Brier score for probability calibration
  3. CalibrationAwareLoss: joint CE + Brier + soft-ECE
  4. Early stopping callback with AUROC+Brier monitoring

This module is a pure PyTorch training utility with no subprocess calls.
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("s15.calibration_losses")


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced binary classification."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class BrierLoss(nn.Module):
    """Differentiable Brier score loss: mean((sigmoid(logit) - y)^2)."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return ((probs - targets) ** 2).mean()


class SoftECELoss(nn.Module):
    """Differentiable soft ECE using Gaussian kernel bin assignments."""

    def __init__(self, n_bins: int = 10, bandwidth: float = 0.05):
        super().__init__()
        self.n_bins = n_bins
        self.bandwidth = bandwidth
        bin_centers = torch.linspace(0.5 / n_bins, 1.0 - 0.5 / n_bins, n_bins)
        self.register_buffer("bin_centers", bin_centers)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).squeeze()
        targets = targets.squeeze()

        diff = probs.unsqueeze(1) - self.bin_centers.unsqueeze(0)
        weights = torch.exp(-0.5 * (diff / self.bandwidth) ** 2)
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

        weighted_probs = (weights * probs.unsqueeze(1)).sum(dim=0)
        weighted_targets = (weights * targets.unsqueeze(1)).sum(dim=0)
        bin_counts = weights.sum(dim=0)

        avg_probs = weighted_probs / (bin_counts + 1e-8)
        avg_targets = weighted_targets / (bin_counts + 1e-8)

        bin_fracs = bin_counts / (bin_counts.sum() + 1e-8)
        ece = (bin_fracs * (avg_probs - avg_targets).abs()).sum()
        return ece


class CalibrationAwareLoss(nn.Module):
    """Joint loss: CE + lambda_brier * Brier + lambda_ece * SoftECE."""

    def __init__(
        self,
        pos_weight: float = 1.0,
        lambda_brier: float = 1.0,
        lambda_ece: float = 0.5,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_brier = lambda_brier
        self.lambda_ece = lambda_ece

        if use_focal:
            alpha = pos_weight / (1.0 + pos_weight)
            self.ce_loss = FocalLoss(alpha=alpha, gamma=focal_gamma)
        else:
            self.ce_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight)
            )

        self.brier_loss = BrierLoss()
        self.ece_loss = SoftECELoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets)
        brier = self.brier_loss(logits, targets)
        ece = self.ece_loss(logits, targets)
        return ce + self.lambda_brier * brier + self.lambda_ece * ece


class CalibrationEarlyStopping:
    """Early stopping based on AUROC + Brier score.

    Monitors: AUROC - lambda_brier * Brier (higher is better).
    """

    def __init__(
        self,
        patience: int = 10,
        lambda_brier: float = 1.0,
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.lambda_brier = lambda_brier
        self.min_delta = min_delta
        self.best_score: float = -np.inf
        self.best_epoch: int = 0
        self.counter: int = 0
        self.best_state: Optional[dict] = None

    def step(
        self,
        epoch: int,
        auroc: float,
        brier: float,
        model_state: Optional[dict] = None,
    ) -> bool:
        """Check if training should stop. Returns True if patience exhausted."""
        composite = auroc - self.lambda_brier * brier

        if composite > self.best_score + self.min_delta:
            self.best_score = composite
            self.best_epoch = epoch
            self.counter = 0
            if model_state is not None:
                self.best_state = copy.deepcopy(model_state)
            return False

        self.counter += 1
        if self.counter >= self.patience:
            logger.info(
                "Early stopping at epoch %d (best epoch %d: composite=%.4f)",
                epoch, self.best_epoch, self.best_score,
            )
            return True
        return False


def compute_val_calibration(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> dict:
    """Run model on validation set and return calibration metrics.

    Returns dict with brier, auroc, ece, mean_pred, obs_rate.
    """
    model.train(False)
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)

            logits = model(x, mask)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_probs_np = np.concatenate(all_probs)
    all_labels_np = np.concatenate(all_labels)

    from sklearn.metrics import brier_score_loss, roc_auc_score

    result = {
        "brier": float(brier_score_loss(all_labels_np, all_probs_np)),
        "mean_pred": float(np.mean(all_probs_np)),
        "obs_rate": float(np.mean(all_labels_np)),
    }

    if len(np.unique(all_labels_np)) >= 2:
        result["auroc"] = float(roc_auc_score(all_labels_np, all_probs_np))
    else:
        result["auroc"] = None

    # 10-bin ECE
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (all_probs_np >= lo) & (all_probs_np <= hi)
        else:
            in_bin = (all_probs_np >= lo) & (all_probs_np < hi)
        if not in_bin.any():
            continue
        frac = in_bin.mean()
        gap = abs(all_labels_np[in_bin].mean() - all_probs_np[in_bin].mean())
        ece += gap * frac

    result["ece"] = float(ece)
    return result
