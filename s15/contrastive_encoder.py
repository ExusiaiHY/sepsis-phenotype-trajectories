"""
contrastive_encoder.py - Encoder with projection head + stochastic window sampling.

Reuses ICUTransformerEncoder from s1/encoder.py.
Adds: ProjectionHead, StochasticWindowSampler, ContrastivePretrainModel.

Design decisions (D007):
  - Projection head output (64d) used only for NT-Xent, discarded after pretraining
  - Encoder output (128d) is the patient embedding for clustering/probes
  - Windows: W=30h, stochastic start positions, overlap 12-24h
  - Lambda warmup: 0→0.5 over 10 epochs
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from s1.encoder import ICUTransformerEncoder, MaskedValuePredictor


# ============================================================
# Projection Head (used only for contrastive loss)
# ============================================================

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head (SimCLR-style).
    Maps encoder embeddings to a lower-dimensional space for NT-Xent.
    Discarded after pretraining.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Stochastic Window Sampler
# ============================================================

class StochasticWindowSampler:
    """
    Sample two overlapping temporal windows per patient.

    Rule (D007):
      W = 30 (view length in hours)
      start1 ~ Uniform{0, 1, ..., 12}
      gap ~ Uniform{6, 7, ..., min(18, 18 - start1)}
      start2 = start1 + gap
      overlap = 30 - gap  ∈ [12, 24]
    """

    def __init__(self, seq_len: int = 48, view_len: int = 30):
        self.seq_len = seq_len
        self.view_len = view_len
        self.max_start = seq_len - view_len  # 18

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (starts1, starts2) arrays of shape (batch_size,).
        Each element is an integer start index for a view window.
        """
        max_start1 = min(self.max_start, self.max_start)  # 18, but limited by gap constraint

        starts1 = np.random.randint(0, 13, size=batch_size)  # [0, 12]

        starts2 = np.zeros(batch_size, dtype=int)
        for i in range(batch_size):
            s1 = starts1[i]
            gap_min = 6
            gap_max = min(self.max_start, self.max_start - s1)  # ensure start2 ≤ 18

            if gap_max < gap_min:
                # Fallback: use maximum valid gap
                gap_max = gap_min
                starts1[i] = max(0, self.max_start - gap_min)
                s1 = starts1[i]

            gap = np.random.randint(gap_min, gap_max + 1)
            starts2[i] = s1 + gap

        return starts1, starts2

    def slice_views(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Slice two stochastic views from each sample.

        Parameters
        ----------
        x:    (B, T, F) continuous values
        mask: (B, T, F) observation masks

        Returns
        -------
        x1, mask1:         (B, W, F) view 1
        x2, mask2:         (B, W, F) view 2
        pos_indices1:      (B,) start indices for view 1 (for positional encoding)
        pos_indices2:      (B,) start indices for view 2
        """
        B = x.shape[0]
        W = self.view_len

        starts1, starts2 = self.sample(B)

        x1 = torch.zeros(B, W, x.shape[2], device=x.device, dtype=x.dtype)
        x2 = torch.zeros(B, W, x.shape[2], device=x.device, dtype=x.dtype)
        m1 = torch.zeros(B, W, mask.shape[2], device=mask.device, dtype=mask.dtype)
        m2 = torch.zeros(B, W, mask.shape[2], device=mask.device, dtype=mask.dtype)

        for i in range(B):
            s1, s2 = int(starts1[i]), int(starts2[i])
            x1[i] = x[i, s1:s1 + W]
            x2[i] = x[i, s2:s2 + W]
            m1[i] = mask[i, s1:s1 + W]
            m2[i] = mask[i, s2:s2 + W]

        pos1 = torch.from_numpy(starts1).long().to(x.device)
        pos2 = torch.from_numpy(starts2).long().to(x.device)

        return x1, m1, x2, m2, pos1, pos2


# ============================================================
# NT-Xent Loss
# ============================================================

def nt_xent_loss(
    p1: torch.Tensor,
    p2: torch.Tensor,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Symmetric NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    Parameters
    ----------
    p1, p2: (B, D) projected embeddings for view 1 and view 2

    Returns
    -------
    loss: scalar
    stats: dict with cosine similarity statistics
    """
    B = p1.shape[0]

    # Normalize
    p1 = F.normalize(p1, dim=-1)
    p2 = F.normalize(p2, dim=-1)

    # Concatenate: [p1_0, ..., p1_{B-1}, p2_0, ..., p2_{B-1}]
    z = torch.cat([p1, p2], dim=0)  # (2B, D)

    # Similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity (diagonal)
    mask_self = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask_self, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_indices = torch.arange(2 * B, device=z.device)
    pos_indices[:B] += B
    pos_indices[B:] -= B

    # Cross-entropy loss
    labels = pos_indices
    loss = F.cross_entropy(sim, labels)

    # Stats (detached)
    with torch.no_grad():
        cos_pos = F.cosine_similarity(p1, p2, dim=-1).mean().item()
        # Average cosine of negatives: mean of off-diagonal
        sim_detached = torch.mm(p1, p2.t())  # (B, B)
        neg_mask = ~torch.eye(B, device=z.device, dtype=torch.bool)
        cos_neg = sim_detached[neg_mask].mean().item()

    stats = {"cos_pos": cos_pos, "cos_neg": cos_neg}
    return loss, stats


# ============================================================
# Full Pretraining Model
# ============================================================

class ContrastivePretrainModel(nn.Module):
    """
    Combined masked reconstruction + contrastive window pretraining.

    Forward pass:
      1. Sample two stochastic windows from each patient
      2. Apply independent masking corruption to view1
      3. Encode both views → embeddings z1, z2
      4. Project embeddings → p1, p2 (for contrastive loss only)
      5. Predict masked values in view1 (reconstruction loss)
      6. Compute NT-Xent on (p1, p2) (contrastive loss)
      7. L_total = L_masked + lambda * L_contrastive
    """

    def __init__(
        self,
        n_features: int = 21,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 48,
        view_len: int = 30,
        mask_ratio: float = 0.15,
        temperature: float = 0.1,
        proj_dim: int = 64,
    ):
        super().__init__()

        self.encoder = ICUTransformerEncoder(
            n_features=n_features, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout, max_seq_len=max_seq_len,
        )
        self.predictor = MaskedValuePredictor(d_model, n_features)
        self.projection = ProjectionHead(d_model, d_model, proj_dim)
        self.sampler = StochasticWindowSampler(seq_len=max_seq_len, view_len=view_len)

        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.n_features = n_features
        self.view_len = view_len

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        x:    (B, 48, F) continuous values
        mask: (B, 48, F) observation masks

        Returns
        -------
        dict with: loss_masked, loss_contrastive, stats
        """
        # Step 1: Sample windows
        x1, m1, x2, m2, _, _ = self.sampler.slice_views(x, mask)

        # Step 2: Masked corruption on view1 (for reconstruction loss)
        mask_targets = self._sample_mask_targets(m1)
        x1_corrupted = x1.clone()
        m1_corrupted = m1.clone()
        x1_corrupted[mask_targets == 1] = 0.0
        m1_corrupted[mask_targets == 1] = 0.0

        # Step 3: Encode both views
        z1, seq1 = self.encoder(x1_corrupted, m1_corrupted, return_sequence=True)
        z2 = self.encoder(x2, m2)  # no corruption on view2

        # Step 4: Reconstruction loss on view1
        predictions = self.predictor(seq1)
        n_masked = mask_targets.sum()
        if n_masked > 0:
            loss_masked = ((predictions - x1) ** 2 * mask_targets).sum() / n_masked
        else:
            loss_masked = torch.tensor(0.0, device=x.device, requires_grad=True)

        # Step 5: Contrastive loss via projection head
        p1 = self.projection(z1)
        p2 = self.projection(z2)
        loss_contrastive, contrast_stats = nt_xent_loss(p1, p2, self.temperature)

        # Step 6: Embedding health stats
        with torch.no_grad():
            norms = torch.norm(z1, dim=-1)
            stats = {
                **contrast_stats,
                "embedding_norm_mean": norms.mean().item(),
                "embedding_norm_std": norms.std().item(),
            }

        return {
            "loss_masked": loss_masked,
            "loss_contrastive": loss_contrastive,
            "stats": stats,
        }

    def _sample_mask_targets(self, mask: torch.Tensor) -> torch.Tensor:
        rand = torch.rand_like(mask)
        return ((rand < self.mask_ratio) & (mask > 0.5)).float()
