"""
encoder.py - Transformer encoder for ICU time-series with mask-aware input.

Architecture:
  Input: concat([continuous, mask], dim=-1)  → (B, 48, 42)
  Input projection: Linear(42, d_model)      → (B, 48, 128)
  Positional encoding: learnable             → (B, 48, 128)
  Transformer encoder: L layers, H heads     → (B, 48, 128)
  Pooling: weighted mean by observation density → (B, 128)

The mask is a first-class input channel, not a summary scalar.
This lets the encoder learn variable-level missingness patterns.

d_model=128 provides moderate capacity for a 12K-patient cohort.
It is 4x the PCA baseline (32 dims), giving the encoder room to
learn richer representations without being so large that it overfits.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class ICUTransformerEncoder(nn.Module):
    """
    Self-supervised Transformer encoder for hourly ICU time-series.

    Input:
      x:    (B, T, F)  continuous measurements (F=21)
      mask: (B, T, F)  observation masks (1=observed, 0=imputed)

    Output:
      embedding: (B, d_model)  patient-level representation
      sequence:  (B, T, d_model)  per-timestep representations (for pretraining head)
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
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Input: concat([x, mask]) → 2 * n_features channels
        self.input_proj = nn.Linear(n_features * 2, d_model)

        # Learnable positional encoding for fixed 48h window
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:    (B, T, 21) continuous values
        mask: (B, T, 21) observation masks
        return_sequence: if True, also return per-timestep output

        Returns
        -------
        embedding: (B, d_model)
        sequence:  (B, T, d_model) only if return_sequence=True
        """
        B, T, F = x.shape

        # Concatenate values and masks as joint input
        x_input = torch.cat([x, mask], dim=-1)  # (B, T, 42)

        # Project to model dimension
        h = self.input_proj(x_input)  # (B, T, d_model)

        # Add positional encoding
        h = h + self.pos_embedding[:, :T, :]

        # Transformer forward
        h = self.transformer(h)  # (B, T, d_model)
        h = self.output_norm(h)

        # Weighted mean pooling by per-timestep observation density
        # obs_density: fraction of features observed at each timestep
        obs_density = mask.mean(dim=-1, keepdim=True)  # (B, T, 1)
        # Floor at small epsilon so fully-missing timesteps still contribute minimally
        weights = obs_density.clamp(min=1e-6)           # (B, T, 1)
        embedding = (h * weights).sum(dim=1) / weights.sum(dim=1)  # (B, d_model)

        if return_sequence:
            return embedding, h
        return embedding


class MaskedValuePredictor(nn.Module):
    """
    Pretraining head: predict original values at masked positions.

    Takes per-timestep encoder output and projects back to feature space.
    """

    def __init__(self, d_model: int = 128, n_features: int = 21):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sequence: (B, T, d_model) per-timestep encoder output

        Returns
        -------
        predictions: (B, T, n_features) predicted values
        """
        return self.head(sequence)


class MaskedPretrainModel(nn.Module):
    """
    Full pretraining model: encoder + prediction head.

    Masking procedure:
      1. Select 15% of observed positions (mask=1) uniformly at random
      2. Zero out those positions in both x and mask
      3. Encoder sees corrupted input
      4. Prediction head outputs values for all positions
      5. Loss = MSE only on the masked-out observed positions
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
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.encoder = ICUTransformerEncoder(
            n_features=n_features, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout, max_seq_len=max_seq_len,
        )
        self.predictor = MaskedValuePredictor(d_model, n_features)
        self.mask_ratio = mask_ratio

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:    (B, T, F) original continuous values
        mask: (B, T, F) original observation masks

        Returns
        -------
        loss: scalar MSE on masked positions
        predictions: (B, T, F) predicted values (for debugging)
        mask_targets: (B, T, F) which positions were masked
        """
        B, T, F = x.shape

        # Generate masking targets: select mask_ratio of observed positions
        mask_targets = self._generate_mask_targets(mask)  # (B, T, F) binary

        # Corrupt input
        x_masked = x.clone()
        mask_input = mask.clone()
        x_masked[mask_targets == 1] = 0.0
        mask_input[mask_targets == 1] = 0.0

        # Forward through encoder
        _, sequence = self.encoder(x_masked, mask_input, return_sequence=True)

        # Predict
        predictions = self.predictor(sequence)  # (B, T, F)

        # Loss: MSE only on masked observed positions
        n_masked = mask_targets.sum()
        if n_masked > 0:
            diff = (predictions - x) ** 2
            loss = (diff * mask_targets).sum() / n_masked
        else:
            loss = torch.tensor(0.0, device=x.device, requires_grad=True)

        return loss, predictions, mask_targets

    def _generate_mask_targets(self, mask: torch.Tensor) -> torch.Tensor:
        """Select mask_ratio fraction of observed positions to mask out."""
        # Only mask positions where mask=1 (actually observed)
        rand = torch.rand_like(mask)
        # Positions eligible for masking: observed AND randomly selected
        mask_targets = (rand < self.mask_ratio) & (mask > 0.5)
        return mask_targets.float()
