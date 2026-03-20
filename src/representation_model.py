"""
representation_model.py - Representation learning module

Responsibilities:
1. MVP: Wrap statistical features + PCA dimensionality reduction as baseline
2. V2: Implement self-supervised patient trajectory encoder
   - Masked Event Modeling (BERT-style masking pretraining)
   - Temporal contrastive learning
   - Transformer / GRU encoder

Current version (MVP):
  Provides interface definitions and baseline wrappers only.
  Self-supervised components will be implemented in V2.
  Designed so V2 replacement only requires modifying this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Any

from utils import setup_logger, timer

logger = setup_logger(__name__)


# ============================================================
# Base Representation Interface
# ============================================================

class BaseRepresentation:
    """Base class for all representation methods, defining a unified interface."""

    def fit(self, time_series_3d: np.ndarray, **kwargs) -> "BaseRepresentation":
        raise NotImplementedError

    def transform(self, time_series_3d: np.ndarray) -> np.ndarray:
        """Return (n_patients, embedding_dim) representation matrix."""
        raise NotImplementedError

    def fit_transform(self, time_series_3d: np.ndarray, **kwargs) -> np.ndarray:
        return self.fit(time_series_3d, **kwargs).transform(time_series_3d)


# ============================================================
# MVP Baseline: Statistical Features + PCA
# ============================================================

class StatisticalRepresentation(BaseRepresentation):
    """
    MVP representation method:
    Uses statistical features from feature_engineering,
    then applies PCA to reduce to specified dimensionality.

    Not self-supervised learning, but provides a strong baseline.
    YAIB 2023 showed that well-designed statistical features + traditional ML
    often match deep models in ICU prediction tasks.
    """

    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self._fitted = False

    def fit(self, feature_matrix: np.ndarray, **kwargs) -> "StatisticalRepresentation":
        X = self.scaler.fit_transform(feature_matrix)
        # Adaptive: PCA dims cannot exceed min(n_samples, n_features)
        max_dim = min(X.shape[0], X.shape[1])
        if self.n_components > max_dim:
            logger.warning(f"PCA dims {self.n_components} exceeds data limit {max_dim}, auto-adjusting")
            self.n_components = max_dim - 1
            self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        self._fitted = True
        explained = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA fitted: {self.n_components} dims, "
                     f"explained variance ratio {explained:.1%}")
        return self

    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Please call fit() first")
        X = self.scaler.transform(feature_matrix)
        return self.pca.transform(X)


# ============================================================
# V2 Reserved: Self-Supervised Trajectory Encoder
# ============================================================

class SelfSupervisedEncoder(BaseRepresentation):
    """
    V2 self-supervised patient trajectory encoder (to be implemented).

    Pretraining task options:
    1. Masked Event Modeling - randomly mask timesteps, predict missing values
    2. Next-Visit Prediction - given first t steps, predict step t+1
    3. Temporal Contrastive Learning - different windows of same patient as positives

    Encoder architecture options:
    - Transformer: suitable for long sequences, parallelizable training
    - GRU-D: native support for irregular time intervals and missing values
    - Mamba (EHRMamba): linear complexity, suitable for very long sequences
    """

    def __init__(self, config: dict):
        self.config = config
        rep_cfg = config["features"]["representation"]
        self.model_type = rep_cfg["model_type"]
        self.embedding_dim = rep_cfg["embedding_dim"]
        self.pretrain_task = rep_cfg["pretrain_task"]
        logger.info(f"Self-supervised encoder initialized: {self.model_type}, "
                     f"pretrain task: {self.pretrain_task}, "
                     f"embedding dim: {self.embedding_dim}")

    def pretrain(self, time_series_3d: np.ndarray, **kwargs):
        """Self-supervised pretraining (V2 implementation)."""
        raise NotImplementedError(
            "Self-supervised pretraining not yet implemented.\n"
            "Please use StatisticalRepresentation as baseline.\n"
            "V2 will implement Masked Event Modeling + Transformer encoder."
        )

    def fit(self, time_series_3d: np.ndarray, **kwargs):
        return self.pretrain(time_series_3d, **kwargs)

    def transform(self, time_series_3d: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Please complete pretraining first")


# ============================================================
# Factory Function
# ============================================================

def get_representation_model(
    config: dict, mode: str = "statistical"
) -> BaseRepresentation:
    """
    Return the appropriate representation model based on mode.

    Parameters
    ----------
    config : dict
    mode : str
        "statistical" - MVP: statistical features + PCA
        "self_supervised" - V2: self-supervised encoder
    """
    if mode == "statistical":
        dim = config["features"]["representation"]["embedding_dim"]
        return StatisticalRepresentation(n_components=min(dim, 32))
    elif mode == "self_supervised":
        return SelfSupervisedEncoder(config)
    else:
        raise ValueError(f"Unsupported representation mode: {mode}")
