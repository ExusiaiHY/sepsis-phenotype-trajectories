"""
domain_adaptation.py - Lightweight center/domain alignment utilities for S6.

Supported methods:
  - coral: covariance alignment against the pooled reference distribution
  - mean_std: per-group mean/std matching
  - dann: lightweight gradient-reversal domain adaptation with reconstruction

The DANN path is intentionally conservative:
  - it works on the existing patient-level covariate matrix
  - it keeps output dimensionality unchanged
  - reconstruction loss prevents aggressive collapse/over-alignment
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("s6.domain_adaptation")


def _matrix_power_symmetric(matrix: np.ndarray, power: float, eps: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, eps, None)
    scaled = np.diag(np.power(vals, power))
    return vecs @ scaled @ vecs.T


def _covariance(matrix: np.ndarray, reg: float) -> np.ndarray:
    if len(matrix) <= 1:
        dim = matrix.shape[1]
        return np.eye(dim, dtype=np.float64) * reg
    cov = np.cov(matrix, rowvar=False).astype(np.float64, copy=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    dim = cov.shape[0]
    return cov + np.eye(dim, dtype=np.float64) * reg


def _coral_align(source: np.ndarray, target_mean: np.ndarray, target_cov: np.ndarray, reg: float) -> np.ndarray:
    source_mean = source.mean(axis=0)
    source_cov = _covariance(source, reg=reg)
    whiten = _matrix_power_symmetric(source_cov, power=-0.5, eps=reg)
    recolor = _matrix_power_symmetric(target_cov, power=0.5, eps=reg)
    centered = source - source_mean
    return centered @ whiten @ recolor + target_mean


def _weighted_group_mean_gap(X: np.ndarray, group_ids: np.ndarray, reference_mean: np.ndarray) -> float:
    total = len(X)
    gap = 0.0
    for group in np.unique(group_ids):
        mask = group_ids == group
        if not np.any(mask):
            continue
        group_mean = X[mask].mean(axis=0)
        gap += float(np.linalg.norm(group_mean - reference_mean)) * (mask.sum() / total)
    return gap


def _blend_alignment(original: np.ndarray, aligned: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    if alpha <= 0.0:
        return original
    if alpha >= 1.0:
        return aligned
    return ((1.0 - alpha) * original) + (alpha * aligned)


def _apply_alignment_method(
    source: np.ndarray,
    *,
    method: str,
    reference_mean: np.ndarray,
    reference_cov: np.ndarray,
    reference_std: np.ndarray,
    reg: float,
) -> np.ndarray:
    method = str(method).lower()
    if method == "coral":
        return _coral_align(source, reference_mean, reference_cov, reg=reg)
    if method == "mean_std":
        source_mean = source.mean(axis=0)
        source_std = np.clip(source.std(axis=0), reg, None)
        return ((source - source_mean) / source_std) * np.clip(reference_std, reg, None) + reference_mean
    raise ValueError(f"Unsupported domain adaptation method: {method}")


def _safe_domain_probe_accuracy(
    X: np.ndarray,
    group_ids: np.ndarray,
    *,
    random_state: int,
) -> float | None:
    unique_groups, counts = np.unique(group_ids, return_counts=True)
    if len(unique_groups) < 2 or counts.min() < 2:
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            group_ids,
            test_size=0.25,
            random_state=random_state,
            stratify=group_ids,
        )
        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        probe.fit(X_train, y_train)
        predictions = probe.predict(X_test)
        return float(balanced_accuracy_score(y_test, predictions))
    except Exception as exc:
        logger.warning("Domain probe accuracy skipped: %s", exc)
        return None


def _linear_warmup(epoch_idx: int, total_epochs: int, target: float, warmup_epochs: int) -> float:
    if target <= 0.0:
        return 0.0
    if warmup_epochs <= 1:
        return target
    progress = min(1.0, float(epoch_idx + 1) / float(min(total_epochs, warmup_epochs)))
    return target * progress


def _run_dann_alignment(
    X: np.ndarray,
    domain_codes: np.ndarray,
    *,
    config: dict,
) -> tuple[np.ndarray, dict]:
    try:
        import torch
        from torch import nn
        from torch.autograd import Function
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    except ImportError as exc:
        return X, {
            "applied": False,
            "reason": f"torch_not_installed: {exc}",
        }

    class _GradientReversal(Function):
        @staticmethod
        def forward(ctx, input_tensor, lambd):  # type: ignore[override]
            ctx.lambd = float(lambd)
            return input_tensor.view_as(input_tensor)

        @staticmethod
        def backward(ctx, grad_output):  # type: ignore[override]
            return -ctx.lambd * grad_output, None

    def _grl(input_tensor: "torch.Tensor", lambd: float) -> "torch.Tensor":
        return _GradientReversal.apply(input_tensor, lambd)

    def _geometry_anchor_loss(
        aligned_x: "torch.Tensor",
        batch_x: "torch.Tensor",
        batch_d: "torch.Tensor",
    ) -> "torch.Tensor":
        losses = [torch.mean((aligned_x.mean(dim=0) - batch_x.mean(dim=0)) ** 2)]
        for domain_code in torch.unique(batch_d):
            mask = batch_d == domain_code
            if int(mask.sum().item()) == 0:
                continue
            losses.append(
                torch.mean((aligned_x[mask].mean(dim=0) - batch_x[mask].mean(dim=0)) ** 2)
            )
        return torch.stack(losses).mean()

    class _ResidualDomainAdapter(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            embedding_dim: int,
            n_domains: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
                nn.ReLU(),
            )
            self.adapter = nn.Linear(embedding_dim, input_dim)
            self.domain_head = nn.Sequential(
                nn.Linear(embedding_dim, max(hidden_dim // 2, n_domains)),
                nn.ReLU(),
                nn.Linear(max(hidden_dim // 2, n_domains), n_domains),
            )

        def forward(self, inputs: "torch.Tensor", grl_lambda: float) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            embedding = self.encoder(inputs)
            aligned = inputs + self.adapter(embedding)
            domain_logits = self.domain_head(_grl(embedding, grl_lambda))
            return aligned, embedding, domain_logits

    seed = int(config.get("random_state", 42))
    batch_size = int(config.get("batch_size", 256))
    epochs = int(config.get("epochs", 20))
    patience = max(1, int(config.get("patience", 4)))
    hidden_dim = int(config.get("hidden_dim", 64))
    embedding_dim = int(config.get("embedding_dim", max(16, min(32, X.shape[1]))))
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    dropout = float(config.get("dropout", 0.1))
    lambda_domain = float(config.get("lambda_domain", 0.3))
    lambda_recon = float(config.get("lambda_recon", 1.0))
    lambda_geometry = float(config.get("lambda_geometry", 0.0))
    warmup_epochs = int(config.get("warmup_epochs", max(2, min(5, epochs // 2 or 1))))
    val_fraction = float(config.get("val_fraction", 0.15))

    if len(np.unique(domain_codes)) < 2:
        return X, {
            "applied": False,
            "reason": "single_domain_after_filtering",
        }

    device_cfg = str(config.get("device", "cpu")).lower()
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X = np.asarray(X, dtype=np.float32)
    domain_codes = np.asarray(domain_codes, dtype=np.int64)
    feature_mean = X.mean(axis=0, keepdims=True)
    feature_std = np.clip(X.std(axis=0, keepdims=True), 1e-6, None)
    X_norm = (X - feature_mean) / feature_std

    if len(X_norm) < 10:
        val_fraction = 0.0

    if val_fraction > 0.0:
        idx = np.arange(len(X_norm))
        train_idx, val_idx = train_test_split(
            idx,
            test_size=val_fraction,
            random_state=seed,
            stratify=domain_codes,
        )
    else:
        train_idx = np.arange(len(X_norm))
        val_idx = np.array([], dtype=np.int64)

    X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32)
    d_train = torch.tensor(domain_codes[train_idx], dtype=torch.long)

    if len(val_idx) > 0:
        X_val = torch.tensor(X_norm[val_idx], dtype=torch.float32)
        d_val = torch.tensor(domain_codes[val_idx], dtype=torch.long)
        val_loader = DataLoader(
            TensorDataset(X_val, d_val),
            batch_size=min(batch_size, len(val_idx)),
            shuffle=False,
        )
    else:
        val_loader = None

    train_domain_counts = np.bincount(domain_codes[train_idx])
    class_weights = train_domain_counts.sum() / np.clip(
        train_domain_counts.astype(np.float32) * len(train_domain_counts),
        1.0,
        None,
    )
    sample_weights = class_weights[domain_codes[train_idx]]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        TensorDataset(X_train, d_train),
        batch_size=min(batch_size, len(train_idx)),
        sampler=sampler,
    )

    model = _ResidualDomainAdapter(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_domains=int(np.unique(domain_codes).size),
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []

    for epoch_idx in range(epochs):
        model.train()
        grl_lambda = _linear_warmup(epoch_idx, epochs, lambda_domain, warmup_epochs)

        train_recon = 0.0
        train_domain = 0.0
        train_geometry = 0.0
        train_correct = 0
        train_total = 0
        for batch_x, batch_d in train_loader:
            batch_x = batch_x.to(device)
            batch_d = batch_d.to(device)
            optimizer.zero_grad()
            aligned_x, _, domain_logits = model(batch_x, grl_lambda)
            recon = mse_loss(aligned_x, batch_x)
            domain = ce_loss(domain_logits, batch_d)
            geometry = _geometry_anchor_loss(aligned_x, batch_x, batch_d)
            loss = (lambda_recon * recon) + domain + (lambda_geometry * geometry)
            loss.backward()
            optimizer.step()

            train_recon += float(recon.item()) * len(batch_x)
            train_domain += float(domain.item()) * len(batch_x)
            train_geometry += float(geometry.item()) * len(batch_x)
            train_correct += float(
                balanced_accuracy_score(
                    batch_d.detach().cpu().numpy(),
                    domain_logits.argmax(dim=1).detach().cpu().numpy(),
                )
            ) * len(batch_x)
            train_total += len(batch_x)

        epoch_stats = {
            "epoch": epoch_idx + 1,
            "grl_lambda": round(grl_lambda, 4),
            "train_recon_loss": train_recon / max(train_total, 1),
            "train_domain_loss": train_domain / max(train_total, 1),
            "train_geometry_loss": train_geometry / max(train_total, 1),
            "train_domain_accuracy": train_correct / max(train_total, 1),
        }

        if val_loader is not None:
            model.eval()
            val_recon = 0.0
            val_domain = 0.0
            val_geometry = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_x, batch_d in val_loader:
                    batch_x = batch_x.to(device)
                    batch_d = batch_d.to(device)
                    aligned_x, _, domain_logits = model(batch_x, 0.0)
                    recon = mse_loss(aligned_x, batch_x)
                    domain = ce_loss(domain_logits, batch_d)
                    geometry = _geometry_anchor_loss(aligned_x, batch_x, batch_d)
                    val_recon += float(recon.item()) * len(batch_x)
                    val_domain += float(domain.item()) * len(batch_x)
                    val_geometry += float(geometry.item()) * len(batch_x)
                    val_correct += float(
                        balanced_accuracy_score(
                            batch_d.detach().cpu().numpy(),
                            domain_logits.argmax(dim=1).detach().cpu().numpy(),
                        )
                    ) * len(batch_x)
                    val_total += len(batch_x)

            epoch_stats.update(
                {
                    "val_recon_loss": val_recon / max(val_total, 1),
                    "val_domain_loss": val_domain / max(val_total, 1),
                    "val_geometry_loss": val_geometry / max(val_total, 1),
                    "val_domain_accuracy": val_correct / max(val_total, 1),
                }
            )
            monitor_value = (
                (lambda_recon * epoch_stats["val_recon_loss"])
                + epoch_stats["val_domain_loss"]
                + (lambda_geometry * epoch_stats["val_geometry_loss"])
            )
        else:
            monitor_value = (
                (lambda_recon * epoch_stats["train_recon_loss"])
                + epoch_stats["train_domain_loss"]
                + (lambda_geometry * epoch_stats["train_geometry_loss"])
            )

        history.append(epoch_stats)

        if monitor_value + 1e-6 < best_metric:
            best_metric = monitor_value
            best_epoch = epoch_idx + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_norm, dtype=torch.float32, device=device)
        aligned_norm, _, domain_logits = model(inputs, 0.0)
        aligned = aligned_norm.cpu().numpy() * feature_std + feature_mean
        domain_accuracy_train = float(
            balanced_accuracy_score(domain_codes, domain_logits.argmax(dim=1).cpu().numpy())
        )

    summary = {
        "applied": True,
        "device": device,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "lambda_domain": lambda_domain,
        "lambda_recon": lambda_recon,
        "lambda_geometry": lambda_geometry,
        "warmup_epochs": warmup_epochs,
        "best_monitor": round(float(best_metric), 6),
        "last_epoch": {
            k: round(float(v), 6) if isinstance(v, float) else v
            for k, v in history[-1].items()
        },
        "domain_classifier_accuracy_final": round(domain_accuracy_train, 4),
    }
    return aligned.astype(np.float32, copy=False), summary


def align_covariates_by_group(
    X: np.ndarray,
    group_ids: np.ndarray,
    *,
    output_dir: Path | None = None,
    config: dict | None = None,
) -> dict:
    cfg = {
        "enabled": False,
        "method": "coral",
        "min_group_size": 200,
        "reg": 1e-3,
        "alpha": 1.0,
        "prealign_method": None,
        "prealign_alpha": 1.0,
        "random_state": 42,
        "device": "cpu",
        "batch_size": 256,
        "epochs": 20,
        "patience": 4,
        "hidden_dim": 64,
        "embedding_dim": 24,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "lambda_domain": 0.3,
        "lambda_recon": 1.0,
        "lambda_geometry": 0.0,
        "warmup_epochs": 5,
        "val_fraction": 0.15,
    }
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})

    X = np.asarray(X, dtype=np.float64)
    group_ids = np.asarray(group_ids).astype(str)
    summary = {
        "enabled": bool(cfg.get("enabled", False)),
        "method": str(cfg.get("method", "coral")).lower(),
        "min_group_size": int(cfg["min_group_size"]),
        "reg": float(cfg["reg"]),
        "alpha": float(cfg["alpha"]),
        "prealign_method": None,
        "prealign_alpha": None,
        "random_state": int(cfg["random_state"]),
    }

    if not cfg.get("enabled", False):
        return {
            "X_aligned": X.astype(np.float32, copy=False),
            "summary": {**summary, "reason": "disabled_by_config"},
        }

    unique_groups = np.unique(group_ids)
    if len(unique_groups) < 2:
        return {
            "X_aligned": X.astype(np.float32, copy=False),
            "summary": {**summary, "reason": "single_group_only", "n_groups": int(len(unique_groups))},
        }

    method = str(cfg["method"]).lower()
    reg = float(cfg["reg"])
    min_group_size = int(cfg["min_group_size"])
    alpha = float(cfg["alpha"])
    prealign_method = str(cfg.get("prealign_method") or "").lower()
    prealign_enabled = prealign_method not in {"", "none", "identity"}
    prealign_alpha = float(cfg.get("prealign_alpha", 1.0))
    random_state = int(cfg["random_state"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"domain adaptation alpha must be in [0, 1], got {alpha}")
    if not 0.0 <= prealign_alpha <= 1.0:
        raise ValueError(f"domain prealignment alpha must be in [0, 1], got {prealign_alpha}")

    reference_mean = X.mean(axis=0)
    reference_cov = _covariance(X, reg=reg)
    reference_std = np.clip(X.std(axis=0), reg, None)
    X_aligned = X.copy()
    group_summaries = []

    mean_gap_before = _weighted_group_mean_gap(X, group_ids, reference_mean)
    probe_before = _safe_domain_probe_accuracy(X, group_ids, random_state=random_state)
    supported_groups = [group for group in unique_groups if int((group_ids == group).sum()) >= min_group_size]
    supported_mask = np.isin(group_ids, supported_groups)

    if method == "dann":
        if len(supported_groups) < 2:
            summary.update(
                {
                    "n_groups": int(len(unique_groups)),
                    "n_supported_groups": int(len(supported_groups)),
                    "reason": "insufficient_supported_groups_for_dann",
                }
            )
            return {
                "X_aligned": X.astype(np.float32, copy=False),
                "summary": summary,
            }

        summary["prealign_method"] = prealign_method if prealign_enabled else None
        summary["prealign_alpha"] = round(prealign_alpha, 4) if prealign_enabled else None
        supported_labels = group_ids[supported_mask]
        label_map = {label: idx for idx, label in enumerate(np.unique(supported_labels))}
        domain_codes = np.array([label_map[label] for label in supported_labels], dtype=np.int64)
        dann_input_supported = X[supported_mask].copy()
        if prealign_enabled:
            for group in supported_groups:
                mask = supported_labels == group
                fully_prealigned = _apply_alignment_method(
                    dann_input_supported[mask],
                    method=prealign_method,
                    reference_mean=reference_mean,
                    reference_cov=reference_cov,
                    reference_std=reference_std,
                    reg=reg,
                )
                dann_input_supported[mask] = _blend_alignment(
                    dann_input_supported[mask],
                    fully_prealigned,
                    alpha=prealign_alpha,
                )

        X_input = X.copy()
        X_input[supported_mask] = dann_input_supported
        mean_gap_input = _weighted_group_mean_gap(X_input, group_ids, reference_mean)
        probe_input = _safe_domain_probe_accuracy(X_input, group_ids, random_state=random_state)
        fully_aligned_supported, adapter_summary = _run_dann_alignment(
            dann_input_supported.astype(np.float32, copy=False),
            domain_codes,
            config=cfg,
        )
        if adapter_summary.get("applied"):
            transformed_supported = _blend_alignment(
                dann_input_supported,
                fully_aligned_supported.astype(np.float64, copy=False),
                alpha=alpha,
            )
            X_aligned[supported_mask] = transformed_supported
        else:
            transformed_supported = dann_input_supported

        for group in unique_groups:
            mask = group_ids == group
            n = int(mask.sum())
            applied = bool(group in supported_groups and adapter_summary.get("applied"))
            input_shift = None
            if group in supported_groups:
                input_shift = float(np.linalg.norm(X_input[mask].mean(axis=0) - reference_mean))
            record = {
                "group": str(group),
                "n": n,
                "applied": applied,
                "mean_shift_l2_before": round(float(np.linalg.norm(X[mask].mean(axis=0) - reference_mean)), 4),
                "mean_shift_l2_after": round(float(np.linalg.norm(X_aligned[mask].mean(axis=0) - reference_mean)), 4),
            }
            if input_shift is not None:
                record["mean_shift_l2_input"] = round(input_shift, 4)
            if not applied:
                record["reason"] = "below_min_group_size" if group not in supported_groups else adapter_summary.get("reason")
            else:
                record["alpha"] = round(alpha, 4)
            group_summaries.append(record)

        mean_gap_after = _weighted_group_mean_gap(X_aligned, group_ids, reference_mean)
        probe_after = _safe_domain_probe_accuracy(X_aligned, group_ids, random_state=random_state)
        summary.update(
            {
                "n_groups": int(len(unique_groups)),
                "n_supported_groups": int(len(supported_groups)),
                "trained_samples": int(supported_mask.sum()),
                "groups": group_summaries,
                "weighted_group_mean_gap_before": round(mean_gap_before, 4),
                "weighted_group_mean_gap_input": round(mean_gap_input, 4),
                "weighted_group_mean_gap_after": round(mean_gap_after, 4),
                "improved_mean_gap": bool(mean_gap_after < mean_gap_before),
                "domain_probe_accuracy_before": None if probe_before is None else round(probe_before, 4),
                "domain_probe_accuracy_input": None if probe_input is None else round(probe_input, 4),
                "domain_probe_accuracy_after": None if probe_after is None else round(probe_after, 4),
                "improved_domain_probe": None if (probe_before is None or probe_after is None) else bool(probe_after < probe_before),
                "adapter_summary": adapter_summary,
            }
        )
    else:
        for group in unique_groups:
            mask = group_ids == group
            n = int(mask.sum())
            if n < min_group_size:
                group_summaries.append(
                    {"group": str(group), "n": n, "applied": False, "reason": "below_min_group_size"}
                )
                continue

            X_group = X[mask]
            fully_aligned = _apply_alignment_method(
                X_group,
                method=method,
                reference_mean=reference_mean,
                reference_cov=reference_cov,
                reference_std=reference_std,
                reg=reg,
            )

            transformed = _blend_alignment(X_group, fully_aligned, alpha=alpha)
            X_aligned[mask] = transformed
            group_summaries.append(
                {
                    "group": str(group),
                    "n": n,
                    "applied": True,
                    "alpha": round(alpha, 4),
                    "mean_shift_l2_before": round(float(np.linalg.norm(X_group.mean(axis=0) - reference_mean)), 4),
                    "mean_shift_l2_after": round(float(np.linalg.norm(transformed.mean(axis=0) - reference_mean)), 4),
                }
            )

        mean_gap_after = _weighted_group_mean_gap(X_aligned, group_ids, reference_mean)
        probe_after = _safe_domain_probe_accuracy(X_aligned, group_ids, random_state=random_state)
        summary.update(
            {
                "n_groups": int(len(unique_groups)),
                "n_supported_groups": int(len(supported_groups)),
                "groups": group_summaries,
                "weighted_group_mean_gap_before": round(mean_gap_before, 4),
                "weighted_group_mean_gap_after": round(mean_gap_after, 4),
                "improved_mean_gap": bool(mean_gap_after < mean_gap_before),
                "domain_probe_accuracy_before": None if probe_before is None else round(probe_before, 4),
                "domain_probe_accuracy_after": None if probe_after is None else round(probe_after, 4),
                "improved_domain_probe": None if (probe_before is None or probe_after is None) else bool(probe_after < probe_before),
            }
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "domain_adaptation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        summary["report_path"] = str(summary_path)

    return {
        "X_aligned": X_aligned.astype(np.float32, copy=False),
        "summary": summary,
    }
