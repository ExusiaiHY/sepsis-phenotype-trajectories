from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s4.treatment_aware_model import TreatmentAwareEncoder
from s5.realtime_model import CausalTCNStudentEncoder
from s6.subtype_metadata import (
    TASK_ALIASES,
    build_classification_task_metadata,
    build_regression_task_metadata,
    default_metadata_bundle,
)


class MaskedNPZMultitaskStudent(nn.Module):
    def __init__(
        self,
        *,
        n_cont_features: int,
        n_treat_features: int,
        classification_tasks: list[dict[str, Any]],
        regression_tasks: list[dict[str, Any]],
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
    ) -> dict[str, Any]:
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


class SepsisSubtypeInferenceEngine:
    def __init__(
        self,
        *,
        model_dir: Path | str,
        schema_path: Path | str | None = None,
        device: str = "cpu",
    ):
        self.model_dir = Path(model_dir)
        self.schema_path = Path(schema_path) if schema_path is not None else self._default_schema_path()
        self.device = str(device)
        self.checkpoint_path = self.model_dir / "multitask_student.pt"
        self.report_path = self.model_dir / "multitask_student_report.json"
        self.feat_mean_path = self.model_dir / "feat_mean.npy"
        self.feat_std_path = self.model_dir / "feat_std.npy"

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.config = dict(checkpoint["config"])
        self.feat_mean_path, self.feat_std_path = self._resolve_feature_stats_paths()
        self.schema = self._load_schema()

        self.classification_tasks = build_classification_task_metadata(
            [dict(task) for task in self.config.get("classification_tasks", [])],
            self.schema.get("classification_lookup", {}),
        )
        self.regression_tasks = build_regression_task_metadata(
            [dict(task) for task in self.config.get("regression_tasks", [])],
            self.schema.get("regression_lookup", {}),
        )
        self.metadata_bundle = default_metadata_bundle(
            self.classification_tasks,
            self.regression_tasks,
            {
                **self.schema.get("classification_lookup", {}),
                **self.schema.get("regression_lookup", {}),
            },
        )

        self.model = MaskedNPZMultitaskStudent(
            n_cont_features=int(self.config["n_cont_features"]),
            n_treat_features=int(self.config["n_treat_features"]),
            classification_tasks=self.classification_tasks,
            regression_tasks=self.regression_tasks,
            note_dim=int(self.config.get("note_dim", 0)),
            student_arch=self.config.get("student_arch", "transformer"),
            student_d_model=int(self.config.get("student_d_model", 64)),
            teacher_dim=int(self.config.get("teacher_dim", 0)),
            n_heads=int(self.config.get("n_heads", 4)),
            n_layers=int(self.config.get("n_layers", 1)),
            d_ff=int(self.config.get("d_ff", 128)),
            dropout=float(self.config.get("dropout", 0.1)),
            max_seq_len=int(self.config.get("max_seq_len", 48)),
            treatment_layers=int(self.config.get("treatment_layers", 1)),
            head_hidden_dim=int(self.config.get("head_hidden_dim", 64)),
            head_dropout=float(self.config.get("head_dropout", 0.1)),
            tcn_kernel_size=int(self.config.get("tcn_kernel_size", 3)),
            tcn_dilations=tuple(int(value) for value in self.config.get("tcn_dilations", [1, 2, 4, 8])),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.feat_mean = np.load(self.feat_mean_path).astype(np.float32)
        self.feat_std = np.load(self.feat_std_path).astype(np.float32)
        self.report = json.loads(self.report_path.read_text(encoding="utf-8")) if self.report_path.exists() else {}
        self.mortality_threshold = float(
            self.report.get("threshold_selection", {}).get("selected_threshold", 0.5)
        )

        self.classification_task_map = {task["task_name"]: task for task in self.classification_tasks}
        self.regression_task_map = {task["task_name"]: task for task in self.regression_tasks}

    def _default_schema_path(self) -> Path:
        candidate = self.model_dir / "sepsis_multitask_schema.json"
        if candidate.exists():
            return candidate
        return Path(__file__).resolve().parent.parent / "data" / "processed_mimic_enhanced" / "sepsis_multitask_schema.json"

    def _resolve_feature_stats_paths(self) -> tuple[Path, Path]:
        if self.feat_mean_path.exists() and self.feat_std_path.exists():
            return self.feat_mean_path, self.feat_std_path

        project_root = Path(__file__).resolve().parent.parent
        fallback_dirs = [
            project_root / "data" / "s6_multitask_smoke_20260403_npz",
            project_root / "data" / "s6_multitask_smoke_20260403",
            project_root / "data" / "s6_multitask_smoke",
            project_root / "data" / "s6_multitask_mimic_cloud_v2",
        ]
        n_cont = int(self.config["n_cont_features"])
        for fallback_dir in fallback_dirs:
            mean_path = fallback_dir / "feat_mean.npy"
            std_path = fallback_dir / "feat_std.npy"
            if not mean_path.exists() or not std_path.exists():
                continue
            try:
                if np.load(mean_path).shape[0] == n_cont and np.load(std_path).shape[0] == n_cont:
                    return mean_path, std_path
            except Exception:
                continue

        recomputed_mean, recomputed_std = self._recompute_feature_stats()
        np.save(self.feat_mean_path, recomputed_mean)
        np.save(self.feat_std_path, recomputed_std)
        return self.feat_mean_path, self.feat_std_path

    def _recompute_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        project_root = Path(__file__).resolve().parent.parent
        time_series_path = project_root / "data" / "processed_mimic_enhanced" / "time_series_enhanced.npy"
        if not time_series_path.exists():
            raise FileNotFoundError(
                f"Feature normalization arrays not found under {self.model_dir}, and {time_series_path} is unavailable for recomputation"
            )
        time_series = np.load(time_series_path, mmap_mode="r")
        n_cont = int(self.config["n_cont_features"])
        continuous = time_series[:, :, :n_cont]
        feat_mean = np.zeros(n_cont, dtype=np.float32)
        feat_std = np.ones(n_cont, dtype=np.float32)
        for feature_idx in range(n_cont):
            values = continuous[:, :, feature_idx]
            valid = values[np.isfinite(values)]
            if valid.size == 0:
                continue
            feat_mean[feature_idx] = float(valid.mean())
            feat_std[feature_idx] = max(float(valid.std()), 1.0e-6)
        return feat_mean, feat_std

    def _load_schema(self) -> dict[str, Any]:
        if self.schema_path.exists():
            schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        else:
            schema = {}
        return {
            "raw": schema,
            "classification_lookup": {
                str(task["name"]): dict(task) for task in schema.get("classification_tasks", [])
            },
            "regression_lookup": {
                str(task["name"]): dict(task) for task in schema.get("regression_tasks", [])
            },
            "legacy_aliases": dict(schema.get("legacy_aliases", TASK_ALIASES)),
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": True,
            "device": self.device,
            "model_dir": str(self.model_dir),
            "model": self.model_metadata(),
        }

    def model_metadata(self) -> dict[str, Any]:
        deployment = self.report.get("deployment", {})
        training = self.report.get("training", {})
        return {
            "name": "S6 Masked-NPZ Mainline",
            "targets_source": self.report.get("targets_source", "unknown"),
            "student_arch": self.config.get("student_arch"),
            "student_d_model": self.config.get("student_d_model"),
            "max_seq_len": self.config.get("max_seq_len"),
            "n_cont_features": self.config.get("n_cont_features"),
            "n_treat_features": self.config.get("n_treat_features"),
            "param_count": deployment.get("float_n_parameters"),
            "cpu_latency_ms_per_sample": deployment.get("cpu_latency_ms_per_sample"),
            "epochs_trained": training.get("epochs_trained"),
            "threshold_selection": self.report.get("threshold_selection", {}),
            "report_path": str(self.report_path) if self.report_path.exists() else None,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "contract_version": "s6-masked-npz-v1",
            "model": self.model_metadata(),
            "families": self.metadata_bundle["families"],
            "classification_tasks": self.classification_tasks,
            "regression_tasks": self.regression_tasks,
            "legacy_aliases": self.schema.get("legacy_aliases", TASK_ALIASES),
            "frontend_notes": [
                "clinical phenotype family = alpha / beta / gamma / delta",
                "trajectory phenotype family = Trajectory A / B / C / D",
                "these two families should be rendered separately in the frontend",
            ],
        }

    def preprocess(self, batch: np.ndarray) -> dict[str, torch.Tensor]:
        n_cont = int(self.config["n_cont_features"])
        n_treat = int(self.config["n_treat_features"])
        total_f = n_cont + n_treat
        max_seq_len = int(self.config.get("max_seq_len", batch.shape[1]))
        if batch.ndim != 3:
            raise ValueError(f"time_series must be 3D (B, T, F), got shape {batch.shape}")
        if batch.shape[-1] != total_f:
            raise ValueError(f"Expected last dim={total_f}, got {batch.shape[-1]}")
        if batch.shape[1] > max_seq_len:
            raise ValueError(f"Sequence length {batch.shape[1]} exceeds max_seq_len={max_seq_len}")
        continuous = batch[:, :, :n_cont].astype(np.float32, copy=False)
        treatments = batch[:, :, n_cont:n_cont + n_treat].astype(np.float32, copy=False)
        mask_cont = np.isfinite(continuous).astype(np.float32)
        mask_treat = np.isfinite(treatments).astype(np.float32)
        continuous = np.nan_to_num(continuous, nan=0.0)
        treatments = np.nan_to_num(treatments, nan=0.0)
        continuous = (continuous - self.feat_mean[None, None, :]) / np.maximum(self.feat_std[None, None, :], 1.0e-6)
        return {
            "x": torch.from_numpy(continuous).to(self.device),
            "mask": torch.from_numpy(mask_cont).to(self.device),
            "treatments": torch.from_numpy(treatments).to(self.device),
            "treatment_mask": torch.from_numpy(mask_treat).to(self.device),
        }

    def predict_batch(
        self,
        batch: np.ndarray,
        *,
        mortality_threshold: float | None = None,
    ) -> dict[str, Any]:
        threshold = float(self.mortality_threshold if mortality_threshold is None else mortality_threshold)
        inputs = self.preprocess(batch)
        with torch.no_grad():
            output = self.model(**inputs)
        mortality_probs = torch.sigmoid(output["logits_mortality"]).detach().cpu().numpy()
        classification_probs = {
            name: F.softmax(logits, dim=-1).detach().cpu().numpy()
            for name, logits in output["classification_logits"].items()
        }
        regression_outputs = {
            name: values.detach().cpu().numpy()
            for name, values in output["regression_outputs"].items()
        }

        predictions = []
        batch_size = batch.shape[0]
        for row_idx in range(batch_size):
            classification_predictions = {
                task_name: self._format_classification_prediction(task_name, probs[row_idx])
                for task_name, probs in classification_probs.items()
            }
            regression_predictions = {
                task_name: self._format_regression_prediction(task_name, float(values[row_idx]))
                for task_name, values in regression_outputs.items()
            }
            families = self._build_family_view(classification_predictions, regression_predictions)
            predictions.append(
                {
                    "mortality": {
                        "probability": round(float(mortality_probs[row_idx]), 6),
                        "threshold": threshold,
                        "predicted_positive": bool(float(mortality_probs[row_idx]) >= threshold),
                    },
                    "classification_tasks": classification_predictions,
                    "regression_tasks": regression_predictions,
                    "families": families,
                    "legacy_outputs": self._build_legacy_outputs(classification_predictions),
                }
            )

        return {
            "model": self.model_metadata(),
            "predictions": predictions,
        }

    def _format_classification_prediction(self, task_name: str, probabilities: np.ndarray) -> dict[str, Any]:
        task_meta = self.classification_task_map[task_name]
        predicted_index = int(np.argmax(probabilities))
        class_probs = []
        for class_meta, probability in zip(task_meta["classes"], probabilities.tolist()):
            class_probs.append(
                {
                    "index": class_meta["index"],
                    "label": class_meta["label"],
                    "display_name": class_meta.get("display_name", class_meta["label"]),
                    "display_name_zh": class_meta.get("display_name_zh", class_meta["label"]),
                    "description": class_meta.get("description", ""),
                    "probability": round(float(probability), 6),
                    "recommendations": class_meta.get("recommendations", []),
                    "monitoring": class_meta.get("monitoring", []),
                }
            )
        predicted_class = class_probs[predicted_index]
        return {
            "task_name": task_name,
            "display_name": task_meta.get("display_name", task_name),
            "display_name_zh": task_meta.get("display_name_zh", task_name),
            "family_id": task_meta.get("family_id"),
            "family_display_name": task_meta.get("family_display_name", task_meta.get("family_id")),
            "family_display_name_zh": task_meta.get("family_display_name_zh", task_meta.get("family_display_name", task_meta.get("family_id"))),
            "description": task_meta.get("description", ""),
            "frontend_note": task_meta.get("frontend_note", ""),
            "predicted_index": predicted_index,
            "predicted_label": predicted_class["label"],
            "predicted_display_name": predicted_class["display_name"],
            "predicted_display_name_zh": predicted_class.get("display_name_zh"),
            "predicted_probability": predicted_class["probability"],
            "probabilities": class_probs,
        }

    def _format_regression_prediction(self, task_name: str, value: float) -> dict[str, Any]:
        task_meta = self.regression_task_map[task_name]
        return {
            "task_name": task_name,
            "display_name": task_meta.get("display_name", task_name),
            "display_name_zh": task_meta.get("display_name_zh", task_name),
            "family_id": task_meta.get("family_id"),
            "description": task_meta.get("description", ""),
            "value": round(float(value), 6),
        }

    def _build_family_view(
        self,
        classification_predictions: dict[str, dict[str, Any]],
        regression_predictions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        family_view = {
            "gold_standard": {
                "gold_mals": classification_predictions.get("gold_mals"),
                "gold_immunoparalysis": classification_predictions.get("gold_immunoparalysis"),
            },
            "immune_state": classification_predictions.get("proxy_immune_state"),
            "clinical_phenotype": classification_predictions.get("proxy_clinical_phenotype"),
            "trajectory_phenotype": classification_predictions.get("proxy_trajectory_phenotype"),
            "fluid_strategy": classification_predictions.get("proxy_fluid_strategy"),
            "scores": {
                "immune": {
                    key: regression_predictions[key]
                    for key in ["score_mals", "score_immunoparalysis"]
                    if key in regression_predictions
                },
                "clinical_phenotype": {
                    key: regression_predictions[key]
                    for key in ["score_alpha", "score_beta", "score_gamma", "score_delta"]
                    if key in regression_predictions
                },
                "trajectory_phenotype": {
                    key: regression_predictions[key]
                    for key in ["score_trajectory_a", "score_trajectory_b", "score_trajectory_c", "score_trajectory_d"]
                    if key in regression_predictions
                },
                "fluid_strategy": {
                    key: regression_predictions[key]
                    for key in ["score_restrictive_fluid_benefit", "score_resuscitation_fluid_benefit"]
                    if key in regression_predictions
                },
            },
        }
        return family_view

    def _build_legacy_outputs(self, classification_predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        legacy = {}
        for alias, task_name in self.schema.get("legacy_aliases", TASK_ALIASES).items():
            prediction = classification_predictions.get(task_name)
            if prediction is None:
                continue
            legacy[alias] = {
                "task_name": task_name,
                "probabilities": [item["probability"] for item in prediction["probabilities"]],
                "predicted_label": prediction["predicted_label"],
            }
        return legacy
