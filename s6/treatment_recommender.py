"""
treatment_recommender.py - Rule-based treatment recommendation engine for sepsis subtype outputs.

Supports both the new masked-NPZ multi-task output schema and the legacy
immune / organ / fluid three-head interface.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from s6.subtype_metadata import RULE_LABEL_ALIASES

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "treatment_rules.json"


def _normalize(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


class SubtypeTreatmentRecommender:
    """Recommender that maps subtype predictions to frontend-friendly treatment guidance."""

    def __init__(self, rules_path: Path | str | None = None):
        self.rules_path = Path(rules_path) if rules_path is not None else DEFAULT_RULES_PATH
        if self.rules_path.exists():
            self.rules = json.loads(self.rules_path.read_text(encoding="utf-8"))
        else:
            self.rules = {
                "subtypes": {},
                "combination_rules": {},
                "disclaimer": (
                    "Recommendations are generated from subtype metadata and should be reviewed "
                    "by qualified clinicians before use."
                ),
            }

    def recommend(
        self,
        *,
        immune_probs: np.ndarray | list[float],
        organ_probs: np.ndarray | list[float],
        fluid_probs: np.ndarray | list[float],
        mortality_prob: float | None = None,
        top_k: int = 2,
        prob_threshold: float = 0.3,
    ) -> dict[str, Any]:
        prediction = {
            "mortality": {
                "probability": None if mortality_prob is None else float(mortality_prob),
            },
            "classification_tasks": {
                "proxy_immune_state": self._legacy_task_prediction(
                    task_name="proxy_immune_state",
                    labels=["Unclassified", "immunoparalysis-like", "MALS-like"],
                    probabilities=np.asarray(immune_probs, dtype=float),
                ),
                "proxy_clinical_phenotype": self._legacy_task_prediction(
                    task_name="proxy_clinical_phenotype",
                    labels=["Unclassified", "alpha-like", "beta-like", "gamma-like", "delta-like"],
                    probabilities=np.asarray(organ_probs, dtype=float),
                ),
                "proxy_fluid_strategy": self._legacy_task_prediction(
                    task_name="proxy_fluid_strategy",
                    labels=["Unclassified", "restrictive-fluid-benefit-like", "resuscitation-fluid-benefit-like"],
                    probabilities=np.asarray(fluid_probs, dtype=float),
                ),
            },
        }
        return self.recommend_from_prediction(prediction, top_k=top_k, prob_threshold=prob_threshold)

    def recommend_from_prediction(
        self,
        prediction: dict[str, Any],
        *,
        top_k: int = 2,
        prob_threshold: float = 0.3,
    ) -> dict[str, Any]:
        classification_tasks = prediction.get("classification_tasks", {})
        mortality_prob = prediction.get("mortality", {}).get("probability")

        family_recommendations: dict[str, Any] = {}
        action_plan: list[str] = []
        monitoring_plan: list[str] = []
        alerts: list[dict[str, Any]] = []

        task_order = [
            "gold_mals",
            "gold_immunoparalysis",
            "proxy_immune_state",
            "proxy_clinical_phenotype",
            "proxy_trajectory_phenotype",
            "proxy_fluid_strategy",
        ]

        for task_name in task_order:
            task_prediction = classification_tasks.get(task_name)
            if not task_prediction:
                continue
            active_classes = self._active_classes(task_prediction, prob_threshold=prob_threshold, top_k=top_k)
            if not active_classes:
                continue
            items = [self._build_item(task_name, active) for active in active_classes]
            family_id = str(task_prediction.get("family_id", task_name))
            family_recommendations[family_id] = {
                "family_id": family_id,
                "family_display_name": task_prediction.get("family_display_name", family_id),
                "family_display_name_zh": task_prediction.get("family_display_name_zh", family_id),
                "task_name": task_name,
                "top_prediction": {
                    "label": task_prediction.get("predicted_label"),
                    "display_name": task_prediction.get("predicted_display_name"),
                    "display_name_zh": task_prediction.get("predicted_display_name_zh"),
                    "probability": task_prediction.get("predicted_probability"),
                },
                "items": items,
            }
            for item in items:
                action_plan.extend(item.get("actions", []))
                monitoring_plan.extend(item.get("monitoring", []))

        alerts.extend(self._mortality_alerts(mortality_prob))
        alerts.extend(self._combination_alerts(classification_tasks))

        return {
            "probability_threshold": prob_threshold,
            "family_recommendations": family_recommendations,
            "action_plan": _dedupe(action_plan),
            "monitoring_plan": _dedupe(monitoring_plan),
            "alerts": alerts,
            "summary": self._build_summary(classification_tasks, mortality_prob, family_recommendations, alerts),
            "disclaimer": self.rules.get("disclaimer", ""),
        }

    def _legacy_task_prediction(
        self,
        *,
        task_name: str,
        labels: list[str],
        probabilities: np.ndarray,
    ) -> dict[str, Any]:
        predicted_index = int(np.argmax(probabilities))
        classes = []
        for idx, (label, probability) in enumerate(zip(labels, probabilities.tolist())):
            classes.append(
                {
                    "index": idx,
                    "label": label,
                    "display_name": label,
                    "display_name_zh": label,
                    "description": "",
                    "probability": round(float(probability), 6),
                    "recommendations": [],
                    "monitoring": [],
                }
            )
        top = classes[predicted_index]
        family_map = {
            "proxy_immune_state": ("immune_state", "Immune State", "免疫状态"),
            "proxy_clinical_phenotype": ("clinical_phenotype", "Clinical Phenotype", "临床器官表型"),
            "proxy_fluid_strategy": ("fluid_strategy", "Fluid Strategy", "液体策略"),
        }
        family_id, family_display_name, family_display_name_zh = family_map[task_name]
        return {
            "task_name": task_name,
            "family_id": family_id,
            "family_display_name": family_display_name,
            "family_display_name_zh": family_display_name_zh,
            "predicted_index": predicted_index,
            "predicted_label": top["label"],
            "predicted_display_name": top["display_name"],
            "predicted_display_name_zh": top["display_name_zh"],
            "predicted_probability": top["probability"],
            "probabilities": classes,
        }

    def _active_classes(
        self,
        task_prediction: dict[str, Any],
        *,
        prob_threshold: float,
        top_k: int,
    ) -> list[dict[str, Any]]:
        classes = sorted(
            task_prediction.get("probabilities", []),
            key=lambda item: float(item.get("probability", 0.0)),
            reverse=True,
        )
        active = [item for item in classes if float(item.get("probability", 0.0)) >= prob_threshold][:top_k]
        if not active and classes:
            active = classes[:1]
        return active

    def _build_item(self, task_name: str, class_entry: dict[str, Any]) -> dict[str, Any]:
        rule = self._lookup_rule(task_name, str(class_entry.get("label", "")))
        fallback_actions = list(class_entry.get("recommendations", []))
        fallback_monitoring = list(class_entry.get("monitoring", []))

        if rule is not None:
            treatments = rule.get("treatments", [])
            actions = []
            monitoring = []
            for treatment in treatments:
                actions.append(f"{treatment.get('category', '治疗')}: {treatment.get('name', '')}".strip(": "))
                actions.extend(treatment.get("actions", []))
                monitoring.extend(treatment.get("monitoring", []))
            actions.extend(fallback_actions)
            monitoring.extend(fallback_monitoring)
            return {
                "label": class_entry.get("label"),
                "display_name": rule.get("display_name", class_entry.get("display_name")),
                "display_name_zh": class_entry.get("display_name_zh"),
                "probability": class_entry.get("probability"),
                "description": class_entry.get("description", ""),
                "evidence_level": rule.get("evidence_level"),
                "mortality_reference": rule.get("mortality_reference"),
                "actions": _dedupe(actions),
                "monitoring": _dedupe(monitoring),
                "source": "config_rule",
            }

        return {
            "label": class_entry.get("label"),
            "display_name": class_entry.get("display_name"),
            "display_name_zh": class_entry.get("display_name_zh"),
            "probability": class_entry.get("probability"),
            "description": class_entry.get("description", ""),
            "evidence_level": "metadata_default",
            "mortality_reference": None,
            "actions": _dedupe(fallback_actions),
            "monitoring": _dedupe(fallback_monitoring),
            "source": "metadata_default",
        }

    def _lookup_rule(self, task_name: str, label: str) -> dict[str, Any] | None:
        section_name = {
            "proxy_immune_state": "immune",
            "proxy_clinical_phenotype": "organ",
            "proxy_fluid_strategy": "fluid",
        }.get(task_name)
        if section_name is None:
            return None
        subtype_rules = self.rules.get("subtypes", {}).get(section_name, {})
        normalized_label = _normalize(label)
        candidates = [label]
        alias_map = RULE_LABEL_ALIASES.get(task_name, {})
        candidates.extend(alias_map.get(normalized_label, []))
        for candidate in candidates:
            for rule_key, rule_value in subtype_rules.items():
                if _normalize(rule_key) == _normalize(candidate):
                    return rule_value
        return None

    def _mortality_alerts(self, mortality_prob: float | None) -> list[dict[str, Any]]:
        if mortality_prob is None:
            return []
        if mortality_prob >= 0.7:
            return [{
                "level": "high",
                "title": "High Mortality Risk",
                "title_zh": "高死亡风险",
                "note": f"Predicted 28-day mortality risk is {mortality_prob:.1%}. Escalate monitoring and senior review.",
            }]
        if mortality_prob >= 0.4:
            return [{
                "level": "medium",
                "title": "Moderate Mortality Risk",
                "title_zh": "中等死亡风险",
                "note": f"Predicted 28-day mortality risk is {mortality_prob:.1%}. Maintain close reassessment.",
            }]
        return []

    def _combination_alerts(self, classification_tasks: dict[str, Any]) -> list[dict[str, Any]]:
        immune = classification_tasks.get("proxy_immune_state", {}).get("predicted_label")
        organ = classification_tasks.get("proxy_clinical_phenotype", {}).get("predicted_label")
        if immune is None or organ is None:
            return []
        special = self.rules.get("combination_rules", {}).get("special_combinations", {})
        candidates = [
            f"{immune} + {organ}",
            f"{_label_to_legacy_name('proxy_immune_state', immune)} + {_label_to_legacy_name('proxy_clinical_phenotype', organ)}",
        ]
        for candidate in candidates:
            for rule_key, rule_value in special.items():
                if _normalize(rule_key) == _normalize(candidate):
                    return [{
                        "level": rule_value.get("risk_level", "high"),
                        "title": "Subtype Combination Alert",
                        "title_zh": "亚型组合提示",
                        "note": rule_value.get("note", ""),
                        "combination": candidate,
                    }]
        return []

    def _build_summary(
        self,
        classification_tasks: dict[str, Any],
        mortality_prob: float | None,
        family_recommendations: dict[str, Any],
        alerts: list[dict[str, Any]],
    ) -> str:
        parts = []
        immune = classification_tasks.get("proxy_immune_state")
        clinical = classification_tasks.get("proxy_clinical_phenotype")
        trajectory = classification_tasks.get("proxy_trajectory_phenotype")
        fluid = classification_tasks.get("proxy_fluid_strategy")
        if immune:
            parts.append(f"免疫状态: {immune.get('predicted_display_name_zh') or immune.get('predicted_display_name')}")
        if clinical:
            parts.append(f"临床器官表型: {clinical.get('predicted_display_name_zh') or clinical.get('predicted_display_name')}")
        if trajectory:
            parts.append(f"生命体征轨迹: {trajectory.get('predicted_display_name_zh') or trajectory.get('predicted_display_name')}")
        if fluid:
            parts.append(f"液体策略: {fluid.get('predicted_display_name_zh') or fluid.get('predicted_display_name')}")
        summary = "；".join(parts) if parts else "未检测到可解释的亚型输出。"
        if mortality_prob is not None:
            summary += f" 28天死亡风险约 {mortality_prob:.1%}。"
        if alerts:
            summary += " 需关注高风险提示。"
        if family_recommendations:
            summary += " 推荐动作已按亚型家族分组返回。"
        return summary


def recommend_from_model_output(
    model_output: dict[str, Any],
    recommender: SubtypeTreatmentRecommender | None = None,
    temperature: float = 1.0,
) -> dict[str, Any]:
    import torch

    if recommender is None:
        recommender = SubtypeTreatmentRecommender()

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    if "classification_logits" in model_output:
        mortality_prob = None
        if "logits_mortality" in model_output:
            val = _to_numpy(model_output["logits_mortality"])
            mortality_prob = float(1.0 / (1.0 + np.exp(-np.ravel(val)[0] / max(temperature, 1.0e-3))))
        classification_tasks = {}
        for task_name, logits in model_output["classification_logits"].items():
            probs = _softmax(_to_numpy(logits))
            if probs.ndim == 2:
                probs = probs[0]
            labels = [f"class-{idx}" for idx in range(probs.shape[-1])]
            predicted_index = int(np.argmax(probs))
            class_entries = [
                {
                    "index": idx,
                    "label": label,
                    "display_name": label,
                    "display_name_zh": label,
                    "probability": round(float(prob), 6),
                }
                for idx, (label, prob) in enumerate(zip(labels, probs.tolist()))
            ]
            classification_tasks[str(task_name)] = {
                "task_name": str(task_name),
                "family_id": str(task_name),
                "family_display_name": str(task_name),
                "family_display_name_zh": str(task_name),
                "predicted_index": predicted_index,
                "predicted_label": class_entries[predicted_index]["label"],
                "predicted_display_name": class_entries[predicted_index]["display_name"],
                "predicted_display_name_zh": class_entries[predicted_index]["display_name_zh"],
                "predicted_probability": class_entries[predicted_index]["probability"],
                "probabilities": class_entries,
            }
        return recommender.recommend_from_prediction(
            {
                "mortality": {"probability": mortality_prob},
                "classification_tasks": classification_tasks,
            }
        )

    mortality_prob = None
    if "logits_mortality" in model_output:
        val = _to_numpy(model_output["logits_mortality"])
        mortality_prob = float(1.0 / (1.0 + np.exp(-np.ravel(val)[0] / max(temperature, 1.0e-3))))

    immune_probs = _softmax(_to_numpy(model_output["logits_immune"]))
    organ_probs = _softmax(_to_numpy(model_output["logits_organ"]))
    fluid_probs = _softmax(_to_numpy(model_output["logits_fluid"]))

    if immune_probs.ndim == 2:
        immune_probs = immune_probs[0]
    if organ_probs.ndim == 2:
        organ_probs = organ_probs[0]
    if fluid_probs.ndim == 2:
        fluid_probs = fluid_probs[0]

    return recommender.recommend(
        immune_probs=immune_probs,
        organ_probs=organ_probs,
        fluid_probs=fluid_probs,
        mortality_prob=mortality_prob,
    )


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if not item:
            continue
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _label_to_legacy_name(task_name: str, label: str) -> str:
    normalized = _normalize(label)
    alias_map = RULE_LABEL_ALIASES.get(task_name, {})
    aliases = alias_map.get(normalized, [])
    if aliases:
        return aliases[0]
    return label


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

