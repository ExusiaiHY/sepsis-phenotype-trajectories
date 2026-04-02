"""
treatment_recommender.py - Rule-based treatment recommendation engine for sepsis subtypes.

Maps multi-task model outputs to evidence-based recommendations using a structured
JSON rule base. Produces human-readable, auditable recommendations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "treatment_rules.json"

IMMUNE_NAMES = ["Unclassified", "EIL-like", "MAS-like"]
ORGAN_NAMES = ["Unclassified", "alpha-like", "beta-like", "gamma-like", "delta-like"]
FLUID_NAMES = ["Unclassified", "low_benefit", "high_benefit"]


class SubtypeTreatmentRecommender:
    """Recommender loaded from config/treatment_rules.json."""

    def __init__(self, rules_path: Path | str | None = None):
        self.rules_path = Path(rules_path) if rules_path is not None else DEFAULT_RULES_PATH
        with open(self.rules_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

    def recommend(
        self,
        *,
        immune_probs: np.ndarray | list[float],
        organ_probs: np.ndarray | list[float],
        fluid_probs: np.ndarray | list[float],
        mortality_prob: float | None = None,
        top_k: int = 2,
        prob_threshold: float = 0.3,
    ) -> dict:
        """
        Generate recommendations from subtype probability vectors.

        Parameters
        ----------
        immune_probs : array-like, shape (3,)
        organ_probs : array-like, shape (5,)
        fluid_probs : array-like, shape (3,)
        mortality_prob : float | None
        top_k : int
            Number of top predicted subtypes to include per dimension.
        prob_threshold : float
            Minimum probability to report a subtype as "active".

        Returns
        -------
        dict with structured recommendations and human-readable summary.
        """
        immune_probs = np.asarray(immune_probs, dtype=float)
        organ_probs = np.asarray(organ_probs, dtype=float)
        fluid_probs = np.asarray(fluid_probs, dtype=float)

        immune_pred = int(immune_probs.argmax())
        organ_pred = int(organ_probs.argmax())
        fluid_pred = int(fluid_probs.argmax())

        # Build active subtype list (above threshold)
        active = {
            "immune": [
                IMMUNE_NAMES[i]
                for i in np.argsort(immune_probs)[::-1][:top_k]
                if immune_probs[i] >= prob_threshold
            ],
            "organ": [
                ORGAN_NAMES[i]
                for i in np.argsort(organ_probs)[::-1][:top_k]
                if organ_probs[i] >= prob_threshold
            ],
            "fluid": [
                FLUID_NAMES[i]
                for i in np.argsort(fluid_probs)[::-1][:top_k]
                if fluid_probs[i] >= prob_threshold
            ],
        }

        # Retrieve recommendations per active subtype
        recs = {
            "immune": [],
            "organ": [],
            "fluid": [],
        }
        for st in active["immune"]:
            rule = self.rules["subtypes"]["immune"].get(st)
            if rule:
                recs["immune"].append({"subtype": st, **self._summarize_rule(rule)})
        for st in active["organ"]:
            rule = self.rules["subtypes"]["organ"].get(st)
            if rule:
                recs["organ"].append({"subtype": st, **self._summarize_rule(rule)})
        for st in active["fluid"]:
            rule = self.rules["subtypes"]["fluid"].get(st)
            if rule:
                recs["fluid"].append({"subtype": st, **self._summarize_rule(rule)})

        # Special combination alerts
        combo_alerts = []
        immune_name = IMMUNE_NAMES[immune_pred]
        organ_name = ORGAN_NAMES[organ_pred]
        special = self.rules.get("combination_rules", {}).get("special_combinations", {})
        combo_key = f"{immune_name} + {organ_name}"
        if combo_key in special:
            combo_alerts.append({
                "combination": combo_key,
                **special[combo_key],
            })

        summary = self._build_summary(
            immune_pred=immune_pred,
            organ_pred=organ_pred,
            fluid_pred=fluid_pred,
            immune_probs=immune_probs,
            organ_probs=organ_probs,
            fluid_probs=fluid_probs,
            mortality_prob=mortality_prob,
            recs=recs,
            combo_alerts=combo_alerts,
        )

        return {
            "active_subtypes": active,
            "predictions": {
                "immune": {"label": IMMUNE_NAMES[immune_pred], "probability": round(float(immune_probs[immune_pred]), 4)},
                "organ": {"label": ORGAN_NAMES[organ_pred], "probability": round(float(organ_probs[organ_pred]), 4)},
                "fluid": {"label": FLUID_NAMES[fluid_pred], "probability": round(float(fluid_probs[fluid_pred]), 4)},
                "mortality": None if mortality_prob is None else round(float(mortality_prob), 4),
            },
            "recommendations": recs,
            "combination_alerts": combo_alerts,
            "human_summary": summary,
            "disclaimer": self.rules.get("disclaimer", ""),
        }

    def _summarize_rule(self, rule: dict) -> dict:
        """Extract actionable fields from a rule entry."""
        return {
            "display_name": rule.get("display_name", ""),
            "evidence_level": rule.get("evidence_level", "N/A"),
            "mortality_reference": rule.get("mortality_reference", "N/A"),
            "treatments": rule.get("treatments", []),
        }

    def _build_summary(
        self,
        *,
        immune_pred: int,
        organ_pred: int,
        fluid_pred: int,
        immune_probs: np.ndarray,
        organ_probs: np.ndarray,
        fluid_probs: np.ndarray,
        mortality_prob: float | None,
        recs: dict,
        combo_alerts: list[dict],
    ) -> str:
        lines = []
        lines.append(
            f"该患者最可能的亚型组合为："
            f"免疫-{IMMUNE_NAMES[immune_pred]}({immune_probs[immune_pred]:.1%}) / "
            f"器官-{ORGAN_NAMES[organ_pred]}({organ_probs[organ_pred]:.1%}) / "
            f"液体-{FLUID_NAMES[fluid_pred]}({fluid_probs[fluid_pred]:.1%})。"
        )
        if mortality_prob is not None:
            lines.append(f"模型估计的28天死亡风险为 {mortality_prob:.1%}。")

        if combo_alerts:
            lines.append("⚠️ 特殊风险提示：")
            for alert in combo_alerts:
                lines.append(f"  - {alert['combination']}：{alert.get('note', '')}")

        lines.append("\n推荐治疗方案：")
        for dim, items in recs.items():
            if not items:
                continue
            lines.append(f"\n【{self._dim_display(dim)}】")
            for item in items:
                lines.append(f"  ▪ {item['display_name']} (证据级别: {item['evidence_level']})")
                for tx in item.get("treatments", []):
                    lines.append(f"    - {tx.get('category', '')}: {tx.get('name', '')}")
                    if "rationale" in tx:
                        lines.append(f"      依据：{tx['rationale']}")
                    if "actions" in tx:
                        lines.append(f"      措施：{'; '.join(tx['actions'])}")
                    if "monitoring" in tx:
                        lines.append(f"      监测：{'; '.join(tx['monitoring'])}")

        lines.append(f"\n{self.rules.get('disclaimer', '')}")
        return "\n".join(lines)

    def _dim_display(self, dim: str) -> str:
        return {"immune": "免疫内型", "organ": "器官主导型", "fluid": "液体策略"}.get(dim, dim)


def recommend_from_model_output(
    model_output: dict,
    recommender: SubtypeTreatmentRecommender | None = None,
    temperature: float = 1.0,
) -> dict:
    """
    Convenience wrapper that consumes raw multi-task model logits/probs.

    model_output expected keys:
      - logits_mortality (tensor or float)
      - logits_immune   (tensor, shape (3,))
      - logits_organ    (tensor, shape (5,))
      - logits_fluid    (tensor, shape (3,))
    """
    import torch

    if recommender is None:
        recommender = SubtypeTreatmentRecommender()

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    mortality_prob = None
    if "logits_mortality" in model_output:
        val = _to_numpy(model_output["logits_mortality"])
        if val.ndim == 0:
            mortality_prob = float(1.0 / (1.0 + np.exp(-val / max(temperature, 1e-3))))
        else:
            mortality_prob = float(1.0 / (1.0 + np.exp(-val[0] / max(temperature, 1e-3))))

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


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)
