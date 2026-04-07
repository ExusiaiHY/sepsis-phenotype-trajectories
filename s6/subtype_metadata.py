from __future__ import annotations

from copy import deepcopy
from typing import Any


def _normalize(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


FAMILY_METADATA: dict[str, dict[str, Any]] = {
    "gold_standard": {
        "family_id": "gold_standard",
        "display_name": "Gold Standard Labels",
        "display_name_zh": "金标准标签家族",
        "order": 10,
        "frontend_note": (
            "Gold tasks are biomarker-anchored targets. They should be displayed as evidence-backed labels "
            "when the corresponding biomarker is available, not as always-observed bedside labels."
        ),
    },
    "immune_state": {
        "family_id": "immune_state",
        "display_name": "Immune State",
        "display_name_zh": "免疫状态家族",
        "order": 20,
        "frontend_note": "This family summarizes immune activation versus immune suppression patterns.",
    },
    "clinical_phenotype": {
        "family_id": "clinical_phenotype",
        "display_name": "Clinical Phenotype",
        "display_name_zh": "临床器官表型家族",
        "order": 30,
        "frontend_note": (
            "alpha / beta / gamma / delta refers to the clinical organ-dysfunction phenotype family. "
            "It is different from Trajectory A / B / C / D."
        ),
    },
    "trajectory_phenotype": {
        "family_id": "trajectory_phenotype",
        "display_name": "Trajectory Phenotype",
        "display_name_zh": "早期生命体征轨迹家族",
        "order": 40,
        "frontend_note": (
            "Trajectory A / B / C / D refers to early vital-sign trajectory groups from the first ICU hours. "
            "It is distinct from alpha / beta / gamma / delta clinical phenotypes."
        ),
    },
    "fluid_strategy": {
        "family_id": "fluid_strategy",
        "display_name": "Fluid Strategy",
        "display_name_zh": "液体策略家族",
        "order": 50,
        "frontend_note": "This family estimates whether restrictive versus resuscitation-focused fluid strategy is more favorable.",
    },
    "regression_score": {
        "family_id": "regression_score",
        "display_name": "Continuous Scores",
        "display_name_zh": "连续风险评分",
        "order": 60,
        "frontend_note": "Continuous scores are 0-1 subtype affinity or treatment-benefit scores, not hard labels.",
    },
}


CLASS_DEFINITIONS: dict[str, dict[str, dict[str, Any]]] = {
    "gold_mals": {
        "negative": {
            "display_name": "MALS Negative",
            "display_name_zh": "MALS 阴性",
            "description": "No ferritin-anchored evidence for macrophage activation-like syndrome.",
            "recommendations": [
                "Continue standard sepsis bundle and reassess if ferritin or inflammatory markers rise.",
                "Monitor platelets, liver function, and coagulation if clinical hyperinflammation emerges.",
            ],
            "monitoring": ["Ferritin trend if clinically indicated", "Platelets", "Liver function", "DIC markers"],
        },
        "positive": {
            "display_name": "MALS Positive",
            "display_name_zh": "MALS 阳性",
            "description": "Ferritin-anchored macrophage activation-like syndrome with hyperinflammatory risk.",
            "recommendations": [
                "Escalate hyperinflammation workup with ferritin, platelets, liver injury markers, and coagulation panel.",
                "Discuss immune-targeted or cytokine-modulating therapy in expert ICU/hematology review pathways.",
                "Intensify surveillance for DIC and multi-organ dysfunction.",
            ],
            "monitoring": ["Ferritin", "Platelets", "ALT/AST", "Bilirubin", "Fibrinogen", "D-dimer"],
        },
    },
    "gold_immunoparalysis": {
        "negative": {
            "display_name": "Immunoparalysis Negative",
            "display_name_zh": "免疫麻痹阴性",
            "description": "No direct biomarker evidence for immunoparalysis.",
            "recommendations": [
                "Maintain standard infection source control and reassess if recurrent infection pattern appears.",
            ],
            "monitoring": ["Secondary infection surveillance", "WBC and lymphocyte trend"],
        },
        "positive": {
            "display_name": "Immunoparalysis Positive",
            "display_name_zh": "免疫麻痹阳性",
            "description": "mHLA-DR-anchored immune exhaustion pattern with secondary infection risk.",
            "recommendations": [
                "Escalate surveillance for secondary or nosocomial infection.",
                "Consider immune-restorative discussion pathways if supported by local protocol and specialist review.",
                "Prefer culture-guided antimicrobial optimization over empiric escalation alone.",
            ],
            "monitoring": ["mHLA-DR if available", "Lymphocytes", "Culture results", "New infection foci"],
        },
    },
    "proxy_immune_state": {
        "unclassified": {
            "display_name": "Immune State Unclassified",
            "display_name_zh": "免疫状态未分类",
            "description": "No dominant immune pattern detected.",
            "recommendations": [
                "Follow standard sepsis bundle and trend inflammatory markers.",
            ],
            "monitoring": ["CRP/PCT trend", "Culture results"],
        },
        "immunoparalysis-like": {
            "display_name": "Immunoparalysis-like",
            "display_name_zh": "免疫麻痹样",
            "description": "Proxy immune-suppressed phenotype with recurrent infection vulnerability.",
            "recommendations": [
                "Screen aggressively for secondary infection and resistant pathogens.",
                "Review antimicrobial de-escalation or escalation against culture data rather than physiology alone.",
                "Discuss immune-restorative strategies when local protocol supports it.",
            ],
            "monitoring": ["Lymphocyte count", "Culture positivity", "Ventilator-associated infection signs"],
        },
        "mals-like": {
            "display_name": "MALS-like",
            "display_name_zh": "MALS 样",
            "description": "Proxy hyperinflammatory phenotype with cytokine-storm and coagulopathy risk.",
            "recommendations": [
                "Trend ferritin, platelets, liver injury markers, and coagulation profile early.",
                "Escalate monitoring for shock, DIC, and rapid organ dysfunction progression.",
                "Consider expert review for hyperinflammation-directed therapy.",
            ],
            "monitoring": ["Ferritin", "Platelets", "Liver function", "DIC markers", "Lactate"],
        },
    },
    "proxy_clinical_phenotype": {
        "unclassified": {
            "display_name": "Clinical Phenotype Unclassified",
            "display_name_zh": "临床器官表型未分类",
            "description": "No dominant organ-dysfunction phenotype detected.",
            "recommendations": [
                "Apply standard sepsis resuscitation and serial organ-function reassessment.",
            ],
            "monitoring": ["SOFA trend", "Lactate", "Urine output"],
        },
        "alpha-like": {
            "display_name": "Alpha-like",
            "display_name_zh": "α型 临床表型",
            "description": "Liver and kidney injury dominant phenotype.",
            "recommendations": [
                "Prioritize hepatic and renal protective strategy.",
                "Avoid nephrotoxic exposures and reassess renal replacement support early if worsening.",
                "Use moderated fluid resuscitation guided by perfusion and congestion status.",
            ],
            "monitoring": ["Creatinine", "Urine output", "ALT/AST", "Bilirubin"],
        },
        "beta-like": {
            "display_name": "Beta-like",
            "display_name_zh": "β型 临床表型",
            "description": "Cardiorenal dysfunction dominant phenotype with fluid intolerance risk.",
            "recommendations": [
                "Use restrictive or tightly titrated fluid strategy with early vasopressor consideration.",
                "Assess cardiac function and fluid responsiveness dynamically.",
                "Watch for worsening congestion or renal hypoperfusion.",
            ],
            "monitoring": ["MAP", "Urine output", "Bedside echo", "Fluid balance"],
        },
        "gamma-like": {
            "display_name": "Gamma-like",
            "display_name_zh": "γ型 临床表型",
            "description": "Respiratory failure dominant phenotype, often pneumonia-linked.",
            "recommendations": [
                "Use lung-protective respiratory support strategy.",
                "Escalate pulmonary source control and oxygenation monitoring.",
                "Evaluate need for mechanical ventilation and adjunct ARDS support.",
            ],
            "monitoring": ["SpO2", "PaO2/FiO2", "Chest imaging", "Ventilator parameters"],
        },
        "delta-like": {
            "display_name": "Delta-like",
            "display_name_zh": "δ型 临床表型",
            "description": "Severe multi-organ failure phenotype with shock burden.",
            "recommendations": [
                "Escalate to full shock and multi-organ support pathway immediately.",
                "Prioritize vasopressor-guided perfusion support and frequent organ reassessment.",
                "Coordinate multidisciplinary ICU review for renal, respiratory, and coagulation support.",
            ],
            "monitoring": ["MAP", "Lactate", "Platelets", "D-dimer", "Ventilator need", "Renal support need"],
        },
    },
    "proxy_trajectory_phenotype": {
        "unclassified": {
            "display_name": "Trajectory Unclassified",
            "display_name_zh": "轨迹未分类",
            "description": "No dominant early vital-sign trajectory group detected.",
            "recommendations": [
                "Continue standard bedside reassessment and trend vital-sign trajectory.",
            ],
            "monitoring": ["Hourly vitals", "MAP trend", "Temperature trend"],
        },
        "group-a": {
            "display_name": "Trajectory A",
            "display_name_zh": "Trajectory A 早期炎症风暴型",
            "description": "High fever, tachycardia, tachypnea, and hypotension early after ICU presentation.",
            "recommendations": [
                "Start early fluid resuscitation and broad-spectrum antibiotics promptly.",
                "Trend hemodynamics closely because early shock progression is possible.",
            ],
            "monitoring": ["Temperature", "Heart rate", "Respiratory rate", "MAP", "Lactate"],
        },
        "group-b": {
            "display_name": "Trajectory B",
            "display_name_zh": "Trajectory B 高血压合并感染型",
            "description": "Fever and tachycardia with relatively preserved or elevated blood pressure, often in older comorbid patients.",
            "recommendations": [
                "Treat infection aggressively while avoiding reflexive over-resuscitation.",
                "Balance blood-pressure control, preload, and comorbidity burden during sepsis care.",
            ],
            "monitoring": ["Blood pressure", "Heart rate", "Temperature", "Fluid balance"],
        },
        "group-c": {
            "display_name": "Trajectory C",
            "display_name_zh": "Trajectory C 相对稳定型",
            "description": "Relatively normal vitals and lower early physiologic derangement.",
            "recommendations": [
                "Maintain standard sepsis therapy and monitor for delayed deterioration.",
                "Avoid unnecessary escalation when perfusion and organ function remain stable.",
            ],
            "monitoring": ["Routine vitals", "Infection source control", "Reassessment interval"],
        },
        "group-d": {
            "display_name": "Trajectory D",
            "display_name_zh": "Trajectory D 低反应休克型",
            "description": "Low temperature, low heart rate or respiratory drive, and hypotension in frail or shock-prone patients.",
            "recommendations": [
                "Treat as high-risk shock trajectory with early vasopressor readiness.",
                "Use cautious fluid strategy and reassess perfusion response frequently.",
                "Escalate organ-support preparation early in elderly or multimorbid patients.",
            ],
            "monitoring": ["MAP", "Urine output", "Lactate", "Mental status", "Temperature"],
        },
    },
    "proxy_fluid_strategy": {
        "unclassified": {
            "display_name": "Fluid Strategy Unclassified",
            "display_name_zh": "液体策略未分类",
            "description": "No dominant fluid strategy preference inferred.",
            "recommendations": [
                "Use guideline-based initial fluid resuscitation and dynamic reassessment.",
            ],
            "monitoring": ["Fluid balance", "Perfusion response", "Congestion signs"],
        },
        "restrictive-fluid-benefit-like": {
            "display_name": "Restrictive Fluid Benefit-like",
            "display_name_zh": "限制性液体获益样",
            "description": "Pattern associated with greater benefit from restrictive fluid management.",
            "recommendations": [
                "Favor restrictive or tightly titrated fluid strategy.",
                "Use vasopressors and dynamic hemodynamic assessment rather than repeated large boluses.",
                "Watch carefully for pulmonary edema and cardiorenal congestion.",
            ],
            "monitoring": ["Net fluid balance", "MAP", "Ultrasound response", "Oxygenation"],
        },
        "resuscitation-fluid-benefit-like": {
            "display_name": "Resuscitation Fluid Benefit-like",
            "display_name_zh": "积极复苏液体获益样",
            "description": "Pattern associated with greater benefit from active early fluid resuscitation.",
            "recommendations": [
                "Use active crystalloid resuscitation early if perfusion deficit is present.",
                "Reassess after each fluid step using lactate, MAP, urine output, or dynamic response tests.",
            ],
            "monitoring": ["MAP", "Lactate", "Urine output", "Fluid responsiveness"],
        },
    },
}


TASK_METADATA: dict[str, dict[str, Any]] = {
    "gold_mals": {
        "task_name": "gold_mals",
        "display_name": "Gold MALS",
        "display_name_zh": "金标准 MALS",
        "family_id": "gold_standard",
        "description": "Ferritin-anchored MALS binary task",
        "classes": ["negative", "positive"],
    },
    "gold_immunoparalysis": {
        "task_name": "gold_immunoparalysis",
        "display_name": "Gold Immunoparalysis",
        "display_name_zh": "金标准免疫麻痹",
        "family_id": "gold_standard",
        "description": "mHLA-DR-anchored immunoparalysis binary task",
        "classes": ["negative", "positive"],
    },
    "proxy_immune_state": {
        "task_name": "proxy_immune_state",
        "display_name": "Proxy Immune State",
        "display_name_zh": "代理免疫状态",
        "family_id": "immune_state",
        "description": "Proxy immune-state task when direct immunophenotyping is absent",
        "classes": ["Unclassified", "immunoparalysis-like", "MALS-like"],
    },
    "proxy_clinical_phenotype": {
        "task_name": "proxy_clinical_phenotype",
        "display_name": "Proxy Clinical Phenotype",
        "display_name_zh": "代理临床器官表型",
        "family_id": "clinical_phenotype",
        "description": "Proxy alpha / beta / gamma / delta phenotype task",
        "classes": ["Unclassified", "alpha-like", "beta-like", "gamma-like", "delta-like"],
    },
    "proxy_trajectory_phenotype": {
        "task_name": "proxy_trajectory_phenotype",
        "display_name": "Proxy Trajectory Phenotype",
        "display_name_zh": "代理生命体征轨迹表型",
        "family_id": "trajectory_phenotype",
        "description": "Proxy early vital-sign trajectory A / B / C / D task",
        "classes": ["Unclassified", "group-a", "group-b", "group-c", "group-d"],
    },
    "proxy_fluid_strategy": {
        "task_name": "proxy_fluid_strategy",
        "display_name": "Proxy Fluid Strategy",
        "display_name_zh": "代理液体策略",
        "family_id": "fluid_strategy",
        "description": "Proxy fluid-strategy benefit task",
        "classes": ["Unclassified", "restrictive-fluid-benefit-like", "resuscitation-fluid-benefit-like"],
    },
}


REGRESSION_TASK_METADATA: dict[str, dict[str, Any]] = {
    "score_mals": {
        "task_name": "score_mals",
        "display_name": "MALS Affinity Score",
        "display_name_zh": "MALS 亲和评分",
        "family_id": "regression_score",
        "description": "Continuous MALS affinity score on 0-1 scale",
    },
    "score_immunoparalysis": {
        "task_name": "score_immunoparalysis",
        "display_name": "Immunoparalysis Affinity Score",
        "display_name_zh": "免疫麻痹亲和评分",
        "family_id": "regression_score",
        "description": "Continuous immunoparalysis affinity score on 0-1 scale",
    },
    "score_alpha": {
        "task_name": "score_alpha",
        "display_name": "Alpha Score",
        "display_name_zh": "α型评分",
        "family_id": "regression_score",
        "description": "Continuous alpha clinical phenotype score",
    },
    "score_beta": {
        "task_name": "score_beta",
        "display_name": "Beta Score",
        "display_name_zh": "β型评分",
        "family_id": "regression_score",
        "description": "Continuous beta clinical phenotype score",
    },
    "score_gamma": {
        "task_name": "score_gamma",
        "display_name": "Gamma Score",
        "display_name_zh": "γ型评分",
        "family_id": "regression_score",
        "description": "Continuous gamma clinical phenotype score",
    },
    "score_delta": {
        "task_name": "score_delta",
        "display_name": "Delta Score",
        "display_name_zh": "δ型评分",
        "family_id": "regression_score",
        "description": "Continuous delta clinical phenotype score",
    },
    "score_trajectory_a": {
        "task_name": "score_trajectory_a",
        "display_name": "Trajectory A Score",
        "display_name_zh": "Trajectory A 评分",
        "family_id": "regression_score",
        "description": "Continuous early vital-sign Trajectory A score",
    },
    "score_trajectory_b": {
        "task_name": "score_trajectory_b",
        "display_name": "Trajectory B Score",
        "display_name_zh": "Trajectory B 评分",
        "family_id": "regression_score",
        "description": "Continuous early vital-sign Trajectory B score",
    },
    "score_trajectory_c": {
        "task_name": "score_trajectory_c",
        "display_name": "Trajectory C Score",
        "display_name_zh": "Trajectory C 评分",
        "family_id": "regression_score",
        "description": "Continuous early vital-sign Trajectory C score",
    },
    "score_trajectory_d": {
        "task_name": "score_trajectory_d",
        "display_name": "Trajectory D Score",
        "display_name_zh": "Trajectory D 评分",
        "family_id": "regression_score",
        "description": "Continuous early vital-sign Trajectory D score",
    },
    "score_restrictive_fluid_benefit": {
        "task_name": "score_restrictive_fluid_benefit",
        "display_name": "Restrictive Fluid Benefit Score",
        "display_name_zh": "限制性液体获益评分",
        "family_id": "regression_score",
        "description": "Continuous score for expected restrictive-fluid benefit",
    },
    "score_resuscitation_fluid_benefit": {
        "task_name": "score_resuscitation_fluid_benefit",
        "display_name": "Resuscitation Fluid Benefit Score",
        "display_name_zh": "积极复苏液体获益评分",
        "family_id": "regression_score",
        "description": "Continuous score for expected active resuscitation-fluid benefit",
    },
}


TASK_ALIASES = {
    "immune_subtype": "proxy_immune_state",
    "organ_subtype": "proxy_clinical_phenotype",
    "fluid_benefit_proxy": "proxy_fluid_strategy",
}


RULE_LABEL_ALIASES = {
    "proxy_immune_state": {
        "immunoparalysis-like": ["EIL-like"],
        "mals-like": ["MAS-like"],
        "unclassified": ["Unclassified"],
    },
    "proxy_clinical_phenotype": {
        "alpha-like": ["alpha-like"],
        "beta-like": ["beta-like"],
        "gamma-like": ["gamma-like"],
        "delta-like": ["delta-like"],
        "unclassified": ["Unclassified"],
    },
    "proxy_fluid_strategy": {
        "restrictive-fluid-benefit-like": ["low_benefit"],
        "resuscitation-fluid-benefit-like": ["high_benefit"],
        "unclassified": ["Unclassified"],
    },
}


def _build_class_entry(task_name: str, label: str, index: int) -> dict[str, Any]:
    normalized = _normalize(label)
    task_defs = CLASS_DEFINITIONS.get(task_name, {})
    class_meta = deepcopy(task_defs.get(normalized, {}))
    if not class_meta:
        class_meta = {
            "display_name": label,
            "display_name_zh": label,
            "description": "",
            "recommendations": [],
            "monitoring": [],
        }
    class_meta["index"] = int(index)
    class_meta["label"] = label
    class_meta["normalized_label"] = normalized
    return class_meta


def get_family_metadata(family_id: str) -> dict[str, Any]:
    return deepcopy(FAMILY_METADATA.get(family_id, {"family_id": family_id, "display_name": family_id}))


def get_task_metadata(task_name: str) -> dict[str, Any]:
    return deepcopy(TASK_METADATA.get(task_name, {
        "task_name": task_name,
        "display_name": task_name,
        "display_name_zh": task_name,
        "family_id": "unknown",
        "description": "",
        "classes": [],
    }))


def get_regression_task_metadata(task_name: str) -> dict[str, Any]:
    return deepcopy(REGRESSION_TASK_METADATA.get(task_name, {
        "task_name": task_name,
        "display_name": task_name,
        "display_name_zh": task_name,
        "family_id": "regression_score",
        "description": "",
    }))


def build_classification_task_metadata(
    configured_tasks: list[dict[str, Any]],
    schema_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    schema_lookup = schema_lookup or {}
    task_entries: list[dict[str, Any]] = []
    for task in configured_tasks:
        task_name = str(task["name"])
        base = get_task_metadata(task_name)
        schema_task = schema_lookup.get(task_name, {})
        labels = schema_task.get("classes") or base.get("classes") or [f"class-{idx}" for idx in range(int(task["n_classes"]))]
        classes = [_build_class_entry(task_name, str(label), idx) for idx, label in enumerate(labels)]
        family = get_family_metadata(base.get("family_id", "unknown"))
        task_entries.append(
            {
                **base,
                **task,
                "family_id": family.get("family_id", base.get("family_id", "unknown")),
                "family_display_name": family.get("display_name", family.get("family_id")),
                "family_display_name_zh": family.get("display_name_zh", family.get("display_name", family.get("family_id"))),
                "family_order": family.get("order", 999),
                "frontend_note": family.get("frontend_note", ""),
                "classes": classes,
                "description": schema_task.get("description", task.get("description", base.get("description", ""))),
            }
        )
    return task_entries


def build_regression_task_metadata(
    configured_tasks: list[dict[str, Any]],
    schema_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    schema_lookup = schema_lookup or {}
    task_entries: list[dict[str, Any]] = []
    for task in configured_tasks:
        task_name = str(task["name"])
        base = get_regression_task_metadata(task_name)
        schema_task = schema_lookup.get(task_name, {})
        family = get_family_metadata(base.get("family_id", "regression_score"))
        task_entries.append(
            {
                **base,
                **task,
                "family_id": family.get("family_id", base.get("family_id", "regression_score")),
                "family_display_name": family.get("display_name", family.get("family_id")),
                "family_display_name_zh": family.get("display_name_zh", family.get("display_name", family.get("family_id"))),
                "family_order": family.get("order", 999),
                "frontend_note": family.get("frontend_note", ""),
                "description": schema_task.get("description", task.get("description", base.get("description", ""))),
            }
        )
    return task_entries


def default_metadata_bundle(
    classification_tasks: list[dict[str, Any]],
    regression_tasks: list[dict[str, Any]],
    schema_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    classification = build_classification_task_metadata(classification_tasks, schema_lookup)
    regression = build_regression_task_metadata(regression_tasks, schema_lookup)
    families: dict[str, dict[str, Any]] = {}
    for task in classification + regression:
        family_id = str(task.get("family_id", "unknown"))
        families[family_id] = get_family_metadata(family_id)
    return {
        "families": sorted(families.values(), key=lambda item: (item.get("order", 999), item["family_id"])),
        "classification_tasks": classification,
        "regression_tasks": regression,
        "legacy_aliases": deepcopy(TASK_ALIASES),
    }
