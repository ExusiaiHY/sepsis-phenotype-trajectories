"""Helpers for preparing Stage 5 source-specific fine-tune runs."""
from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import yaml

EXPECTED_TRIGGER_STEP = "start_source_specific_full_finetune"

DEFAULT_EXPECTED_OUTPUTS = {
    "training_config": "train_config.yaml",
    "prep_manifest": "prep_manifest.json",
    "model_artifact": "realtime_student.pt",
    "report_json": "realtime_student_report.json",
    "best_checkpoint": "checkpoints/student_best.pt",
}


def build_s5_finetune_prep_bundle(
    *,
    template_config_path: Path,
    project_root: Path,
    run_name: str | None = None,
    trigger_report_path: Path | None = None,
    output_root: Path | None = None,
    allow_untriggered: bool = False,
    overwrite: bool = False,
) -> dict:
    """Build a concrete Stage 5 fine-tune prep bundle without launching training."""
    project_root = Path(project_root).resolve()
    template_config_path = _resolve_project_path(template_config_path, project_root)
    cfg = _load_yaml(template_config_path)

    paths_cfg = dict(cfg.get("paths", {}))
    adaptation_cfg = dict(cfg.get("adaptation", {}))
    trigger_report_path = _resolve_project_path(
        trigger_report_path or paths_cfg.get("trigger_report"),
        project_root,
    )
    trigger_report = _load_json(trigger_report_path)
    _validate_trigger_report(trigger_report, allow_untriggered=allow_untriggered)

    source = str(adaptation_cfg.get("source", trigger_report.get("source", "unknown_source")))
    source_key = str(adaptation_cfg.get("source_key", trigger_report.get("source_key", source)))
    run_prefix = str(adaptation_cfg.get("run_name_prefix", source))
    concrete_run_name = str(run_name or f"{run_prefix}_{datetime.now().strftime('%Y%m%d')}")
    output_root = _resolve_project_path(output_root or adaptation_cfg.get("output_root", "data/s5_mimic_adapt"), project_root)
    run_dir = output_root / concrete_run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Prepared run directory already exists and is not empty: {run_dir}")

    base_student_artifact = _resolve_optional_project_path(paths_cfg.get("base_student_artifact"), project_root)
    base_student_report = _resolve_optional_project_path(paths_cfg.get("base_student_report"), project_root)
    expected_outputs = dict(DEFAULT_EXPECTED_OUTPUTS)
    expected_outputs.update(dict(adaptation_cfg.get("expected_outputs", {})))

    _require_existing_paths(
        {
            "template_config": template_config_path,
            "trigger_report": trigger_report_path,
            "s0_dir": _resolve_project_path(paths_cfg["s0_dir"], project_root),
            "treatment_dir": _resolve_project_path(paths_cfg["treatment_dir"], project_root),
            "teacher_embeddings": _resolve_optional_project_path(paths_cfg.get("teacher_embeddings"), project_root),
            "teacher_probabilities": _resolve_optional_project_path(paths_cfg.get("teacher_probabilities"), project_root),
            "note_embeddings": _resolve_optional_project_path(paths_cfg.get("note_embeddings"), project_root),
            "base_student_artifact": base_student_artifact,
            "base_student_report": base_student_report,
        }
    )

    train_config = copy.deepcopy(cfg)
    train_paths = train_config.setdefault("paths", {})
    train_paths["output_dir"] = _relative_to_project(run_dir, project_root)
    train_paths["trigger_report"] = _relative_to_project(trigger_report_path, project_root)
    if base_student_artifact is not None:
        train_paths["init_checkpoint"] = _relative_to_project(base_student_artifact, project_root)
    if base_student_artifact is not None:
        train_paths["base_student_artifact"] = _relative_to_project(base_student_artifact, project_root)
    if base_student_report is not None:
        train_paths["base_student_report"] = _relative_to_project(base_student_report, project_root)

    adaptation_block = train_config.setdefault("adaptation", {})
    adaptation_block.update(
        {
            "source": source,
            "source_key": source_key,
            "mode": "source_specific_full_finetune_prep",
            "prepared_run_name": concrete_run_name,
            "prepared_on": datetime.now().strftime("%Y-%m-%d"),
            "overwrite": bool(overwrite),
            "output_root": _relative_to_project(output_root, project_root),
            "run_dir": _relative_to_project(run_dir, project_root),
            "expected_outputs": expected_outputs,
            "trigger_snapshot": {
                "triggered": bool(trigger_report.get("triggered", False)),
                "next_step": str(trigger_report.get("next_step", "")),
                "production_policy_ready": bool(trigger_report.get("production_policy_ready", False)),
                "shadow_policy_ready": bool(trigger_report.get("shadow_policy_ready", False)),
                "offline_model_ready": bool(trigger_report.get("offline_model_ready", False)),
                "search_exhausted": bool(trigger_report.get("search_exhausted", False)),
            },
        }
    )

    train_config_path = run_dir / expected_outputs["training_config"]
    prep_manifest_path = run_dir / expected_outputs["prep_manifest"]
    launch_config_arg = _relative_to_project(train_config_path, project_root)
    launch_command = f"./.venv/bin/python scripts/s5_distill_realtime.py --config {launch_config_arg}"

    manifest = {
        "status": "prepared",
        "mode": "source_specific_full_finetune_prep",
        "source": source,
        "source_key": source_key,
        "run_name": concrete_run_name,
        "run_dir": str(run_dir),
        "prepared_on": datetime.now().strftime("%Y-%m-%d"),
        "trigger_report": {
            "path": str(trigger_report_path),
            "triggered": bool(trigger_report.get("triggered", False)),
            "next_step": str(trigger_report.get("next_step", "")),
        },
        "base_artifacts": {
            "template_config": str(template_config_path),
            "base_student_artifact": None if base_student_artifact is None else str(base_student_artifact),
            "base_student_report": None if base_student_report is None else str(base_student_report),
        },
        "expected_outputs": {
            key: str(run_dir / value)
            for key, value in expected_outputs.items()
        },
        "launch": {
            "cwd": str(project_root),
            "entrypoint": "scripts/s5_distill_realtime.py",
            "config_path": str(train_config_path),
            "command": launch_command,
            "notes": "Preparation only. This does not start training. The generated config is warm-start ready.",
        },
        "metrics_snapshot": trigger_report.get("metrics_snapshot", {}),
    }

    return {
        "project_root": project_root,
        "run_dir": run_dir,
        "train_config": train_config,
        "train_config_path": train_config_path,
        "prep_manifest": manifest,
        "prep_manifest_path": prep_manifest_path,
    }


def write_s5_finetune_prep_artifacts(bundle: dict) -> dict:
    """Write a concrete fine-tune prep bundle to disk."""
    run_dir = Path(bundle["run_dir"])
    train_config_path = Path(bundle["train_config_path"])
    prep_manifest_path = Path(bundle["prep_manifest_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _write_yaml(train_config_path, bundle["train_config"])
    _write_json(prep_manifest_path, bundle["prep_manifest"])
    return {
        "run_dir": str(run_dir),
        "train_config_path": str(train_config_path),
        "prep_manifest_path": str(prep_manifest_path),
    }


def _validate_trigger_report(trigger_report: dict, *, allow_untriggered: bool) -> None:
    if allow_untriggered:
        return
    triggered = bool(trigger_report.get("triggered", False))
    next_step = str(trigger_report.get("next_step", ""))
    if not triggered or next_step != EXPECTED_TRIGGER_STEP:
        raise ValueError(
            "Fine-tune prep requires a triggered adaptation report with "
            f"next_step={EXPECTED_TRIGGER_STEP!r}; got triggered={triggered} next_step={next_step!r}."
        )


def _require_existing_paths(paths: dict[str, Path | None]) -> None:
    for name, path in paths.items():
        if path is None:
            continue
        if not Path(path).exists():
            raise FileNotFoundError(f"Required path for {name} does not exist: {path}")


def _relative_to_project(path: Path, project_root: Path) -> str:
    path = Path(path).resolve()
    project_root = Path(project_root).resolve()
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _resolve_project_path(path_value: str | Path | None, project_root: Path) -> Path:
    if path_value is None:
        raise ValueError("Expected a path value, got None")
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (Path(project_root).resolve() / path)


def _resolve_optional_project_path(path_value: str | Path | None, project_root: Path) -> Path | None:
    if path_value in {None, ""}:
        return None
    return _resolve_project_path(path_value, project_root)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return dict(payload or {})


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
