#!/usr/bin/env python3
"""
Run a sequential masked-NPZ S6 sweep and write a compact summary table.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "s6_train_masked_npz_cloud.py"


def _score_summary(report: dict) -> float:
    test = report["splits"]["test"]
    mortality_auroc = float(test["mortality"]["auroc"])
    gold_mals = test["classification"]["gold_mals"]
    traj = test["classification"]["proxy_trajectory_phenotype"]
    clinical = test["classification"]["proxy_clinical_phenotype"]
    fluid = test["classification"]["proxy_fluid_strategy"]
    reg = test["regression"]["score_restrictive_fluid_benefit"]

    total = mortality_auroc
    if gold_mals["auroc"] is not None:
        total += 0.75 * float(gold_mals["auroc"])
    if clinical["macro_f1"] is not None:
        total += 0.5 * float(clinical["macro_f1"])
    if traj["macro_f1"] is not None:
        total += 0.75 * float(traj["macro_f1"])
    if fluid["macro_f1"] is not None:
        total += 0.25 * float(fluid["macro_f1"])
    if reg["rmse"] is not None:
        total += 0.25 * (1.0 / (1.0 + float(reg["rmse"])))
    return round(total, 4)


def _flatten_result(run_cfg: dict, report: dict) -> dict:
    test = report["splits"]["test"]
    return {
        "name": run_cfg["name"],
        "student_arch": run_cfg["student_arch"],
        "student_d_model": int(run_cfg["student_d_model"]),
        "lambda_gold": float(run_cfg["lambda_gold"]),
        "lambda_trajectory": float(run_cfg["lambda_trajectory"]),
        "lambda_regression": float(run_cfg["lambda_regression"]),
        "param_count": int(report["deployment"]["float_n_parameters"]),
        "mortality_auroc": round(float(test["mortality"]["auroc"]), 4),
        "gold_mals_auroc": test["classification"]["gold_mals"]["auroc"],
        "clinical_macro_f1": test["classification"]["proxy_clinical_phenotype"]["macro_f1"],
        "trajectory_macro_f1": test["classification"]["proxy_trajectory_phenotype"]["macro_f1"],
        "fluid_macro_f1": test["classification"]["proxy_fluid_strategy"]["macro_f1"],
        "restrictive_fluid_rmse": test["regression"]["score_restrictive_fluid_benefit"]["rmse"],
        "composite_score": _score_summary(report),
        "report_path": str(Path(run_cfg["output_dir"]) / "multitask_student_report.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential masked-NPZ S6 sweep")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--runs-root", type=str, required=True)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    runs_root = PROJECT_ROOT / args.runs_root
    runs_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shared = dict(manifest.get("shared", {}))
    results: list[dict] = []

    for run_cfg in manifest.get("runs", []):
        cfg = dict(shared)
        cfg.update(run_cfg)
        run_dir = runs_root / cfg["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg["output_dir"] = str(run_dir.relative_to(PROJECT_ROOT))

        cmd = [
            args.python_bin,
            str(TRAIN_SCRIPT),
            "--data-dir",
            str(cfg["data_dir"]),
            "--output-dir",
            str(cfg["output_dir"]),
            "--epochs",
            str(cfg["epochs"]),
            "--batch-size",
            str(cfg["batch_size"]),
            "--device",
            str(cfg["device"]),
            "--student-arch",
            str(cfg["student_arch"]),
            "--student-d-model",
            str(cfg["student_d_model"]),
            "--patience",
            str(cfg["patience"]),
            "--lr",
            str(cfg["lr"]),
            "--dropout",
            str(cfg["dropout"]),
            "--lambda-mortality",
            str(cfg["lambda_mortality"]),
            "--lambda-immune",
            str(cfg["lambda_immune"]),
            "--lambda-organ",
            str(cfg["lambda_organ"]),
            "--lambda-fluid",
            str(cfg["lambda_fluid"]),
            "--lambda-gold",
            str(cfg["lambda_gold"]),
            "--lambda-trajectory",
            str(cfg["lambda_trajectory"]),
            "--lambda-regression",
            str(cfg["lambda_regression"]),
            "--seed",
            str(cfg["seed"]),
        ]

        run_log = run_dir / "run.log"
        with run_log.open("w", encoding="utf-8") as logf:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, stdout=logf, stderr=subprocess.STDOUT)

        report_path = run_dir / "multitask_student_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        results.append(_flatten_result(cfg, report))

        (runs_root / "sweep_summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    results = sorted(results, key=lambda row: row["composite_score"], reverse=True)
    (runs_root / "sweep_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "name,student_arch,student_d_model,lambda_gold,lambda_trajectory,lambda_regression,param_count,mortality_auroc,gold_mals_auroc,clinical_macro_f1,trajectory_macro_f1,fluid_macro_f1,restrictive_fluid_rmse,composite_score"
    ]
    for row in results:
        lines.append(
            ",".join(
                [
                    str(row["name"]),
                    str(row["student_arch"]),
                    str(row["student_d_model"]),
                    str(row["lambda_gold"]),
                    str(row["lambda_trajectory"]),
                    str(row["lambda_regression"]),
                    str(row["param_count"]),
                    str(row["mortality_auroc"]),
                    str(row["gold_mals_auroc"]),
                    str(row["clinical_macro_f1"]),
                    str(row["trajectory_macro_f1"]),
                    str(row["fluid_macro_f1"]),
                    str(row["restrictive_fluid_rmse"]),
                    str(row["composite_score"]),
                ]
            )
        )
    (runs_root / "sweep_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
