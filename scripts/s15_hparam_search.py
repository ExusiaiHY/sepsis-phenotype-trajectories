#!/usr/bin/env python3
"""
s15_hparam_search.py - Systematic hyperparameter search for downstream models.

How to run:
  cd project
  ./.venv/bin/python scripts/s15_hparam_search.py --mode advanced
  ./.venv/bin/python scripts/s15_hparam_search.py --mode finetune
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s15.advanced_classifier import train_advanced_mortality_classifier
from s15.finetune_supervised import train_end_to_end_classifier


def get_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Systematic hyperparameter search for S1.5 downstream models")
    parser.add_argument("--config", default="config/s15_trainval_config.yaml")
    parser.add_argument("--mode", default="advanced", choices=["advanced", "finetune"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold-metric", default="accuracy",
                        choices=["accuracy", "balanced_accuracy", "f1"])
    return parser.parse_args()


def main():
    args = parse_args()
    with open(PROJECT_ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=getattr(logging, cfg.get("runtime", {}).get("log_level", "INFO")),
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("scripts.s15_hparam_search")

    s0_dir = PROJECT_ROOT / cfg["paths"]["s0_dir"]
    s15_dir = PROJECT_ROOT / cfg["paths"]["s15_dir"]
    output_dir = Path(args.output_dir) if args.output_dir else (s15_dir / f"hparam_search_{args.mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "advanced":
        report = run_advanced_search(
            s0_dir=s0_dir,
            s15_dir=s15_dir,
            output_dir=output_dir,
            threshold_metric=args.threshold_metric,
        )
    else:
        report = run_finetune_search(
            s0_dir=s0_dir,
            s15_dir=s15_dir,
            output_dir=output_dir,
            device=args.device or get_device(cfg.get("runtime", {}).get("device", "auto")),
            seed=cfg.get("pretraining", {}).get("seed", 42),
            threshold_metric=args.threshold_metric,
        )

    report_path = output_dir / "search_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved search report to %s", report_path)
    for metric_name, leader in report["leaders"].items():
        logger.info(
            "%s leader: run=%s val=%.4f test=%.4f",
            metric_name,
            leader["run_name"],
            leader["val_metric"],
            leader["test_metric"],
        )


def run_advanced_search(
    *,
    s0_dir: Path,
    s15_dir: Path,
    output_dir: Path,
    threshold_metric: str,
) -> dict:
    embeddings_path = s15_dir / "embeddings_s15.npy"
    candidates = []
    run_id = 0

    for feature_set in ("stats_mask_proxy_static", "fused_all"):
        for max_depth in (3, 5, 7):
            for learning_rate in (0.03, 0.05):
                for max_iter in (200, 400):
                    candidates.append({
                        "model_type": "hgb",
                        "feature_set": feature_set,
                        "hgb_max_depth": max_depth,
                        "hgb_learning_rate": learning_rate,
                        "hgb_max_iter": max_iter,
                        "threshold_metric": threshold_metric,
                    })

    for feature_set in ("embeddings", "embeddings_static", "fused_all"):
        candidates.append({
            "model_type": "logreg",
            "feature_set": feature_set,
            "threshold_metric": threshold_metric,
        })

    for max_depth in (3, 5):
        for learning_rate in (0.03, 0.05):
            for max_iter in (200, 400):
                candidates.append({
                    "model_type": "hgb_ensemble",
                    "feature_set": "fused_all",
                    "hgb_max_depth": max_depth,
                    "hgb_learning_rate": learning_rate,
                    "hgb_max_iter": max_iter,
                    "threshold_metric": threshold_metric,
                })

    results = []
    for candidate in candidates:
        run_id += 1
        run_name = _advanced_run_name(run_id, candidate)
        run_output = output_dir / run_name
        report = train_advanced_mortality_classifier(
            s0_dir=s0_dir,
            splits_path=s0_dir / "splits.json",
            output_dir=run_output,
            embeddings_path=embeddings_path if embeddings_path.exists() else None,
            label_col="mortality_inhospital",
            model_type=candidate["model_type"],
            feature_set=candidate["feature_set"],
            threshold_metric=candidate["threshold_metric"],
            hgb_max_depth=candidate.get("hgb_max_depth", 5),
            hgb_learning_rate=candidate.get("hgb_learning_rate", 0.05),
            hgb_max_iter=candidate.get("hgb_max_iter", 300),
        )
        results.append(_summarize_run(run_name, candidate, report))

    return _build_search_report("advanced", results, threshold_metric)


def run_finetune_search(
    *,
    s0_dir: Path,
    s15_dir: Path,
    output_dir: Path,
    device: str,
    seed: int,
    threshold_metric: str,
) -> dict:
    checkpoint = s15_dir / "checkpoints" / "pretrain_best.pt"
    aux_dir = PROJECT_ROOT / "data/s19_bridge"
    candidates = [
        {
            "use_aux": False,
            "epochs": 10,
            "aux_epochs": 0,
            "lr_encoder": 1.0e-4,
            "lr_head": 7.5e-4,
            "head_dropout": 0.2,
            "head_hidden_dim": 128,
        },
        {
            "use_aux": False,
            "epochs": 12,
            "aux_epochs": 0,
            "lr_encoder": 2.0e-4,
            "lr_head": 1.0e-3,
            "head_dropout": 0.3,
            "head_hidden_dim": 128,
        },
        {
            "use_aux": True,
            "epochs": 12,
            "aux_epochs": 3,
            "lr_encoder": 1.5e-4,
            "lr_head": 1.0e-3,
            "head_dropout": 0.3,
            "head_hidden_dim": 128,
        },
        {
            "use_aux": True,
            "epochs": 12,
            "aux_epochs": 4,
            "lr_encoder": 1.0e-4,
            "lr_head": 7.5e-4,
            "head_dropout": 0.4,
            "head_hidden_dim": 256,
        },
    ]

    results = []
    for run_id, candidate in enumerate(candidates, start=1):
        run_name = _finetune_run_name(run_id, candidate)
        run_output = output_dir / run_name
        report = train_end_to_end_classifier(
            s0_dir=s0_dir,
            output_dir=run_output,
            pretrained_checkpoint=checkpoint if checkpoint.exists() else None,
            aux_data_dir=aux_dir if candidate["use_aux"] and aux_dir.exists() else None,
            batch_size=128,
            epochs=candidate["epochs"],
            aux_epochs=candidate["aux_epochs"],
            lr_encoder=candidate["lr_encoder"],
            lr_head=candidate["lr_head"],
            weight_decay=1.0e-4,
            patience=3,
            freeze_encoder_epochs=1,
            grad_clip=1.0,
            threshold_metric=threshold_metric,
            monitor_metric=threshold_metric if threshold_metric != "f1" else "f1",
            head_hidden_dim=candidate["head_hidden_dim"],
            head_dropout=candidate["head_dropout"],
            device=device,
            seed=seed,
        )
        results.append(_summarize_run(run_name, candidate, report["main_task"]))

    return _build_search_report("finetune", results, threshold_metric)


def _advanced_run_name(run_id: int, candidate: dict) -> str:
    if candidate["model_type"] == "logreg":
        return f"{run_id:02d}_logreg_{candidate['feature_set']}"
    if candidate["model_type"] == "hgb_ensemble":
        return (
            f"{run_id:02d}_ensemble_d{candidate['hgb_max_depth']}"
            f"_lr{str(candidate['hgb_learning_rate']).replace('.', 'p')}"
            f"_iter{candidate['hgb_max_iter']}"
        )
    return (
        f"{run_id:02d}_hgb_{candidate['feature_set']}"
        f"_d{candidate['hgb_max_depth']}"
        f"_lr{str(candidate['hgb_learning_rate']).replace('.', 'p')}"
        f"_iter{candidate['hgb_max_iter']}"
    )


def _finetune_run_name(run_id: int, candidate: dict) -> str:
    aux = "aux" if candidate["use_aux"] else "noaux"
    return (
        f"{run_id:02d}_{aux}"
        f"_lre{str(candidate['lr_encoder']).replace('.', 'p')}"
        f"_lrh{str(candidate['lr_head']).replace('.', 'p')}"
        f"_drop{str(candidate['head_dropout']).replace('.', 'p')}"
        f"_hid{candidate['head_hidden_dim']}"
    )


def _summarize_run(run_name: str, candidate: dict, report: dict) -> dict:
    val_metrics = report["splits"]["val"]
    test_metrics = report["splits"]["test"]
    return {
        "run_name": run_name,
        "config": candidate,
        "threshold": report["threshold_selection"]["selected_threshold"],
        "val": val_metrics,
        "test": test_metrics,
    }


def _build_search_report(mode: str, results: list[dict], threshold_metric: str) -> dict:
    leaders = {}
    for metric_name in ("accuracy", "balanced_accuracy", "auroc"):
        best = max(
            results,
            key=lambda row: (
                row["val"][metric_name] if row["val"][metric_name] is not None else -1.0,
                row["val"]["balanced_accuracy"] if row["val"]["balanced_accuracy"] is not None else -1.0,
                row["val"]["auroc"] if row["val"]["auroc"] is not None else -1.0,
            ),
        )
        leaders[metric_name] = {
            "run_name": best["run_name"],
            "config": best["config"],
            "threshold": best["threshold"],
            "val_metric": best["val"][metric_name],
            "test_metric": best["test"][metric_name],
            "test_metrics": best["test"],
        }

    return {
        "mode": mode,
        "threshold_metric": threshold_metric,
        "n_candidates": len(results),
        "leaders": leaders,
        "results": results,
    }


if __name__ == "__main__":
    main()
