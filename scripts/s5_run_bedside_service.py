#!/usr/bin/env python3
"""s5_run_bedside_service.py - Launch the bedside runtime HTTP service."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s5.bedside_service import BedsideMonitoringService, run_bedside_service


def _resolve(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def get_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 5 bedside monitoring HTTP service")
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--phenotype-centroids", default=None)
    parser.add_argument("--treatment-feature-names", default=None)
    parser.add_argument("--dashboard-dir", default="outputs/reports/s5_bedside_dashboards")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-snapshots", type=int, default=72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    service = BedsideMonitoringService.from_artifacts(
        model_artifact_path=_resolve(args.model_artifact),
        policy_path=_resolve(args.policy_path),
        phenotype_centroids_path=_resolve(args.phenotype_centroids),
        treatment_feature_names_path=_resolve(args.treatment_feature_names),
        device=get_device(args.device),
        max_snapshots=max(1, int(args.max_snapshots)),
        dashboard_dir=_resolve(args.dashboard_dir),
    )
    server = run_bedside_service(
        service=service,
        host=str(args.host),
        port=int(args.port),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.getLogger("s5.bedside").info("Shutting down bedside service")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
