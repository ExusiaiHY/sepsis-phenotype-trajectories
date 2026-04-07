#!/usr/bin/env python3
"""s5_run_silent_deployment.py - Replay frozen S5 artifacts in bedside silent mode."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s5.silent_deployment import run_silent_deployment_replay


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
    parser = argparse.ArgumentParser(description="Replay Stage 5 bedside silent deployment")
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--s0-dir", required=True)
    parser.add_argument("--treatment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--note-embeddings", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-history-hours", type=int, default=6)
    parser.add_argument("--landmark-hours", nargs="*", type=int, default=[6, 12, 24, 36, 48])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--save-replay-bundle", action="store_true")
    parser.add_argument("--replay-bundle-name", default="replay_bundle.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    summary = run_silent_deployment_replay(
        model_artifact_path=_resolve(args.model_artifact),
        s0_dir=_resolve(args.s0_dir),
        treatment_dir=_resolve(args.treatment_dir),
        output_dir=_resolve(args.output_dir),
        policy_path=_resolve(args.policy_path),
        note_embeddings_path=_resolve(args.note_embeddings),
        split=args.split,
        min_history_hours=args.min_history_hours,
        landmark_hours=tuple(args.landmark_hours),
        batch_size=max(1, int(args.batch_size)),
        device=get_device(args.device),
        max_patients=args.max_patients,
        save_replay_bundle=bool(args.save_replay_bundle),
        replay_bundle_name=str(args.replay_bundle_name),
    )
    logging.getLogger("s5.silent").info(
        "Silent deployment complete. split=%s n=%s alert_rate=%s alerts_per_patient_day=%s sample_patient=%s",
        summary["split"],
        summary["n_patients"],
        summary["patient_alert_rate"],
        summary["alerts_per_patient_day"],
        summary["sample_patient_id"],
    )


if __name__ == "__main__":
    main()
