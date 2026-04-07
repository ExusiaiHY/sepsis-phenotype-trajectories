#!/usr/bin/env python3
"""s5_assess_adaptation_trigger.py - Decide whether Stage 5 should escalate to source-specific fine-tuning."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s5.reporting import (
    build_s5_adaptation_trigger_report,
    write_s5_adaptation_trigger_artifacts,
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess whether Stage 5 should escalate to source-specific full fine-tuning")
    parser.add_argument("--validation-summary", required=True)
    parser.add_argument("--policy-best-json", required=True)
    parser.add_argument("--policy-candidates-csv", required=True)
    parser.add_argument("--shadow-replay-summary", required=True)
    parser.add_argument("--trigger-config", required=True)
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    bundle = build_s5_adaptation_trigger_report(
        validation_summary_path=_resolve(args.validation_summary),
        policy_best_json_path=_resolve(args.policy_best_json),
        policy_candidates_csv_path=_resolve(args.policy_candidates_csv),
        shadow_replay_summary_path=_resolve(args.shadow_replay_summary),
        trigger_config_path=_resolve(args.trigger_config),
        source_key=str(args.source_key),
    )
    artifacts = write_s5_adaptation_trigger_artifacts(
        bundle,
        reports_dir=_resolve(args.output_dir),
    )

    logging.getLogger("s5.trigger").info(
        "Adaptation trigger assessed. source=%s triggered=%s next_step=%s summary=%s",
        bundle["source"],
        int(bundle["triggered"]),
        bundle["next_step"],
        artifacts["summary_json"],
    )


if __name__ == "__main__":
    main()
