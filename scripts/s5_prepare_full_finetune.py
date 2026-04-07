#!/usr/bin/env python3
"""s5_prepare_full_finetune.py - Materialize a Stage 5 source-specific fine-tune run directory."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s5.finetune_prep import build_s5_finetune_prep_bundle, write_s5_finetune_prep_artifacts


def _resolve(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a Stage 5 source-specific full fine-tune run directory without starting training"
    )
    parser.add_argument("--template-config", default="config/s5_mimic_full_finetune_prep.yaml")
    parser.add_argument("--trigger-report", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--allow-untriggered", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    bundle = build_s5_finetune_prep_bundle(
        template_config_path=_resolve(args.template_config),
        project_root=PROJECT_ROOT,
        run_name=args.run_name,
        trigger_report_path=_resolve(args.trigger_report),
        output_root=_resolve(args.output_root),
        allow_untriggered=bool(args.allow_untriggered),
        overwrite=bool(args.overwrite),
    )
    artifacts = write_s5_finetune_prep_artifacts(bundle)
    manifest = bundle["prep_manifest"]

    logging.getLogger("s5.finetune_prep").info(
        "Prepared Stage 5 fine-tune run. source=%s run_name=%s config=%s manifest=%s",
        manifest["source"],
        manifest["run_name"],
        artifacts["train_config_path"],
        artifacts["prep_manifest_path"],
    )


if __name__ == "__main__":
    main()
