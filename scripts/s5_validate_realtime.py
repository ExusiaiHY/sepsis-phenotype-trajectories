#!/usr/bin/env python3
"""s5_validate_realtime.py - Generate reproducible Stage 5 validation artifacts."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s5.reporting import build_s5_validation_summary, write_s5_validation_artifacts


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 5 validation summary and figures")
    parser.add_argument("--mimic-report", default="data/s5/realtime_mimic/realtime_student_report.json")
    parser.add_argument("--eicu-report", default="data/s5/realtime_eicu/realtime_student_report.json")
    parser.add_argument("--reports-dir", default="outputs/reports/s5")
    parser.add_argument("--figures-dir", default="docs/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    bundle = build_s5_validation_summary(
        mimic_report_path=_resolve(args.mimic_report),
        eicu_report_path=_resolve(args.eicu_report),
    )
    artifacts = write_s5_validation_artifacts(
        bundle,
        reports_dir=_resolve(args.reports_dir),
        figures_dir=_resolve(args.figures_dir),
    )
    logging.getLogger("s5.validation").info("Wrote Stage 5 validation artifacts: %s", artifacts)


if __name__ == "__main__":
    main()
