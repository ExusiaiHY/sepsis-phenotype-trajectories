#!/usr/bin/env python3
"""s4_closeout_report.py - Generate reproducible Stage 4 closeout artifacts."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s4.reporting import build_s4_closeout_summary, write_s4_closeout_artifacts


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 4 closeout summary and figures")
    parser.add_argument("--mimic-treatment-report", default="data/s4/mimic_treatment_aware/treatment_aware_report.json")
    parser.add_argument("--eicu-treatment-report", default="data/s4/eicu_treatment_aware/treatment_aware_report.json")
    parser.add_argument("--mimic-causal-report", default="data/s4/mimic_causal/causal_analysis_report.json")
    parser.add_argument("--eicu-causal-report", default="data/s4/eicu_causal/causal_analysis_report.json")
    parser.add_argument("--reports-dir", default="outputs/reports/s4")
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
    bundle = build_s4_closeout_summary(
        mimic_treatment_report_path=_resolve(args.mimic_treatment_report),
        eicu_treatment_report_path=_resolve(args.eicu_treatment_report),
        mimic_causal_report_path=_resolve(args.mimic_causal_report),
        eicu_causal_report_path=_resolve(args.eicu_causal_report),
    )
    artifacts = write_s4_closeout_artifacts(
        bundle,
        reports_dir=_resolve(args.reports_dir),
        figures_dir=_resolve(args.figures_dir),
    )
    logging.getLogger("s4.closeout").info("Wrote Stage 4 closeout artifacts: %s", artifacts)


if __name__ == "__main__":
    main()
