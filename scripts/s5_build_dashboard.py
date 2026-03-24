#!/usr/bin/env python3
"""s5_build_dashboard.py - Render a demo bedside dashboard from JSON snapshots."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s5.dashboard import render_clinical_dashboard_html


def _resolve(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def parse_args():
    parser = argparse.ArgumentParser(description="Render Stage 5 dashboard HTML")
    parser.add_argument("--config", default="config/s5_config.yaml")
    parser.add_argument("--snapshots-json", required=True)
    parser.add_argument("--output-html", default=None)
    parser.add_argument("--patient-id", default="demo-patient")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    with open(_resolve(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    snapshots = json.loads(Path(args.snapshots_json).read_text(encoding="utf-8"))
    html = render_clinical_dashboard_html(
        patient_id=args.patient_id,
        snapshots=snapshots,
        output_path=_resolve(args.output_html or cfg["paths"]["dashboard_html"]),
        model_meta=cfg.get("model", {}),
    )
    logging.getLogger("s5.dashboard").info("Dashboard rendered (%s chars).", len(html))


if __name__ == "__main__":
    main()
