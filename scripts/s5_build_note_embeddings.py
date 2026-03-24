#!/usr/bin/env python3
"""s5_build_note_embeddings.py - Build hourly note embeddings for Stage 5."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s5.text_features import build_eicu_note_embedding_tensor


def _resolve(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stage 5 note embeddings")
    parser.add_argument("--config", default="config/s5_config.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--cohort-static", default=None)
    parser.add_argument("--output-path", default=None)
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
    report = build_eicu_note_embedding_tensor(
        raw_dir=_resolve(args.raw_dir or cfg["notes"]["raw_dir"]),
        cohort_static_path=_resolve(args.cohort_static or cfg["notes"]["cohort_static"]),
        output_path=_resolve(args.output_path or cfg["notes"]["output_path"]),
        n_hours=int(cfg["notes"].get("n_hours", 48)),
        n_features=int(cfg["notes"].get("n_features", 512)),
        n_components=int(cfg["notes"].get("n_components", 16)),
    )
    logging.getLogger("s5.notes").info("Saved note embeddings: %s", report["output_path"])


if __name__ == "__main__":
    main()
