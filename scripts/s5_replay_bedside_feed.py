#!/usr/bin/env python3
"""s5_replay_bedside_feed.py - Replay an auditable prospective bedside feed."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s5.bedside_adapter import (
    iter_stage5_bedside_feed,
    load_bedside_feed_jsonl,
    replay_bedside_feed,
    tee_bedside_feed_jsonl,
)
from s5.bedside_service import BedsideMonitoringService


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
    parser = argparse.ArgumentParser(description="Replay Stage 5 bedside feed with audit outputs")
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feed-jsonl", default=None)
    parser.add_argument("--s0-dir", default=None)
    parser.add_argument("--treatment-dir", default=None)
    parser.add_argument("--note-embeddings", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--feed-order", choices=["round_robin", "patient_major"], default="round_robin")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--write-feed-jsonl", default=None)
    parser.add_argument("--phenotype-centroids", default=None)
    parser.add_argument("--treatment-feature-names", default=None)
    parser.add_argument("--dashboard-dir", default=None)
    parser.add_argument("--render-dashboard-on-alert", action="store_true")
    parser.add_argument("--render-final-dashboards", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-snapshots", type=int, default=72)
    return parser.parse_args()


def _build_events(args: argparse.Namespace):
    feed_jsonl = _resolve(args.feed_jsonl)
    s0_dir = _resolve(args.s0_dir)
    treatment_dir = _resolve(args.treatment_dir)
    note_embeddings = _resolve(args.note_embeddings)
    if feed_jsonl is not None:
        if s0_dir is not None or treatment_dir is not None:
            raise ValueError("Use either --feed-jsonl or (--s0-dir and --treatment-dir), not both")
        return load_bedside_feed_jsonl(feed_jsonl)
    if s0_dir is None or treatment_dir is None:
        raise ValueError("Either --feed-jsonl or both --s0-dir and --treatment-dir are required")
    return iter_stage5_bedside_feed(
        s0_dir=s0_dir,
        treatment_dir=treatment_dir,
        note_embeddings_path=note_embeddings,
        split=args.split,
        max_patients=args.max_patients,
        max_events=args.max_events,
        order=args.feed_order,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    output_dir = _resolve(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir = _resolve(args.dashboard_dir) or (output_dir / "dashboards")
    audit_jsonl_path = output_dir / "bedside_audit.jsonl"

    service = BedsideMonitoringService.from_artifacts(
        model_artifact_path=_resolve(args.model_artifact),
        policy_path=_resolve(args.policy_path),
        phenotype_centroids_path=_resolve(args.phenotype_centroids),
        treatment_feature_names_path=_resolve(args.treatment_feature_names),
        device=get_device(args.device),
        max_snapshots=max(1, int(args.max_snapshots)),
        dashboard_dir=dashboard_dir,
    )
    if service.note_dim > 0 and args.feed_jsonl is None and args.note_embeddings is None:
        raise ValueError("Loaded artifact requires note embeddings, but --note-embeddings was not provided")

    events = _build_events(args)
    if args.write_feed_jsonl is not None:
        events = tee_bedside_feed_jsonl(_resolve(args.write_feed_jsonl), events)

    summary = replay_bedside_feed(
        service=service,
        events=events,
        output_dir=output_dir,
        audit_jsonl_path=audit_jsonl_path,
        dashboard_dir=dashboard_dir,
        render_dashboard_on_alert=bool(args.render_dashboard_on_alert),
        render_final_dashboards=bool(args.render_final_dashboards),
    )
    logging.getLogger("s5.bedside.replay").info(
        "Replay complete. events=%s patients=%s alert_events=%s active_final=%s summary=%s",
        summary["n_events"],
        summary["n_patients_seen"],
        summary["n_alert_events"],
        summary["n_active_patients_final"],
        summary["artifacts"]["summary_json"],
    )


if __name__ == "__main__":
    main()
