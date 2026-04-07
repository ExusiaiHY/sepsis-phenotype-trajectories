#!/usr/bin/env python3
"""s5_optimize_alert_policy.py - Optimize bedside alert policies from replay bundles."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s5.deployment_policy import (
    DEFAULT_POLICY_GRID,
    POLICY_GRID_PRESETS,
    build_policy_grid,
    evaluate_policy_grid,
    load_replay_bundle,
    select_best_policy,
    write_policy_artifacts,
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _parse_max_alerts(values: list[str] | None) -> list[int | None]:
    if not values:
        return list(DEFAULT_POLICY_GRID["max_alerts_per_stay"])
    parsed: list[int | None] = []
    for value in values:
        if value.lower() == "none":
            parsed.append(None)
        else:
            parsed.append(max(1, int(value)))
    return parsed


def _dedupe_sorted(values: list[float], extra: float | None = None) -> list[float]:
    merged = list(values)
    if extra is not None:
        merged.append(float(extra))
    return sorted({round(float(value), 4) for value in merged})


def _dedupe_sorted_int(values: list[int], extra: int | None = None) -> list[int]:
    merged = list(values)
    if extra is not None:
        merged.append(int(extra))
    return sorted({max(1, int(value)) for value in merged})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Stage 5 bedside alert policy")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--grid-profile", choices=sorted(POLICY_GRID_PRESETS.keys()), default="default")
    parser.add_argument("--ranking-mode", choices=["balanced", "burden_first"], default="balanced")
    parser.add_argument("--enter-thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--hysteresis-gaps", nargs="*", type=float, default=None)
    parser.add_argument("--min-consecutive-hours", nargs="*", type=int, default=None)
    parser.add_argument("--refractory-hours", nargs="*", type=int, default=None)
    parser.add_argument("--max-alerts-per-stay", nargs="*", default=None)
    parser.add_argument("--min-history-hours", nargs="*", type=int, default=None)
    parser.add_argument("--landmark-hours", nargs="*", type=int, default=[6, 12, 24, 36, 48])
    parser.add_argument("--max-negative-patient-alert-rate", type=float, default=0.25)
    parser.add_argument("--max-alert-events-per-patient-day", type=float, default=1.0)
    parser.add_argument("--min-positive-patient-alert-rate", type=float, default=0.6)
    parser.add_argument("--min-positive-alert-rate-24h", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    bundle = load_replay_bundle(_resolve(args.bundle))
    source_name = args.source_name or Path(args.bundle).stem
    baseline_threshold = float(bundle.get("threshold", 0.5))
    baseline_min_history = int(bundle.get("min_history_hours", 6))
    grid = POLICY_GRID_PRESETS[args.grid_profile]
    min_history_candidates = _dedupe_sorted_int(
        list(args.min_history_hours or grid.get("min_history_hours", [baseline_min_history])),
        extra=baseline_min_history,
    )

    candidates = build_policy_grid(
        enter_thresholds=_dedupe_sorted(
            args.enter_thresholds or list(grid["enter_thresholds"]),
            extra=baseline_threshold,
        ),
        min_history_hours=min_history_candidates,
        hysteresis_gaps=args.hysteresis_gaps or list(grid["hysteresis_gaps"]),
        min_consecutive_hours=args.min_consecutive_hours or list(grid["min_consecutive_hours"]),
        refractory_hours=args.refractory_hours or list(grid["refractory_hours"]),
        max_alerts_per_stay=_parse_max_alerts(args.max_alerts_per_stay) if args.max_alerts_per_stay else list(grid["max_alerts_per_stay"]),
    )
    policy_frame = evaluate_policy_grid(
        labels=bundle["labels"],
        risk_matrix=bundle["risk_matrix"],
        active_hours=bundle["active_hours"],
        candidates=candidates,
        landmark_hours=tuple(args.landmark_hours),
    )
    best_policy, ranked_candidates = select_best_policy(
        policy_frame,
        max_negative_patient_alert_rate=args.max_negative_patient_alert_rate,
        max_alert_events_per_patient_day=args.max_alert_events_per_patient_day,
        min_positive_patient_alert_rate=args.min_positive_patient_alert_rate,
        min_positive_alert_rate_24h=args.min_positive_alert_rate_24h,
        ranking_mode=args.ranking_mode,
    )
    constraints = {
        "max_negative_patient_alert_rate": float(args.max_negative_patient_alert_rate),
        "max_alert_events_per_patient_day": float(args.max_alert_events_per_patient_day),
        "min_positive_patient_alert_rate": float(args.min_positive_patient_alert_rate),
        "min_positive_alert_rate_24h": float(args.min_positive_alert_rate_24h),
        "min_history_hours": min_history_candidates if len(min_history_candidates) > 1 else min_history_candidates[0],
        "grid_profile": args.grid_profile,
        "ranking_mode": args.ranking_mode,
    }
    artifacts = write_policy_artifacts(
        output_dir=_resolve(args.output_dir),
        source_name=source_name,
        ranked_candidates=ranked_candidates,
        best_policy=best_policy,
        constraints=constraints,
    )

    preview_columns = [
        "policy_name",
        "feasible",
        "constraint_penalty",
        "enter_threshold",
        "exit_threshold",
        "min_consecutive_hours",
        "refractory_hours",
        "max_alerts_per_stay",
        "positive_patient_alert_rate",
        "negative_patient_alert_rate",
        "alert_events_per_patient_day",
        "positive_alert_rate_at_24h",
        "median_first_alert_hour_positive",
    ]
    top_k = max(1, int(args.top_k))
    preview = ranked_candidates.head(top_k).loc[:, preview_columns]
    logging.getLogger("s5.policy").info(
        "Policy sweep complete. source=%s n_candidates=%s feasible=%s best_policy=%s candidates_csv=%s best_policy_json=%s",
        source_name,
        len(ranked_candidates),
        int(ranked_candidates["feasible"].sum()),
        best_policy["policy_name"],
        artifacts["candidates_csv"],
        artifacts["best_policy_json"],
    )
    logging.getLogger("s5.policy").info("Top %s candidates:\n%s", top_k, preview.to_string(index=False))


if __name__ == "__main__":
    main()
