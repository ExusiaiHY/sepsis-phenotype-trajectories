"""Stage 5: lightweight real-time phenotype monitoring and bedside interface."""

from s5.bedside_adapter import (
    BedsideFeedAdapter,
    iter_stage5_bedside_feed,
    load_bedside_feed_jsonl,
    replay_bedside_feed,
    tee_bedside_feed_jsonl,
    write_bedside_feed_jsonl,
)
from s5.bedside_service import (
    BedsideMonitoringService,
    dispatch_bedside_request,
    make_bedside_request_handler,
    run_bedside_service,
)
from s5.dashboard import render_clinical_dashboard_html
from s5.deployment_policy import (
    build_policy_grid,
    evaluate_alert_policy,
    evaluate_policy_grid,
    load_policy_artifact,
    load_replay_bundle,
    select_best_policy,
    simulate_alert_policy,
)
from s5.realtime_model import (
    RealtimeAlertPolicy,
    RealtimePatientBuffer,
    RealtimePhenotypeMonitor,
    RealtimeStudentClassifier,
    distill_realtime_student,
    estimate_cpu_latency_ms,
    load_realtime_student_artifact,
    quantize_realtime_model,
)
from s5.silent_deployment import run_silent_deployment_replay
from s5.text_features import (
    build_eicu_note_embedding_tensor,
    build_hourly_note_embeddings,
)

__all__ = [
    "BedsideFeedAdapter",
    "BedsideMonitoringService",
    "RealtimeAlertPolicy",
    "RealtimePatientBuffer",
    "RealtimePhenotypeMonitor",
    "RealtimeStudentClassifier",
    "build_policy_grid",
    "build_eicu_note_embedding_tensor",
    "build_hourly_note_embeddings",
    "distill_realtime_student",
    "estimate_cpu_latency_ms",
    "evaluate_alert_policy",
    "evaluate_policy_grid",
    "dispatch_bedside_request",
    "iter_stage5_bedside_feed",
    "load_bedside_feed_jsonl",
    "load_policy_artifact",
    "load_replay_bundle",
    "make_bedside_request_handler",
    "load_realtime_student_artifact",
    "quantize_realtime_model",
    "render_clinical_dashboard_html",
    "replay_bedside_feed",
    "run_bedside_service",
    "run_silent_deployment_replay",
    "select_best_policy",
    "simulate_alert_policy",
    "tee_bedside_feed_jsonl",
    "write_bedside_feed_jsonl",
]
