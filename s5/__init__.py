"""Stage 5: lightweight real-time phenotype monitoring and bedside interface."""

from s5.dashboard import render_clinical_dashboard_html
from s5.realtime_model import (
    RealtimePatientBuffer,
    RealtimePhenotypeMonitor,
    RealtimeStudentClassifier,
    distill_realtime_student,
    estimate_cpu_latency_ms,
    quantize_realtime_model,
)
from s5.text_features import (
    build_eicu_note_embedding_tensor,
    build_hourly_note_embeddings,
)

__all__ = [
    "RealtimePatientBuffer",
    "RealtimePhenotypeMonitor",
    "RealtimeStudentClassifier",
    "build_eicu_note_embedding_tensor",
    "build_hourly_note_embeddings",
    "distill_realtime_student",
    "estimate_cpu_latency_ms",
    "quantize_realtime_model",
    "render_clinical_dashboard_html",
]
