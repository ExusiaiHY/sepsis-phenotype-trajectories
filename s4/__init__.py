"""Stage 4: treatment-aware temporal phenotyping and causal analysis."""

from s4.causal_analysis import (
    build_causal_frame,
    estimate_causal_forest_dml,
    estimate_propensity_score_matching,
    estimate_regression_discontinuity,
    generate_precision_treatment_recommendations,
    run_causal_suite,
)
from s4.treatment_aware_model import (
    TreatmentAwareClassifier,
    TreatmentAwareEncoder,
    extract_treatment_aware_embeddings,
    extract_treatment_aware_rolling_embeddings,
    train_treatment_aware_classifier,
)
from s4.treatment_features import (
    TREATMENT_FEATURES,
    build_treatment_feature_bundle,
    load_treatment_bundle,
)

__all__ = [
    "TREATMENT_FEATURES",
    "TreatmentAwareClassifier",
    "TreatmentAwareEncoder",
    "build_causal_frame",
    "build_treatment_feature_bundle",
    "estimate_causal_forest_dml",
    "estimate_propensity_score_matching",
    "estimate_regression_discontinuity",
    "extract_treatment_aware_embeddings",
    "extract_treatment_aware_rolling_embeddings",
    "generate_precision_treatment_recommendations",
    "load_treatment_bundle",
    "run_causal_suite",
    "train_treatment_aware_classifier",
]
