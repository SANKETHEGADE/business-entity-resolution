"""Accuracy, Reranking, and Threshold Tuning Layer.

Owner: Accuracy / Reranking / Tuning module.
Adds an isolated post-scoring layer on top of the existing pipeline:
- Enhanced pairwise similarity features (name, address, postal, interaction, TF-IDF).
- Second-stage reranker combining baseline model probability with enhanced features.
- Confidence scoring and conservative false-positive filtering.
- Threshold tuning optimized for Macro F0.5.
- Singleton-protected calibrated margin decision policy.
"""

from .accuracy_features import (
    AccuracyFeatureExtractor,
    compute_pair_accuracy_features,
    ACCURACY_FEATURE_NAMES,
)
from .reranker import (
    AccuracyReranker,
    run_accuracy_reranker_train,
    run_accuracy_reranker_test,
)
from .threshold_tuning import ThresholdTuner, ThresholdSearchResult
from .evaluation import evaluate_pipeline_comparison

__all__ = [
    "AccuracyFeatureExtractor",
    "compute_pair_accuracy_features",
    "ACCURACY_FEATURE_NAMES",
    "AccuracyReranker",
    "ThresholdTuner",
    "ThresholdSearchResult",
    "evaluate_pipeline_comparison",
    "run_accuracy_reranker_train",
    "run_accuracy_reranker_test",
]

