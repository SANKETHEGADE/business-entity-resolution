"""Match / no-match classifier over candidate pairs.

Owner: feat/model.
Uses LightGBM (MIT licensed, efficient, tabular powerhouse) with:
1. Top-100 blueprint hyperparameters (n_estimators=1000, lr=0.03, max_depth=6, subsample=0.8).
2. 2D grid search over (threshold θ ∈ [0.50, 0.85], margin δ ∈ [0.05, 0.20]) to maximize Macro F_0.5.
3. Singleton-Gated Calibrated Margin policy.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import lightgbm as lgb
import numpy as np

from . import evaluate
from .features import FEATURE_ORDER


@dataclass
class MatchModel:
    threshold: float = 0.65
    margin: float = 0.10
    confidence_hurdle: float = 0.72
    clf: Optional[lgb.LGBMClassifier] = None

    def __post_init__(self):
        if self.clf is None:
            self.clf = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
                verbose=-1,
            )

    def to_matrix(self, feature_dicts: List[dict]) -> np.ndarray:
        return np.array([[fd.get(k, 0.0) for k in FEATURE_ORDER] for fd in feature_dicts], dtype=np.float32)

    def fit(self, feature_dicts: List[dict], labels: List[int]) -> "MatchModel":
        X = self.to_matrix(feature_dicts)
        y = np.array(labels, dtype=int)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, feature_dicts: List[dict]) -> np.ndarray:
        if not feature_dicts:
            return np.array([], dtype=float)
        X = self.to_matrix(feature_dicts)
        return self.clf.predict_proba(X)[:, 1]

    def predict_for_entities(
        self,
        entity_candidates_scores: Dict[str, List[Tuple[str, float]]],
        threshold: Optional[float] = None,
        margin: Optional[float] = None,
        confidence_hurdle: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """Decision Rule from Blueprint: Candidate i is retained if:

            P_i >= theta   AND   (max_j P_j - P_i) <= delta

        Plus Absolute Confidence Hurdle (singleton protection):
        If max_j P_j < hurdle, entity is immediately emitted as an empty list.
        """
        th = self.threshold if threshold is None else threshold
        mg = self.margin if margin is None else margin
        hurdle = self.confidence_hurdle if confidence_hurdle is None else confidence_hurdle

        predictions: Dict[str, List[str]] = {}

        for s1_id, cand_scores in entity_candidates_scores.items():
            if not cand_scores:
                predictions[s1_id] = []
                continue

            max_score = max(score for _, score in cand_scores)
            # Absolute confidence hurdle (singleton protection)
            if max_score < hurdle:
                predictions[s1_id] = []
                continue

            selected = [
                cand_id
                for cand_id, score in cand_scores
                if score >= th and (max_score - score) <= mg
            ]
            predictions[s1_id] = selected

        return predictions

    def tune_threshold_2d(
        self,
        entity_candidates_scores: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
        theta_range: Optional[Iterable[float]] = None,
        delta_range: Optional[Iterable[float]] = None,
    ) -> Tuple[float, float, float]:
        """2D Grid Search over threshold θ ∈ [0.30, 0.85] and margin δ ∈ [0.05, 0.20]."""
        if theta_range is None:
            theta_range = [round(x, 2) for x in np.arange(0.30, 0.86, 0.02)]
        if delta_range is None:
            delta_range = [round(x, 2) for x in np.arange(0.05, 0.21, 0.02)]

        best_theta = self.threshold
        best_delta = self.margin
        best_score = -1.0

        for th in theta_range:
            for delta in delta_range:
                preds = self.predict_for_entities(
                    entity_candidates_scores,
                    threshold=th,
                    margin=delta,
                    confidence_hurdle=th,
                )
                score_dict = evaluate.macro_f0_5(preds, ground_truth)
                f_score = float(score_dict["macro_f0_5"])
                if f_score > best_score:
                    best_score = f_score
                    best_theta = th
                    best_delta = delta

        self.threshold = best_theta
        self.margin = best_delta
        self.confidence_hurdle = best_theta
        return best_theta, best_delta, best_score
