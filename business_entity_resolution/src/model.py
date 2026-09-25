"""Match / no-match classifier over candidate pairs.

Owner: feat/model.
Uses LightGBM (MIT licensed, efficient, tabular powerhouse) with:
1. Macro F0.5-optimized threshold selection.
2. Per-entity relative margin pruning.
3. Singleton preservation policy.
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
    margin: float = 0.15
    clf: Optional[lgb.LGBMClassifier] = None

    def __post_init__(self):
        if self.clf is None:
            self.clf = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
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
    ) -> Dict[str, List[str]]:
        """Apply precision-heavy relative margin decision policy per Source-1 entity.

        Keeps candidate if:
        1. prob >= threshold
        2. prob >= (max_prob_for_entity - margin)
        If no candidate exceeds threshold, entity is treated as a singleton (returns empty list).
        """
        th = self.threshold if threshold is None else threshold
        mg = self.margin if margin is None else margin

        predictions: Dict[str, List[str]] = {}

        for s1_id, cand_scores in entity_candidates_scores.items():
            if not cand_scores:
                predictions[s1_id] = []
                continue

            max_score = max(score for _, score in cand_scores)
            if max_score < th:
                # Singleton protection: no candidate is confident enough
                predictions[s1_id] = []
                continue

            selected = [
                cand_id
                for cand_id, score in cand_scores
                if score >= th and score >= (max_score - mg)
            ]
            predictions[s1_id] = selected

        return predictions

    def tune_threshold(
        self,
        entity_candidates_scores: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
        threshold_candidates: Optional[Iterable[float]] = None,
    ) -> Tuple[float, float]:
        """Grid search decision threshold to maximize macro F_0.5 score."""
        if threshold_candidates is None:
            threshold_candidates = [round(x, 2) for x in np.arange(0.35, 0.90, 0.05)]

        best_th = self.threshold
        best_score = -1.0

        for th in threshold_candidates:
            preds = self.predict_for_entities(entity_candidates_scores, threshold=th, margin=self.margin)
            score_dict = evaluate.macro_f0_5(preds, ground_truth)
            f_score = float(score_dict["macro_f0_5"])
            if f_score > best_score:
                best_score = f_score
                best_th = th

        self.threshold = best_th
        return best_th, best_score
