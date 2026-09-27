"""Match / no-match classifier over candidate pairs.

Owner: feat/model.
Uses LightGBM (MIT licensed) with:
1. Improved hyperparameters: colsample_bytree, reg_alpha/lambda, min_child_samples.
2. Early stopping on a held-out split to prevent overfitting.
3. 2D grid search over (threshold θ, margin δ) to maximize Macro F_0.5.
4. Singleton-Gated Calibrated Margin policy.
5. Save/load trained model for reuse across runs.
"""
import os
from dataclasses import dataclass, field
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
    clf: Optional[lgb.LGBMClassifier] = field(default=None, repr=False)

    def __post_init__(self):
        if self.clf is None:
            self.clf = lgb.LGBMClassifier(
                n_estimators=2000,          # more trees; early stopping will find optimal
                learning_rate=0.02,         # lower LR for better generalization
                max_depth=7,                # slightly deeper for richer features
                num_leaves=63,              # 2^(max_depth) - 1
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,       # feature bagging per tree
                min_child_samples=20,       # regularize leaf size
                reg_alpha=0.1,              # L1 regularization
                reg_lambda=1.0,             # L2 regularization
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def to_matrix(self, feature_dicts: List[dict]) -> np.ndarray:
        return np.array(
            [[fd.get(k, 0.0) for k in FEATURE_ORDER] for fd in feature_dicts],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        feature_dicts: List[dict],
        labels: List[int],
        eval_feature_dicts: Optional[List[dict]] = None,
        eval_labels: Optional[List[int]] = None,
        early_stopping_rounds: int = 50,
    ) -> "MatchModel":
        """Train LightGBM with optional early stopping on an eval set.

        If eval_feature_dicts / eval_labels are provided, early stopping is
        applied to avoid overfitting. Otherwise trains for the full n_estimators.
        """
        X = self.to_matrix(feature_dicts)
        y = np.array(labels, dtype=int)

        callbacks = [lgb.log_evaluation(period=100)]

        if eval_feature_dicts and eval_labels:
            X_eval = self.to_matrix(eval_feature_dicts)
            y_eval = np.array(eval_labels, dtype=int)
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True))
            self.clf.fit(
                X, y,
                eval_set=[(X_eval, y_eval)],
                eval_metric="binary_logloss",
                callbacks=callbacks,
            )
        else:
            self.clf.fit(X, y, callbacks=callbacks)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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

            # Absolute confidence hurdle — singleton protection
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

    # ------------------------------------------------------------------
    # Threshold Tuning
    # ------------------------------------------------------------------

    def tune_threshold_2d(
        self,
        entity_candidates_scores: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
        theta_range: Optional[Iterable[float]] = None,
        delta_range: Optional[Iterable[float]] = None,
    ) -> Tuple[float, float, float]:
        """2D Grid Search over threshold θ and margin δ to maximize Macro F_0.5."""
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the trained LightGBM booster to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.clf.booster_.save_model(path)
        print(f"  [Model] Saved booster to {path}")

    def load(self, path: str) -> "MatchModel":
        """Load a previously saved LightGBM booster from disk."""
        booster = lgb.Booster(model_file=path)
        # Wrap in a sklearn-compatible shim
        self.clf = lgb.LGBMClassifier()
        self.clf._Booster = booster
        self.clf._n_features = booster.num_feature()
        self.clf.fitted_ = True
        print(f"  [Model] Loaded booster from {path}")
        return self

    def feature_importance(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """Return top-N features by LightGBM split importance."""
        importances = self.clf.booster_.feature_importance(importance_type="split")
        paired = sorted(
            zip(FEATURE_ORDER, importances), key=lambda x: x[1], reverse=True
        )
        return paired[:top_n]
