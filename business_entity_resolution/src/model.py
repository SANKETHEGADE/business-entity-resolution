"""Match / no-match classifier over candidate pairs.

Owner: feat/model branch.

Constraints from the challenge:
- Final model must be MIT/Apache-2.0 licensed and <= 8B parameters. A
  scikit-learn classifier over hand-engineered features (this stub) trivially
  satisfies that; if the team wants an embedding/transformer component later,
  double-check its license and parameter count before adopting it.
- F_0.5 is precision-heavy and computed *per Source-1 entity* (not globally),
  so:
    - Tune the decision threshold on validation data using the macro F_0.5
      scorer in evaluate.py, not plain accuracy/AUC.
    - Consider constraining predictions per Source-1 entity (e.g. only keep
      candidates above threshold AND within some margin of the top score)
      rather than a single global cutoff, since a business with many
      look-alikes needs a stricter bar than one with an obvious unique match.

This module intentionally does not hardcode a training loop shape yet --
fill in once features.py has a settled feature set and blocking.py has
established candidate recall.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from . import config

FEATURE_ORDER: List[str] = [
    "name_exact_match",
    "name_token_jaccard",
    "name_fuzzy_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "address_fuzzy_ratio",
    "address_token_sort_ratio",
    "country_match",
    "name_len_diff",
]


@dataclass
class MatchModel:
    threshold: float = 0.5
    clf: GradientBoostingClassifier = None

    def __post_init__(self):
        if self.clf is None:
            self.clf = GradientBoostingClassifier(random_state=config.RANDOM_SEED)

    def to_matrix(self, feature_dicts: List[dict]) -> np.ndarray:
        return np.array([[fd[k] for k in FEATURE_ORDER] for fd in feature_dicts], dtype=float)

    def fit(self, feature_dicts: List[dict], labels: List[int]) -> "MatchModel":
        X = self.to_matrix(feature_dicts)
        self.clf.fit(X, labels)
        return self

    def predict_proba(self, feature_dicts: List[dict]) -> np.ndarray:
        X = self.to_matrix(feature_dicts)
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, feature_dicts: List[dict]) -> np.ndarray:
        return (self.predict_proba(feature_dicts) >= self.threshold).astype(int)
