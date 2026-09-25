"""Local validation scorer: F_0.5, macro-averaged per Source-1 entity.

Mirrors the challenge's evaluation exactly, so a held-out slice of train
scored here should be a reasonable proxy for the leaderboard:

- Precision/recall/F_0.5 computed per Source-1 entity, then macro-averaged.
- A true singleton (no ground-truth matches) scores 1.0 if predicted empty,
  0.0 if anything is predicted for it.
- Entities with ground-truth matches but zero predictions score 0.0
  (precision undefined -> treated as 0 recall achieved, F_0.5 = 0).

This file has no dependency on the rest of the pipeline on purpose -- it
should be trustworthy as an independent check on whatever the model
produces.
"""
from typing import Dict, Iterable, Mapping, Set


def _f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(predicted: Set[str], truth: Set[str]) -> float:
    if not truth:
        # True singleton: full credit only for an exact empty prediction.
        return 1.0 if not predicted else 0.0

    if not predicted:
        return 0.0  # recall = 0 -> F_0.5 = 0, regardless of precision

    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return _f_beta(precision, recall, beta=0.5)


def macro_f0_5(
    predictions: Mapping[str, Iterable[str]],
    ground_truth: Mapping[str, Set[str]],
) -> Dict[str, object]:
    """Score every Source-1 entity present in `ground_truth`.

    Any entity missing from `predictions` is treated as an empty prediction
    (matches how a real leaderboard would penalize a dropped row, except a
    truly missing row is a hard format violation -- validate submission
    completeness separately, this only checks matching quality).
    """
    scores = {}
    for s1_id, truth in ground_truth.items():
        predicted = set(predictions.get(s1_id, []))
        scores[s1_id] = score_entity(predicted, truth)

    macro = sum(scores.values()) / len(scores) if scores else 0.0

    n_singletons = sum(1 for t in ground_truth.values() if not t)
    n_matched = len(ground_truth) - n_singletons

    return {
        "macro_f0_5": macro,
        "n_entities": len(ground_truth),
        "n_singletons": n_singletons,
        "n_with_matches": n_matched,
        "per_entity_scores": scores,
    }


if __name__ == "__main__":
    # Tiny smoke test against the worked example in the problem statement.
    pred = {"S1-00001": ["S2-00047", "S2-00193", "S3-00812"]}
    truth = {"S1-00001": {"S2-00047", "S3-00812"}}
    result = macro_f0_5(pred, truth)
    print(result)
    assert abs(result["macro_f0_5"] - 0.7143) < 1e-3, "doesn't match worked example"
    print("OK: matches the problem statement's worked example (~0.714)")
