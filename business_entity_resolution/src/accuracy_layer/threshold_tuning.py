"""Threshold tuning and validation split module for Macro F0.5 optimization.

Owner: Accuracy / Reranking / Tuning module.
Implements:
1. Leak-free validation split grouped by Source 1 entity_id hash.
2. Comprehensive multi-threshold evaluation (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95).
3. Metric calculations at each threshold:
   - Precision
   - Recall
   - Macro F0.5 (per challenge specification)
   - Number of predicted matches
   - True Positives, False Positives, False Negatives
   - Singleton accuracy (correct singletons vs false merges)
4. Joint tuning of threshold, confidence hurdle (singleton protection), and relative margin.
"""
from dataclasses import dataclass
import hashlib
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .. import config, evaluate


@dataclass
class ThresholdSearchResult:
    threshold: float
    confidence_hurdle: float
    margin: float
    macro_f0_5: float
    macro_precision: float
    macro_recall: float
    n_predicted_matches: int
    n_tp: int
    n_fp: int
    n_fn: int
    n_singletons: int
    n_correct_singletons: int
    n_false_merge_singletons: int

    def __str__(self) -> str:
        return (
            f"Th={self.threshold:.2f} (Hurdle={self.confidence_hurdle:.2f}, Mg={self.margin:.2f}) | "
            f"F0.5={self.macro_f0_5:.4f} | Prec={self.macro_precision:.4f} | Rec={self.macro_recall:.4f} | "
            f"Matches={self.n_predicted_matches} (TP={self.n_tp}, FP={self.n_fp}, FN={self.n_fn}) | "
            f"Singletons={self.n_correct_singletons}/{self.n_singletons}"
        )


def make_entity_hash_split(
    s1_df: pd.DataFrame,
    n_splits: int = 5,
    val_fold: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create leak-free train/validation split by hashing Source 1 entity_id."""
    val_mask = s1_df[config.COL_ENTITY_ID].apply(
        lambda x: int(hashlib.md5(str(x).encode("utf-8")).hexdigest(), 16) % n_splits == val_fold
    )
    train_df = s1_df[~val_mask].copy().reset_index(drop=True)
    val_df = s1_df[val_mask].copy().reset_index(drop=True)
    return train_df, val_df


class ThresholdTuner:
    """Evaluates and tunes decision thresholds and confidence hurdles for Macro F0.5."""

    DEFAULT_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    def evaluate_predictions(
        self,
        predictions: Dict[str, List[str]],
        ground_truth: Dict[str, Set[str]],
        threshold: float = 0.65,
        confidence_hurdle: float = 0.70,
        margin: float = 0.12,
    ) -> ThresholdSearchResult:
        """Compute full challenge metrics, pair metrics, and singleton metrics."""
        score_dict = evaluate.macro_f0_5(predictions, ground_truth)
        macro_f0_5 = float(score_dict["macro_f0_5"])

        tot_tp, tot_fp, tot_fn = 0, 0, 0
        n_predicted = 0
        per_entity_prec = []
        per_entity_rec = []

        n_singletons = sum(1 for t in ground_truth.values() if not t)
        correct_singletons = 0
        false_merge_singletons = 0

        for s1_id, truth in ground_truth.items():
            pred_set = set(predictions.get(s1_id, []))
            n_predicted += len(pred_set)

            if not truth:
                if not pred_set:
                    correct_singletons += 1
                else:
                    false_merge_singletons += 1

            tp = len(pred_set & truth)
            fp = len(pred_set - truth)
            fn = len(truth - pred_set)

            tot_tp += tp
            tot_fp += fp
            tot_fn += fn

            if pred_set:
                per_entity_prec.append(tp / len(pred_set))
            if truth:
                per_entity_rec.append(tp / len(truth))

        macro_p = float(np.mean(per_entity_prec)) if per_entity_prec else 1.0
        macro_r = float(np.mean(per_entity_rec)) if per_entity_rec else 0.0

        return ThresholdSearchResult(
            threshold=round(threshold, 3),
            confidence_hurdle=round(confidence_hurdle, 3),
            margin=round(margin, 3),
            macro_f0_5=macro_f0_5,
            macro_precision=macro_p,
            macro_recall=macro_r,
            n_predicted_matches=n_predicted,
            n_tp=tot_tp,
            n_fp=tot_fp,
            n_fn=tot_fn,
            n_singletons=n_singletons,
            n_correct_singletons=correct_singletons,
            n_false_merge_singletons=false_merge_singletons,
        )

    def tune(
        self,
        predict_fn: Callable[[float, float, float], Dict[str, List[str]]],
        ground_truth: Dict[str, Set[str]],
        thresholds: Optional[List[float]] = None,
        hurdles: Optional[List[float]] = None,
        margins: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> Tuple[ThresholdSearchResult, List[ThresholdSearchResult]]:
        """Search threshold grid and return optimal result."""
        if thresholds is None:
            thresholds = self.DEFAULT_THRESHOLDS
        if hurdles is None:
            hurdles = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        if margins is None:
            margins = [0.10, 0.12, 0.15]


        results: List[ThresholdSearchResult] = []
        best_result: Optional[ThresholdSearchResult] = None

        if verbose:
            print("\n" + "=" * 95)
            print("THRESHOLD TUNING GRID SEARCH FOR MACRO F0.5 (PRECISION-CONSCIOUS)")
            print("=" * 95)
            print(
                f"{'Thresh':>7} | {'Hurdle':>7} | {'Margin':>7} | {'Macro F0.5':>10} | "
                f"{'Precision':>9} | {'Recall':>8} | {'Matches':>7} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'Singles':>7}"
            )
            print("-" * 95)

        for th in thresholds:
            for hurdle in hurdles:
                for mg in margins:
                    preds = predict_fn(th, mg, hurdle)
                    res = self.evaluate_predictions(preds, ground_truth, th, hurdle, mg)
                    results.append(res)

                    if best_result is None or (
                        res.macro_f0_5 > best_result.macro_f0_5
                        or (
                            abs(res.macro_f0_5 - best_result.macro_f0_5) < 1e-4
                            and res.macro_precision > best_result.macro_precision
                        )
                    ):
                        best_result = res

                    if verbose and hurdle == 0.70 and mg == 0.12:
                        print(
                            f"{res.threshold:7.2f} | {res.confidence_hurdle:7.2f} | {res.margin:7.2f} | "
                            f"{res.macro_f0_5:10.4f} | {res.macro_precision:9.4f} | {res.macro_recall:8.4f} | "
                            f"{res.n_predicted_matches:7d} | {res.n_tp:4d} | {res.n_fp:4d} | {res.n_fn:4d} | "
                            f"{res.n_correct_singletons:3d}/{res.n_singletons:<3d}"
                        )

        if verbose and best_result:
            print("-" * 95)
            print(f"Optimal Configuration Selected:")
            print(f"  Decision Threshold:    {best_result.threshold:.2f}")
            print(f"  Confidence Hurdle:     {best_result.confidence_hurdle:.2f}")
            print(f"  Relative Margin:       {best_result.margin:.2f}")
            print(f"  Validation Macro F0.5: {best_result.macro_f0_5:.4f}")
            print(f"  Validation Precision:  {best_result.macro_precision:.4f}")
            print(f"  Validation Recall:     {best_result.macro_recall:.4f}")
            print(f"  Predicted Matches:     {best_result.n_predicted_matches} (TP={best_result.n_tp}, FP={best_result.n_fp}, FN={best_result.n_fn})")
            print(f"  Singleton Accuracy:    {best_result.n_correct_singletons} / {best_result.n_singletons}")
            print("=" * 95)

        return best_result, results
