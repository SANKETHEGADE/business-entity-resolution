"""Comprehensive evaluation comparing baseline pipeline vs improved accuracy/reranking layer.

Owner: Accuracy / Reranking / Tuning module.
Produces complete Before-vs-After benchmark:
- Baseline Precision, Recall, Macro F0.5, Predicted Matches, FP, FN, Singletons.
- Improved Precision, Recall, Macro F0.5, Predicted Matches, FP, FN, Singletons.
- Absolute and relative F0.5 improvement.
- Evaluation of ensemble vs ML reranker.
- Standalone CLI entrypoint: python -m src.accuracy_layer.evaluation [--sample]
"""
import argparse
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from tqdm import tqdm

from .. import blocking, config, evaluate, features, io_utils, model, pipeline
from .accuracy_features import AccuracyFeatureExtractor, ACCURACY_FEATURE_NAMES
from .reranker import AccuracyReranker
from .threshold_tuning import ThresholdTuner, make_entity_hash_split


def featurize_candidates_with_accuracy_layer(
    s1_df: pd.DataFrame,
    candidates: Dict[str, Set[str]],
    other_records: Dict[str, dict],
    baseline_probs_by_pair: Dict[Tuple[str, str], float],
    extractor: AccuracyFeatureExtractor,
    ground_truth: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[List[dict], List[int], Dict[str, List[Tuple[str, dict]]]]:
    """Build dataset with both baseline features and enhanced accuracy features."""
    s1_records = pipeline.extract_record_maps(s1_df)

    feat_dicts: List[dict] = []
    labels: List[int] = []
    entity_cand_features: Dict[str, List[Tuple[str, dict]]] = {}

    for s1_id in s1_df[config.COL_ENTITY_ID]:
        s1_meta = s1_records[s1_id]
        cand_ids = list(candidates.get(s1_id, []))
        entity_cand_features[s1_id] = []

        true_matches = ground_truth.get(s1_id, set()) if ground_truth else set()

        for cand_id in cand_ids:
            cand_meta = other_records.get(cand_id)
            if not cand_meta:
                continue

            b_prob = baseline_probs_by_pair.get((s1_id, cand_id), 0.5)

            fd = extractor.extract_pair(
                name_a=s1_meta["name"],
                addr_a=s1_meta["address"],
                country_a=s1_meta["country"],
                name_b=cand_meta["name"],
                addr_b=cand_meta["address"],
                country_b=cand_meta["country"],
                cand_id=cand_id,
                s1_id=s1_id,
                baseline_prob=b_prob,
            )
            feat_dicts.append(fd)
            entity_cand_features[s1_id].append((cand_id, fd))

            if ground_truth is not None:
                labels.append(1 if cand_id in true_matches else 0)

    return feat_dicts, labels, entity_cand_features


def run_evaluation(sample: bool = True, val_fold: int = 0) -> dict:
    """Run full benchmark comparing baseline pipeline vs improved accuracy layer."""
    if sample or not config.TRAIN_FILES["S1"].exists():
        print("[Accuracy Layer] Running on sample dataset...")
        s1_file = config.DATA_DIR / "samples" / "sample_source1.tsv"
        s2_file = config.DATA_DIR / "samples" / "sample_source2.tsv"
        s3_file = config.DATA_DIR / "samples" / "sample_source3.tsv"
        gt_file = config.DATA_DIR / "samples" / "sample_ground_truth.tsv"
    else:
        print("[Accuracy Layer] Running on train dataset...")
        s1_file = config.TRAIN_FILES["S1"]
        s2_file = config.TRAIN_FILES["S2"]
        s3_file = config.TRAIN_FILES["S3"]
        gt_file = config.TRAIN_GROUND_TRUTH

    s1 = io_utils.read_source(s1_file)
    s2 = io_utils.read_source(s2_file)
    s3 = io_utils.read_source(s3_file)
    gt_df = io_utils.read_ground_truth(gt_file)
    ground_truth = io_utils.ground_truth_to_sets(gt_df)

    # 1. Leak-Free Validation Split
    train_s1, val_s1 = make_entity_hash_split(s1, n_splits=5, val_fold=val_fold)
    val_gt = {sid: ground_truth.get(sid, set()) for sid in val_s1[config.COL_ENTITY_ID]}
    print(f"Data Loaded: S1={len(s1)} (Train S1={len(train_s1)}, Val S1={len(val_s1)})")
    n_val_singles = sum(1 for t in val_gt.values() if not t)
    print(f"Singletons in Validation Split: {n_val_singles} / {len(val_gt)}")

    # 2. Candidate Generation (using existing blocking strategy)
    cand_s2_tr = blocking.generate_candidates(train_s1, s2, config.COL_ENTITY_ID, config.COL_NAME, top_k=25)
    cand_s3_tr = blocking.generate_candidates(train_s1, s3, config.COL_ENTITY_ID, config.COL_NAME, top_k=25)
    train_cands = {sid: cand_s2_tr.get(sid, set()) | cand_s3_tr.get(sid, set()) for sid in train_s1[config.COL_ENTITY_ID]}

    cand_s2_v = blocking.generate_candidates(val_s1, s2, config.COL_ENTITY_ID, config.COL_NAME, top_k=25)
    cand_s3_v = blocking.generate_candidates(val_s1, s3, config.COL_ENTITY_ID, config.COL_NAME, top_k=25)
    val_cands = {sid: cand_s2_v.get(sid, set()) | cand_s3_v.get(sid, set()) for sid in val_s1[config.COL_ENTITY_ID]}

    other_records = {**pipeline.extract_record_maps(s2), **pipeline.extract_record_maps(s3)}

    # 3. Baseline Feature Generation & Model Training
    print("\n--- Training Baseline Pipeline Model ---")
    X_tr_base, y_tr, train_pair_idx = pipeline.build_pair_dataset(train_s1, train_cands, other_records, ground_truth)
    X_val_base, y_val, val_pair_idx = pipeline.build_pair_dataset(val_s1, val_cands, other_records, ground_truth)

    base_matcher = model.MatchModel()
    base_matcher.fit(X_tr_base, y_tr)

    train_base_probs = base_matcher.predict_proba(X_tr_base)
    val_base_probs = base_matcher.predict_proba(X_val_base)

    # Index baseline probabilities by (s1_id, cand_id)
    train_prob_map = {}
    p_idx = 0
    for s1_id in train_s1[config.COL_ENTITY_ID]:
        for cid in train_pair_idx.get(s1_id, []):
            train_prob_map[(s1_id, cid)] = float(train_base_probs[p_idx])
            p_idx += 1

    val_prob_map = {}
    val_base_entity_scores = {}
    p_idx = 0
    for s1_id in val_s1[config.COL_ENTITY_ID]:
        cand_list = val_pair_idx.get(s1_id, [])
        scored_pairs = []
        for cid in cand_list:
            p = float(val_base_probs[p_idx])
            val_prob_map[(s1_id, cid)] = p
            scored_pairs.append((cid, p))
            p_idx += 1
        val_base_entity_scores[s1_id] = scored_pairs

    # Baseline Threshold Tuning & Evaluation
    base_th, base_f05 = base_matcher.tune_threshold(val_base_entity_scores, val_gt)
    base_preds = base_matcher.predict_for_entities(val_base_entity_scores, threshold=base_th)

    tuner = ThresholdTuner()
    baseline_result = tuner.evaluate_predictions(base_preds, val_gt, threshold=base_th)

    # 4. Enhanced Accuracy Layer: Featurize & Train Reranker
    print("\n--- Training Accuracy & Reranking Layer ---")
    extractor = AccuracyFeatureExtractor()
    all_names = list(s1[config.COL_NAME]) + list(s2[config.COL_NAME]) + list(s3[config.COL_NAME])
    all_addrs = list(s1[config.COL_ADDRESS]) + list(s2[config.COL_ADDRESS]) + list(s3[config.COL_ADDRESS])
    extractor.fit_tfidf(all_names, all_addrs)
    extractor.precompute_vectors({**pipeline.extract_record_maps(s1), **other_records})

    X_tr_rerank, y_tr_rerank, _ = featurize_candidates_with_accuracy_layer(
        train_s1, train_cands, other_records, train_prob_map, extractor, ground_truth
    )
    X_val_rerank, y_val_rerank, val_cands_with_feats = featurize_candidates_with_accuracy_layer(
        val_s1, val_cands, other_records, val_prob_map, extractor, ground_truth
    )

    reranker = AccuracyReranker(mode="ml", feature_extractor=extractor)
    reranker.fit(X_tr_rerank, y_tr_rerank)

    # Rescore validation pairs with reranker
    val_rerank_scores = reranker.rerank_entity_candidates(val_cands_with_feats)

    # 5. Tune Threshold on Reranker
    def predict_reranker(th: float, mg: float, hurdle: float) -> Dict[str, List[str]]:
        return reranker.predict_for_entities(
            val_rerank_scores, threshold=th, margin=mg, confidence_hurdle=hurdle
        )

    best_rerank_result, _ = tuner.tune(
        predict_fn=predict_reranker,
        ground_truth=val_gt,
        verbose=True,
    )

    # 6. Ensemble Evaluation (Requirement 5)
    print("\n--- Evaluating Ensemble Mode vs ML Mode ---")
    ensemble_reranker = AccuracyReranker(mode="ensemble", clf=reranker.clf, feature_extractor=extractor)
    val_ensemble_scores = ensemble_reranker.rerank_entity_candidates(val_cands_with_feats)

    def predict_ensemble(th: float, mg: float, hurdle: float) -> Dict[str, List[str]]:
        return ensemble_reranker.predict_for_entities(
            val_ensemble_scores, threshold=th, margin=mg, confidence_hurdle=hurdle
        )

    best_ensemble_result, _ = tuner.tune(
        predict_fn=predict_ensemble,
        ground_truth=val_gt,
        verbose=False,
    )
    print(f"ML Reranker F0.5: {best_rerank_result.macro_f0_5:.4f}")
    print(f"Ensemble Reranker F0.5: {best_ensemble_result.macro_f0_5:.4f}")

    # Choose best performer
    if best_ensemble_result.macro_f0_5 > best_rerank_result.macro_f0_5:
        chosen_result = best_ensemble_result
        chosen_mode = "Ensemble"
    else:
        chosen_result = best_rerank_result
        chosen_mode = "ML (LightGBM Reranker)"

    # 7. Final Before vs After Comparison Report
    f05_diff = chosen_result.macro_f0_5 - baseline_result.macro_f0_5
    f05_pct = (f05_diff / baseline_result.macro_f0_5) * 100 if baseline_result.macro_f0_5 > 0 else 0.0

    print("\n" + "=" * 80)
    print("FINAL BEFORE-VS-AFTER VALIDATION COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Baseline Pipeline':<20} | {'Improved Accuracy Layer':<20}")
    print("-" * 80)
    print(f"{'Validation Macro F0.5':<30} | {baseline_result.macro_f0_5:<20.4f} | {chosen_result.macro_f0_5:<20.4f}")
    print(f"{'Macro Precision':<30} | {baseline_result.macro_precision:<20.4f} | {chosen_result.macro_precision:<20.4f}")
    print(f"{'Macro Recall':<30} | {baseline_result.macro_recall:<20.4f} | {chosen_result.macro_recall:<20.4f}")
    print(f"{'Decision Threshold':<30} | {baseline_result.threshold:<20.2f} | {chosen_result.threshold:<20.2f}")
    print(f"{'Confidence Hurdle (Singleton)':<30} | {baseline_result.confidence_hurdle:<20.2f} | {chosen_result.confidence_hurdle:<20.2f}")
    print(f"{'Predicted Matches':<30} | {baseline_result.n_predicted_matches:<20d} | {chosen_result.n_predicted_matches:<20d}")
    print(f"{'True Positives (TP)':<30} | {baseline_result.n_tp:<20d} | {chosen_result.n_tp:<20d}")
    print(f"{'False Positives (FP)':<30} | {baseline_result.n_fp:<20d} | {chosen_result.n_fp:<20d}")
    print(f"{'False Negatives (FN)':<30} | {baseline_result.n_fn:<20d} | {chosen_result.n_fn:<20d}")
    print(f"{'Singletons Correct/Total':<30} | {f'{baseline_result.n_correct_singletons}/{baseline_result.n_singletons}':<20} | {f'{chosen_result.n_correct_singletons}/{chosen_result.n_singletons}':<20}")
    print(f"{'Selected Strategy':<30} | {'LightGBM Base':<20} | {chosen_mode:<20}")
    print("-" * 80)
    print(f"Absolute F0.5 Improvement:  {f05_diff:+.4f}")
    print(f"Relative F0.5 Improvement:  {f05_pct:+.2f}%")
    print("=" * 80 + "\n")

    return {
        "baseline": baseline_result,
        "improved": chosen_result,
        "delta_f0_5": f05_diff,
        "chosen_mode": chosen_mode,
    }


# Alias for package export
evaluate_pipeline_comparison = run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy layer comparison.")
    parser.add_argument("--sample", action="store_true", default=True, help="Run on sample dataset.")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold (0-4).")
    args = parser.parse_args()

    run_evaluation(sample=args.sample, val_fold=args.fold)


if __name__ == "__main__":
    main()

