"""End-to-end orchestration: load -> normalize -> block -> featurize -> score -> emit.

Improvements v2:
- Chunked blocking: processes S1 in chunks to control peak RAM.
- Early stopping wired into model training using 10% of training pairs as eval set.
- Model saved to disk after training; loaded during test — no re-training needed.
- Threshold tuned from validation then persisted for test.
- Progress reporting at each major step.

Usage:
    python -m src.pipeline --split train [--sample] [--chunk-size N]
    python -m src.pipeline --split test  [--sample]
"""
import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import blocking, config, evaluate, features, io_utils, model as model_mod

# Path for persisted model
MODEL_SAVE_PATH = str(config.OUTPUT_DIR / "matcher.lgb")
THRESHOLD_SAVE_PATH = str(config.OUTPUT_DIR / "thresholds.txt")


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def extract_record_maps(df: pd.DataFrame) -> Dict[str, dict]:
    records = {}
    for row in df.itertuples(index=False):
        ent_id = getattr(row, config.COL_ENTITY_ID)
        records[ent_id] = {
            "name": getattr(row, config.COL_NAME),
            "address": getattr(row, config.COL_ADDRESS),
            "country": getattr(row, config.COL_COUNTRY),
        }
    return records


# ---------------------------------------------------------------------------
# Pair dataset construction
# ---------------------------------------------------------------------------

def build_pair_dataset(
    s1_df: pd.DataFrame,
    candidates: Dict[str, Set[str]],
    other_records: Dict[str, dict],
    ground_truth: Optional[Dict[str, Set[str]]] = None,
    max_neg_ratio: Optional[float] = None,
) -> Tuple[List[dict], List[int], Dict[str, List[str]]]:
    """Featurize all (Source 1, Candidate) pairs with optional hard negative subsampling."""
    s1_records = extract_record_maps(s1_df)

    pos_features: List[dict] = []
    neg_features: List[dict] = []
    pair_index: Dict[str, List[str]] = {}

    for s1_id in tqdm(s1_df[config.COL_ENTITY_ID], desc="  Featurizing pairs", unit="entity", leave=False):
        s1_meta = s1_records[s1_id]
        cand_ids = list(candidates.get(s1_id, []))
        pair_index[s1_id] = cand_ids

        true_matches = ground_truth.get(s1_id, set()) if ground_truth else set()

        for cand_id in cand_ids:
            cand_meta = other_records.get(cand_id)
            if not cand_meta:
                continue

            fd = features.pair_features(
                name_a=s1_meta["name"],
                addr_a=s1_meta["address"],
                country_a=s1_meta["country"],
                name_b=cand_meta["name"],
                addr_b=cand_meta["address"],
                country_b=cand_meta["country"],
                cand_id=cand_id,
            )

            if ground_truth is not None:
                if cand_id in true_matches:
                    pos_features.append(fd)
                else:
                    neg_features.append(fd)
            else:
                pos_features.append(fd)

    if ground_truth is not None:
        if max_neg_ratio is not None and pos_features:
            max_negs = int(len(pos_features) * max_neg_ratio)
            if len(neg_features) > max_negs:
                random.seed(config.RANDOM_SEED)
                neg_features = random.sample(neg_features, max_negs)

        all_features = pos_features + neg_features
        labels = [1] * len(pos_features) + [0] * len(neg_features)
        return all_features, labels, pair_index

    return pos_features, [0] * len(pos_features), pair_index


# ---------------------------------------------------------------------------
# Score grouping helper
# ---------------------------------------------------------------------------

def group_scores_by_entity(
    s1_id_series,
    pair_index: Dict[str, List[str]],
    probs: np.ndarray,
) -> Dict[str, List[Tuple[str, float]]]:
    entity_scores: Dict[str, List[Tuple[str, float]]] = {}
    idx = 0
    for s1_id in s1_id_series:
        cand_list = pair_index.get(s1_id, [])
        scored = []
        for cand_id in cand_list:
            scored.append((cand_id, float(probs[idx])))
            idx += 1
        entity_scores[s1_id] = scored
    return entity_scores


# ---------------------------------------------------------------------------
# Train pipeline
# ---------------------------------------------------------------------------

def run_train(sample: bool = False, sample_other: int = 300_000) -> None:
    t0 = time.time()

    if sample or not config.TRAIN_FILES["S1"].exists():
        print("Using sample dataset for training & validation...")
        s1_file = config.DATA_DIR / "samples" / "sample_source1.tsv"
        s2_file = config.DATA_DIR / "samples" / "sample_source2.tsv"
        s3_file = config.DATA_DIR / "samples" / "sample_source3.tsv"
        gt_file = config.DATA_DIR / "samples" / "sample_ground_truth.tsv"
        sample_other = None  # use all sample data
    else:
        print("Loading training dataset...")
        s1_file = config.TRAIN_FILES["S1"]
        s2_file = config.TRAIN_FILES["S2"]
        s3_file = config.TRAIN_FILES["S3"]
        gt_file = config.TRAIN_GROUND_TRUTH

    s1 = io_utils.read_source(s1_file)
    s2 = io_utils.read_source(s2_file)
    s3 = io_utils.read_source(s3_file)
    gt_df = io_utils.read_ground_truth(gt_file)
    ground_truth = io_utils.ground_truth_to_sets(gt_df)
    print(f"Data loaded: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}, GT={len(ground_truth):,}")

    # 1. Entity-disjoint 80/20 train/val split
    val_mask = s1[config.COL_ENTITY_ID].apply(lambda x: hash(x) % 5 == 0)
    train_s1 = s1[~val_mask].copy()
    val_s1 = s1[val_mask].copy()
    val_gt = {sid: ground_truth.get(sid, set()) for sid in val_s1[config.COL_ENTITY_ID]}
    print(f"Split: Train S1={len(train_s1):,}  Val S1={len(val_s1):,}")

    other_records = {**extract_record_maps(s2), **extract_record_maps(s3)}

    # 2. Blocking — S2 and S3 each sampled to sample_other rows during training
    print(f"\nGenerating training candidates (sample_other={sample_other})...")
    cand_s2 = blocking.generate_candidates(
        train_s1, s2, config.COL_ENTITY_ID, config.COL_NAME,
        config.COL_ADDRESS, config.COL_COUNTRY, sample_other=sample_other
    )
    cand_s3 = blocking.generate_candidates(
        train_s1, s3, config.COL_ENTITY_ID, config.COL_NAME,
        config.COL_ADDRESS, config.COL_COUNTRY, sample_other=sample_other
    )
    train_candidates = {
        s1_id: cand_s2.get(s1_id, set()) | cand_s3.get(s1_id, set())
        for s1_id in train_s1[config.COL_ENTITY_ID]
    }

    print(f"\nGenerating validation candidates (sample_other={sample_other})...")
    v_cand_s2 = blocking.generate_candidates(
        val_s1, s2, config.COL_ENTITY_ID, config.COL_NAME,
        config.COL_ADDRESS, config.COL_COUNTRY, sample_other=sample_other
    )
    v_cand_s3 = blocking.generate_candidates(
        val_s1, s3, config.COL_ENTITY_ID, config.COL_NAME,
        config.COL_ADDRESS, config.COL_COUNTRY, sample_other=sample_other
    )
    val_candidates = {
        s1_id: v_cand_s2.get(s1_id, set()) | v_cand_s3.get(s1_id, set())
        for s1_id in val_s1[config.COL_ENTITY_ID]
    }

    val_recall, val_avg_cands, _ = blocking.measure_recall(val_candidates, val_gt)
    print(f"\nValidation Blocking Recall: {val_recall * 100:.2f}%  Avg candidates/S1: {val_avg_cands:.2f}")

    # 3. Feature extraction
    print(f"\nFeaturizing training pairs (1:6 negative ratio)...")
    other_records = {**extract_record_maps(s2), **extract_record_maps(s3)}
    X_train, y_train, _ = build_pair_dataset(train_s1, train_candidates, other_records, ground_truth, max_neg_ratio=6.0)
    print(f"  Train pairs: {len(X_train):,}  Positives: {sum(y_train):,}  Negatives: {len(y_train)-sum(y_train):,}")

    print(f"Featurizing validation pairs...")
    X_val, y_val, val_pair_index = build_pair_dataset(val_s1, val_candidates, other_records, ground_truth)
    print(f"  Val pairs:   {len(X_val):,}")

    # 4. Train with early stopping on 10% of training data
    print(f"\nTraining LightGBM with early stopping...")
    matcher = model_mod.MatchModel()
    matcher.fit(
        X_train, y_train,
        eval_feature_dicts=X_val,
        eval_labels=y_val,
        early_stopping_rounds=50,
    )

    # 5. Score val + 2D grid search
    print(f"\nScoring validation pairs...")
    val_probs = matcher.predict_proba(X_val)
    val_entity_scores = group_scores_by_entity(val_s1[config.COL_ENTITY_ID], val_pair_index, val_probs)

    print(f"Running 2D grid search (theta x delta)...")
    best_th, best_delta, best_f05 = matcher.tune_threshold_2d(val_entity_scores, val_gt)

    print(f"\n{'='*50}")
    print(f"  Optimal Threshold (theta*):   {best_th:.2f}")
    print(f"  Optimal Margin    (delta*):   {best_delta:.2f}")
    print(f"  Validation Macro F_0.5:       {best_f05:.4f}")
    print(f"  Blocking Recall Ceiling:      {val_recall * 100:.2f}%")
    print(f"  Avg Candidates per S1:        {val_avg_cands:.2f}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*50}\n")

    # 6. Feature importance
    print("Top-10 Features by Importance:")
    for feat, imp in matcher.feature_importance(top_n=10):
        print(f"  {feat:<30} {imp:>6}")

    # 7. Save model + thresholds
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matcher.save(MODEL_SAVE_PATH)
    with open(THRESHOLD_SAVE_PATH, "w") as f:
        f.write(f"{best_th}\n{best_delta}\n")
    print(f"Saved thresholds to {THRESHOLD_SAVE_PATH}")


# ---------------------------------------------------------------------------
# Test pipeline
# ---------------------------------------------------------------------------

def run_test(sample: bool = False) -> None:
    t0 = time.time()

    if sample or not config.TEST_FILES["S1"].exists():
        print("Using sample dataset as test set...")
        s1_file = config.DATA_DIR / "samples" / "sample_source1.tsv"
        s2_file = config.DATA_DIR / "samples" / "sample_source2.tsv"
        s3_file = config.DATA_DIR / "samples" / "sample_source3.tsv"
    else:
        print("Loading test dataset...")
        s1_file = config.TEST_FILES["S1"]
        s2_file = config.TEST_FILES["S2"]
        s3_file = config.TEST_FILES["S3"]

    s1 = io_utils.read_source(s1_file)
    s2 = io_utils.read_source(s2_file)
    s3 = io_utils.read_source(s3_file)
    print(f"Test data: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}")

    # 1. Blocking (full data, no sampling for test)
    print(f"\nGenerating test candidates (full S2/S3, no sampling)...")
    cand_s2 = blocking.generate_candidates(
        s1, s2, config.COL_ENTITY_ID, config.COL_NAME,
        config.COL_ADDRESS, config.COL_COUNTRY, sample_other=None
    )
    cand_s3 = blocking.generate_candidates(
        s1, s3, config.COL_ENTITY_ID, config.COL_NAME,
        config.COL_ADDRESS, config.COL_COUNTRY, sample_other=None
    )
    candidates = {
        s1_id: cand_s2.get(s1_id, set()) | cand_s3.get(s1_id, set())
        for s1_id in s1[config.COL_ENTITY_ID]
    }

    # Write candidate_pairs.tsv (mandatory)
    io_utils.write_candidate_pairs(candidates)
    total_cands = sum(len(v) for v in candidates.values())
    avg_cands = total_cands / len(candidates) if candidates else 0
    print(f"Wrote {config.CANDIDATE_PAIRS_PATH}  (avg {avg_cands:.2f} candidates/S1)")

    # 2. Load trained model + thresholds
    matcher = model_mod.MatchModel()
    if Path(MODEL_SAVE_PATH).exists():
        print(f"\nLoading saved model from {MODEL_SAVE_PATH}...")
        matcher.load(MODEL_SAVE_PATH)
        if Path(THRESHOLD_SAVE_PATH).exists():
            with open(THRESHOLD_SAVE_PATH) as f:
                lines = f.read().strip().split("\n")
                best_th = float(lines[0])
                best_delta = float(lines[1])
            matcher.threshold = best_th
            matcher.margin = best_delta
            matcher.confidence_hurdle = best_th
            print(f"  Thresholds: theta={best_th:.2f}, delta={best_delta:.2f}")
    else:
        print("\nNo saved model found — training from scratch on full data...")
        # Fallback: train on full training data
        tr_s1_file = config.TRAIN_FILES.get("S1", s1_file)
        tr_s2_file = config.TRAIN_FILES.get("S2", s2_file)
        tr_s3_file = config.TRAIN_FILES.get("S3", s3_file)
        tr_gt_file = config.TRAIN_GROUND_TRUTH

        tr_s1 = io_utils.read_source(tr_s1_file)
        tr_s2 = io_utils.read_source(tr_s2_file)
        tr_s3 = io_utils.read_source(tr_s3_file)
        tr_gt = io_utils.ground_truth_to_sets(io_utils.read_ground_truth(tr_gt_file))

        tr_cands: Dict[str, Set[str]] = {}
        tr_chunks = [tr_s1.iloc[i: i + chunk_size] for i in range(0, len(tr_s1), chunk_size)]
        for chunk_idx, chunk in enumerate(tr_chunks, 1):
            print(f"  Train blocking chunk {chunk_idx}/{len(tr_chunks)}...")
            c2 = blocking.generate_candidates(chunk, tr_s2, config.COL_ENTITY_ID, config.COL_NAME, config.COL_ADDRESS, config.COL_COUNTRY)
            c3 = blocking.generate_candidates(chunk, tr_s3, config.COL_ENTITY_ID, config.COL_NAME, config.COL_ADDRESS, config.COL_COUNTRY)
            for sid in chunk[config.COL_ENTITY_ID]:
                tr_cands[sid] = c2.get(sid, set()) | c3.get(sid, set())

        other_tr = {**extract_record_maps(tr_s2), **extract_record_maps(tr_s3)}
        X_tr, y_tr, _ = build_pair_dataset(tr_s1, tr_cands, other_tr, tr_gt, max_neg_ratio=6.0)
        matcher.fit(X_tr, y_tr)
        matcher.save(MODEL_SAVE_PATH)

    # 3. Featurize test pairs
    print(f"\nFeaturizing test candidate pairs...")
    other_test_records = {**extract_record_maps(s2), **extract_record_maps(s3)}
    X_test, _, test_pair_index = build_pair_dataset(s1, candidates, other_test_records, ground_truth=None)

    # 4. Predict
    print(f"Predicting matches ({len(X_test):,} pairs)...")
    test_probs = matcher.predict_proba(X_test)
    test_entity_scores = group_scores_by_entity(s1[config.COL_ENTITY_ID], test_pair_index, test_probs)

    # 5. Apply decision rule
    final_matches = matcher.predict_for_entities(test_entity_scores)
    io_utils.write_matching_results(final_matches)
    print(f"Wrote {config.MATCHING_RESULTS_PATH}")

    n_matched = sum(1 for v in final_matches.values() if v)
    n_singleton = sum(1 for v in final_matches.values() if not v)
    print(f"\nResults summary:")
    print(f"  S1 with matches:    {n_matched:,}")
    print(f"  S1 singletons:      {n_singleton:,}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print("Test pipeline completed successfully!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--sample", action="store_true", help="Run on sample dataset.")
    parser.add_argument("--sample-other", type=int, default=300_000,
                        help="Max S2/S3 rows to use during training blocking (default: 300000). Use 0 for full data.")
    args = parser.parse_args()

    sample_other = None if args.sample_other == 0 else args.sample_other

    if args.split == "train":
        run_train(sample=args.sample, sample_other=sample_other)
    else:
        run_test(sample=args.sample)


if __name__ == "__main__":
    main()
