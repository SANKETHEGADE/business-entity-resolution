"""End-to-end orchestration: load -> normalize -> block -> featurize -> score -> emit.

Usage:
    python -m src.pipeline --split train [--sample]
    python -m src.pipeline --split test
"""
import argparse
from typing import Dict, List, Set, Tuple
import pandas as pd
from tqdm import tqdm

from . import blocking, config, evaluate, features, io_utils, model


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


def build_pair_dataset(
    s1_df: pd.DataFrame,
    candidates: Dict[str, Set[str]],
    other_records: Dict[str, dict],
    ground_truth: Dict[str, Set[str]] = None,
) -> Tuple[List[dict], List[int], Dict[str, List[str]]]:
    """Featurize all (Source 1, Candidate) pairs."""
    s1_records = extract_record_maps(s1_df)

    feat_dicts: List[dict] = []
    labels: List[int] = []
    pair_index: Dict[str, List[str]] = {}

    for s1_id in s1_df[config.COL_ENTITY_ID]:
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
            feat_dicts.append(fd)
            if ground_truth is not None:
                labels.append(1 if cand_id in true_matches else 0)

    return feat_dicts, labels, pair_index


def run_train(sample: bool = False) -> None:
    if sample or not config.TRAIN_FILES["S1"].exists():
        print("Using sample dataset for training & validation...")
        s1_file = config.DATA_DIR / "samples" / "sample_source1.tsv"
        s2_file = config.DATA_DIR / "samples" / "sample_source2.tsv"
        s3_file = config.DATA_DIR / "samples" / "sample_source3.tsv"
        gt_file = config.DATA_DIR / "samples" / "sample_ground_truth.tsv"
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

    print(f"Data loaded: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}, GT={len(ground_truth)}")

    # 1. Train/Validation Split (80/20 by Source 1 entity_id hash to prevent leakage)
    val_mask = s1[config.COL_ENTITY_ID].apply(lambda x: hash(x) % 5 == 0)
    train_s1 = s1[~val_mask].copy()
    val_s1 = s1[val_mask].copy()

    val_gt = {sid: ground_truth.get(sid, set()) for sid in val_s1[config.COL_ENTITY_ID]}
    print(f"Split sizes: Train S1={len(train_s1)}, Val S1={len(val_s1)}")

    # 2. Blocking on Train S1
    print("Generating candidates for training set...")
    cand_s2 = blocking.generate_candidates(
        train_s1, s2, config.COL_ENTITY_ID, config.COL_NAME, config.COL_ADDRESS, config.COL_COUNTRY, top_k=25
    )
    cand_s3 = blocking.generate_candidates(
        train_s1, s3, config.COL_ENTITY_ID, config.COL_NAME, config.COL_ADDRESS, config.COL_COUNTRY, top_k=25
    )
    train_candidates = {
        s1_id: cand_s2.get(s1_id, set()) | cand_s3.get(s1_id, set())
        for s1_id in train_s1[config.COL_ENTITY_ID]
    }

    # 3. Blocking on Validation S1
    print("Generating candidates for validation set...")
    v_cand_s2 = blocking.generate_candidates(
        val_s1, s2, config.COL_ENTITY_ID, config.COL_NAME, config.COL_ADDRESS, config.COL_COUNTRY, top_k=25
    )
    v_cand_s3 = blocking.generate_candidates(
        val_s1, s3, config.COL_ENTITY_ID, config.COL_NAME, config.COL_ADDRESS, config.COL_COUNTRY, top_k=25
    )
    val_candidates = {
        s1_id: v_cand_s2.get(s1_id, set()) | v_cand_s3.get(s1_id, set())
        for s1_id in val_s1[config.COL_ENTITY_ID]
    }

    val_recall, val_avg_cands, _ = blocking.measure_recall(val_candidates, val_gt)
    print(f"Validation Blocking Recall Ceiling: {val_recall * 100:.2f}% (Avg candidates/S1: {val_avg_cands:.1f})")

    # 4. Feature Extraction
    print("Featurizing training candidate pairs...")
    other_records = {**extract_record_maps(s2), **extract_record_maps(s3)}
    X_train_dicts, y_train, _ = build_pair_dataset(train_s1, train_candidates, other_records, ground_truth)
    print(f"  Train pairs: {len(X_train_dicts)} (Positives: {sum(y_train)}, Negatives: {len(y_train) - sum(y_train)})")

    print("Featurizing validation candidate pairs...")
    X_val_dicts, y_val, val_pair_index = build_pair_dataset(val_s1, val_candidates, other_records, ground_truth)

    # 5. Train Model
    print("Training LightGBM Matcher...")
    matcher = model.MatchModel()
    matcher.fit(X_train_dicts, y_train)

    # 6. Score Validation Pairs
    print("Scoring validation pairs...")
    val_probs = matcher.predict_proba(X_val_dicts)

    # Group scores by S1 entity
    val_entity_scores: Dict[str, List[Tuple[str, float]]] = {}
    idx = 0
    for s1_id in val_s1[config.COL_ENTITY_ID]:
        cand_list = val_pair_index.get(s1_id, [])
        scored_pairs = []
        for cand_id in cand_list:
            scored_pairs.append((cand_id, float(val_probs[idx])))
            idx += 1
        val_entity_scores[s1_id] = scored_pairs

    # 7. Tune Threshold on Validation for Macro F0.5
    best_th, best_f05 = matcher.tune_threshold(val_entity_scores, val_gt)
    print(f"\n==========================================")
    print(f"Optimal Threshold (Macro F_0.5): {best_th:.2f}")
    print(f"Validation Macro F_0.5 Score:    {best_f05:.4f}")
    print(f"==========================================")

    val_preds = matcher.predict_for_entities(val_entity_scores, threshold=best_th)
    score_report = evaluate.macro_f0_5(val_preds, val_gt)
    print(f"Singletons in Val: {score_report['n_singletons']} / {score_report['n_entities']}")


def run_test(sample: bool = False) -> None:
    if sample or not config.TEST_FILES["S1"].exists():
        print("Using sample dataset as test set...")
        s1_file = config.DATA_DIR / "samples" / "sample_source1.tsv"
        s2_file = config.DATA_DIR / "samples" / "sample_source2.tsv"
        s3_file = config.DATA_DIR / "samples" / "sample_source3.tsv"
        train_s1_file = s1_file
        train_gt_file = config.DATA_DIR / "samples" / "sample_ground_truth.tsv"
    else:
        print("Loading test dataset...")
        s1_file = config.TEST_FILES["S1"]
        s2_file = config.TEST_FILES["S2"]
        s3_file = config.TEST_FILES["S3"]
        train_s1_file = config.TRAIN_FILES["S1"]
        train_gt_file = config.TRAIN_GROUND_TRUTH

    s1 = io_utils.read_source(s1_file)
    s2 = io_utils.read_source(s2_file)
    s3 = io_utils.read_source(s3_file)
    print(f"Test data loaded: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}")

    # 1. Blocking on Test
    print("Generating candidates for test set...")
    cand_s2 = blocking.generate_candidates(
        s1, s2, config.COL_ENTITY_ID, config.COL_NAME, config.COL_ADDRESS, config.COL_COUNTRY, top_k=25
    )
    cand_s3 = blocking.generate_candidates(
        s1, s3, config.COL_ENTITY_ID, config.COL_NAME, config.COL_ADDRESS, config.COL_COUNTRY, top_k=25
    )
    candidates = {
        s1_id: cand_s2.get(s1_id, set()) | cand_s3.get(s1_id, set())
        for s1_id in s1[config.COL_ENTITY_ID]
    }

    # Write candidate pairs TSV (mandatory evaluation requirement)
    io_utils.write_candidate_pairs(candidates)
    print(f"Wrote {config.CANDIDATE_PAIRS_PATH}")

    # 2. Train on full training data
    print("Training production model on training data...")
    tr_s1 = io_utils.read_source(train_s1_file)
    tr_s2 = io_utils.read_source(config.TRAIN_FILES["S2"] if config.TRAIN_FILES["S2"].exists() else s2_file)
    tr_s3 = io_utils.read_source(config.TRAIN_FILES["S3"] if config.TRAIN_FILES["S3"].exists() else s3_file)
    tr_gt = io_utils.ground_truth_to_sets(io_utils.read_ground_truth(train_gt_file))

    tr_c2 = blocking.generate_candidates(tr_s1, tr_s2, config.COL_ENTITY_ID, config.COL_NAME, config.COL_ADDRESS, config.COL_COUNTRY, top_k=25)
    tr_c3 = blocking.generate_candidates(tr_s1, tr_s3, config.COL_ENTITY_ID, config.COL_NAME, config.COL_ADDRESS, config.COL_COUNTRY, top_k=25)
    tr_cands = {sid: tr_c2.get(sid, set()) | tr_c3.get(sid, set()) for sid in tr_s1[config.COL_ENTITY_ID]}

    other_train_records = {**extract_record_maps(tr_s2), **extract_record_maps(tr_s3)}
    X_train, y_train, _ = build_pair_dataset(tr_s1, tr_cands, other_train_records, tr_gt)

    matcher = model.MatchModel()
    matcher.fit(X_train, y_train)

    # 3. Featurize & Predict on Test
    print("Featurizing test candidate pairs...")
    other_test_records = {**extract_record_maps(s2), **extract_record_maps(s3)}
    X_test, _, test_pair_index = build_pair_dataset(s1, candidates, other_test_records, ground_truth=None)

    print("Predicting matches for test set...")
    test_probs = matcher.predict_proba(X_test)

    test_entity_scores: Dict[str, List[Tuple[str, float]]] = {}
    idx = 0
    for s1_id in s1[config.COL_ENTITY_ID]:
        cand_list = test_pair_index.get(s1_id, [])
        scored_pairs = []
        for cand_id in cand_list:
            scored_pairs.append((cand_id, float(test_probs[idx])))
            idx += 1
        test_entity_scores[s1_id] = scored_pairs

    # Apply precision-heavy thresholding
    final_matches = matcher.predict_for_entities(test_entity_scores, threshold=0.70)
    io_utils.write_matching_results(final_matches)
    print(f"Wrote {config.MATCHING_RESULTS_PATH}")
    print("Test pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--sample", action="store_true", help="Run on sample dataset.")
    args = parser.parse_args()

    if args.split == "train":
        run_train(sample=args.sample)
    else:
        run_test(sample=args.sample)


if __name__ == "__main__":
    main()
