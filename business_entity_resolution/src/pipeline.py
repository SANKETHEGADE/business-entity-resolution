"""End-to-end orchestration: load -> normalize -> block -> featurize -> score -> emit.

Owner: feat/eval branch (integration point for everyone else's modules).

This is deliberately thin -- it should just call into the other modules in
order and handle train-vs-test branching, so each area's logic stays testable
in isolation. Fill in the TODOs as blocking/features/model land.

Usage:
    python3 -m src.pipeline --split train   # trains + reports held-out F_0.5
    python3 -m src.pipeline --split test    # writes output/*.tsv for submission
"""
import argparse

from . import blocking, config, evaluate, io_utils


def run_train(sample: bool = False) -> None:
    print("Loading train sources...")
    s1 = io_utils.read_source(config.TRAIN_FILES["S1"])
    s2 = io_utils.read_source(config.TRAIN_FILES["S2"])
    s3 = io_utils.read_source(config.TRAIN_FILES["S3"])
    gt_df = io_utils.read_ground_truth(config.TRAIN_GROUND_TRUTH)
    ground_truth = io_utils.ground_truth_to_sets(gt_df)
    print(f"  S1={len(s1)}  S2={len(s2)}  S3={len(s3)}  ground_truth={len(ground_truth)}")

    if sample:
        s1 = s1.head(2000)

    # TODO: proper train/val split (e.g. by hashing entity_id) before tuning
    # anything, so the held-out score is a trustworthy leaderboard proxy.

    print("Generating candidates (name-token blocking, S1 x S2)...")
    cand_s2 = blocking.generate_candidates(s1, s2, config.COL_ENTITY_ID, config.COL_NAME)
    print("Generating candidates (name-token blocking, S1 x S3)...")
    cand_s3 = blocking.generate_candidates(s1, s3, config.COL_ENTITY_ID, config.COL_NAME)

    candidates = {
        s1_id: cand_s2.get(s1_id, set()) | cand_s3.get(s1_id, set())
        for s1_id in s1[config.COL_ENTITY_ID]
    }

    recall = blocking.measure_recall(candidates, ground_truth)
    print(f"Blocking recall ceiling on train: {recall:.4f}")

    # TODO: features.pair_features + model.MatchModel go here once the
    # candidate set's recall/size trade-off looks reasonable. For now, report
    # blocking recall only -- that's the number to watch first.


def run_test() -> None:
    print("Loading test sources...")
    s1 = io_utils.read_source(config.TEST_FILES["S1"])
    s2 = io_utils.read_source(config.TEST_FILES["S2"])
    s3 = io_utils.read_source(config.TEST_FILES["S3"])
    print(f"  S1={len(s1)}  S2={len(s2)}  S3={len(s3)}")

    print("Generating candidates (name-token blocking, S1 x S2)...")
    cand_s2 = blocking.generate_candidates(s1, s2, config.COL_ENTITY_ID, config.COL_NAME)
    print("Generating candidates (name-token blocking, S1 x S3)...")
    cand_s3 = blocking.generate_candidates(s1, s3, config.COL_ENTITY_ID, config.COL_NAME)

    candidates = {
        s1_id: cand_s2.get(s1_id, set()) | cand_s3.get(s1_id, set())
        for s1_id in s1[config.COL_ENTITY_ID]
    }
    io_utils.write_candidate_pairs(candidates)
    print(f"Wrote {config.CANDIDATE_PAIRS_PATH}")

    # TODO: score candidates with the trained model and write matching_results.tsv.
    # Placeholder below writes an all-singleton submission so the format is
    # valid end-to-end even before the model exists.
    empty_matches = {s1_id: [] for s1_id in s1[config.COL_ENTITY_ID]}
    io_utils.write_matching_results(empty_matches)
    print(f"Wrote {config.MATCHING_RESULTS_PATH} (placeholder: all singletons)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--sample", action="store_true", help="Run on a small slice of train for a fast smoke test.")
    args = parser.parse_args()

    if args.split == "train":
        run_train(sample=args.sample)
    else:
        run_test()


if __name__ == "__main__":
    main()
