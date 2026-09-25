"""Fast unit tests -- no data files required except the checked-in sample/.

Run with: pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate, normalize, io_utils, blocking, config

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "dataset" / "samples"


def test_f0_5_worked_example():
    pred = {"S1-00001": ["S2-00047", "S2-00193", "S3-00812"]}
    truth = {"S1-00001": {"S2-00047", "S3-00812"}}
    result = evaluate.macro_f0_5(pred, truth)
    assert abs(result["macro_f0_5"] - 0.7143) < 1e-3


def test_f0_5_singleton_full_credit():
    assert evaluate.score_entity(set(), set()) == 1.0


def test_f0_5_singleton_false_merge_zero_credit():
    assert evaluate.score_entity({"S2-1"}, set()) == 0.0


def test_f0_5_missed_entity_zero_credit():
    assert evaluate.score_entity(set(), {"S2-1"}) == 0.0


def test_normalize_name_folds_legal_suffix():
    assert normalize.normalize_name("Acme Corporation") == normalize.normalize_name("Acme Corp")


def test_normalize_handles_empty_and_none():
    assert normalize.normalize_name("") == ""
    assert normalize.normalize_name(None) == ""
    assert normalize.normalize_address(None) == ""


def test_sample_dataset_round_trip():
    """The checked-in sample/ should load cleanly and be internally consistent
    (every ground-truth id actually exists in the sample source files)."""
    s1 = io_utils.read_source(SAMPLE_DIR / "sample_source1.tsv")
    s2 = io_utils.read_source(SAMPLE_DIR / "sample_source2.tsv")
    s3 = io_utils.read_source(SAMPLE_DIR / "sample_source3.tsv")
    gt = io_utils.ground_truth_to_sets(io_utils.read_ground_truth(SAMPLE_DIR / "sample_ground_truth.tsv"))

    all_s2_ids = set(s2[config.COL_ENTITY_ID])
    all_s3_ids = set(s3[config.COL_ENTITY_ID])
    all_s1_ids = set(s1[config.COL_ENTITY_ID])

    assert set(gt.keys()) <= all_s1_ids
    for matched in gt.values():
        for eid in matched:
            assert eid in all_s2_ids or eid in all_s3_ids


def test_blocking_recall_on_sample_is_reasonable():
    s1 = io_utils.read_source(SAMPLE_DIR / "sample_source1.tsv")
    s2 = io_utils.read_source(SAMPLE_DIR / "sample_source2.tsv")
    s3 = io_utils.read_source(SAMPLE_DIR / "sample_source3.tsv")
    gt = io_utils.ground_truth_to_sets(io_utils.read_ground_truth(SAMPLE_DIR / "sample_ground_truth.tsv"))

    cand_s2 = blocking.generate_candidates(s1, s2, config.COL_ENTITY_ID, config.COL_NAME)
    cand_s3 = blocking.generate_candidates(s1, s3, config.COL_ENTITY_ID, config.COL_NAME)
    candidates = {sid: cand_s2.get(sid, set()) | cand_s3.get(sid, set()) for sid in s1[config.COL_ENTITY_ID]}

    recall = blocking.measure_recall(candidates, gt)
    # Regression floor for the naive token-index baseline -- if this drops,
    # something in normalize/blocking broke, not a "go tune it" signal.
    assert recall > 0.7
