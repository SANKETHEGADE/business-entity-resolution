"""Shared paths and constants for the entity-resolution pipeline.

Keep every hardcoded path in one place so `pipeline.py` and notebooks agree
on where things live.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = ROOT / "output"

TRAIN_FILES = {
    "S1": TRAIN_DIR / "train_source1.tsv",
    "S2": TRAIN_DIR / "train_source2.tsv",
    "S3": TRAIN_DIR / "train_source3.tsv",
}
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_FILES = {
    "S1": TEST_DIR / "test_source1.tsv",
    "S2": TEST_DIR / "test_source2.tsv",
    "S3": TEST_DIR / "test_source3.tsv",
}

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Column names, kept in one place so a typo doesn't silently break a join.
COL_ENTITY_ID = "entity_id"
COL_NAME = "business_name"
COL_ADDRESS = "business_address"
COL_COUNTRY = "country"

COL_S1_ID = "source1_entity_id"
COL_MATCHED_IDS = "matched_entity_ids"
COL_CANDIDATE_IDS = "candidate_entity_ids"

RANDOM_SEED = 42

# NOTE: country is an OPEN SET (test adds France, unseen in train).
# Never hardcode a {US, India} allowlist anywhere in the pipeline -- if you
# need country-specific normalization rules, dispatch on the string value
# and fall back to a generic path for anything not explicitly handled.
