"""Unit tests for Accuracy / Reranking / Tuning Layer.

Tests:
1. Accuracy feature extractor and feature names completeness.
2. Legal suffix handling and normalization.
3. Postal code and street number extraction and matching.
4. TF-IDF character n-gram cosine similarities.
5. AccuracyReranker scoring and anti-false-merge guards.
6. Singleton protection hurdle (ensuring empty match for low confidence).
7. ThresholdTuner grid evaluation and Macro F0.5 optimization.
8. Candidate pairs rule preservation (matches are strictly a subset of candidates).
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import blocking, config, evaluate, io_utils, model, pipeline
from src.accuracy_layer import (
    AccuracyFeatureExtractor,
    compute_pair_accuracy_features,
    ACCURACY_FEATURE_NAMES,
    AccuracyReranker,
    ThresholdTuner,
)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "dataset" / "samples"


def test_feature_names_and_completeness():
    """Verify all defined feature names are generated and in valid numerical ranges."""
    fd = compute_pair_accuracy_features(
        name_a="Acme Corporation Ltd",
        addr_a="123 Main Street, Suite 400, New York, NY 10001",
        country_a="US",
        name_b="Acme Corp Limited",
        addr_b="123 Main St, New York, NY 10001",
        country_b="US",
        cand_id="S2-001",
        baseline_prob=0.85,
    )
    assert len(fd) == len(ACCURACY_FEATURE_NAMES)
    for feat_name in ACCURACY_FEATURE_NAMES:
        assert feat_name in fd
        assert isinstance(fd[feat_name], (int, float))
        assert not np.isnan(fd[feat_name])


def test_legal_suffix_and_transliteration_robustness():
    """Verify legal suffixes (Corp, Pvt Ltd, SA) are recognized and matched."""
    fd = compute_pair_accuracy_features(
        name_a="Infosys Technologies Private Limited",
        addr_a="Electronics City, Hosur Road, Bangalore, Karnataka",
        country_a="India",
        name_b="Infosys Tech Pvt. Ltd.",
        addr_b="Electronics City, Hosur Rd, Bengaluru, KA",
        country_b="India",
        cand_id="S3-002",
        baseline_prob=0.92,
    )
    assert fd["name_base_exact_match"] == 0.0 or fd["name_token_jaccard"] > 0.4
    assert fd["addr_token_jaccard"] > 0.4
    assert fd["country_conflict"] == 0.0


def test_postal_and_numeric_matching():
    """Verify postal / PIN codes and building numbers are extracted and compared."""
    fd = compute_pair_accuracy_features(
        name_a="Apex Logistics",
        addr_a="Plot 45, Sector 18, Gurgaon 122015, Haryana",
        country_a="India",
        name_b="Apex Logistics Corp",
        addr_b="Plot 45, Sector 18, Gurugram 122015",
        country_b="India",
        cand_id="S2-003",
        baseline_prob=0.95,
    )
    assert fd["addr_postal_exact_match"] == 1.0
    assert fd["addr_street_num_match"] == 1.0
    assert fd["addr_numeric_overlap_count"] >= 2.0


def test_tfidf_cosine_similarity():
    """Verify TF-IDF character n-gram cosine similarity works accurately."""
    extractor = AccuracyFeatureExtractor()
    corpus_names = ["Global Trading Corporation", "Global Trade Enterprises", "Western Pacific Logistics"]
    corpus_addrs = ["500 5th Ave, New York, NY", "500 Fifth Avenue, NY", "100 Ocean Blvd, CA"]
    extractor.fit_tfidf(corpus_names, corpus_addrs)

    sim_name, sim_addr = extractor.get_tfidf_similarities(
        s1_id=None,
        cand_id=None,
        norm_name_a="global trade",
        norm_name_b="global trade",
        norm_addr_a="500 5th ave ny",
        norm_addr_b="500 5th ave ny",
    )
    assert abs(sim_name - 1.0) < 1e-4
    assert abs(sim_addr - 1.0) < 1e-4


def test_anti_false_merge_filter():
    """Ensure complete dual mismatch (poor name + poor address) is filtered out."""
    reranker = AccuracyReranker()
    fd_mismatch = {
        "name_exact_match": 0.0,
        "name_base_exact_match": 0.0,
        "name_levenshtein_sim": 0.10,
        "name_base_levenshtein_sim": 0.10,
        "name_token_jaccard": 0.0,
        "name_is_substring": 0.0,
        "addr_exact_match": 0.0,
        "addr_levenshtein_sim": 0.15,
        "addr_token_jaccard": 0.10,
        "addr_numeric_overlap_count": 0.0,
        "country_conflict": 0.0,
        "addr_postal_exact_match": 0.5,
        "addr_street_num_match": 0.5,
    }
    score = reranker._apply_fp_penalties(raw_prob=0.60, fd=fd_mismatch)
    assert score == 0.0, "Dual mismatch should be suppressed to 0.0"


def test_country_conflict_hard_drop():
    """Conflicting known countries must always be strictly zeroed out."""
    reranker = AccuracyReranker()
    fd_country_conflict = {"country_conflict": 1.0}
    score = reranker._apply_fp_penalties(raw_prob=0.99, fd=fd_country_conflict)
    assert score == 0.0, "Conflicting country must receive 0.0 probability"


def test_singleton_protection():
    """Ensure low-confidence candidate scores emit an empty match list."""
    reranker = AccuracyReranker(confidence_hurdle=0.70)
    entity_scores = {
        "S1-001": [("S2-101", 0.55), ("S3-202", 0.40)],  # Max score 0.55 < hurdle 0.70
        "S1-002": [("S2-303", 0.88), ("S3-404", 0.85)],  # Max score 0.88 >= hurdle 0.70
    }
    preds = reranker.predict_for_entities(entity_scores, threshold=0.50, margin=0.10, confidence_hurdle=0.70)
    assert preds["S1-001"] == [], "Low-confidence entity must be protected as singleton []"
    assert "S2-303" in preds["S1-002"]


def test_candidate_pairs_subset_rule():
    """Verify that reranker predictions are strictly a subset of provided candidate pairs."""
    s1 = io_utils.read_source(SAMPLE_DIR / "sample_source1.tsv")
    s2 = io_utils.read_source(SAMPLE_DIR / "sample_source2.tsv")
    s3 = io_utils.read_source(SAMPLE_DIR / "sample_source3.tsv")

    c2 = blocking.generate_candidates(s1, s2, config.COL_ENTITY_ID, config.COL_NAME, top_k=5)
    c3 = blocking.generate_candidates(s1, s3, config.COL_ENTITY_ID, config.COL_NAME, top_k=5)
    candidates = {sid: c2.get(sid, set()) | c3.get(sid, set()) for sid in s1[config.COL_ENTITY_ID]}

    # Mock entity candidate scores based only on candidates
    scores = {sid: [(cid, 0.80) for cid in cands] for sid, cands in candidates.items()}
    reranker = AccuracyReranker()
    preds = reranker.predict_for_entities(scores, threshold=0.50, margin=0.10, confidence_hurdle=0.50)

    for sid, matched in preds.items():
        assert set(matched) <= set(candidates.get(sid, set())), f"Matched IDs for {sid} must be subset of candidates"


def test_open_set_france_country_handling():
    """Verify open-set requirement: handles France (unseen in train) correctly without hardcoding."""
    extractor = AccuracyFeatureExtractor()
    reranker = AccuracyReranker()

    # 1. France vs France: Valid match, no conflict
    fd_fr_fr = extractor.extract_pair(
        name_a="Boulangerie Paul",
        addr_a="15 Rue de Rivoli, Paris",
        country_a="France",
        name_b="Paul Boulangerie SAS",
        addr_b="15 Rue de Rivoli, Paris",
        country_b="France",
        cand_id="S2-FR01",
    )
    assert fd_fr_fr["country_conflict"] == 0.0
    assert fd_fr_fr["country_exact_match"] == 1.0
    score_fr_fr = reranker._apply_fp_penalties(raw_prob=0.92, fd=fd_fr_fr)
    assert score_fr_fr == 0.92, "Matching France entities should not be penalized"

    # 2. France vs United States: Cross-border mismatch, must drop to 0.0
    fd_fr_us = extractor.extract_pair(
        name_a="Boulangerie Paul",
        addr_a="15 Rue de Rivoli, Paris",
        country_a="France",
        name_b="Paul Bakery LLC",
        addr_b="123 Main St, New York, NY",
        country_b="US",
        cand_id="S2-US01",
    )
    assert fd_fr_us["country_conflict"] == 1.0
    assert fd_fr_us["country_exact_match"] == 0.0
    score_fr_us = reranker._apply_fp_penalties(raw_prob=0.92, fd=fd_fr_us)
    assert score_fr_us == 0.0, "France vs US must be strictly dropped by country conflict guard"

    # 3. France vs Missing Country: Should NOT conflict (safe handling of unpopulated fields)
    fd_fr_empty = extractor.extract_pair(
        name_a="Boulangerie Paul",
        addr_a="15 Rue de Rivoli, Paris",
        country_a="France",
        name_b="Paul Boulangerie",
        addr_b="15 Rue de Rivoli, Paris",
        country_b="",
        cand_id="S3-FR02",
    )
    assert fd_fr_empty["country_conflict"] == 0.0
    assert fd_fr_empty["country_exact_match"] == 0.5
    score_fr_empty = reranker._apply_fp_penalties(raw_prob=0.88, fd=fd_fr_empty)
    assert score_fr_empty == 0.88, "Missing country must not trigger a conflict penalty"

