"""Pairwise similarity features for candidate pairs.

Owner: feat/features.
Computes 18-22 dense tabular features per candidate pair matching the Top-100 blueprint:
- Name Match Signals: Exact match, RapidFuzz ratios (ratio, partial, token_sort, token_set),
  Jaro-Winkler similarity, length difference ratio.
- Address Signals: Numeric token Jaccard, token sort ratio, common street number match boolean.
- Structural & Missingness Indicators: Missing name flag, missing address flag, source indicator (S2=1, S3=0).
- Cross-country match status (1.0 if matching or unknown, 0.0 if conflicting).
- Optional frozen all-MiniLM-L6-v2 embedding cosine similarity.
"""
from typing import Dict, List, Optional, Set
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from . import normalize

FEATURE_ORDER: List[str] = [
    # Name Match Signals
    "name_exact_match",
    "name_base_exact_match",
    "name_fuzzy_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_len_diff_ratio",
    "name_token_jaccard",
    "name_char3_jaccard",
    # Address Signals
    "addr_numeric_jaccard",
    "addr_token_sort_ratio",
    "addr_fuzzy_ratio",
    "common_street_number_match",
    # Structural & Missingness Indicators
    "name_missing",
    "addr_missing",
    "is_s2",
    # Country Compatibility
    "country_match",
    # Optional Semantic Neural Features
    "name_emb_cosine",
    "addr_emb_cosine",
]


def _char_ngrams(s: str, n: int = 3) -> Set[str]:
    clean = s.replace(" ", "")
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def _jaccard(s1: Set[str], s2: Set[str]) -> float:
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def pair_features(
    name_a: str,
    addr_a: str,
    country_a: str,
    name_b: str,
    addr_b: str,
    country_b: str,
    cand_id: str,
    emb_name_sim: float = 0.0,
    emb_addr_sim: float = 0.0,
) -> Dict[str, float]:
    """Compute dense 18-22 tabular features between Source 1 entity and a candidate."""
    # 1. Normalized strings
    norm_name_a = normalize.normalize_name(name_a, strip_suffixes=False)
    norm_name_b = normalize.normalize_name(name_b, strip_suffixes=False)

    base_name_a = normalize.normalize_name(name_a, strip_suffixes=True)
    base_name_b = normalize.normalize_name(name_b, strip_suffixes=True)

    norm_addr_a = normalize.normalize_address(addr_a)
    norm_addr_b = normalize.normalize_address(addr_b)

    # 2. Missingness Indicators
    name_missing = float(not norm_name_a or not norm_name_b)
    addr_missing = float(not norm_addr_a or not norm_addr_b)

    # 3. Name Match Signals
    len_a = len(norm_name_a)
    len_b = len(norm_name_b)
    max_len = max(len_a, len_b)
    len_diff_ratio = abs(len_a - len_b) / max_len if max_len > 0 else 0.0

    toks_a = normalize.name_tokens(name_a)
    toks_b = normalize.name_tokens(name_b)

    char3_a = _char_ngrams(norm_name_a, 3)
    char3_b = _char_ngrams(norm_name_b, 3)

    jw_sim = JaroWinkler.similarity(norm_name_a, norm_name_b) if norm_name_a and norm_name_b else 0.0

    # 4. Address & Numeric Signals
    nums_a = normalize.extract_numeric_tokens(addr_a)
    nums_b = normalize.extract_numeric_tokens(addr_b)
    numeric_jaccard = _jaccard(nums_a, nums_b)

    # Check common street number match (e.g., first number in address)
    common_street_number = 0.0
    if nums_a and nums_b:
        first_num_a = next(iter(nums_a))
        first_num_b = next(iter(nums_b))
        if first_num_a == first_num_b:
            common_street_number = 1.0
        elif nums_a & nums_b:
            common_street_number = 1.0

    # 5. Cross-country match status (1.0 if matching or unknown, 0.0 if conflicting)
    c_a = (country_a or "").strip().casefold()
    c_b = (country_b or "").strip().casefold()
    if not c_a or not c_b:
        country_match = 1.0
    else:
        country_match = 1.0 if c_a == c_b else 0.0

    # 6. Source indicator: 1 if candidate is from S2, 0 if from S3
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0

    return {
        "name_exact_match": float(norm_name_a == norm_name_b and bool(norm_name_a)),
        "name_base_exact_match": float(base_name_a == base_name_b and bool(base_name_a)),
        "name_fuzzy_ratio": fuzz.ratio(norm_name_a, norm_name_b) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(norm_name_a, norm_name_b) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(norm_name_a, norm_name_b) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(norm_name_a, norm_name_b) / 100.0,
        "name_jaro_winkler": float(jw_sim),
        "name_len_diff_ratio": float(len_diff_ratio),
        "name_token_jaccard": _jaccard(toks_a, toks_b),
        "name_char3_jaccard": _jaccard(char3_a, char3_b),
        "addr_numeric_jaccard": numeric_jaccard,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(norm_addr_a, norm_addr_b) / 100.0,
        "addr_fuzzy_ratio": fuzz.ratio(norm_addr_a, norm_addr_b) / 100.0,
        "common_street_number_match": common_street_number,
        "name_missing": name_missing,
        "addr_missing": addr_missing,
        "is_s2": is_s2,
        "country_match": country_match,
        "name_emb_cosine": float(emb_name_sim),
        "addr_emb_cosine": float(emb_addr_sim),
    }
