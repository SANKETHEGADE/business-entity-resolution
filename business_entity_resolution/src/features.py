"""Pairwise similarity features for candidate pairs.

Owner: feat/features.
Computes 24 dense tabular features per candidate pair:
- Name Match Signals: Exact match, RapidFuzz ratios (ratio, partial, token_sort, token_set),
  Jaro-Winkler similarity, length difference ratio, token Jaccard, char-3 Jaccard,
  bigram Jaccard, abbreviation match, first-token exact match.
- Address Signals: Numeric token Jaccard, postal Jaccard, token sort ratio, fuzzy ratio,
  common street number match boolean, address city/locality match.
- Structural & Missingness Indicators: Missing name flag, missing address flag,
  source indicator (S2=1, S3=0), cross-country match status.
- Optional frozen all-MiniLM-L6-v2 (Apache-2.0) embedding cosine similarity for name & address.
"""
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from . import normalize

FEATURE_ORDER: List[str] = [
    # --- Name Match Signals ---
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
    "name_bigram_jaccard",
    "name_abbrev_match",
    "name_first_token_exact",
    # --- Address Signals ---
    "addr_numeric_jaccard",
    "addr_postal_jaccard",
    "addr_token_sort_ratio",
    "addr_fuzzy_ratio",
    "common_street_number_match",
    "addr_city_match",
    # --- Structural & Missingness ---
    "name_missing",
    "addr_missing",
    "is_s2",
    # --- Country Compatibility ---
    "country_match",
    # --- Optional Semantic Neural Features ---
    "name_emb_cosine",
    "addr_emb_cosine",
]


def _char_ngrams(s: str, n: int = 3) -> Set[str]:
    clean = s.replace(" ", "")
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i: i + n] for i in range(len(clean) - n + 1)}


def _jaccard(s1: Set, s2: Set) -> float:
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def _first_token(name: str) -> str:
    parts = name.split()
    return parts[0] if parts else ""


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
    """Compute 24-feature dense vector between a Source 1 entity and a candidate."""

    # ------------------------------------------------------------------ #
    # 1. Normalized strings
    # ------------------------------------------------------------------ #
    norm_name_a = normalize.normalize_name(name_a, strip_suffixes=False)
    norm_name_b = normalize.normalize_name(name_b, strip_suffixes=False)
    base_name_a = normalize.normalize_name(name_a, strip_suffixes=True)
    base_name_b = normalize.normalize_name(name_b, strip_suffixes=True)

    norm_addr_a = normalize.normalize_address(addr_a)
    norm_addr_b = normalize.normalize_address(addr_b)

    # ------------------------------------------------------------------ #
    # 2. Missingness
    # ------------------------------------------------------------------ #
    name_missing = float(not norm_name_a or not norm_name_b)
    addr_missing = float(not norm_addr_a or not norm_addr_b)

    # ------------------------------------------------------------------ #
    # 3. Name features
    # ------------------------------------------------------------------ #
    len_a = len(norm_name_a)
    len_b = len(norm_name_b)
    max_len = max(len_a, len_b)
    len_diff_ratio = abs(len_a - len_b) / max_len if max_len > 0 else 0.0

    toks_a = normalize.name_tokens(name_a)
    toks_b = normalize.name_tokens(name_b)

    char3_a = _char_ngrams(norm_name_a, 3)
    char3_b = _char_ngrams(norm_name_b, 3)

    bigrams_a = normalize.name_bigrams(name_a)
    bigrams_b = normalize.name_bigrams(name_b)

    abbrev_a = normalize.name_abbreviation(name_a)
    abbrev_b = normalize.name_abbreviation(name_b)
    abbrev_match = float(
        bool(abbrev_a) and bool(abbrev_b) and (abbrev_a == abbrev_b)
    )

    first_a = _first_token(base_name_a)
    first_b = _first_token(base_name_b)
    first_token_exact = float(bool(first_a) and first_a == first_b)

    jw_sim = (
        JaroWinkler.similarity(norm_name_a, norm_name_b)
        if norm_name_a and norm_name_b
        else 0.0
    )

    # ------------------------------------------------------------------ #
    # 4. Address & numeric features
    # ------------------------------------------------------------------ #
    nums_a = normalize.extract_numeric_tokens(addr_a)
    nums_b = normalize.extract_numeric_tokens(addr_b)
    numeric_jaccard = _jaccard(nums_a, nums_b)

    postals_a = normalize.extract_postal_tokens(addr_a)
    postals_b = normalize.extract_postal_tokens(addr_b)
    postal_jaccard = _jaccard(postals_a, postals_b)

    # Common street number (any shared numeric token)
    common_street_number = float(bool(nums_a & nums_b))

    # City/locality match: look for a shared long address token (len >= 4)
    addr_toks_a = normalize.address_tokens(addr_a)
    addr_toks_b = normalize.address_tokens(addr_b)
    long_toks_a = {t for t in addr_toks_a if len(t) >= 4}
    long_toks_b = {t for t in addr_toks_b if len(t) >= 4}
    addr_city_match = float(bool(long_toks_a & long_toks_b))

    # ------------------------------------------------------------------ #
    # 5. Country compatibility
    # ------------------------------------------------------------------ #
    c_a = normalize.normalize_country(country_a)
    c_b = normalize.normalize_country(country_b)
    if not c_a or not c_b:
        country_match = 1.0
    else:
        country_match = 1.0 if c_a == c_b else 0.0

    # ------------------------------------------------------------------ #
    # 6. Source indicator
    # ------------------------------------------------------------------ #
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0

    # ------------------------------------------------------------------ #
    # 7. Assemble
    # ------------------------------------------------------------------ #
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
        "name_bigram_jaccard": _jaccard(bigrams_a, bigrams_b),
        "name_abbrev_match": abbrev_match,
        "name_first_token_exact": first_token_exact,
        "addr_numeric_jaccard": numeric_jaccard,
        "addr_postal_jaccard": postal_jaccard,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(norm_addr_a, norm_addr_b) / 100.0,
        "addr_fuzzy_ratio": fuzz.ratio(norm_addr_a, norm_addr_b) / 100.0,
        "common_street_number_match": common_street_number,
        "addr_city_match": addr_city_match,
        "name_missing": name_missing,
        "addr_missing": addr_missing,
        "is_s2": is_s2,
        "country_match": country_match,
        "name_emb_cosine": float(emb_name_sim),
        "addr_emb_cosine": float(emb_addr_sim),
    }
