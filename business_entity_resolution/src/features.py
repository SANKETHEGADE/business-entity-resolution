"""Pairwise similarity features for candidate pairs.

Owner: feat/features.
Computes comprehensive string, token, character-ngram, numeric/postal,
and optional sentence-embedding cosine similarities across (Source 1, Candidate).
"""
from typing import Dict, List, Optional, Set
import numpy as np
from rapidfuzz import fuzz

from . import normalize

FEATURE_ORDER: List[str] = [
    # Name features
    "name_exact_match",
    "name_base_exact_match",
    "name_fuzzy_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_token_jaccard",
    "name_char3_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "name_is_substring",
    "name_missing",
    # Address features
    "addr_fuzzy_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_token_jaccard",
    "addr_numeric_jaccard",
    "addr_numeric_common_count",
    "addr_missing",
    # Country & Metadata
    "country_match",
    "is_s2",
    "is_s3",
    # Embedding cosine features (frozen all-MiniLM-L6-v2)
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
    """Compute rich pairwise feature vector between Source 1 entity and a candidate."""
    # 1. Normalized variants
    norm_name_a = normalize.normalize_name(name_a, strip_suffixes=False)
    norm_name_b = normalize.normalize_name(name_b, strip_suffixes=False)

    base_name_a = normalize.normalize_name(name_a, strip_suffixes=True)
    base_name_b = normalize.normalize_name(name_b, strip_suffixes=True)

    norm_addr_a = normalize.normalize_address(addr_a)
    norm_addr_b = normalize.normalize_address(addr_b)

    # 2. Missing flags
    name_missing = float(not norm_name_a or not norm_name_b)
    addr_missing = float(not norm_addr_a or not norm_addr_b)

    # 3. Name features
    toks_a = normalize.name_tokens(name_a)
    toks_b = normalize.name_tokens(name_b)

    char3_a = _char_ngrams(norm_name_a, 3)
    char3_b = _char_ngrams(norm_name_b, 3)

    len_a = len(norm_name_a)
    len_b = len(norm_name_b)
    len_diff = abs(len_a - len_b)
    len_ratio = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 0.0

    is_substring = 0.0
    if len_a >= 3 and len_b >= 3:
        if norm_name_a in norm_name_b or norm_name_b in norm_name_a:
            is_substring = 1.0

    # 4. Address & Numeric features
    nums_a = normalize.extract_numeric_tokens(addr_a)
    nums_b = normalize.extract_numeric_tokens(addr_b)
    common_nums = nums_a & nums_b

    addr_toks_a = normalize.address_tokens(addr_a)
    addr_toks_b = normalize.address_tokens(addr_b)

    # 5. Country compatibility (open set: US, India, France, etc.)
    c_a = (country_a or "").strip().casefold()
    c_b = (country_b or "").strip().casefold()
    if not c_a or not c_b:
        country_match = 1.0  # unknown country is neutral
    else:
        country_match = 1.0 if c_a == c_b else 0.0

    # 6. Source identifier
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    return {
        "name_exact_match": float(norm_name_a == norm_name_b and bool(norm_name_a)),
        "name_base_exact_match": float(base_name_a == base_name_b and bool(base_name_a)),
        "name_fuzzy_ratio": fuzz.ratio(norm_name_a, norm_name_b) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(norm_name_a, norm_name_b) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(norm_name_a, norm_name_b) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(norm_name_a, norm_name_b) / 100.0,
        "name_token_jaccard": _jaccard(toks_a, toks_b),
        "name_char3_jaccard": _jaccard(char3_a, char3_b),
        "name_len_diff": float(len_diff),
        "name_len_ratio": float(len_ratio),
        "name_is_substring": is_substring,
        "name_missing": name_missing,
        "addr_fuzzy_ratio": fuzz.ratio(norm_addr_a, norm_addr_b) / 100.0,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(norm_addr_a, norm_addr_b) / 100.0,
        "addr_token_set_ratio": fuzz.token_set_ratio(norm_addr_a, norm_addr_b) / 100.0,
        "addr_token_jaccard": _jaccard(addr_toks_a, addr_toks_b),
        "addr_numeric_jaccard": _jaccard(nums_a, nums_b),
        "addr_numeric_common_count": float(len(common_nums)),
        "addr_missing": addr_missing,
        "country_match": country_match,
        "is_s2": is_s2,
        "is_s3": is_s3,
        "name_emb_cosine": float(emb_name_sim),
        "addr_emb_cosine": float(emb_addr_sim),
    }
