"""Pairwise similarity features for (Source-1, candidate) pairs.

Owner: feat/model branch (shared with model.py).

Each row fed to the classifier is one candidate pair. Keep features generic
across countries/scripts -- don't one-hot `country` directly since test adds
an unseen value (France); if country is used, do it in a way that degrades
gracefully to an "unknown" bucket rather than crashing or silently zeroing
out (e.g. `country_match: bool`, not a fixed-vocabulary one-hot).

Candidate feature families to implement:
- Name: exact match (normalized), token Jaccard, character n-gram Jaccard,
  Levenshtein / rapidfuzz ratio, token-sort ratio (handles word-order swaps).
- Address: same family of string similarities on normalized address; also
  consider partial-token overlap since addresses are often truncated.
- Structural: length difference of name/address, whether one name is a
  substring/abbreviation of the other (e.g. acronym detection).
- country_match: same normalized country string or not (still informative
  even though it's an open set).
"""
from typing import Dict

from rapidfuzz import fuzz

from . import normalize


def pair_features(name_a: str, addr_a: str, country_a: str, name_b: str, addr_b: str, country_b: str) -> Dict[str, float]:
    norm_name_a, norm_name_b = normalize.normalize_name(name_a), normalize.normalize_name(name_b)
    norm_addr_a, norm_addr_b = normalize.normalize_address(addr_a), normalize.normalize_address(addr_b)

    tokens_a, tokens_b = normalize.name_tokens(name_a), normalize.name_tokens(name_b)
    union = tokens_a | tokens_b
    jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0

    return {
        "name_exact_match": float(norm_name_a == norm_name_b and bool(norm_name_a)),
        "name_token_jaccard": jaccard,
        "name_fuzzy_ratio": fuzz.ratio(norm_name_a, norm_name_b) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(norm_name_a, norm_name_b) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(norm_name_a, norm_name_b) / 100.0,
        "address_fuzzy_ratio": fuzz.ratio(norm_addr_a, norm_addr_b) / 100.0,
        "address_token_sort_ratio": fuzz.token_sort_ratio(norm_addr_a, norm_addr_b) / 100.0,
        "country_match": float((country_a or "").strip().casefold() == (country_b or "").strip().casefold()),
        "name_len_diff": abs(len(norm_name_a) - len(norm_name_b)),
    }
