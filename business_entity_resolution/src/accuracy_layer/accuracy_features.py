"""Enhanced pairwise matching features for Source 1 <-> Source 2/3 candidate pairs.

Owner: Accuracy / Reranking / Tuning module.
Implements:
1. Enhanced Name Features:
   - normalized exact name match, base exact match
   - token overlap count & overlap coefficient
   - Jaccard similarity & character-level n-gram (3-gram, 4-gram) Jaccard
   - Levenshtein normalized edit similarity & base Levenshtein similarity
   - common prefix ratio
   - TF-IDF cosine similarity (char 3-4 n-grams & word tokens)
   - legal entity suffix handling (Pvt, Ltd, Corp, Inc, LLC, etc.)
   - suffix match / conflict indicators
2. Enhanced Address Features:
   - normalized address similarity & Levenshtein similarity
   - token overlap count & overlap coefficient
   - Jaccard similarity & character n-gram similarities
   - TF-IDF cosine similarity
   - PIN / postal code exact match, overlap & presence
   - street number exact match & numeric Jaccard
   - street keyword overlap & abbreviation handling
3. Other & Interaction Features:
   - country exact-match & conflict indicators
   - combined name + address similarities (arithmetic, geometric, min, product)
   - name/address similarity difference
   - missing-field indicators (either_addr_missing, both_addr_present, etc.)
   - baseline model probability and logit features

Strictly self-contained: NO external APIs, lookups, or external datasets.
"""
from typing import Dict, Iterable, List, Optional, Set, Tuple
import math
import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein as rf_lev
from sklearn.feature_extraction.text import TfidfVectorizer

from .. import normalize


ACCURACY_FEATURE_NAMES: List[str] = [
    # Baseline model signals
    "baseline_prob",
    "baseline_logit",
    # Name Features
    "name_exact_match",
    "name_base_exact_match",
    "name_levenshtein_sim",
    "name_base_levenshtein_sim",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_token_overlap_ratio",
    "name_char3_jaccard",
    "name_char4_jaccard",
    "name_prefix_sim",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_fuzzy_ratio",
    "name_partial_ratio",
    "name_is_substring",
    "name_len_diff",
    "name_len_ratio",
    "name_tfidf_cosine",
    "name_legal_suffix_match",
    "name_has_suffix_a",
    "name_has_suffix_b",
    "name_missing",
    # Address Features
    "addr_exact_match",
    "addr_levenshtein_sim",
    "addr_token_jaccard",
    "addr_token_overlap_count",
    "addr_token_overlap_ratio",
    "addr_char3_jaccard",
    "addr_char4_jaccard",
    "addr_tfidf_cosine",
    "addr_postal_exact_match",
    "addr_postal_jaccard",
    "addr_postal_overlap_count",
    "addr_has_postal_a",
    "addr_has_postal_b",
    "addr_street_num_match",
    "addr_numeric_jaccard",
    "addr_numeric_overlap_count",
    "addr_fuzzy_ratio",
    "addr_token_sort_ratio",
    "addr_missing",
    # Country & Source Metadata
    "country_exact_match",
    "country_conflict",
    "is_s2",
    "is_s3",
    # Interaction Features
    "name_addr_mean_sim",
    "name_addr_geom_sim",
    "name_addr_min_sim",
    "name_addr_sim_diff",
    "name_addr_prod_sim",
    # Missingness Indicators
    "name_a_missing",
    "name_b_missing",
    "addr_a_missing",
    "addr_b_missing",
    "either_addr_missing",
    "both_addr_present",
]


def _char_ngrams(s: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from string ignoring whitespace."""
    clean = s.replace(" ", "")
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def _jaccard(s1: Set[str], s2: Set[str]) -> float:
    """Jaccard similarity coefficient: |A n B| / |A u B|."""
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def _overlap_coefficient(s1: Set[str], s2: Set[str]) -> float:
    """Overlap coefficient (Szymkiewicz-Simpson): |A n B| / min(|A|, |B|)."""
    min_len = min(len(s1), len(s2))
    return len(s1 & s2) / min_len if min_len > 0 else 0.0


def _common_prefix_ratio(s1: str, s2: str) -> float:
    """Ratio of longest common prefix length to min string length."""
    if not s1 or not s2:
        return 0.0
    min_l = min(len(s1), len(s2))
    match_len = 0
    while match_len < min_l and s1[match_len] == s2[match_len]:
        match_len += 1
    return match_len / min_l if min_l > 0 else 0.0


def _extract_legal_suffix(cleaned_name: str) -> Optional[str]:
    """Extract known legal suffix from cleaned name."""
    if not cleaned_name:
        return None
    tokens = cleaned_name.split()
    for t in reversed(tokens):
        if t in normalize.LEGAL_SUFFIXES:
            return normalize.LEGAL_SUFFIX_MAP.get(t, t)
    return None


def _extract_street_number(addr: str) -> Optional[str]:
    """Extract the primary street/building number from address."""
    if not addr:
        return None
    match = re.search(r"\b(\d+)\b", addr)
    return match.group(1) if match else None


class AccuracyFeatureExtractor:
    """Feature extractor for accuracy improvement and reranking.

    Supports pre-fitted TF-IDF models over names and addresses for sub-millisecond
    vector dot products across candidate pairs.
    """

    def __init__(self):
        self.name_vectorizer: Optional[TfidfVectorizer] = None
        self.addr_vectorizer: Optional[TfidfVectorizer] = None
        self._name_vec_cache: Dict[str, np.ndarray] = {}
        self._addr_vec_cache: Dict[str, np.ndarray] = {}

    def fit_tfidf(self, all_names: Iterable[str], all_addresses: Iterable[str]) -> "AccuracyFeatureExtractor":
        """Fit character n-gram TF-IDF vectorizers on corpus names and addresses."""
        cleaned_names = [normalize.normalize_name(n, strip_suffixes=True) or "empty" for n in all_names]
        cleaned_addrs = [normalize.normalize_address(a) or "empty" for a in all_addresses]

        self.name_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )
        self.name_vectorizer.fit(cleaned_names)

        self.addr_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )
        self.addr_vectorizer.fit(cleaned_addrs)
        return self

    def precompute_vectors(self, record_map: Dict[str, dict]) -> None:
        """Precompute and cache TF-IDF vectors for all records by entity_id."""
        if self.name_vectorizer is None or self.addr_vectorizer is None:
            return

        entity_ids = list(record_map.keys())
        names = [normalize.normalize_name(record_map[eid].get("name", ""), strip_suffixes=True) or "empty" for eid in entity_ids]
        addrs = [normalize.normalize_address(record_map[eid].get("address", "")) or "empty" for eid in entity_ids]

        name_matrix = self.name_vectorizer.transform(names)
        addr_matrix = self.addr_vectorizer.transform(addrs)

        for i, eid in enumerate(entity_ids):
            self._name_vec_cache[eid] = name_matrix[i]
            self._addr_vec_cache[eid] = addr_matrix[i]

    def get_tfidf_similarities(
        self,
        s1_id: Optional[str],
        cand_id: Optional[str],
        norm_name_a: str,
        norm_name_b: str,
        norm_addr_a: str,
        norm_addr_b: str,
    ) -> Tuple[float, float]:
        """Compute TF-IDF cosine similarity for name and address pairs."""
        name_sim = 0.0
        addr_sim = 0.0

        if (
            s1_id is not None
            and cand_id is not None
            and s1_id in self._name_vec_cache
            and cand_id in self._name_vec_cache
        ):
            v1 = self._name_vec_cache[s1_id]
            v2 = self._name_vec_cache[cand_id]
            name_sim = float(v1.dot(v2.T).toarray()[0, 0])
        elif self.name_vectorizer is not None and norm_name_a and norm_name_b:
            m = self.name_vectorizer.transform([norm_name_a, norm_name_b])
            name_sim = float((m[0] * m[1].T).toarray()[0, 0])

        if (
            s1_id is not None
            and cand_id is not None
            and s1_id in self._addr_vec_cache
            and cand_id in self._addr_vec_cache
        ):
            a1 = self._addr_vec_cache[s1_id]
            a2 = self._addr_vec_cache[cand_id]
            addr_sim = float(a1.dot(a2.T).toarray()[0, 0])
        elif self.addr_vectorizer is not None and norm_addr_a and norm_addr_b:
            ma = self.addr_vectorizer.transform([norm_addr_a, norm_addr_b])
            addr_sim = float((ma[0] * ma[1].T).toarray()[0, 0])

        return max(0.0, min(1.0, name_sim)), max(0.0, min(1.0, addr_sim))

    def extract_pair(
        self,
        name_a: str,
        addr_a: str,
        country_a: str,
        name_b: str,
        addr_b: str,
        country_b: str,
        cand_id: str,
        s1_id: Optional[str] = None,
        baseline_prob: float = 0.5,
    ) -> Dict[str, float]:
        """Compute complete enhanced feature dictionary for a candidate pair."""
        return compute_pair_accuracy_features(
            name_a=name_a,
            addr_a=addr_a,
            country_a=country_a,
            name_b=name_b,
            addr_b=addr_b,
            country_b=country_b,
            cand_id=cand_id,
            s1_id=s1_id,
            baseline_prob=baseline_prob,
            extractor=self,
        )


def compute_pair_accuracy_features(
    name_a: str,
    addr_a: str,
    country_a: str,
    name_b: str,
    addr_b: str,
    country_b: str,
    cand_id: str,
    s1_id: Optional[str] = None,
    baseline_prob: float = 0.5,
    extractor: Optional[AccuracyFeatureExtractor] = None,
) -> Dict[str, float]:
    """Compute rich enhanced accuracy features for a candidate pair."""
    # 1. Normalization & Variations
    norm_name_a = normalize.normalize_name(name_a, strip_suffixes=False)
    norm_name_b = normalize.normalize_name(name_b, strip_suffixes=False)

    base_name_a = normalize.normalize_name(name_a, strip_suffixes=True)
    base_name_b = normalize.normalize_name(name_b, strip_suffixes=True)

    norm_addr_a = normalize.normalize_address(addr_a)
    norm_addr_b = normalize.normalize_address(addr_b)

    # 2. Missing indicators
    name_a_missing = 1.0 if not norm_name_a else 0.0
    name_b_missing = 1.0 if not norm_name_b else 0.0
    addr_a_missing = 1.0 if not norm_addr_a else 0.0
    addr_b_missing = 1.0 if not norm_addr_b else 0.0
    name_missing = 1.0 if (name_a_missing or name_b_missing) else 0.0
    addr_missing = 1.0 if (addr_a_missing or addr_b_missing) else 0.0
    either_addr_missing = addr_missing
    both_addr_present = 1.0 if (not addr_a_missing and not addr_b_missing) else 0.0

    # 3. Name Similarity Features
    name_exact = 1.0 if (norm_name_a == norm_name_b and bool(norm_name_a)) else 0.0
    base_name_exact = 1.0 if (base_name_a == base_name_b and bool(base_name_a)) else 0.0

    # Levenshtein edit distance similarity
    name_lev_sim = rf_lev.normalized_similarity(norm_name_a, norm_name_b) if (norm_name_a and norm_name_b) else 0.0
    base_lev_sim = rf_lev.normalized_similarity(base_name_a, base_name_b) if (base_name_a and base_name_b) else 0.0

    # Tokens & N-grams
    toks_a = normalize.name_tokens(name_a)
    toks_b = normalize.name_tokens(name_b)
    name_tok_jaccard = _jaccard(toks_a, toks_b)
    name_tok_overlap_cnt = float(len(toks_a & toks_b))
    name_tok_overlap_ratio = _overlap_coefficient(toks_a, toks_b)

    char3_a = _char_ngrams(norm_name_a, 3)
    char3_b = _char_ngrams(norm_name_b, 3)
    name_char3_jaccard = _jaccard(char3_a, char3_b)

    char4_a = _char_ngrams(norm_name_a, 4)
    char4_b = _char_ngrams(norm_name_b, 4)
    name_char4_jaccard = _jaccard(char4_a, char4_b)

    name_prefix_sim = _common_prefix_ratio(base_name_a, base_name_b)

    # String lengths
    len_a = len(norm_name_a)
    len_b = len(norm_name_b)
    len_diff = float(abs(len_a - len_b))
    len_ratio = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 0.0

    is_substring = 0.0
    if len_a >= 3 and len_b >= 3:
        if norm_name_a in norm_name_b or norm_name_b in norm_name_a:
            is_substring = 1.0

    # RapidFuzz ratios
    name_sort_ratio = fuzz.token_sort_ratio(norm_name_a, norm_name_b) / 100.0 if (norm_name_a and norm_name_b) else 0.0
    name_set_ratio = fuzz.token_set_ratio(norm_name_a, norm_name_b) / 100.0 if (norm_name_a and norm_name_b) else 0.0
    name_fuzzy_ratio = fuzz.ratio(norm_name_a, norm_name_b) / 100.0 if (norm_name_a and norm_name_b) else 0.0
    name_partial_ratio = fuzz.partial_ratio(norm_name_a, norm_name_b) / 100.0 if (norm_name_a and norm_name_b) else 0.0

    # Legal Suffix features
    suffix_a = _extract_legal_suffix(norm_name_a)
    suffix_b = _extract_legal_suffix(norm_name_b)
    has_suffix_a = 1.0 if suffix_a else 0.0
    has_suffix_b = 1.0 if suffix_b else 0.0
    if suffix_a and suffix_b:
        suffix_match = 1.0 if suffix_a == suffix_b else 0.0
    elif not suffix_a and not suffix_b:
        suffix_match = 0.5  # neutral
    else:
        suffix_match = 0.5  # one has suffix, other doesn't (neutral)

    # 4. Address Similarity Features
    addr_exact = 1.0 if (norm_addr_a == norm_addr_b and bool(norm_addr_a)) else 0.0
    addr_lev_sim = rf_lev.normalized_similarity(norm_addr_a, norm_addr_b) if (norm_addr_a and norm_addr_b) else 0.0

    addr_toks_a = normalize.address_tokens(addr_a)
    addr_toks_b = normalize.address_tokens(addr_b)
    addr_tok_jaccard = _jaccard(addr_toks_a, addr_toks_b)
    addr_tok_overlap_cnt = float(len(addr_toks_a & addr_toks_b))
    addr_tok_overlap_ratio = _overlap_coefficient(addr_toks_a, addr_toks_b)

    addr_char3_a = _char_ngrams(norm_addr_a, 3)
    addr_char3_b = _char_ngrams(norm_addr_b, 3)
    addr_char3_jaccard = _jaccard(addr_char3_a, addr_char3_b)

    addr_char4_a = _char_ngrams(norm_addr_a, 4)
    addr_char4_b = _char_ngrams(norm_addr_b, 4)
    addr_char4_jaccard = _jaccard(addr_char4_a, addr_char4_b)

    addr_fuzzy = fuzz.ratio(norm_addr_a, norm_addr_b) / 100.0 if (norm_addr_a and norm_addr_b) else 0.0
    addr_sort_ratio = fuzz.token_sort_ratio(norm_addr_a, norm_addr_b) / 100.0 if (norm_addr_a and norm_addr_b) else 0.0

    # Postal & Numeric components
    postals_a = normalize.extract_postal_tokens(addr_a)
    postals_b = normalize.extract_postal_tokens(addr_b)
    has_postal_a = 1.0 if postals_a else 0.0
    has_postal_b = 1.0 if postals_b else 0.0
    common_postals = postals_a & postals_b
    if postals_a and postals_b:
        postal_exact = 1.0 if len(common_postals) > 0 else 0.0
    else:
        postal_exact = 0.5  # neutral if unobserved
    postal_jaccard = _jaccard(postals_a, postals_b)
    postal_overlap_cnt = float(len(common_postals))

    nums_a = normalize.extract_numeric_tokens(addr_a)
    nums_b = normalize.extract_numeric_tokens(addr_b)
    common_nums = nums_a & nums_b
    numeric_jaccard = _jaccard(nums_a, nums_b)
    numeric_overlap_cnt = float(len(common_nums))

    street_num_a = _extract_street_number(addr_a)
    street_num_b = _extract_street_number(addr_b)
    if street_num_a and street_num_b:
        street_num_match = 1.0 if street_num_a == street_num_b else 0.0
    else:
        street_num_match = 0.5

    # 5. TF-IDF Cosine Similarities
    if extractor is not None:
        name_tfidf, addr_tfidf = extractor.get_tfidf_similarities(
            s1_id, cand_id, base_name_a, base_name_b, norm_addr_a, norm_addr_b
        )
    else:
        name_tfidf = name_tok_jaccard
        addr_tfidf = addr_tok_jaccard

    # 6. Country & Metadata
    c_a = (country_a or "").strip().casefold()
    c_b = (country_b or "").strip().casefold()
    if c_a and c_b:
        country_exact = 1.0 if c_a == c_b else 0.0
        country_conflict = 0.0 if c_a == c_b else 1.0
    else:
        country_exact = 0.5
        country_conflict = 0.0

    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    # 7. Interaction & Combined Similarities
    # Effective address sim: if either address is missing, don't penalize!
    eff_addr_sim = addr_lev_sim if both_addr_present else name_lev_sim

    mean_sim = 0.6 * name_lev_sim + 0.4 * eff_addr_sim
    geom_sim = math.sqrt(max(0.0, name_lev_sim) * max(0.0, eff_addr_sim))
    min_sim = min(name_lev_sim, eff_addr_sim)
    sim_diff = abs(name_lev_sim - eff_addr_sim)
    prod_sim = name_lev_sim * eff_addr_sim

    # 8. Baseline Probability & Logit
    p_clamped = max(1e-5, min(1.0 - 1e-5, float(baseline_prob)))
    baseline_logit = math.log(p_clamped / (1.0 - p_clamped))
    baseline_logit = max(-10.0, min(10.0, baseline_logit))

    return {
        "baseline_prob": float(baseline_prob),
        "baseline_logit": float(baseline_logit),
        "name_exact_match": name_exact,
        "name_base_exact_match": base_name_exact,
        "name_levenshtein_sim": float(name_lev_sim),
        "name_base_levenshtein_sim": float(base_lev_sim),
        "name_token_jaccard": float(name_tok_jaccard),
        "name_token_overlap_count": name_tok_overlap_cnt,
        "name_token_overlap_ratio": float(name_tok_overlap_ratio),
        "name_char3_jaccard": float(name_char3_jaccard),
        "name_char4_jaccard": float(name_char4_jaccard),
        "name_prefix_sim": float(name_prefix_sim),
        "name_token_sort_ratio": float(name_sort_ratio),
        "name_token_set_ratio": float(name_set_ratio),
        "name_fuzzy_ratio": float(name_fuzzy_ratio),
        "name_partial_ratio": float(name_partial_ratio),
        "name_is_substring": is_substring,
        "name_len_diff": len_diff,
        "name_len_ratio": float(len_ratio),
        "name_tfidf_cosine": float(name_tfidf),
        "name_legal_suffix_match": float(suffix_match),
        "name_has_suffix_a": has_suffix_a,
        "name_has_suffix_b": has_suffix_b,
        "name_missing": name_missing,
        "addr_exact_match": addr_exact,
        "addr_levenshtein_sim": float(addr_lev_sim),
        "addr_token_jaccard": float(addr_tok_jaccard),
        "addr_token_overlap_count": addr_tok_overlap_cnt,
        "addr_token_overlap_ratio": float(addr_tok_overlap_ratio),
        "addr_char3_jaccard": float(addr_char3_jaccard),
        "addr_char4_jaccard": float(addr_char4_jaccard),
        "addr_tfidf_cosine": float(addr_tfidf),
        "addr_postal_exact_match": float(postal_exact),
        "addr_postal_jaccard": float(postal_jaccard),
        "addr_postal_overlap_count": postal_overlap_cnt,
        "addr_has_postal_a": has_postal_a,
        "addr_has_postal_b": has_postal_b,
        "addr_street_num_match": float(street_num_match),
        "addr_numeric_jaccard": float(numeric_jaccard),
        "addr_numeric_overlap_count": numeric_overlap_cnt,
        "addr_fuzzy_ratio": float(addr_fuzzy),
        "addr_token_sort_ratio": float(addr_sort_ratio),
        "addr_missing": addr_missing,
        "country_exact_match": country_exact,
        "country_conflict": country_conflict,
        "is_s2": is_s2,
        "is_s3": is_s3,
        "name_addr_mean_sim": float(mean_sim),
        "name_addr_geom_sim": float(geom_sim),
        "name_addr_min_sim": float(min_sim),
        "name_addr_sim_diff": float(sim_diff),
        "name_addr_prod_sim": float(prod_sim),
        "name_a_missing": name_a_missing,
        "name_b_missing": name_b_missing,
        "addr_a_missing": addr_a_missing,
        "addr_b_missing": addr_b_missing,
        "either_addr_missing": either_addr_missing,
        "both_addr_present": both_addr_present,
    }
