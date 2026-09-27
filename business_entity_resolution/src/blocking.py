"""High-Recall, High-Selectivity Blocking Engine — Scaled for Millions of Records.

Owner: feat/blocking.
Design objectives:
1. High Recall (>96% upper bound on matching pairs).
2. Minimal Candidate Set Size: Keep candidates strictly bounded (Top-K per S1)
   to maximize the candidate-size reduction ratio rewarded by Amazon evaluators.
3. Scalable to 10M+ records using TF-IDF sparse matrix multiplication instead
   of Python-loop inverted indices.

Architecture (v2 — Production Scale):
    Channel A: TF-IDF cosine similarity on normalized business names (char n-gram + word).
    Channel B: Numeric/Postal anchor exact co-occurrence (lightweight dict lookup).
    Channel C: Address token TF-IDF cosine similarity.
    Merge    : Score fusion -> adaptive top-K per S1 entity.

Performance target: < 10 min for 2M S1 x 10M S2+S3 on Colab T4 (CPU).
"""
import gc
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from . import normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_series(series: pd.Series) -> List[str]:
    """Vectorized name normalization over a pandas Series."""
    return [
        normalize.normalize_name(v, strip_suffixes=True) if v else ""
        for v in series.fillna("")
    ]


def _normalize_addr_series(series: pd.Series) -> List[str]:
    return [
        normalize.normalize_address(v) if v else ""
        for v in series.fillna("")
    ]


def _country_series(series: pd.Series) -> List[str]:
    return [
        normalize.normalize_country(v) if v else ""
        for v in series.fillna("")
    ]


def _batch_cosine_topk(
    query_matrix: sp.csr_matrix,
    index_matrix: sp.csr_matrix,
    top_k: int,
    batch_size: int = 2048,
) -> List[List[Tuple[int, float]]]:
    """Compute cosine similarity between query and index matrices in batches.

    Returns list of (index_row_idx, score) lists, one per query row,
    containing up to top_k results with score > 0.
    """
    n_queries = query_matrix.shape[0]
    results: List[List[Tuple[int, float]]] = [[] for _ in range(n_queries)]

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = query_matrix[start:end]  # shape: (batch, features)

        # (batch, n_index)  — sparse dot product
        sims = batch.dot(index_matrix.T)

        if sp.issparse(sims):
            sims = sims.toarray()

        # For each query in batch, pick top_k
        for local_i, row_scores in enumerate(sims):
            global_i = start + local_i
            # argpartition is O(n) — much faster than full argsort on large arrays
            if len(row_scores) <= top_k:
                idxs = np.where(row_scores > 0.0)[0]
            else:
                # Get top_k candidates efficiently
                part = np.argpartition(row_scores, -top_k)[-top_k:]
                idxs = part[row_scores[part] > 0.0]

            if len(idxs) == 0:
                continue

            top_pairs = sorted(
                [(int(j), float(row_scores[j])) for j in idxs],
                key=lambda x: x[1],
                reverse=True,
            )
            results[global_i] = top_pairs[:top_k]

    return results


def _build_numeric_index(
    df: pd.DataFrame,
    id_col: str,
    addr_col: str,
) -> Dict[str, List[str]]:
    """Build exact-match inverted index on numeric/postal tokens."""
    index: Dict[str, List[str]] = defaultdict(list)
    for row in df.itertuples(index=False):
        ent_id = getattr(row, id_col)
        addr = getattr(row, addr_col, "") or ""
        for num in normalize.extract_numeric_tokens(addr):
            if len(num) >= 4:  # skip trivial short numbers
                index[num].append(ent_id)
        for postal in normalize.extract_postal_tokens(addr):
            if isinstance(postal, str) and len(postal) >= 4:
                index[postal].append(ent_id)
    return dict(index)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def generate_candidates(
    source1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    id_col: str,
    name_col: str,
    addr_col: str = "business_address",
    country_col: str = "country",
    top_k: int = 10,
    name_weight: float = 3.0,
    addr_weight: float = 1.5,
    numeric_bonus: float = 2.0,
    min_score: float = 0.15,
    batch_size: int = 2048,
) -> Dict[str, Set[str]]:
    """Generate high-probability candidate set using TF-IDF sparse matrix multiplication.

    Scales to millions of records. Typical runtime: 5-15 min for 2M x 10M on Colab CPU.

    Args:
        source1_df:    Source 1 dataframe (query side).
        other_df:      Source 2 or 3 dataframe (index side).
        id_col:        Entity ID column name.
        name_col:      Business name column name.
        addr_col:      Address column name.
        country_col:   Country column name.
        top_k:         Maximum candidates per S1 entity.
        name_weight:   Score weight for name TF-IDF channel.
        addr_weight:   Score weight for address TF-IDF channel.
        numeric_bonus: Score bonus for shared numeric/postal tokens.
        min_score:     Minimum fused score to keep a candidate.
        batch_size:    Rows per batch for matrix multiplication.

    Returns:
        Dict mapping each S1 entity ID to a set of candidate entity IDs.
    """
    # Resolve column names
    actual_addr = addr_col if addr_col in other_df.columns else "business_address"
    actual_country = country_col if country_col in other_df.columns else "country"

    other_ids: List[str] = list(other_df[id_col])
    s1_ids: List[str] = list(source1_df[id_col])
    n_other = len(other_ids)
    n_s1 = len(s1_ids)

    print(f"  [Blocking] S1={n_s1:,}, Other={n_other:,} — building TF-IDF indices...")

    # ------------------------------------------------------------------
    # 1. Normalize text
    # ------------------------------------------------------------------
    s1_names = _normalize_series(source1_df[name_col])
    s1_addrs = _normalize_addr_series(source1_df[actual_addr] if actual_addr in source1_df.columns else pd.Series([""] * n_s1))
    s1_countries = _country_series(source1_df[actual_country] if actual_country in source1_df.columns else pd.Series([""] * n_s1))

    other_names = _normalize_series(other_df[name_col])
    other_addrs = _normalize_addr_series(other_df[actual_addr])
    other_countries = _country_series(other_df[actual_country] if actual_country in other_df.columns else pd.Series([""] * n_other))

    # ------------------------------------------------------------------
    # 2. Name TF-IDF — fit ONLY on index side (other_df), transform both
    #    Use char 3-gram only + max_features cap to control RAM.
    #    For very large corpora, fit on a random sample to save memory.
    # ------------------------------------------------------------------
    print(f"  [Blocking] Fitting Name TF-IDF vectorizer (max_features=150k)...")
    FIT_SAMPLE = 300_000  # max rows used to fit vocabulary
    fit_names = other_names
    if len(fit_names) > FIT_SAMPLE:
        rng = np.random.default_rng(42)
        idx_sample = rng.choice(len(fit_names), size=FIT_SAMPLE, replace=False)
        fit_names = [fit_names[i] for i in idx_sample]

    name_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),   # char 3-gram only — best recall/RAM tradeoff
        min_df=3,
        max_df=0.9,
        max_features=150_000,  # hard cap on vocabulary size
        sublinear_tf=True,
        dtype=np.float32,
    )
    name_vectorizer.fit(fit_names)
    del fit_names
    gc.collect()

    s1_name_mat = name_vectorizer.transform(s1_names)       # (n_s1, vocab)
    other_name_mat = name_vectorizer.transform(other_names)  # (n_other, vocab)
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Address TF-IDF (word unigram only — keeps RAM low)
    # ------------------------------------------------------------------
    print(f"  [Blocking] Fitting Address TF-IDF vectorizer (max_features=50k)...")
    non_empty_count = sum(1 for a in other_addrs if a.strip())
    use_addr_channel = non_empty_count > 100

    if use_addr_channel:
        fit_addrs = other_addrs
        if len(fit_addrs) > FIT_SAMPLE:
            rng2 = np.random.default_rng(99)
            idx2 = rng2.choice(len(fit_addrs), size=FIT_SAMPLE, replace=False)
            fit_addrs = [fit_addrs[i] for i in idx2]

        addr_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),   # unigram only for address
            min_df=3,
            max_df=0.9,
            max_features=50_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        addr_vectorizer.fit(fit_addrs)
        del fit_addrs
        gc.collect()

        s1_addr_mat = addr_vectorizer.transform(s1_addrs)
        other_addr_mat = addr_vectorizer.transform(other_addrs)
        gc.collect()

    # ------------------------------------------------------------------
    # 4. Numeric/Postal exact-match index
    # ------------------------------------------------------------------
    print(f"  [Blocking] Building numeric/postal anchor index...")
    numeric_index = _build_numeric_index(other_df, id_col, actual_addr)
    # Map other entity ID -> positional index in other_ids list
    other_id_to_pos = {eid: pos for pos, eid in enumerate(other_ids)}

    # ------------------------------------------------------------------
    # 5. Batch cosine similarity: Name channel
    # ------------------------------------------------------------------
    print(f"  [Blocking] Computing Name cosine similarities (batch_size={batch_size})...")
    name_topk = _batch_cosine_topk(s1_name_mat, other_name_mat, top_k * 3, batch_size)

    # ------------------------------------------------------------------
    # 6. Batch cosine similarity: Address channel
    # ------------------------------------------------------------------
    addr_topk: List[List[Tuple[int, float]]] = [[] for _ in range(n_s1)]
    if use_addr_channel:
        print(f"  [Blocking] Computing Address cosine similarities (batch_size={batch_size})...")
        addr_topk = _batch_cosine_topk(s1_addr_mat, other_addr_mat, top_k * 2, batch_size)

    # ------------------------------------------------------------------
    # 7. Fuse scores + numeric bonus + country filter -> top-K
    # ------------------------------------------------------------------
    print(f"  [Blocking] Fusing channels and applying country filter...")
    candidates: Dict[str, Set[str]] = {}

    for i, s1_id in enumerate(s1_ids):
        s1_country = s1_countries[i]
        s1_addr_text = s1_addrs[i]

        # Accumulate fused scores per candidate
        fused: Dict[int, float] = {}

        for pos, score in name_topk[i]:
            fused[pos] = fused.get(pos, 0.0) + name_weight * score

        for pos, score in addr_topk[i]:
            fused[pos] = fused.get(pos, 0.0) + addr_weight * score

        # Numeric bonus: exact numeric token overlap
        s1_nums = normalize.extract_numeric_tokens(s1_addr_text)
        s1_postals = normalize.extract_postal_tokens(s1_addr_text)
        for token in s1_nums | s1_postals:
            if len(str(token)) < 4:
                continue
            for match_id in numeric_index.get(str(token), []):
                pos = other_id_to_pos.get(match_id, -1)
                if pos >= 0:
                    fused[pos] = fused.get(pos, 0.0) + numeric_bonus

        if not fused:
            candidates[s1_id] = set()
            continue

        # Country filter: hard-drop conflicting known countries
        filtered = {}
        for pos, score in fused.items():
            cand_country = other_countries[pos]
            if s1_country and cand_country and s1_country != cand_country:
                continue  # Hard drop
            filtered[pos] = score

        if not filtered:
            candidates[s1_id] = set()
            continue

        # Minimum score gate + top-K
        sorted_cands = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        max_score = sorted_cands[0][1]

        if max_score < min_score:
            candidates[s1_id] = set()
            continue

        # Dynamic floor: keep candidates within 40% of top score
        floor = max(min_score, max_score * 0.40)
        selected = {
            other_ids[pos]
            for pos, score in sorted_cands[:top_k]
            if score >= floor
        }
        candidates[s1_id] = selected

    return candidates


# ---------------------------------------------------------------------------
# Recall measurement
# ---------------------------------------------------------------------------

class RecallResult(float):
    avg_candidates: float = 0.0
    total_candidates: int = 0

    def __iter__(self):
        yield float(self)
        yield self.avg_candidates
        yield self.total_candidates


def measure_recall(
    candidates: Dict[str, Iterable[str]],
    ground_truth: Dict[str, set],
) -> RecallResult:
    """Calculate recall ceiling, average candidates per entity, and total candidate count."""
    total_true = 0
    total_found = 0
    cand_counts = []

    for s1_id, truth in ground_truth.items():
        cand = set(candidates.get(s1_id, ()))
        cand_counts.append(len(cand))
        if not truth:
            continue
        total_true += len(truth)
        total_found += len(truth & cand)

    recall = total_found / total_true if total_true else 1.0
    res = RecallResult(recall)
    res.avg_candidates = sum(cand_counts) / len(cand_counts) if cand_counts else 0.0
    res.total_candidates = sum(cand_counts)
    return res
