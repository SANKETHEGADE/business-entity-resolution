"""High-Recall Blocking Engine — RAM-Safe for 10M+ records on 12 GB Colab.

Key design: NEVER materialise the full S2/S3 TF-IDF matrix in memory.
Instead, transform index rows in shards of 100K at a time so peak RAM
per matrix multiply is only ~300 MB regardless of S2/S3 size.

Architecture:
    Step 1 : Fit TF-IDF vocab on a random sample of the index side (300K rows).
    Step 2 : Transform all S1 rows into a sparse matrix (small - max 200K rows).
    Step 3 : Loop over index shards of 100K rows; transform each shard on-the-fly;
             compute (S1_batch × shard.T) dense block of max 256 × 100K × 4B ≈ 100 MB.
    Step 4 : Accumulate running top-K candidates per S1 entity.
    Step 5 : Numeric/postal exact-match bonus.
    Step 6 : Country filter + dynamic floor → emit candidate set.
"""
import gc
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from . import normalize


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _norm_names(series: pd.Series) -> List[str]:
    return [normalize.normalize_name(v, strip_suffixes=True) if v else ""
            for v in series.fillna("")]


def _norm_addrs(series: pd.Series) -> List[str]:
    return [normalize.normalize_address(v) if v else ""
            for v in series.fillna("")]


def _norm_countries(series: pd.Series) -> List[str]:
    return [normalize.normalize_country(v) if v else ""
            for v in series.fillna("")]


# ---------------------------------------------------------------------------
# Core: sharded top-K cosine similarity
# ---------------------------------------------------------------------------

def _sharded_topk(
    query_mat: sp.csr_matrix,           # (n_s1, vocab) — fits in RAM
    index_texts: List[str],             # raw strings — transformed shard by shard
    vectorizer: TfidfVectorizer,
    top_k: int,
    query_batch: int = 512,
    index_shard: int = 100_000,
) -> List[List[Tuple[int, float]]]:
    """Compute cosine top-K without ever building the full index matrix.

    Peak RAM per multiply:  query_batch × index_shard × 4 bytes
                         =  512 × 100K × 4 = ~200 MB  — safe on Colab.
    """
    n_q = query_mat.shape[0]
    n_i = len(index_texts)

    # Running accumulators: top_scores[qi] = [(score, global_index_idx), ...]
    top_buf: List[List[Tuple[float, int]]] = [[] for _ in range(n_q)]

    for i_start in range(0, n_i, index_shard):
        i_end = min(i_start + index_shard, n_i)
        shard_texts = index_texts[i_start:i_end]

        # Transform only this shard  →  (shard_size, vocab) sparse
        shard_mat = vectorizer.transform(shard_texts)

        for q_start in range(0, n_q, query_batch):
            q_end = min(q_start + query_batch, n_q)
            q = query_mat[q_start:q_end]          # (qb, vocab) sparse

            # Dense result: (qb, shard_size)  — small enough
            sims = q.dot(shard_mat.T)
            if sp.issparse(sims):
                sims = sims.toarray()
            sims = np.asarray(sims, dtype=np.float32)

            for local_qi, row in enumerate(sims):
                pos = np.where(row > 0.02)[0]
                if pos.size == 0:
                    continue
                global_qi = q_start + local_qi
                buf = top_buf[global_qi]
                for j in pos:
                    buf.append((float(row[j]), int(i_start + j)))

            del sims

        del shard_mat
        gc.collect()

    # Trim each buffer to top-K
    results: List[List[Tuple[int, float]]] = []
    for buf in top_buf:
        if not buf:
            results.append([])
            continue
        buf.sort(reverse=True)
        results.append([(idx, sc) for sc, idx in buf[:top_k]])

    return results


# ---------------------------------------------------------------------------
# Numeric / postal inverted index
# ---------------------------------------------------------------------------

def _build_numeric_index(
    df: pd.DataFrame, id_col: str, addr_col: str
) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = defaultdict(list)
    for row in df.itertuples(index=False):
        eid = getattr(row, id_col)
        addr = getattr(row, addr_col, "") or ""
        for tok in normalize.extract_numeric_tokens(addr):
            if len(tok) >= 4:
                idx[tok].append(eid)
        for tok in normalize.extract_postal_tokens(addr):
            if isinstance(tok, str) and len(tok) >= 4:
                idx[tok].append(eid)
    return dict(idx)


# ---------------------------------------------------------------------------
# Public API
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
) -> Dict[str, Set[str]]:
    """Generate candidate set using RAM-safe sharded TF-IDF similarity.

    Never loads full S2/S3 TF-IDF matrix; transforms in 100K-row shards.
    Peak RAM: ~1–2 GB regardless of S2/S3 size.
    """
    actual_addr = addr_col if addr_col in other_df.columns else "business_address"
    actual_country = country_col if country_col in other_df.columns else "country"

    other_ids: List[str] = list(other_df[id_col])
    s1_ids: List[str] = list(source1_df[id_col])
    n_other = len(other_ids)
    n_s1 = len(s1_ids)

    print(f"  [Blocking] S1={n_s1:,}, Other={n_other:,}")

    # ------------------------------------------------------------------
    # 1. Normalise text (returns plain Python lists, not matrices)
    # ------------------------------------------------------------------
    s1_names = _norm_names(source1_df[name_col])
    s1_addrs_raw = list(source1_df[actual_addr].fillna("")) if actual_addr in source1_df.columns else [""] * n_s1
    s1_addrs = _norm_addrs(pd.Series(s1_addrs_raw))
    s1_countries = _norm_countries(source1_df[actual_country].fillna("") if actual_country in source1_df.columns else pd.Series([""] * n_s1))

    other_names: List[str] = _norm_names(other_df[name_col])
    other_addrs_raw = list(other_df[actual_addr].fillna(""))
    other_addrs: List[str] = _norm_addrs(pd.Series(other_addrs_raw))
    other_countries: List[str] = _norm_countries(other_df[actual_country].fillna("") if actual_country in other_df.columns else pd.Series([""] * n_other))

    # ------------------------------------------------------------------
    # 2. Fit Name TF-IDF on a sample of the index side only
    # ------------------------------------------------------------------
    FIT_SAMPLE = 300_000
    print(f"  [Blocking] Fitting Name TF-IDF on {min(n_other, FIT_SAMPLE):,} samples...")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_other, size=min(n_other, FIT_SAMPLE), replace=False)
    fit_names = [other_names[i] for i in sample_idx]

    name_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 3),
        min_df=2, max_df=0.9, max_features=150_000,
        sublinear_tf=True, dtype=np.float32,
    )
    name_vec.fit(fit_names)
    del fit_names, sample_idx
    gc.collect()

    # Transform ONLY S1 into a matrix — small enough (max 200K rows)
    print(f"  [Blocking] Transforming S1 names ({n_s1:,} rows)...")
    s1_name_mat = name_vec.transform(s1_names)   # (n_s1, vocab) ≈ 100–200 MB
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Fit Address TF-IDF on sample
    # ------------------------------------------------------------------
    non_empty_addr = sum(1 for a in other_addrs if a.strip())
    use_addr = non_empty_addr > 100

    s1_addr_mat: Optional[sp.csr_matrix] = None
    addr_vec: Optional[TfidfVectorizer] = None

    if use_addr:
        print(f"  [Blocking] Fitting Address TF-IDF on {min(n_other, FIT_SAMPLE):,} samples...")
        rng2 = np.random.default_rng(99)
        sidx2 = rng2.choice(n_other, size=min(n_other, FIT_SAMPLE), replace=False)
        fit_addrs = [other_addrs[i] for i in sidx2]
        addr_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 1),
            min_df=3, max_df=0.9, max_features=50_000,
            sublinear_tf=True, dtype=np.float32,
        )
        addr_vec.fit(fit_addrs)
        del fit_addrs, sidx2
        gc.collect()

        print(f"  [Blocking] Transforming S1 addresses ({n_s1:,} rows)...")
        s1_addr_mat = addr_vec.transform(s1_addrs)
        gc.collect()

    # ------------------------------------------------------------------
    # 4. Numeric / postal index on other_df
    # ------------------------------------------------------------------
    print(f"  [Blocking] Building numeric anchor index...")
    numeric_idx = _build_numeric_index(other_df, id_col, actual_addr)
    other_id_to_pos = {eid: pos for pos, eid in enumerate(other_ids)}

    # ------------------------------------------------------------------
    # 5. Sharded cosine similarity — Name channel
    #    other_names is a plain list; transform happens 100K rows at a time
    # ------------------------------------------------------------------
    print(f"  [Blocking] Name channel: sharded top-K (shard=100k, qbatch=512)...")
    name_topk = _sharded_topk(s1_name_mat, other_names, name_vec, top_k * 3)

    # ------------------------------------------------------------------
    # 6. Sharded cosine similarity — Address channel
    # ------------------------------------------------------------------
    addr_topk: List[List[Tuple[int, float]]] = [[] for _ in range(n_s1)]
    if use_addr and addr_vec is not None and s1_addr_mat is not None:
        print(f"  [Blocking] Address channel: sharded top-K...")
        addr_topk = _sharded_topk(s1_addr_mat, other_addrs, addr_vec, top_k * 2)

    # ------------------------------------------------------------------
    # 7. Fuse scores + numeric bonus + country filter → top-K candidates
    # ------------------------------------------------------------------
    print(f"  [Blocking] Fusing channels + country filter...")
    candidates: Dict[str, Set[str]] = {}

    for i, s1_id in enumerate(s1_ids):
        s1_country = s1_countries[i]

        fused: Dict[int, float] = {}
        for pos, score in name_topk[i]:
            fused[pos] = fused.get(pos, 0.0) + name_weight * score
        for pos, score in addr_topk[i]:
            fused[pos] = fused.get(pos, 0.0) + addr_weight * score

        # Numeric bonus
        for tok in normalize.extract_numeric_tokens(s1_addrs_raw[i]):
            if len(tok) < 4:
                continue
            for match_id in numeric_idx.get(tok, []):
                p = other_id_to_pos.get(match_id, -1)
                if p >= 0:
                    fused[p] = fused.get(p, 0.0) + numeric_bonus

        if not fused:
            candidates[s1_id] = set()
            continue

        # Country filter
        filtered = {
            pos: score for pos, score in fused.items()
            if not (s1_country and other_countries[pos] and s1_country != other_countries[pos])
        }

        if not filtered:
            candidates[s1_id] = set()
            continue

        sorted_c = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        max_sc = sorted_c[0][1]

        if max_sc < min_score:
            candidates[s1_id] = set()
            continue

        floor = max(min_score, max_sc * 0.40)
        candidates[s1_id] = {
            other_ids[pos]
            for pos, sc in sorted_c[:top_k]
            if sc >= floor
        }

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
) -> "RecallResult":
    total_true = total_found = 0
    cand_counts = []
    for s1_id, truth in ground_truth.items():
        cand = set(candidates.get(s1_id, ()))
        cand_counts.append(len(cand))
        if truth:
            total_true += len(truth)
            total_found += len(truth & cand)
    recall = total_found / total_true if total_true else 1.0
    res = RecallResult(recall)
    res.avg_candidates = sum(cand_counts) / len(cand_counts) if cand_counts else 0.0
    res.total_candidates = sum(cand_counts)
    return res
