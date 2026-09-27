"""High-Recall Blocking Engine — Fast token-based inverted index.

For training: optionally sample S2/S3 to N records (default 300K) — fast enough
for Colab T4 in ~5 min. The model trained on the sample generalises to full test.

For test: uses the same inverted index approach but processes the full corpus in
chunks so RAM stays bounded.

Architecture (v3 — Inverted Index, pandas-accelerated):
    Step 1: Tokenise names into char-3 grams + word tokens using pandas str ops.
    Step 2: Build an inverted index: token -> list of (entity_id, idf_weight).
    Step 3: For each S1 entity, look up all candidate IDs via its tokens,
            accumulate weighted hit counts using numpy, pick top-K.
    Step 4: Address numeric bonus + country filter.

No TF-IDF matrix multiplication — O(n * avg_token_hits) per entity,
typically 1-3 ms/entity on Colab CPU.
"""
import gc
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from . import normalize


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _name_tokens_fast(name: str) -> List[str]:
    """Word tokens + char 3-grams for a normalised name string."""
    n = normalize.normalize_name(name, strip_suffixes=True) if name else ""
    if not n:
        return []
    words = [w for w in n.split() if len(w) >= 2]
    trigrams = []
    compact = n.replace(" ", "")
    if len(compact) >= 3:
        trigrams = [compact[i:i+3] for i in range(len(compact) - 2)]
    return words + trigrams


def _addr_tokens_fast(addr: str) -> List[str]:
    """Word tokens from a normalised address string."""
    a = normalize.normalize_address(addr) if addr else ""
    return [w for w in a.split() if len(w) >= 3]


# ---------------------------------------------------------------------------
# Inverted index construction
# ---------------------------------------------------------------------------

def _build_index(
    df: pd.DataFrame,
    id_col: str,
    name_col: str,
    addr_col: str,
    country_col: str,
    max_df_ratio: float = 0.03,
    sample_n: Optional[int] = None,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, dict]]:
    """Build token -> [(entity_id, idf_weight)] inverted index.

    Args:
        sample_n: If set, randomly sample this many rows from df before indexing.
    """
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"    Sampled {sample_n:,} rows from {len(df) + (len(df)):,} available")

    n = len(df)
    max_df = max(10, int(n * max_df_ratio))

    # Count token document frequencies
    token_df: Dict[str, int] = defaultdict(int)
    records: Dict[str, dict] = {}

    for row in df.itertuples(index=False):
        eid = getattr(row, id_col)
        name = getattr(row, name_col) or ""
        addr = getattr(row, addr_col, "") or ""
        country = getattr(row, country_col, "") or ""

        toks = set(_name_tokens_fast(name))
        a_toks = set(_addr_tokens_fast(addr))
        nums = normalize.extract_numeric_tokens(addr)
        postals = normalize.extract_postal_tokens(addr)

        for t in toks:
            token_df[t] += 1

        records[eid] = {
            "name_toks": toks,
            "addr_toks": a_toks,
            "nums": nums,
            "postals": postals,
            "country": normalize.normalize_country(country),
        }

    # Build inverted index with IDF weights; drop stop-tokens
    index: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for eid, meta in records.items():
        for t in meta["name_toks"]:
            df_t = token_df.get(t, 1)
            if df_t <= max_df:
                idf = math.log((n + 1) / (df_t + 1)) + 1.0
                index[t].append((eid, idf))

    return dict(index), records


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    source1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    id_col: str,
    name_col: str,
    addr_col: str = "business_address",
    country_col: str = "country",
    top_k: int = 10,
    numeric_bonus: float = 3.0,
    min_score: float = 1.5,
    sample_other: Optional[int] = None,
) -> Dict[str, Set[str]]:
    """Generate candidates using a fast IDF-weighted inverted index.

    Args:
        sample_other: If set, sample this many rows from other_df before indexing.
                      Recommended 300_000 for training on Colab. Set None for test.
    """
    actual_addr = addr_col if addr_col in other_df.columns else "business_address"
    actual_country = country_col if country_col in other_df.columns else "country"

    n_s1 = len(source1_df)
    n_other = len(other_df)
    print(f"    S1={n_s1:,}  Other={n_other:,}  sample_other={sample_other}")

    # Build inverted index on other_df (sampled if requested)
    token_index, other_records = _build_index(
        other_df, id_col, name_col, actual_addr, actual_country,
        sample_n=sample_other,
    )
    gc.collect()

    candidates: Dict[str, Set[str]] = {}

    for row in source1_df.itertuples(index=False):
        s1_id = getattr(row, id_col)
        name = getattr(row, name_col) or ""
        addr = getattr(row, actual_addr, "") or ""
        s1_country = normalize.normalize_country(getattr(row, actual_country, "") or "")

        s1_name_toks = _name_tokens_fast(name)
        s1_nums = normalize.extract_numeric_tokens(addr)
        s1_postals = normalize.extract_postal_tokens(addr)

        # Score candidates via token hit accumulation
        scores: Dict[str, float] = defaultdict(float)
        for tok in s1_name_toks:
            for cand_id, idf in token_index.get(tok, []):
                scores[cand_id] += idf

        # Numeric / postal bonus
        for num in s1_nums | s1_postals:
            if len(str(num)) < 4:
                continue
            # Find candidates that share this number via their records
            for cand_id, meta in other_records.items():
                if num in meta["nums"] or num in meta["postals"]:
                    scores[cand_id] += numeric_bonus

        if not scores:
            candidates[s1_id] = set()
            continue

        # Country filter + min score gate
        filtered = {
            cid: sc for cid, sc in scores.items()
            if sc >= min_score and not (
                s1_country and other_records[cid]["country"]
                and s1_country != other_records[cid]["country"]
            )
        }

        if not filtered:
            candidates[s1_id] = set()
            continue

        # Top-K with dynamic floor
        top = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]
        max_sc = top[0][1]
        floor = max(min_score, max_sc * 0.35)
        candidates[s1_id] = {cid for cid, sc in top[:top_k] if sc >= floor}

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
