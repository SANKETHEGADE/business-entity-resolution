"""Candidate generation: Source-1 entity -> plausible Source-2/3 candidates.

Owner: feat/blocking branch.

This determines the recall ceiling for the whole pipeline -- anything missed
here can never be recovered by the model. At this scale (millions of rows per
source), a naive cross join is not an option; needs an indexed/blocking
approach. Options to evaluate (pick one or combine, see notes):

1. Token/n-gram inverted index on normalized business name: build
   token -> [entity_ids] postings lists per source, then for each Source-1
   entity, union the postings of its name tokens as candidates. Cheap,
   scales linearly, and is the natural first thing to try.
2. Sorted-neighborhood on a blocking key (e.g. first N chars of normalized
   name + country), sliding a window over the sorted list.
3. MinHash/LSH over token shingles for near-duplicate name detection at
   scale, if (1) alone lets too much noisy long-tail through.
4. Address-based blocking (city/postal fragments) as a second independent
   blocking pass, unioned with the name-based candidates -- addresses and
   names fail independently, so combining catches more true matches.

Whatever is used, the exact candidate set that gets passed to the model must
be what's written to candidate_pairs.tsv (see io_utils.write_candidate_pairs).
Measure recall against train_ground_truth.tsv before tuning precision here --
this stage should be recall-generous, the model narrows it down.
"""
from typing import Dict, Iterable, List

import pandas as pd

from . import normalize


def build_name_token_index(df: pd.DataFrame, entity_id_col: str, name_col: str) -> Dict[str, List[str]]:
    """token -> list of entity_ids whose normalized name contains that token."""
    index: Dict[str, List[str]] = {}
    for entity_id, name in zip(df[entity_id_col], df[name_col]):
        for token in normalize.name_tokens(name):
            index.setdefault(token, []).append(entity_id)
    return index


def candidates_for_entity(name: str, index: Dict[str, List[str]]) -> set:
    """Union of postings lists for every token in the (normalized) name."""
    candidates: set = set()
    for token in normalize.name_tokens(name):
        candidates.update(index.get(token, ()))
    return candidates


def generate_candidates(
    source1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    entity_id_col: str,
    name_col: str,
) -> Dict[str, set]:
    """Token-index blocking of one Source-1 dataframe against one other source.

    Returns {source1_entity_id: {candidate_entity_id, ...}}. Callers should
    union results across Source-2 and Source-3 before writing candidate_pairs.tsv.

    TODO (feat/blocking): this is the naive baseline (option 1 above) --
    measure its recall on train before deciding whether options 2-4 are
    needed for entities with very short/generic names.
    """
    index = build_name_token_index(other_df, entity_id_col, name_col)
    result: Dict[str, set] = {}
    for s1_id, s1_name in zip(source1_df[entity_id_col], source1_df[name_col]):
        result[s1_id] = candidates_for_entity(s1_name, index)
    return result


def measure_recall(candidates: Dict[str, Iterable[str]], ground_truth: Dict[str, set]) -> float:
    """Fraction of true matches that appear in the candidate set (upper bound on final recall)."""
    total_true = 0
    total_found = 0
    for s1_id, truth in ground_truth.items():
        if not truth:
            continue
        cand = set(candidates.get(s1_id, ()))
        total_true += len(truth)
        total_found += len(truth & cand)
    return total_found / total_true if total_true else 1.0
