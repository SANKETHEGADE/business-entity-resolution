"""Reading source files and writing the two required submission TSVs.

Both output files share the same shape: one row per Source-1 entity, a
comma-separated ID list in the second column (empty string when there are
no candidates/matches). Keeping the writer in one place avoids four people
writing four slightly-different TSV quoting bugs.
"""
from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd

from . import config


def read_source(path: Path) -> pd.DataFrame:
    """Read one source file. Always pass sep='\\t' explicitly -- a plain
    read_csv silently produces a single garbage column on this data."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,  # empty address fields are legitimate, not NaN
    )
    expected = {config.COL_ENTITY_ID, config.COL_NAME, config.COL_ADDRESS, config.COL_COUNTRY}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def read_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected = {config.COL_S1_ID, config.COL_MATCHED_IDS}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def ground_truth_to_sets(gt: pd.DataFrame) -> Dict[str, set]:
    """source1_entity_id -> set of matched ids (empty set for singletons)."""
    out = {}
    for row in gt.itertuples(index=False):
        s1_id = getattr(row, config.COL_S1_ID)
        raw = getattr(row, config.COL_MATCHED_IDS)
        out[s1_id] = set(raw.split(",")) if raw else set()
    return out


def write_id_list_tsv(
    mapping: Mapping[str, Iterable[str]],
    path: Path,
    id_col: str,
    list_col: str,
) -> None:
    """Write a {source1_id: [ids...]} mapping in the required TSV shape.

    - One row per key, in the order given by `mapping`.
    - No duplicate IDs within a list (deduped here defensively).
    - Empty list -> empty string, not "nan" or "[]".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for s1_id, ids in mapping.items():
            deduped = list(dict.fromkeys(ids))  # preserve order, drop dupes
            f.write(f"{s1_id}\t{','.join(deduped)}\n")


def write_matching_results(mapping: Mapping[str, Iterable[str]], path: Path = config.MATCHING_RESULTS_PATH) -> None:
    write_id_list_tsv(mapping, path, config.COL_S1_ID, config.COL_MATCHED_IDS)


def write_candidate_pairs(mapping: Mapping[str, Iterable[str]], path: Path = config.CANDIDATE_PAIRS_PATH) -> None:
    write_id_list_tsv(mapping, path, config.COL_S1_ID, config.COL_CANDIDATE_IDS)
