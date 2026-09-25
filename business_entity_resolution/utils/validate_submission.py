#!/usr/bin/env python3
"""Submission validator for Business Entity Resolution Challenge.

Stdlib only (no third-party dependencies). Checks both matching_results.tsv
and candidate_pairs.tsv against every challenge rule.

Usage:
    python utils/validate_submission.py \\
        --matching output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv \\
        --test-dir dataset/test
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Set


def read_tsv_rows(path: Path) -> List[List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        return list(reader)


def load_source_ids(source_file: Path) -> Set[str]:
    ids = set()
    with open(source_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return ids
        for row in reader:
            if row:
                ids.add(row[0])
    return ids


def validate_file(
    path: Path,
    expected_header: List[str],
    s1_ids: Set[str],
    valid_other_ids: Set[str],
    file_type: str,
) -> (Dict[str, List[str]], List[str]):
    errors = []
    if not path.exists():
        errors.append(f"{file_type} file not found: {path}")
        return {}, errors

    rows = read_tsv_rows(path)
    if not rows:
        errors.append(f"{file_type} file is empty: {path}")
        return {}, errors

    header = rows[0]
    if header != expected_header:
        errors.append(
            f"{file_type} invalid header. Expected {expected_header}, got {header}"
        )

    seen_s1 = set()
    records: Dict[str, List[str]] = {}

    for line_num, row in enumerate(rows[1:], start=2):
        if len(row) < 1:
            errors.append(f"{file_type} line {line_num}: Empty line")
            continue
        if len(row) > 2:
            errors.append(
                f"{file_type} line {line_num}: Row has {len(row)} columns, expected exactly 1 or 2 tab-separated columns."
            )
            continue

        s1_id = row[0].strip()
        raw_matches = row[1].strip() if len(row) > 1 else ""

        if not s1_id:
            errors.append(f"{file_type} line {line_num}: Empty source1_entity_id")
            continue

        if s1_id in seen_s1:
            errors.append(f"{file_type} line {line_num}: Duplicate source1_entity_id: {s1_id}")
        seen_s1.add(s1_id)

        if s1_ids and s1_id not in s1_ids:
            errors.append(f"{file_type} line {line_num}: Entity {s1_id} is not present in test_source1.tsv")

        if not raw_matches:
            records[s1_id] = []
            continue

        id_list = raw_matches.split(",")
        id_set = set()
        cleaned_ids = []
        for match_id in id_list:
            mid = match_id.strip()
            if not mid:
                continue
            if mid in id_set:
                errors.append(f"{file_type} line {line_num}: Duplicate ID '{mid}' in list for {s1_id}")
            id_set.add(mid)

            if not (mid.startswith("S2-") or mid.startswith("S3-")):
                errors.append(
                    f"{file_type} line {line_num}: Invalid prefix for '{mid}'. Must start with S2- or S3-"
                )

            if valid_other_ids and mid not in valid_other_ids:
                errors.append(
                    f"{file_type} line {line_num}: ID '{mid}' does not exist in test Source 2 or Source 3"
                )

            cleaned_ids.append(mid)

        records[s1_id] = cleaned_ids

    # Check that all test S1 IDs are present
    if s1_ids:
        missing_s1 = s1_ids - seen_s1
        if missing_s1:
            sample_missing = list(missing_s1)[:5]
            errors.append(
                f"{file_type} is missing {len(missing_s1)} Source 1 entities (e.g. {sample_missing})"
            )

    return records, errors


def main():
    parser = argparse.ArgumentParser(description="Validate challenge submissions")
    parser.add_argument("--matching", type=Path, required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", type=Path, required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=Path, required=True, help="Path to test directory (or sample dir)")
    args = parser.parse_args()

    s1_file = args.test_dir / "test_source1.tsv"
    s2_file = args.test_dir / "test_source2.tsv"
    s3_file = args.test_dir / "test_source3.tsv"

    # Fallback for sample dir if test files aren't named with test_ prefix
    if not s1_file.exists() and (args.test_dir / "sample_source1.tsv").exists():
        s1_file = args.test_dir / "sample_source1.tsv"
        s2_file = args.test_dir / "sample_source2.tsv"
        s3_file = args.test_dir / "sample_source3.tsv"

    s1_ids = load_source_ids(s1_file) if s1_file.exists() else set()
    s2_ids = load_source_ids(s2_file) if s2_file.exists() else set()
    s3_ids = load_source_ids(s3_file) if s3_file.exists() else set()
    valid_other_ids = s2_ids | s3_ids

    all_errors = []

    # 1. Validate matching_results.tsv
    matches, matching_errors = validate_file(
        args.matching,
        expected_header=["source1_entity_id", "matched_entity_ids"],
        s1_ids=s1_ids,
        valid_other_ids=valid_other_ids,
        file_type="matching_results.tsv",
    )
    all_errors.extend(matching_errors)

    # 2. Validate candidate_pairs.tsv
    candidates, candidate_errors = validate_file(
        args.candidate,
        expected_header=["source1_entity_id", "candidate_entity_ids"],
        s1_ids=s1_ids,
        valid_other_ids=valid_other_ids,
        file_type="candidate_pairs.tsv",
    )
    all_errors.extend(candidate_errors)

    # 3. Cross-validate: matches must be subset of candidates
    if matches and candidates:
        subset_violations = 0
        for s1_id, match_ids in matches.items():
            cand_set = set(candidates.get(s1_id, []))
            for mid in match_ids:
                if mid not in cand_set:
                    subset_violations += 1
                    if subset_violations <= 5:
                        all_errors.append(
                            f"Subset violation: entity {s1_id} matched '{mid}' but it is not in candidate_pairs.tsv"
                        )
        if subset_violations > 5:
            all_errors.append(f"... and {subset_violations - 5} more subset violations.")

    if all_errors:
        print("FAIL: Found the following issues with your submission:")
        for idx, err in enumerate(all_errors, 1):
            print(f"  {idx}. {err}")
        sys.exit(1)
    else:
        print("PASS: Submissions are valid and ready for submission!")
        sys.exit(0)


if __name__ == "__main__":
    main()
