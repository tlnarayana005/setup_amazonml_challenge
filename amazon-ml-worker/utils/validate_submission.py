"""
Validate submission for Entity Resolution task.

Checks:
1. matching_results.tsv format and coverage
2. candidate_pairs.tsv format
3. All matches are a subset of candidates
4. One row per test S1 entity
5. No duplicate matched IDs
6. Only S2/S3 IDs in matched_entity_ids
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def validate(
    matching_path: str,
    candidate_path: str,
    test_dir: str,
) -> bool:
    errors = []
    warnings = []

    # ── Load test sources ──────────────────────────────────────────────────
    test_base = Path(test_dir)
    test_s1 = pd.read_csv(test_base / "test_source1.tsv", sep="\t", dtype=str)
    test_s2 = pd.read_csv(test_base / "test_source2.tsv", sep="\t", dtype=str) if (test_base / "test_source2.tsv").exists() else pd.DataFrame(columns=["entity_id"])
    test_s3 = pd.read_csv(test_base / "test_source3.tsv", sep="\t", dtype=str) if (test_base / "test_source3.tsv").exists() else pd.DataFrame(columns=["entity_id"])

    test_s1_ids = set(test_s1["entity_id"].astype(str))
    valid_match_ids = set(test_s2["entity_id"].astype(str)) | set(test_s3["entity_id"].astype(str))

    # ── Load matching_results.tsv ──────────────────────────────────────────
    matching = pd.read_csv(matching_path, sep="\t", dtype=str).fillna("")
    print(f"matching_results.tsv: {len(matching)} rows")

    if "source1_entity_id" not in matching.columns:
        errors.append("Missing column 'source1_entity_id' in matching_results.tsv")
    if "matched_entity_ids" not in matching.columns:
        errors.append("Missing column 'matched_entity_ids' in matching_results.tsv")

    if errors:
        _report(errors, warnings)
        return False

    # Check: one row per test S1 entity
    result_s1_ids = set(matching["source1_entity_id"].astype(str))
    missing_s1 = test_s1_ids - result_s1_ids
    extra_s1 = result_s1_ids - test_s1_ids
    if missing_s1:
        errors.append(f"Missing {len(missing_s1)} test S1 entities from matching_results.tsv")
    if extra_s1:
        errors.append(f"{len(extra_s1)} unexpected S1 entities in matching_results.tsv")

    # Check: no duplicate S1 rows
    dup_s1 = matching["source1_entity_id"].duplicated().sum()
    if dup_s1 > 0:
        errors.append(f"{dup_s1} duplicate source1_entity_id rows in matching_results.tsv")

    # Check matched IDs validity
    all_matched_ids = set()
    for _, row in matching.iterrows():
        matched = str(row["matched_entity_ids"]).strip()
        if not matched:
            continue
        ids = [x.strip() for x in matched.split(",") if x.strip()]
        # No duplicates within a row
        if len(ids) != len(set(ids)):
            errors.append(f"Duplicate matched IDs for entity {row['source1_entity_id']}")
        for mid in ids:
            all_matched_ids.add(mid)
            # Must be S2 or S3
            if mid not in valid_match_ids:
                errors.append(f"Matched ID '{mid}' for S1='{row['source1_entity_id']}' is not a valid S2/S3 entity")

    # ── Load candidate_pairs.tsv ───────────────────────────────────────────
    candidates = pd.read_csv(candidate_path, sep="\t", dtype=str)
    print(f"candidate_pairs.tsv: {len(candidates)} rows")

    if "source1_entity_id" not in candidates.columns:
        errors.append("Missing column 'source1_entity_id' in candidate_pairs.tsv")
    if "source2_entity_id" not in candidates.columns:
        errors.append("Missing column 'source2_entity_id' in candidate_pairs.tsv")

    if "source1_entity_id" in candidates.columns and "source2_entity_id" in candidates.columns:
        cand_pairs = set(zip(
            candidates["source1_entity_id"].astype(str),
            candidates["source2_entity_id"].astype(str)
        ))

        # Check: all matches are subset of candidates
        for _, row in matching.iterrows():
            s1 = str(row["source1_entity_id"])
            matched = str(row["matched_entity_ids"]).strip()
            if not matched:
                continue
            for mid in matched.split(","):
                mid = mid.strip()
                if mid and (s1, mid) not in cand_pairs:
                    errors.append(
                        f"Match ({s1}, {mid}) is NOT in candidate_pairs.tsv"
                    )

        # Check: no duplicate candidate pairs
        dup_cands = candidates.duplicated(subset=["source1_entity_id", "source2_entity_id"]).sum()
        if dup_cands > 0:
            warnings.append(f"{dup_cands} duplicate pairs in candidate_pairs.tsv")

    _report(errors, warnings)
    return len(errors) == 0


def _report(errors, warnings):
    print()
    print("=" * 60)
    print("  Submission Validation Report")
    print("=" * 60)
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        print()
        print(f"  VALIDATION FAILED: {len(errors)} errors")
    else:
        print("  VALIDATION PASSED [OK]")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate Entity Resolution submission")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to test data directory")
    args = parser.parse_args()

    valid = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
