#!/usr/bin/env python3
"""Automated Submission Packaging Script.

Validates the output files and generates <team_name>_submission.zip
matching the exact competition structure:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── utils/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md

Usage:
    python utils/make_submission_zip.py --team-name my_team
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
SRC_DIR = ROOT / "src"
UTILS_DIR = ROOT / "utils"
DOC_FILE = ROOT / "Documentation_template.md"
README_FILE = ROOT / "README.md"
REQ_FILE = ROOT / "requirements.txt"


def validate_outputs(test_dir: Path) -> bool:
    val_script = UTILS_DIR / "validate_submission.py"
    matching = OUTPUT_DIR / "matching_results.tsv"
    candidate = OUTPUT_DIR / "candidate_pairs.tsv"

    if not matching.exists() or not candidate.exists():
        print(f"ERROR: Missing output files in {OUTPUT_DIR}. Run the test pipeline first.")
        return False

    cmd = [
        sys.executable,
        str(val_script),
        "--matching", str(matching),
        "--candidate", str(candidate),
        "--test-dir", str(test_dir),
    ]
    res = subprocess.run(cmd)
    return res.returncode == 0


def create_zip(team_name: str) -> Path:
    zip_filename = f"{team_name}_submission.zip"
    zip_path = ROOT.parent / zip_filename

    print(f"Creating submission archive: {zip_path}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. output/
        zf.write(OUTPUT_DIR / "matching_results.tsv", "output/matching_results.tsv")
        zf.write(OUTPUT_DIR / "candidate_pairs.tsv", "output/candidate_pairs.tsv")

        # 2. code/business_entity_resolution/
        if README_FILE.exists():
            zf.write(README_FILE, "code/business_entity_resolution/README.md")
        if REQ_FILE.exists():
            zf.write(REQ_FILE, "code/business_entity_resolution/requirements.txt")

        # Include src/
        for p in SRC_DIR.rglob("*.py"):
            rel = p.relative_to(SRC_DIR)
            zf.write(p, f"code/business_entity_resolution/src/{rel.as_posix()}")

        # Include utils/
        for p in UTILS_DIR.rglob("*.py"):
            rel = p.relative_to(UTILS_DIR)
            zf.write(p, f"code/business_entity_resolution/utils/{rel.as_posix()}")

        # 3. Documentation_template.md
        if DOC_FILE.exists():
            zf.write(DOC_FILE, "Documentation_template.md")

    print(f"Successfully packaged {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.2f} MB)")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Build submission package zip")
    parser.add_argument("--team-name", required=True, help="Your team name for the zip filename")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "dataset" / "test", help="Path to test files directory")
    args = parser.parse_args()

    test_dir = args.test_dir
    if not test_dir.exists() or not any(test_dir.iterdir()):
        print(f"Warning: {test_dir} is empty or not found. Falling back to samples for validation check.")
        test_dir = ROOT / "dataset" / "samples"

    print("Step 1: Validating submission files against rules...")
    if not validate_outputs(test_dir):
        print("\nPackaging aborted due to validation errors.")
        sys.exit(1)

    print("\nStep 2: Building submission archive...")
    zip_path = create_zip(args.team_name)
    print(f"\nALL DONE! Ready to submit {zip_path.name}")


if __name__ == "__main__":
    main()
