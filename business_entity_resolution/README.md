# Business Entity Resolution Challenge

ML pipeline that matches Source 2 / Source 3 business records to the deduplicated
Source 1 reference set, across US, India, and (test-only) France records.

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Data is **not** committed to the repo (too large, and the challenge rules keep
it out of git anyway). Drop the challenge files here so paths match `src/config.py`:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Each teammate keeps their own local copy under `dataset/` (gitignored).

## 2. Pipeline stages

```
src/
├── config.py        # paths, constants
├── io_utils.py       # TSV read/write, submission-format helpers
├── normalize.py      # name & address normalization (suffixes, abbreviations, unicode)
├── blocking.py        # candidate generation (Source1 x {Source2, Source3})
├── features.py        # pairwise similarity features for candidate pairs
├── model.py           # train/predict the match/no-match classifier
├── pipeline.py         # orchestrates: load -> normalize -> block -> featurize -> score -> emit
└── evaluate.py         # local F_0.5 macro-average scorer (train/val split only)
```

Run the whole thing end to end:

```bash
python3 -m src.pipeline --split train        # trains + validates on a held-out slice of train
python3 -m src.pipeline --split test         # produces output/matching_results.tsv + candidate_pairs.tsv
```

Validate the format before submitting (once `utils/validate_submission.py` is added
from the challenge kit):

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## 3. Constraints to keep front of mind

- **No external lookups** — no geocoding APIs, no registry lookups, no internet
  augmentation of any kind. Normalization/similarity must be self-contained.
- **Country is open-set** — test includes France, unseen in train. Nothing in
  the pipeline should branch on a hardcoded `{US, India}` set.
- **Precision-heavy metric (F_0.5, macro per Source-1 entity)** — singletons
  (no match) are a real class worth getting right, not a fallback.
- **Final model** must be MIT/Apache-2.0 licensed and ≤ 8B parameters.
- `candidate_pairs.tsv` must be the exact candidate set fed to the final
  model — a superset of `matching_results.tsv`.

## 4. Team workflow (4 people)

Branch model: `main` is protected (PR + 1 review to merge).

Suggested split, each on its own branch (`feat/<area>`):

| Area | Branch | Owns |
|---|---|---|
| Normalization + EDA | `feat/normalize` | `src/normalize.py`, `notebooks/eda.ipynb` |
| Blocking / candidate generation | `feat/blocking` | `src/blocking.py` |
| Features + model | `feat/model` | `src/features.py`, `src/model.py` |
| Evaluation + submission plumbing | `feat/eval` | `src/evaluate.py`, `src/io_utils.py`, integration in `src/pipeline.py` |

Workflow:
1. `git checkout -b feat/<area>`
2. Commit small, working increments — the interfaces in `src/pipeline.py` are
   the contract between areas, so keep function signatures stable and flag in
   PR description if you need to change one.
3. Open a PR into `main`, tag one teammate for review.
4. Rebase on `main` before merging to avoid big conflict resolutions.

Keep `dataset/` and `output/*.tsv` out of git (see `.gitignore`) — they're
huge and regenerable. Small sample slices for tests can live in
`dataset/samples/` if useful (a few hundred rows, hand-picked or `head -n`).

## 5. Opening in Antigravity

```bash
git clone <your-repo-url>
cd business_entity_resolution
```
Open the folder in Antigravity, then point agents at one `feat/*` branch /
module at a time so parallel agent runs don't collide on the same files.
