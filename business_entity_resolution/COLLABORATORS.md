# Team Collaborators - Business Entity Resolution

Amazon Hackathon | September 2026

---

## Team Members

| Name | GitHub | Role |
|:-----|:-------|:-----|
| **Sanketh Egade** | [@SANKETHEGADE](https://github.com/SANKETHEGADE) | Team Lead - Pipeline Architect |
| **Shubham** | — | Blocking and Candidate Generation (`src/blocking.py`) |
| **Anwin** | — | Feature Engineering and Model (`src/features.py`, `src/model.py`) |
| **Tarun** | — | Normalization and EDA (`src/normalize.py`, `notebooks/`) |

---

## Branch Strategy

Each member works on their own feature branch and opens a PR into `main`.

| Area | Branch | Owns |
|:-----|:-------|:-----|
| Normalization + EDA | `feat/normalize` | `src/normalize.py`, `notebooks/` |
| Blocking / Candidate Generation | `feat/blocking` | `src/blocking.py` |
| Features + Model | `feat/model` | `src/features.py`, `src/model.py` |
| Evaluation + Submission | `feat/eval` | `src/evaluate.py`, `src/pipeline.py`, `utils/` |

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/SANKETHEGADE/business-entity-resolution.git
cd business-entity-resolution/business_entity_resolution

# 2. Install dependencies
pip install -r requirements.txt

# 3. Checkout your branch
git checkout -b feat/<your-area>
```

---

## How to Run

### Train locally on sample data (no dataset needed)

```bash
python -m src.pipeline --split train --sample
```

### Run on full dataset (after placing files in dataset/train/ and dataset/test/)

```bash
python -m src.pipeline --split train
python -m src.pipeline --split test
```

### Validate submission files

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Run unit tests

```bash
python -m pytest tests/
```

---

## Run on Free Cloud GPU (Google Colab / Kaggle)

1. Open `notebooks/run_on_colab.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **T4 GPU** (Runtime -> Change runtime type -> T4 GPU)
3. Run all cells: it will clone the repo, install deps, train, and generate submission files
4. Download `output/matching_results.tsv` for leaderboard upload

---

## Workflow Rules

- **Never commit to `main` directly** — always open a PR from your branch
- **Never commit dataset files** — `dataset/train/` and `dataset/test/` are gitignored
- **Never commit output files** — `output/*.tsv` are gitignored
- `dataset/samples/` — sample files for testing are committed and safe to use
- Keep function signatures in `src/pipeline.py` stable — it is the shared integration point

---

## Key Metric

We are optimized for **Macro F_0.5** (precision-weighted):

```
F_0.5 = (1.25 x Precision x Recall) / (0.25 x Precision + Recall)
```

- Precision is weighted **2x over Recall**
- Singletons (no matches) score **1.0** when correctly predicted as empty, **0.0** if any false match is added
- Candidate set size is also evaluated — **smaller is better**

---

## Repository

**https://github.com/SANKETHEGADE/business-entity-resolution**
