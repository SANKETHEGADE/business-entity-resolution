# Business Entity Resolution — Methodology

## Team

- Names / roles:
- Repo:

## 1. Overview

<!-- One paragraph: what the pipeline does end to end. -->

## 2. Candidate Generation / Blocking Strategy

- Blocking key(s) used:
- Why this choice (recall ceiling vs. candidate-set size trade-off):
- Measured recall on train (via `src/blocking.measure_recall`):
- Reduction ratio (candidates considered / full cross-join size):
- Handling of France (unseen-in-train country) at the blocking stage:

## 3. Normalization

- Name normalization steps:
- Address normalization steps:
- Transliteration / non-Latin script handling:
- Country-agnostic design notes (how the open country-set requirement was respected):

## 4. Model Architecture & Feature Engineering

- Features used (list + brief rationale for each):
- Model type, license, parameter count (must be MIT/Apache-2.0, <= 8B params):
- Training data construction (how positive/negative pairs were derived from
  `train_ground_truth.tsv`, how candidate pairs *not* in ground truth were
  treated as negatives):
- Threshold selection method (how it was tuned against macro F_0.5):
- Per-entity prediction policy (single global threshold vs. relative/top-k
  policy per Source-1 entity, and why):

## 5. Validation

- Train/validation split strategy:
- Held-out macro F_0.5:
- Precision / recall breakdown, singleton accuracy specifically:
- Error analysis (representative false positives / false negatives, patterns noticed):

## 6. Fair Play

- Confirm no external data, APIs, or lookups were used anywhere in the pipeline:
- Confirm final model license and parameter count:

## 7. Reproduction

```bash
# exact commands to regenerate output/matching_results.tsv and output/candidate_pairs.tsv
```

## 8. Known Limitations / Future Work
