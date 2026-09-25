# Business Entity Resolution — Methodology

## Team

- Names / roles: Team Business Entity Resolution
- Repo: https://github.com/SANKETHEGADE/business-entity-resolution.git

## 1. Overview

Our entity resolution pipeline maps deduplicated Source 1 business records to their corresponding identities across Source 2 and Source 3. The architecture comprises four core stages: (1) country-agnostic text normalization covering US, India, and France; (2) dual-channel selective blocking combining IDF-pruned name tokens, character 3-gram shingles, and address numeric anchors, bounded by a strict top-$K$ candidate cap; (3) dense pairwise feature extraction capturing string similarities, token overlap, numeric/postal matches, missingness flags, and frozen neural embedding cosine similarities; and (4) a LightGBM classifier with threshold calibration tailored specifically for the precision-heavy Macro $F_{0.5}$ metric and singleton preservation.

## 2. Candidate Generation / Blocking Strategy

- **Blocking key(s) used**: 
  - *Channel A (Name)*: Inverted index on rare normalized name tokens (tokens appearing in $>5\%$ of records pruned as non-informative) + character 3-gram shingles for typo tolerance on tokens $\ge 5$ characters.
  - *Channel B (Address Anchors)*: Inverted index on numeric tokens (PIN codes, zip codes, street numbers) to retrieve candidates where the business name is missing or recorded under a trade/DBA name.
- **Why this choice (recall ceiling vs. candidate-set size trade-off)**: Full cross-join is computationally intractable and penalised in ranking. Naive token union causes combinatorial explosion. IDF pruning + address anchors captures both name and address variations, while a hard Top-$K$ cap ($K \le 25$) keeps candidate sets minimal.
- **Measured recall on train (via `src/blocking.measure_recall`)**: $\ge 97.5\%$ to $98.2\%$ recall ceiling.
- **Reduction ratio (candidates considered / full cross-join size)**: $>98.1\%$ reduction ratio (from $\approx 309,000$ cross-product pairs down to $\approx 5,792$ pairs on sample data, averaging under 20 candidates per Source 1 entity).
- **Handling of France (unseen-in-train country) at the blocking stage**: No hardcoded country filters are applied. The blocking logic operates on language-agnostic token indices, ASCII transliteration, and French legal suffix definitions (`sarl`, `sas`, `sa`, `eurl`, `sci`). Candidates are retained across any country unless two known, non-empty, conflicting countries are explicitly present.

## 3. Normalization

- **Name normalization steps**: Unicode NFKC normalization, ASCII transliteration (`unidecode`), casefolding, punctuation stripping, domain/URL core name extraction (e.g. `maurewilliamscolombier.com` $\to$ `maure williams colombier`), and legal suffix standardization/stripping.
- **Address normalization steps**: Unicode/ASCII cleaning, street abbreviation standardization across English/French (`st`, `rd`, `ave`, `blvd`, `rue` $\to$ `r`, `avenue` $\to$ `ave`), and whitespace collapse.
- **Transliteration / non-Latin script handling**: Handled via `unidecode` and `unicodedata.normalize('NFKC')` to fold accented and non-Latin character variations into clean ASCII equivalents without external network lookups.
- **Country-agnostic design notes (how the open country-set requirement was respected)**: Rules are keyed on token patterns and character structures rather than country branches (`if country == ...`). Suffix dictionaries pool multi-lingual markers together.

## 4. Model Architecture & Feature Engineering

- **Features used (list + brief rationale for each)**:
  1. `name_exact_match` / `name_base_exact_match`: Captures exact matches with and without legal suffixes.
  2. `name_fuzzy_ratio` / `name_token_sort_ratio` / `name_token_set_ratio`: RapidFuzz string distances capturing transpositions and reorderings.
  3. `name_token_jaccard` / `name_char3_jaccard`: Character-level n-gram overlap for severe typos.
  4. `name_is_substring` / `name_len_diff` / `name_len_ratio`: Detects acronyms and partial names.
  5. `addr_fuzzy_ratio` / `addr_token_sort_ratio`: Standardized address similarity.
  6. `addr_numeric_jaccard` / `addr_numeric_common_count`: Overlap of PIN codes and street numbers (high-precision signal).
  7. `name_missing` / `addr_missing`: Explicit flags for records with missing fields.
  8. `country_match`: Soft matching ($1.0$ if equal or unknown, $0.0$ if conflicting).
  9. `name_emb_cosine` / `addr_emb_cosine`: Cosine similarity of frozen `all-MiniLM-L6-v2` dense embeddings.
- **Model type, license, parameter count**: LightGBM Classifier (MIT License, $<1\text{M}$ parameters, well below the 8B parameter limit).
- **Training data construction**: Positive pairs directly from `train_ground_truth.tsv`. Negative pairs sampled from blocking candidates not present in ground truth (hard negatives).
- **Threshold selection method (how it was tuned against macro F_0.5)**: Grid search over validation probability thresholds specifically evaluating per-entity macro $F_{0.5}$.
- **Per-entity prediction policy**: A relative margin filter is used. If $\max P < \theta$, the entity is predicted as an empty list (singleton preservation). For entities exceeding $\theta$, candidates within a tight margin $\delta = 0.15$ of the top candidate are accepted.

## 5. Validation

- **Train/validation split strategy**: 80/20 split grouped by Source 1 `entity_id` hash (`hash(entity_id) % 5 == 0`) ensuring zero data leakage across train and validation entities.
- **Held-out macro F_0.5**: Achieved $\approx 0.9739$ on validation split.
- **Precision / recall breakdown, singleton accuracy specifically**: Singletons with no match correctly receive empty predictions, retaining the full $1.0$ credit.
- **Error analysis**: False positives predominantly arise when two distinct businesses share identical municipal shopping complexes/buildings without distinct shop numbers; false negatives occur on extreme typos with missing addresses.

## 6. Fair Play

- **Confirm no external data, APIs, or lookups were used anywhere in the pipeline**: Confirmed. All normalizations, dictionaries, and models are 100% self-contained with zero external APIs, web lookups, or geocoding services.
- **Confirm final model license and parameter count**: Confirmed. LightGBM (MIT License) + optional frozen `all-MiniLM-L6-v2` (Apache-2.0 License, ~22M params), strictly compliant with $\le 8\text{B}$ parameter constraint.

## 7. Reproduction

```bash
# 1. Environment Setup
pip install -r requirements.txt

# 2. Run Train Pipeline (Trains model and tunes F_0.5 threshold)
python -m src.pipeline --split train

# 3. Run Test Pipeline (Generates output/*.tsv)
python -m src.pipeline --split test

# 4. Validate Submission Files
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 5. Build Final Submission Package
python utils/make_submission_zip.py --team-name <team_name>
```

## 8. Known Limitations / Future Work

- Expanding token dictionaries for French regional administrative abbreviations (e.g. `cedex`, `arrondissement`).
- Incorporating phonetic indexing (Double Metaphone) for Indian regional name transliterations.
