"""Accuracy Reranker: Second-stage reranking and confidence scoring module.

Owner: Accuracy / Reranking / Tuning module.
Implements:
1. Second-stage reranker that takes candidates and their baseline scores,
   combining them with enhanced pairwise features.
2. Supports ML-based reranking (LightGBM/LogisticRegression on top of baseline),
   weighted ensemble score combination, or hybrid stacking.
3. Conservative false-positive filtering to prevent false merges:
   - Mismatched business name penalty (blocks city/address-only collisions)
   - Conflicting postal/PIN code penalty
   - Conflicting street number penalty
   - Conflicting country hard drop
4. Singleton protection:
   - High confidence hurdle: entities where no candidate reaches the hurdle
     are emitted as empty lists [], earning full 1.0 credit for singletons.
   - Calibrated relative margin pruning.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from .accuracy_features import (
    AccuracyFeatureExtractor,
    ACCURACY_FEATURE_NAMES,
    compute_pair_accuracy_features,
)


@dataclass
class AccuracyReranker:
    """Second-stage reranking model on top of the baseline entity resolution model."""

    mode: str = "ml"  # "ml", "ensemble", or "hybrid"
    threshold: float = 0.65
    margin: float = 0.12
    confidence_hurdle: float = 0.70
    blend_alpha: float = 0.40  # weight on baseline probability in hybrid/ensemble mode
    clf: Optional[Union[lgb.LGBMClassifier, LogisticRegression]] = None
    feature_extractor: Optional[AccuracyFeatureExtractor] = field(default_factory=AccuracyFeatureExtractor)

    def __post_init__(self):
        if self.clf is None:
            # Conservative, well-regularized tree reranker that refines probabilities
            self.clf = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=4,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
                verbose=-1,
            )

    def to_matrix(self, feature_dicts: List[dict]) -> np.ndarray:
        """Convert list of feature dicts to 2D numpy array with fixed column order."""
        return np.array(
            [[fd.get(k, 0.0) for k in ACCURACY_FEATURE_NAMES] for fd in feature_dicts],
            dtype=np.float32,
        )

    def fit(
        self,
        feature_dicts: List[dict],
        labels: List[int],
    ) -> "AccuracyReranker":
        """Fit second-stage reranker on candidate pairs with baseline scores."""
        X = self.to_matrix(feature_dicts)
        y = np.array(labels, dtype=int)
        self.clf.fit(X, y)
        return self

    def _apply_fp_penalties(
        self,
        raw_prob: float,
        fd: dict,
    ) -> float:
        """Apply conservative false-positive filtering rules to avoid false merges.

        Key insight:
        In macro F0.5, false positives are 4x more costly than false negatives.
        If business names have almost zero similarity AND addresses also fail to match,
        the candidate is a false positive (e.g. sharing a common city/state word).
        However, if the address has strong numeric/street overlap, it is likely an
        alternate name or script variation and should NOT be penalized.
        """
        score = raw_prob

        # Hard guard: Conflicting known countries (e.g. US vs India)
        if fd.get("country_conflict", 0.0) > 0.5:
            return 0.0

        # Guard 1: Severe Dual Mismatch (both name AND address fail to match)
        name_sim = max(
            fd.get("name_exact_match", 0.0),
            fd.get("name_base_exact_match", 0.0),
            fd.get("name_levenshtein_sim", 0.0),
            fd.get("name_base_levenshtein_sim", 0.0),
            fd.get("name_token_jaccard", 0.0),
            fd.get("name_is_substring", 0.0) * 0.7,
        )
        addr_sim = max(
            fd.get("addr_exact_match", 0.0),
            fd.get("addr_levenshtein_sim", 0.0),
            fd.get("addr_token_jaccard", 0.0),
            min(1.0, fd.get("addr_numeric_overlap_count", 0.0)),
        )

        # Complete mismatch on both name and address components
        if name_sim < 0.35 and addr_sim < 0.35:
            return 0.0

        # Guard 2: Postal / PIN code conflict
        postal_exact = fd.get("addr_postal_exact_match", 0.5)
        has_postals = fd.get("addr_has_postal_a", 0.0) > 0.5 and fd.get("addr_has_postal_b", 0.0) > 0.5
        name_exact = fd.get("name_exact_match", 0.0) > 0.5 or fd.get("name_base_exact_match", 0.0) > 0.5
        if has_postals and postal_exact < 0.2 and not name_exact:
            score *= 0.50

        # Guard 3: Street number conflict (different building/house numbers on same road)
        street_match = fd.get("addr_street_num_match", 0.5)
        if street_match < 0.2 and not name_exact and addr_sim < 0.60:
            score *= 0.75

        return float(max(0.0, min(1.0, score)))

    def predict_pair_scores(self, feature_dicts: List[dict]) -> np.ndarray:
        """Compute rescaled and filtered reranker scores for a list of candidate pairs."""
        if not feature_dicts:
            return np.array([], dtype=float)

        X = self.to_matrix(feature_dicts)
        ml_probs = self.clf.predict_proba(X)[:, 1]

        scores = []
        for i, fd in enumerate(feature_dicts):
            p_base = fd.get("baseline_prob", 0.5)
            p_ml = float(ml_probs[i])

            if self.mode == "ml":
                # Blended ML probability to anchor on baseline stability while leveraging new features
                combined = self.blend_alpha * p_base + (1.0 - self.blend_alpha) * p_ml
            elif self.mode == "ensemble":
                # Linear blend of baseline + ML probability + weighted feature components
                name_comp = fd.get("name_base_levenshtein_sim", 0.0) * 0.5 + fd.get("name_token_jaccard", 0.0) * 0.5
                addr_comp = fd.get("addr_levenshtein_sim", 0.0) * 0.5 + fd.get("addr_token_jaccard", 0.0) * 0.5
                combined = (
                    0.50 * p_base
                    + 0.30 * p_ml
                    + 0.12 * name_comp
                    + 0.08 * addr_comp
                )
            else:  # "hybrid"
                combined = self.blend_alpha * p_base + (1.0 - self.blend_alpha) * p_ml

            # Apply anti-false-merge guards
            filtered_score = self._apply_fp_penalties(combined, fd)
            scores.append(filtered_score)

        return np.array(scores, dtype=float)


    def rerank_entity_candidates(
        self,
        entity_cands_with_features: Dict[str, List[Tuple[str, dict]]],
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Rerank and score candidates grouped by Source 1 entity ID."""
        entity_scores: Dict[str, List[Tuple[str, float]]] = {}

        # Flatten pairs for fast batch prediction
        all_dicts = []
        pair_tracking = []
        for s1_id, cands in entity_cands_with_features.items():
            for cand_id, fd in cands:
                all_dicts.append(fd)
                pair_tracking.append((s1_id, cand_id))

        if not all_dicts:
            return {s1_id: [] for s1_id in entity_cands_with_features}

        batch_scores = self.predict_pair_scores(all_dicts)

        idx = 0
        for s1_id, cand_id in pair_tracking:
            score = float(batch_scores[idx])
            entity_scores.setdefault(s1_id, []).append((cand_id, score))
            idx += 1

        # Sort each entity's candidates descending by reranked score
        for s1_id in entity_scores:
            entity_scores[s1_id].sort(key=lambda x: x[1], reverse=True)

        # Include entities that had 0 candidates
        for s1_id in entity_cands_with_features:
            if s1_id not in entity_scores:
                entity_scores[s1_id] = []

        return entity_scores

    def predict_for_entities(
        self,
        entity_candidates_scores: Dict[str, List[Tuple[str, float]]],
        threshold: Optional[float] = None,
        margin: Optional[float] = None,
        confidence_hurdle: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """Singleton-Protected Calibrated Margin Decision Policy.

        1. Absolute Confidence Hurdle (Singleton Protection):
           If max score for an entity < confidence_hurdle, immediately emit []
           (ensures 1.0 credit for singletons, avoiding catastrophic 0.0 false merge penalty).
        2. Calibrated Margin Selection:
           For entities passing the hurdle, retain all candidates satisfying:
             score >= threshold AND score >= (max_score - margin).
        """
        th = self.threshold if threshold is None else threshold
        mg = self.margin if margin is None else margin
        hurdle = self.confidence_hurdle if confidence_hurdle is None else confidence_hurdle

        predictions: Dict[str, List[str]] = {}

        for s1_id, cand_scores in entity_candidates_scores.items():
            if not cand_scores:
                predictions[s1_id] = []
                continue

            max_score = max(score for _, score in cand_scores)

            # Singleton protection hurdle
            if max_score < hurdle:
                predictions[s1_id] = []
                continue

            # Retain candidates passing threshold within relative margin OR with high absolute confidence (>= 0.70)
            selected = [
                cand_id
                for cand_id, score in cand_scores
                if score >= th and (score >= (max_score - mg) or score >= 0.70)
            ]
            predictions[s1_id] = selected

        return predictions


def run_accuracy_reranker_train(
    train_s1,
    val_s1,
    s2,
    s3,
    train_candidates: Dict[str, Set[str]],
    val_candidates: Dict[str, Set[str]],
    train_base_probs: Dict[Tuple[str, str], float],
    val_base_probs: Dict[Tuple[str, str], float],
    ground_truth: Dict[str, Set[str]],
    val_gt: Dict[str, Set[str]],
):
    """Bridge for integrating accuracy layer into pipeline.run_train."""
    from .threshold_tuning import ThresholdTuner

    other_records = {}
    for df in [s2, s3]:
        for row in df.itertuples(index=False):
            other_records[getattr(row, "entity_id")] = {
                "name": getattr(row, "business_name"),
                "address": getattr(row, "business_address"),
                "country": getattr(row, "country"),
            }

    extractor = AccuracyFeatureExtractor()
    all_names = list(train_s1["business_name"]) + list(val_s1["business_name"]) + list(s2["business_name"]) + list(s3["business_name"])
    all_addrs = list(train_s1["business_address"]) + list(val_s1["business_address"]) + list(s2["business_address"]) + list(s3["business_address"])
    extractor.fit_tfidf(all_names, all_addrs)
    s1_all = {
        getattr(r, "entity_id"): {
            "name": getattr(r, "business_name"),
            "address": getattr(r, "business_address"),
            "country": getattr(r, "country"),
        }
        for df in [train_s1, val_s1]
        for r in df.itertuples(index=False)
    }
    extractor.precompute_vectors({**s1_all, **other_records})

    # Featurize train and val pairs
    def _build_features(s1_df, cands, prob_map, is_train=True):
        records = {}
        for r in s1_df.itertuples(index=False):
            records[getattr(r, "entity_id")] = {
                "name": getattr(r, "business_name"),
                "address": getattr(r, "business_address"),
                "country": getattr(r, "country"),
            }
        fd_list = []
        lbl_list = []
        entity_cands = {}
        for sid in s1_df["entity_id"]:
            s_meta = records[sid]
            clist = list(cands.get(sid, []))
            entity_cands[sid] = []
            true_set = ground_truth.get(sid, set()) if ground_truth else set()
            for cid in clist:
                cmeta = other_records.get(cid)
                if not cmeta:
                    continue
                p = prob_map.get((sid, cid), 0.5)
                fd = extractor.extract_pair(
                    name_a=s_meta["name"],
                    addr_a=s_meta["address"],
                    country_a=s_meta["country"],
                    name_b=cmeta["name"],
                    addr_b=cmeta["address"],
                    country_b=cmeta["country"],
                    cand_id=cid,
                    s1_id=sid,
                    baseline_prob=p,
                )
                fd_list.append(fd)
                entity_cands[sid].append((cid, fd))
                if is_train:
                    lbl_list.append(1 if cid in true_set else 0)
        return fd_list, lbl_list, entity_cands

    print("\n[Accuracy Layer] Extracting enhanced features for training & validation pairs...")
    X_tr_r, y_tr_r, _ = _build_features(train_s1, train_candidates, train_base_probs, is_train=True)
    X_val_r, y_val_r, val_cands_with_feats = _build_features(val_s1, val_candidates, val_base_probs, is_train=False)

    reranker = AccuracyReranker(mode="ml", blend_alpha=0.50, feature_extractor=extractor)
    reranker.fit(X_tr_r, y_tr_r)

    val_rerank_scores = reranker.rerank_entity_candidates(val_cands_with_feats)

    tuner = ThresholdTuner()
    best_res, _ = tuner.tune(
        lambda th, mg, hurdle: reranker.predict_for_entities(
            val_rerank_scores, threshold=th, margin=mg, confidence_hurdle=hurdle
        ),
        ground_truth=val_gt,
        verbose=True,
    )

    reranker.threshold = best_res.threshold
    reranker.confidence_hurdle = best_res.confidence_hurdle
    reranker.margin = best_res.margin

    return reranker, best_res


def run_accuracy_reranker_test(
    s1,
    s2,
    s3,
    candidates: Dict[str, Set[str]],
    test_base_probs: Dict[Tuple[str, str], float],
    reranker: Optional[AccuracyReranker] = None,
    train_s1=None,
    train_candidates=None,
    train_base_probs=None,
    ground_truth=None,
) -> Dict[str, List[str]]:
    """Bridge for integrating accuracy layer into pipeline.run_test."""
    other_records = {}
    for df in [s2, s3]:
        for row in df.itertuples(index=False):
            other_records[getattr(row, "entity_id")] = {
                "name": getattr(row, "business_name"),
                "address": getattr(row, "business_address"),
                "country": getattr(row, "country"),
            }

    s1_records = {}
    for row in s1.itertuples(index=False):
        s1_records[getattr(row, "entity_id")] = {
            "name": getattr(row, "business_name"),
            "address": getattr(row, "business_address"),
            "country": getattr(row, "country"),
        }

    extractor = reranker.feature_extractor if (reranker and reranker.feature_extractor) else AccuracyFeatureExtractor()
    if extractor.name_vectorizer is None:
        all_names = list(s1["business_name"]) + list(s2["business_name"]) + list(s3["business_name"])
        all_addrs = list(s1["business_address"]) + list(s2["business_address"]) + list(s3["business_address"])
        if train_s1 is not None:
            all_names += list(train_s1["business_name"])
            all_addrs += list(train_s1["business_address"])
        extractor.fit_tfidf(all_names, all_addrs)
        extractor.precompute_vectors({**s1_records, **other_records})

    if reranker is None:
        reranker = AccuracyReranker(mode="ml", threshold=0.35, confidence_hurdle=0.45, margin=0.10, feature_extractor=extractor)
        if train_s1 is not None and train_candidates is not None and train_base_probs is not None:
            print("[Accuracy Layer] Training production reranker on training data...")
            tr_s1_records = {}
            for r in train_s1.itertuples(index=False):
                tr_s1_records[getattr(r, "entity_id")] = {
                    "name": getattr(r, "business_name"),
                    "address": getattr(r, "business_address"),
                    "country": getattr(r, "country"),
                }
            tr_fd_list = []
            tr_lbl_list = []
            for sid in train_s1["entity_id"]:
                sm = tr_s1_records[sid]
                true_set = ground_truth.get(sid, set()) if ground_truth else set()
                for cid in train_candidates.get(sid, []):
                    cm = other_records.get(cid)
                    if not cm:
                        continue
                    p = train_base_probs.get((sid, cid), 0.5)
                    fd = extractor.extract_pair(
                        name_a=sm["name"], addr_a=sm["address"], country_a=sm["country"],
                        name_b=cm["name"], addr_b=cm["address"], country_b=cm["country"],
                        cand_id=cid, s1_id=sid, baseline_prob=p,
                    )
                    tr_fd_list.append(fd)
                    tr_lbl_list.append(1 if cid in true_set else 0)
            reranker.fit(tr_fd_list, tr_lbl_list)

    test_cands_with_feats: Dict[str, List[Tuple[str, dict]]] = {}
    for sid in s1["entity_id"]:
        s_meta = s1_records[sid]
        clist = list(candidates.get(sid, []))
        test_cands_with_feats[sid] = []
        for cid in clist:
            cmeta = other_records.get(cid)
            if not cmeta:
                continue
            p = test_base_probs.get((sid, cid), 0.5)
            fd = extractor.extract_pair(
                name_a=s_meta["name"],
                addr_a=s_meta["address"],
                country_a=s_meta["country"],
                name_b=cmeta["name"],
                addr_b=cmeta["address"],
                country_b=cmeta["country"],
                cand_id=cid,
                s1_id=sid,
                baseline_prob=p,
            )
            test_cands_with_feats[sid].append((cid, fd))

    test_rerank_scores = reranker.rerank_entity_candidates(test_cands_with_feats)
    final_matches = reranker.predict_for_entities(
        test_rerank_scores,
        threshold=reranker.threshold,
        margin=reranker.margin,
        confidence_hurdle=reranker.confidence_hurdle,
    )
    return final_matches


