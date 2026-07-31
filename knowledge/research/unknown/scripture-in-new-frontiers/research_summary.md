## Ranked shortlist

### 1. Mapping-conditioned CatBoost ranker + full-corpus Qwen3 reranking — recommended

**Leak-free features/encodings:** Create one event/class pair for every global `moment_type`. Use raw biometrics, missing flags, parsed time, mapping-only trigger/activity/theme/delivery prototypes, signed and absolute trigger distances, current-and-past lags/deltas/rolling/EWM values, first-observed baseline deltas, peak-to-date/drawdown state, cumulative threshold exposure, and expected progress fit from outer-train sessions only. Mapping word/character TF-IDF is fit from mapping documents only. Keep `session_id`, candidate identity, `moment_type`, `assigned_verse_id`, verse text, and future-session aggregates out of predictors.

**Models + key hyperparameters:** CatBoostRanker `QuerySoftMax`, 900 iterations, depth 5, learning rate 0.025, L2 12, random strength 0.15, bagging temperature 0.4, early stopping 120. Inner LOGO compares only ranker-only and fixed 0.50/0.50 ranker/rules posteriors at temperature 1.0. For retrieval, rerank the full corpus when it has at most 64 rows using Qwen3-Reranker-4B; use Qwen3-Embedding-4B only to prune larger corpora. Qwen3’s official family provides 0.6B and 4B fallbacks and instruction-aware multilingual retrieval/reranking. ([GitHub][3])

**Expected runtime/memory:** 300–720 minutes end to end on RTX 3060 including shared retrieval initialization; batch 1, fp16, length 384, sequential model loading, approximately 8.5–11.5GB peak VRAM and under 10GB host RSS.

**Leakage risk:** candidate identity as a predictor; outer-validation labels entering blend choice; future-derived temporal summaries; assigned-reference contamination of retrieval; or label-bearing cache keys. Enforce outer/inner/nested LOGO and explicit forbidden-column assertions.

**Fallback:** Reject promotion below 0.6403741496598639 or when worst-session/unseen-class gates fail, then restore the exact deterministic rules posterior. For retrieval use Qwen3 0.6B, then BGE-M3/BGE reranker, then TF-IDF.

### 2. Direct causal CatBoost + rules + cross-fitted calibration/transition — strong_single challenger

**Leak-free features/encodings:** Same causal event frame without event/class expansion. `activity_type` stays a raw CatBoost categorical. Expected duration, priors, calibration, transitions, imputers, and any bins are fit on each outer-train partition and applied unchanged to held-out sessions. Cast categoricals to string before filling missing values.

**Models + key hyperparameters:** CatBoost MultiClass, 1200 iterations, depth 6, learning rate 0.02, L2 12, random strength 0.35, bagging temperature 0.5, balanced class weights, early stopping 150. Blend learned/rule probabilities 0.70/0.30. Fit a scalar temperature on complete inner LOGO and accept it only for ECE gain at least 0.01, no macro-F1 loss, and worst-session decline no worse than 0.03. Fit a forward-only transition matrix with additive smoothing 0.5 and strength 0.15; promote it only when grouped gates pass.

**Expected runtime/memory:** 60–180 incremental minutes after shared feature and retrieval caches; tree training is primarily CPU-bound and normally below 6GB VRAM.

**Leakage risk:** fold-unseen classes, globally fitted calibration/transitions, use of true previous labels during validation, or full-session aggregates. Apply the transition filter only to previous predicted posterior state.

**Fallback:** ExtraTrees with fold-fitted median/one-hot preprocessing only if CatBoost is unavailable. Reject calibration or transition independently when honest promotion gates fail.

### 3. Rules + BGE-M3/TF-IDF contract failsafe — reliability floor

**Leak-free features/encodings:** Organizer trigger thresholds, activity compatibility, current/past slopes, observed-range checks, novelty/cooldown state, word 1–2 gram TF-IDF, and character 3–5 gram TF-IDF. No target encoding or assigned-reference feature.

**Models + key hyperparameters:** Deterministic moment state machine; BGE-M3 full-corpus hybrid retrieval at length 256 and optional BGE reranking; TF-IDF capped at 12,000 features with sublinear TF; confidence gate 0.60 and 180-second cooldown. BGE-M3 supports dense, sparse, and multi-vector retrieval over more than 100 languages, making it the strongest broadly reproducible fallback. ([GitHub][2])

**Expected runtime/memory:** under 60 minutes with cached BGE assets; under 10 minutes and below 2GB RAM in TF-IDF-only mode.

**Leakage risk:** organizer replay can be circular because the supplied mapping and event labels share semantics. Label all retrieval scores as proxy evidence and keep them out of the official rubric estimate.

**Fallback:** deterministic rules + TF-IDF + fixed, clearly local safe phrases. This path is never represented as Gloo output and is selected only when learned routes fail validation or dependencies are unavailable.

## Recommendation

Promote pipeline 1 only when it beats the frozen rules floor and stability gates. Pipeline 2 supplies the strongest independent strong_single and calibration evidence. Pipeline 3 guarantees reproducibility and outage behavior. Spend no budget on a full seed × fold × model-family Cartesian product; spend it on one accurate cached retrieval pass, robust safety/API tests, live dual-API evidence, and a polished writeup/video package.
