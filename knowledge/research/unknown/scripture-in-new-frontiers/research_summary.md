# Ranked pipeline shortlist

## 1. Causal CatBoost + cross-fitted calibration + Qwen3 retrieval/reranking

**Leak-free features/encodings:** raw heart rate, zone, effort, recovery, stress, session minute, parsed timestamp, missing flags, activity categorical, trigger-distance/interactions, and within-session past-only lags, deltas, accelerations, rolling/EWM summaries, crossings, elapsed time, and online phase. `session_id` is split metadata only. Exclude `moment_type`, `assigned_verse_id`, translation, verse text, future rows, and full-session aggregates. CatBoost receives `activity_type` as a string categorical; all priors, rule probabilities, transition matrices, and calibration parameters are fit within the outer-train fold.

**Models and concrete hyperparameters:** CatBoost MultiClass, 1000 iterations, depth 6, learning rate 0.025, L2 10, random strength 0.5, bagging temperature 0.5, balanced class weights, early stopping 150. Blend 0.70 learned + 0.30 rules; transition strength 0.18 with smoothing 0.5. Inner-LOGO scalar temperature in [0.5, 5.0] plus prior-logit adjustment 0.25, promoted only for ECE gain ≥0.01 with no macro-F1 loss. Retrieval uses Qwen3-Embedding-4B, top 12, then Qwen3-Reranker-4B over top 8 with sequential fp16 loading at length 384 and batch 1. The official Qwen card supports the 4B/0.6B family, instruction-aware retrieval, and the yes/no reranker formulation. ([Hugging Face][3])

**Expected runtime/memory:** roughly 240–720 minutes on RTX 3060; 8.5–11.5GB peak VRAM with sequential loading, cached embeddings/pair scores, and under 10GB host RSS.

**Leakage risk:** future session information, calibration on outer validation labels, global priors/transitions, assigned-reference use in retrieval queries or cache keys, and full-data backend selection. **Fallback:** shorten/chunk first, then 8/4-bit, Qwen3 0.6B, BGE-M3, TF-IDF; ExtraTrees only if CatBoost is unavailable.

## 2. XGBoost temporal challenger + shared retrieval cache

**Leak-free features/encodings:** identical causal frame; numeric median imputation and missing indicators are fit on each fold; `activity_type` uses fold-fit one-hot encoding with `handle_unknown="ignore"`. `align_features` adds absent columns as NA and ignores test-only extras.

**Models and concrete hyperparameters:** XGBoost `multi:softprob`, 900 trees, depth 4, learning rate 0.025, subsample 0.85, column sample 0.80, min child weight 2, alpha 0.15, lambda 6, histogram tree method, CUDA device, early stopping 100. Inner-LOGO scalar temperature uses the same [0.5, 5.0] promotion gates. Reuse the selected Qwen/BGE/TF-IDF caches. Evaluate a fixed 0.50/0.50 CatBoost–XGBoost probability blend only; promote for ≥0.005 grouped macro-F1 gain, worst-session drop no worse than 0.03, and no per-class recall collapse above 0.20.

**Expected runtime/memory:** 60–180 incremental minutes after shared retrieval caches; usually below 6GB VRAM and 8GB host RAM.

**Leakage risk:** globally fit one-hot/imputation, sparse matrix densification, early stopping against labels outside the fold, and blend selection on non-OOF predictions. **Fallback:** CUDA-to-CPU XGBoost, then sparse ExtraTrees; HistGradientBoosting only after a dense-memory guard.

## 3. Deterministic rules + BGE-M3/TF-IDF safety floor

**Leak-free features/encodings:** organizer threshold catalog, activity compatibility, current/past slopes, observed-range checks, novelty, cooldown, word 1–2 grams, and character 3–5 grams. No target encoder or future feature is permitted.

**Models and concrete hyperparameters:** deterministic state machine; BGE-M3 at length 256, embedding batch 4, reranker batch 2, first-stage top 10, rerank top 8; TF-IDF `min_df=1`, `max_features=12000`, sublinear TF. BGE-M3 supports dense, sparse, and multi-vector retrieval and is therefore a credible hybrid fallback. ([arXiv][2])

**Expected runtime/memory:** under 60 minutes with cached BGE; under 10 minutes and 2GB RAM for TF-IDF-only.

**Leakage risk:** circular-looking replay scores because organizer trigger/mapping semantics are close to labels. Do not present replay as real-user effectiveness. **Fallback:** rules + TF-IDF + fixed, clearly labeled non-generative phrases.

**Recommendation:** promote pipeline 1, retain pipeline 2 only for diversity/blending, and require pipeline 3 for outage and contract reliability. The first engineering action is to repair the plan/kernel hash and candidate-name drift, then implement the planned cross-fitted calibration before any tuning.
