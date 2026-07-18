## Ranked shortlist

### 1. Causal CatBoost + rule gate + Qwen3 dual-reranker cascade

**Leak-free features/encodings:** raw heart rate/zone/effort/recovery/stress/minute, parsed timestamp, missing indicators, activity categorical, static trigger distances, interactions, and within-session current/past lags, deltas, acceleration, rolling means/std, EWM, and threshold crossings. `session_id` is group metadata; `moment_type`, `assigned_verse_id`, translation, verse fields, future rows, and full-session aggregates are excluded. CatBoost receives string categoricals with `"Unknown"` filled safely. Transition counts are fitted inside each fold.

**Models and concrete settings:** CatBoost multiclass with 1,000 iterations, depth 6, learning rate 0.025, L2 10, balanced class weights, early stopping 150; 70/30 learned/rule probability blend; optional transition strength 0.18 only after OOF promotion. Qwen3-Embedding-4B first stage, word/char TF-IDF and structured scores, top-12 candidates, then top-8 reranking. Compare Qwen3-Reranker-4B and Querit-4B only through nested Leave-One-Session-Out retrieval selection. Qwen3’s official cards support instruction-aware 4B embedding/reranking and Apache-2.0 licensing; Querit is a newer Apache-2.0 4B cross-encoder challenger. ([Hugging Face][1])

**Expected runtime/memory:** 240–720 minutes end to end on RTX 3060. Sequential 4B loading, FP16, batch 1, 384 tokens, cached corpus/query embeddings and pair scores; target peak 8.5–11.5GB VRAM and <10GB host RSS.

**Leakage risk:** high if assigned references select retrieval weights/backends on the same session, if temporal features use future rows, or if rule/transition statistics are fitted globally. The nested group protocol and fold-local transforms are mandatory.

**Fallback:** Qwen3 4B at shorter lengths → supported 8/4-bit → Qwen3 0.6B → BGE-M3/BGE reranker → TF-IDF. CatBoost absence falls back to ExtraTrees, but that fallback cannot be presented as equal technical depth.

### 2. XGBoost temporal challenger + shared retrieval

**Leak-free features/encodings:** identical causal frame, with fold-fitted numeric median imputation, missing indicators, and `OneHotEncoder(handle_unknown="ignore")` for activity. Train-only columns are dropped; absent test columns are added as NA by `align_features`.

**Models and concrete settings:** XGBoost `multi:softprob`, 900 estimators, depth 4, learning rate 0.025, subsample 0.85, column sample 0.80, min child weight 2, alpha 0.15, lambda 6, histogram tree method, CUDA with CPU retry, early stopping 100. Reuse the selected retrieval caches. Evaluate a fixed 50/50 probability blend with CatBoost and promote only for at least +0.005 grouped macro-F1 without >0.03 worst-session loss.

**Expected runtime/memory:** 60–180 incremental minutes after shared caches; normally <6GB GPU memory and <8GB RAM.

**Leakage risk:** one-hot category fitting outside folds, validation-only target classes, accidental sparse-to-dense conversion, and blend selection on non-OOF predictions.

**Fallback:** CPU XGBoost, then sparse ExtraTrees or HistGradientBoosting only when safe to densify. Reject the challenger when it lacks stable grouped gain.

### 3. Rules + BGE-M3/TF-IDF contract failsafe

**Leak-free features/encodings:** organizer trigger catalog, activity compatibility, current/past slopes, observed-range checks, cooldown/novelty state, word 1–2 grams, and character 3–5 grams. No learned target encoder.

**Models and concrete settings:** deterministic state machine; BGE-M3 dense/sparse/multi-vector retrieval when an immutable local asset and `FlagEmbedding` are available; BGE reranker or first-stage scoring; otherwise 12,000-feature word/character TF-IDF. Delivery confidence 0.60, cooldown 180 seconds, no generated words in the pure outage path. BGE-M3’s primary paper describes unified dense, sparse, and multi-vector retrieval across 100+ languages, making it a defensible fallback rather than a generic lexical baseline. ([arXiv][7])

**Expected runtime/memory:** <60 minutes with BGE caches; <10 minutes and <2GB RAM for TF-IDF-only mode.

**Leakage risk:** circularity from organizer mapping labels and thresholds, plus misleadingly strong replay metrics. Report it as an organizer-proxy floor and keep a rules-only ablation.

**Fallback:** deterministic rules + TF-IDF + fixed safe phrases. This path is for reliability and contract validation, not the preferred final product unless learned routes fail honest validation.

## Recommendation

Promote pipeline 1, retain pipeline 2 for diversity, and require pipeline 3 as the outage/sanity floor. The decisive improvement over the current implementation is not another broad hyperparameter sweep; it is nested session-level retrieval evaluation, immutable asset locking, and visibly real dual-API evidence. Keep the JITAI-style timing gate and MRT-compatible ledger as technical depth, while explicitly avoiding effectiveness or medical claims. ([arXiv][3])
