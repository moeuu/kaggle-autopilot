## Ranked candidate pipeline shortlist

### 1. Hybrid CatBoost + rule gate + BGE-M3 retrieval + required APIs

**Leak-free features:** raw biometrics; past-only heart-rate, effort, and stress deltas; rolling statistics; recovery deficit; effort-zone interactions; activity category; missing indicators. `session_id` is group-only, and translation is used only after moment prediction. All imputers and encoders are fold-fitted.

**Models and configuration:** CatBoost MultiClass, 700 iterations, depth 6, learning rate 0.035, balanced class weights, 80-round early stopping; probabilities blended 0.70 learned / 0.30 deterministic rule model. Retrieve top eight with BGE-M3 fp16, maximum length 256, batch size 8, plus TF-IDF lexical score. Gloo output is capped at 22 words and schema-validated.

**Runtime/memory:** approximately 30–90 minutes for three-seed grouped CV after embeddings are cached; approximately 3–5 GB VRAM.

**Leakage risk:** low under Leave-One-Session-Out. Main hazards are future-derived session features, accidental use of `assigned_verse_id`, translation-to-moment correlation, and fold-global preprocessing.

**Fallback:** CatBoost plus dynamic threshold rules and train-safe TF-IDF retrieval. This is the recommended final pipeline. BGE-M3’s model card supports multilingual hybrid retrieval and identifies an MIT license. ([Hugging Face][3])

### 2. XGBoost temporal challenger + identical retrieval stack

**Leak-free features:** the same temporal frame, but with fold-fitted one-hot activity encoding and explicit missing indicators.

**Models and configuration:** XGBoost `multi:softprob`, 900 estimators, depth 4, learning rate 0.025, subsample 0.85, column sample 0.80, `reg_lambda=6`, and histogram CUDA training. Reuse the cached BGE-M3 and lexical retrieval features.

**Runtime/memory:** approximately 35–100 minutes; normally under 4 GB VRAM.

**Leakage risk:** low if the encoder is fit inside every fold, but fold-local absent classes and sparse categories can destabilize probabilities.

**Fallback:** CPU XGBoost, then sklearn HistGradientBoosting or ExtraTrees. Blend with CatBoost only when OOF macro-F1 improves by at least 0.005 without materially weakening the worst session.

### 3. Rule-state machine + word/character TF-IDF

**Leak-free features:** organizer thresholds, current/past slopes, activity compatibility, translation filter, and a TF-IDF vocabulary fit only on the permitted retrieval corpus.

**Models and configuration:** deterministic moment scores; word 1–2 grams and character 3–5 grams with a 12,000-feature cap; 180-second cooldown.

**Runtime/memory:** under 10 minutes and 2 GB RAM.

**Leakage risk:** minimal, though hand-tuned rules may inadvertently mirror the tiny sample. Keep every threshold derived from the organizer mapping or frozen plan rather than row-specific labels.

**Fallback:** this pipeline is itself the outage fallback. It provides reliability evidence but should not be the final entry unless learned candidates fail.

A contextual bandit is retained as a logged replay interface only. JITAI and micro-randomized-trial research support the architecture, but the organizer sample contains no genuine intervention outcomes from which to train an adaptive policy. ([arXiv][2])
