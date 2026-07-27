# Ranked shortlist

## 1. Cross-fitted CatBoost + XGBoost + RealMLP, with class-logit calibration

**Leak-free features/encodings:** CatBoost receives raw string categoricals, missing sentinels, numeric values, and deterministic row-wise health/activity features. XGBoost receives 64-bin quantiles, outer-train frequency maps, and four-way inner-cross-fitted multiclass target encodings. RealMLP receives fold-fitted robust scaling, smooth clipping, explicit missing/unknown handling, categorical embeddings, and the same deterministic features. `id` is excluded.

**Models and frozen settings:** CatBoost GPU: 6,000 iterations, learning rate 0.035, depth 8, `l2_leaf_reg=8`, `border_count=254`, early stopping 300. XGBoost CUDA: 6,000 estimators, learning rate 0.025, depth 7, `min_child_weight=8`, `subsample=0.85`, `colsample_bytree=0.90`, `reg_lambda=6`, `max_bin=512`, early stopping 250. RealMLP-style PyTorch: 512/512/256, dropout 0.10, batch 4,096, up to 120 epochs, AdamW at 0.001, patience 15. Five tree folds and three neural folds feed a five-fold cross-fitted probability blend and additive class-logit biases.

**Expected runtime/memory:** 6–10 hours total on RTX 3060, usually below 10 GB VRAM. **Leakage risk:** low if the inner target-encoding and meta-calibration folds are enforced; high if either is collapsed into in-sample tuning. **Fallback:** drop RealMLP only after its local PyTorch implementation and float32/smaller-batch retries fail; retain calibrated CatBoost + XGBoost. RealMLP’s published design and standalone preprocessing support this as the most reproducible high-accuracy core. ([arXiv][2])

## 2. Add local TabPFN-3 as a fourth OOF member

**Leak-free features/encodings:** use raw numerics, explicit missing values, fold-fitted categorical integer vocabularies with unknown `-1`, and only a small deterministic row-feature set. Do not use global scaling, one-hot encoding, external predictions, or test-fitted category statistics.

**Model and frozen settings:** local `TabPFNClassifier`, CUDA, four estimators, three OOF folds, seed 42, low-memory option only when the installed API supports it, test/validation chunks of 16,384 with an 8,192 retry. Probabilities are reordered to the canonical class mapping before saving.

**Expected runtime/memory:** preflight-dependent, approximately 3–7 hours and a 10.5 GB soft VRAM cap. **Leakage risk:** low for local OOF inference; operational/license risk is moderate because current weights require accepted terms and may not already be cached. **Fallback:** two estimators, smaller chunks, then a recorded skip; never call a hosted API. TabPFN-3’s one-million-row target scale makes it the highest-upside additional candidate. ([arXiv][3])

## 3. Guarded boundary refiner over the calibrated blend

**Leak-free features/encodings:** base-model OOF probabilities, log probabilities, top-two margin, entropy, cross-model variance/disagreement, raw deterministic features, and missingness. Generate refiner OOF predictions in five folds; choose the margin gate and blend alpha only through a second cross-fit.

**Model and frozen settings:** shallow XGBoost with 900 estimators, learning rate 0.03, depth 4, `min_child_weight=12`, `subsample=0.8`, `colsample_bytree=0.8`. Query it only for base-margin `<0.30`; accept only at `+0.00015` balanced accuracy with no per-class recall loss over `0.002`.

**Expected runtime/memory:** 40–100 minutes, generally under 6 GB. **Leakage risk:** medium if refiner selection is performed on the same OOF rows; low under the specified nested/cross-fitted acceptance. **Fallback:** keep the calibrated blend unchanged. This is the targeted final squeeze, not a license to rewrite selected test IDs.

## 4. Original-data signal ablations

**Leak-free features/encodings:** `competition_plus_original` may append a locally mounted, public original dataset only after license/schema checks; external rows are weighted 0.25 and never enter validation. `orig_signal_only` trains on competition rows and adds only original-derived class prototypes, distances, quantile offsets, or category likelihoods fitted without validation/test access.

**Models and frozen settings:** screen with three-fold XGBoost/LightGBM at 1,500 rounds before promoting to the full CatBoost/XGBoost core. **Expected runtime/memory:** 30–120 minutes for screening, under 8 GB. **Leakage risk:** low for labels but meaningful distribution-shift and provenance risk. **Fallback:** skip the suite when the file, license, target mapping, or normalized schema is not verified; `competition_only` remains authoritative.
