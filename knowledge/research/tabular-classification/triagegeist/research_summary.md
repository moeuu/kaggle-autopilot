# research_summary.md

## Ranked shortlist

### 1. CatBoost GPU multimodal ensemble anchor
- Leak-free features/encodings:
  - Merge `train/test` with `chief_complaints.csv` and `patient_history.csv` on `patient_id`.
  - Drop only post-triage leakage columns: `disposition`, `ed_los_hours`.
  - Keep `news2_score`; remap `pain_score == -1` to missing; add missingness flags and simple physiologic interactions.
  - Fit all imputations and any learned text settings on train folds only.
- Models + key hyperparameters:
  - `CatBoostClassifier(loss_function="MultiClass", eval_metric="Accuracy", task_type="GPU")`
  - start with `iterations=4000`, `learning_rate=0.03`, `depth=8`, `l2_leaf_reg=6`, `bootstrap_type="Bayesian"`, `random_strength=0.5`
  - `text_features=["chief_complaint_raw"]`, unigram+bigram dictionaries, `BoW` + `NaiveBayes` feature calcers
- Expected runtime/memory:
  - Moderate; ~25-45 min for 5 folds x 1 seed on a typical Kaggle GPU, ~1.5-2.5h for 3 seeds.
- Leakage risk:
  - Low if only post-triage columns are dropped and fold-train fitting is respected.
- Fallback if dependency unavailable:
  - None needed in this repo; `catboost` is already present.

### 2. XGBoost GPU dense fusion model
- Leak-free features/encodings:
  - Dense structured features + train-fit TF-IDF on `chief_complaint_raw`, then TruncatedSVD to 64-128 dims fit on train folds only.
  - Keep pandas categorical dtype for native categorical handling where practical.
- Models + key hyperparameters:
  - `XGBClassifier(objective="multi:softprob", num_class=5, tree_method="hist", device="cuda", enable_categorical=True)`
  - start with `n_estimators=3000`, `learning_rate=0.03`, `max_depth=8`, `min_child_weight=4`, `subsample=0.85`, `colsample_bytree=0.8`, `reg_lambda=2.0`
- Expected runtime/memory:
  - Moderate; ~20-35 min for 5 folds x 1 seed, memory lower than transformer-based text.
- Leakage risk:
  - Low if TF-IDF/SVD are fold-fitted and train-only columns are dropped.
- Fallback if dependency unavailable:
  - LightGBM dense model with the same engineered features.

### 3. TabICL structured-or-dense optional model
- Leak-free features/encodings:
  - Use structured features plus optional dense complaint SVD features; no target encoders.
  - Keep all fold-specific preprocessing fit on train folds only.
- Models + key hyperparameters:
  - `TabICLClassifier` with current stable checkpoint; one main config first, not wide tuning.
  - Prefer structured-only first, then add dense text features as an ablation.
- Expected runtime/memory:
  - Fast inference but checkpoint download/cache can be brittle; GPU preferred.
- Leakage risk:
  - Low from preprocessing, medium operational risk from checkpoint/runtime instability rather than label leakage.
- Fallback if dependency unavailable:
  - Disable TabICL toggle and rely on CatBoost + XGBoost.

### 4. LightGBM sparse TF-IDF fallback
- Leak-free features/encodings:
  - Structured features with train-fit imputers and category casting; complaint TF-IDF fit on train folds only.
- Models + key hyperparameters:
  - `LGBMClassifier(objective="multiclass", num_class=5, learning_rate=0.03, n_estimators=3000, num_leaves=127)`
- Expected runtime/memory:
  - Fast enough, but warning-heavy and less elegant than CatBoost for mixed data.
- Leakage risk:
  - Low if sparse text and imputers are fit only on fold-train data.
- Fallback if dependency unavailable:
  - XGBoost dense fusion without sparse text.

## Recommendation
Start with CatBoost as the primary candidate, add XGBoost for diversity, and gate TabICL behind a runtime toggle. Use weighted probability blending only if OOF accuracy improves over the best single model by at least 0.001; otherwise ship the best single model.
