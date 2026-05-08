# Ranked shortlist for `playground-series-s6e3`

## 1. XGB GPU + orig-source stats + leak-safe bi/tri target encodings
- Leak-free features/encodings: numeric ratios (`TotalCharges/(tenure+1)`, `MonthlyCharges/(tenure+1)`), charges deviation, service count, tenure bins fit on train fold only, train-fold frequency encoding, grouped stats fit on train fold only, cross-fitted target encoding for curated single categorical columns plus a small curated set of bi-gram/tri-gram tokens, and original-source probability/distribution features computed from the public IBM telco source only.
- Models + key hyperparameters: `XGBClassifier(device="cuda", tree_method="hist", learning_rate=0.025, max_depth=6, min_child_weight=10, subsample=0.82, colsample_bytree=0.72, reg_alpha=0.05, reg_lambda=1.2, n_estimators=16000, early_stopping_rounds=350)`.
- Expected runtime/memory: about 90-150 minutes total for 10 folds x 3 seeds on a single common GPU; moderate VRAM and moderate RAM.
- Leakage risk: medium if fold boundaries are violated; low enough if all competition-derived stats are train-fold-only.
- Fallback if dependency unavailable: CPU XGBoost on the same encoded matrix, or reduce to 5 folds/1 seed under `FAST_DEV`.

## 2. CatBoost GPU + raw categoricals + safe engineered numerics
- Leak-free features/encodings: raw string categoricals, numeric ratios, service counts, optional original-source maps, and selected distribution features that do not use competition labels outside the fold.
- Models + key hyperparameters: `CatBoostClassifier(task_type="GPU", loss_function="Logloss", eval_metric="AUC", iterations=12000, learning_rate=0.03, depth=8, l2_leaf_reg=12, random_strength=0.8, bootstrap_type="Bernoulli", subsample=0.8, od_type="Iter", od_wait=350)`.
- Expected runtime/memory: about 90-180 minutes total for 10 folds x 3 seeds on GPU; a bit heavier than XGB.
- Leakage risk: low to medium; simpler preprocessing than TE-heavy XGB, but original-source joins still require schema discipline.
- Fallback if dependency unavailable: disable CatBoost and increase the XGB/LightGBM blend diversity.

## 3. LightGBM on the shared encoded matrix
- Leak-free features/encodings: exactly the same fold-safe encoded table built for XGB so feature generation is shared.
- Models + key hyperparameters: `LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, n_estimators=14000, num_leaves=96, min_data_in_leaf=160, feature_fraction=0.75, bagging_fraction=0.82, bagging_freq=1, lambda_l2=1.0)`.
- Expected runtime/memory: about 20-40 minutes total on CPU; modest RAM.
- Leakage risk: medium for the same reason as XGB if encodings are done incorrectly.
- Fallback if dependency unavailable: skip it rather than adding a niche replacement.

## 4. OOF rank/logit blend of shortlisted base models
- Leak-free features/encodings: no new features; consumes saved OOF/test predictions from the shortlisted pipelines only.
- Models + key hyperparameters: weighted rank blend and weighted logit blend searched over a small deterministic grid, selecting only if OOF ROC AUC improves by at least `0.00015`.
- Expected runtime/memory: negligible relative to training.
- Leakage risk: low if blending uses only OOF predictions and never leaderboard feedback.
- Fallback if dependency unavailable: none needed; pure NumPy/pandas implementation.
