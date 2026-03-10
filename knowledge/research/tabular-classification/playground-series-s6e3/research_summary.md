# Ranked shortlist for `playground-series-s6e3`

## 1. XGB GPU + leak-free target statistics + external-source rates
- **Leak-free features/encodings:** raw numeric columns, arithmetic ratios (`TotalCharges/(tenure+1)`, `MonthlyCharges/(tenure+1)`), service-count features, tenure bins fit on train only, fold-safe frequency encoding fit on train fold only, cross-fitted target encoding for low-cardinality categoricals and a very small pair/trigram shortlist, plus mapped churn-rate features derived from the public IBM-style original source only.
- **Models + key hyperparameters:** `XGBClassifier(device="cuda", tree_method="hist", learning_rate=0.03, max_depth=6, min_child_weight=12, subsample=0.85, colsample_bytree=0.70, reg_alpha=0.1, reg_lambda=1.0, n_estimators=12000, early_stopping_rounds=300)`.
- **Expected runtime/memory:** roughly 25-45 minutes per seed on a common single GPU for 5 folds; moderate VRAM, moderate RAM.
- **Leakage risk:** medium if target stats are mishandled; low if all competition-derived stats are fold-fitted only.
- **Fallback if dependency unavailable:** use CPU XGBoost or switch to LightGBM on the same encoded matrix.

## 2. CatBoost GPU + raw categoricals + numeric arithmetic features
- **Leak-free features/encodings:** raw object/string categoricals, engineered numeric ratios, service-count features, optionally external-source rate maps; no internal target encoding required for the main path.
- **Models + key hyperparameters:** `CatBoostClassifier(task_type="GPU", loss_function="Logloss", eval_metric="AUC", iterations=10000, learning_rate=0.03, depth=8, l2_leaf_reg=12, random_strength=0.8, bootstrap_type="Bernoulli", subsample=0.8, od_type="Iter", od_wait=300)`.
- **Expected runtime/memory:** 30-60 minutes per seed for 5 folds on GPU; a little heavier than XGB but still practical.
- **Leakage risk:** low to medium; lower preprocessing risk than TE-heavy pipelines, but external-source mappings still need documentation.
- **Fallback if dependency unavailable:** drop CatBoost and increase XGB/LGBM diversity.

## 3. LightGBM on the same fold-safe encoded matrix
- **Leak-free features/encodings:** identical encoded matrix to XGB so feature generation cost is shared.
- **Models + key hyperparameters:** `LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, n_estimators=12000, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0)`.
- **Expected runtime/memory:** usually the cheapest pipeline; good CPU fallback and blend candidate.
- **Leakage risk:** medium for the same reason as XGB if encodings are not fold-safe.
- **Fallback if dependency unavailable:** skip it; do not add a niche substitute.

## 4. Torch entity-embedding MLP for diversity only
- **Leak-free features/encodings:** categorical embeddings from train-fit vocabularies, standardized numerics fit on train only, optional discretized numeric-as-categorical branch.
- **Models + key hyperparameters:** 3 hidden layers (`512-256-128`), dropout `0.15`, batch size `4096`, `AdamW`, BCE-with-logits, AUC early stopping.
- **Expected runtime/memory:** 20-40 minutes total on GPU, memory-light.
- **Leakage risk:** low if vocab/scaler fitting is restricted to train folds.
- **Fallback if dependency unavailable:** disable it by default; the tree ensemble is the mainline.

**Recommendation:** implement all four as toggled pipelines in one `kernel.py`, but expect the winner to be XGB + CatBoost, with LightGBM added only if repeated-CV lift is real and stable. Keep pseudo-labeling as a post-shortlist experiment, not as the default baseline.
