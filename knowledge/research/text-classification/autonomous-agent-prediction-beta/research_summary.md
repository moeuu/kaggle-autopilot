# Ranked pipeline shortlist

## 1. Meta-routed native GBDT portfolio — recommended final

**Leak-free features/encodings:** raw strings for CatBoost; fold-train category maps for LightGBM; train-fitted numeric medians, missing indicators, row aggregates, and parsed ordinal suffixes. **Models:** CatBoost (`iterations=1200`, `depth=7`, `learning_rate=0.035`), LightGBM (`n_estimators=1600`, `num_leaves=31`, `learning_rate=0.025`), ExtraTrees (`n_estimators=600`), and regularized logistic regression. Add rank/logit/weighted OOF blends only when cross-fitted AUC improves by at least 0.0005. **Runtime/memory:** about 15–35 CPU minutes per hidden task and under roughly 6 GB RAM, with a quick CatBoost submission in the first 3–5 minutes. **Leakage risk:** low if every transform is fold-fit and `row_id` is excluded. **Fallback:** CatBoost → HistGradientBoosting on ordinal/frequency features → logistic regression; never a constant/sample-copy output.

## 2. Cross-fitted encoded XGBoost specialist

**Leak-free features/encodings:** per-fold frequency/log-frequency, rare flags, smoothed target means, ordinal parsing, quantile bins, and missingness; unknown categories receive a reserved code. **Models:** XGBoost histogram classifier (`n_estimators=1400`, `max_depth=6`, `learning_rate=0.025`, `min_child_weight=4`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_lambda=5`) plus an encoded LightGBM variant. **Runtime/memory:** roughly 8–20 CPU minutes and under 5 GB RAM after cached fold transforms. **Leakage risk:** medium/high if target encoding is not cross-fitted; enforce fold isolation and smoothing. **Fallback:** frequency-only XGBoost or sklearn HistGradientBoosting if XGBoost categorical/version support fails.

## 3. Learned task router and schedule policy

**Leak-free features/encodings:** task meta-features computed from training data only: row/column counts, class balance, categorical fraction/cardinality summaries, missingness, numeric distribution summaries, duplicate rate, and cheap baseline OOF. **Models:** standardized ridge utility model with candidate-family interactions (`alpha=10`) trained on task-candidate records under leave-one-dataset-out evaluation. **Runtime/memory:** outer local build about 6–12 hours depending on promoted candidates; packaged inference is milliseconds and tiny memory. **Leakage risk:** meta-overfit to only 16 tasks; mitigate with LODO, normalized within-task utility, regularization, and a fixed-policy ablation. **Fallback:** deterministic candidate priority table learned from aggregate task ranks.

## 4. TabICLv2/RealMLP GPU research challenger

**Leak-free features/encodings:** train-only preprocessing, explicit unknown/missing categories, no use of development solution labels inside a held-out task. **Models:** official TabICLv2 checkpoint (`ensemble_size=8`, `batch_size=8`, `max_rows=10000`, fp16) and compact RealMLP defaults. **Runtime/memory:** about 1–4 local GPU hours across eligible tasks, targeting 8–11 GB VRAM; not assumed feasible in the 60-minute CPU agent. **Leakage risk:** low algorithmically but high operational risk from checkpoints, packaging, and train/test preprocessing mismatch. **Fallback:** do not package it; use its outer results only to adjust the router, then retain the GBDT portfolio.
