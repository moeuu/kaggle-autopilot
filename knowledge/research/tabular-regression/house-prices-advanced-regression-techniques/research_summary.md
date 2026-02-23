# research_summary.md (content)
Ranked shortlist of candidate pipelines for **house-prices-advanced-regression-techniques** (log-RMSE / RMSLE-style metric). All pipelines are leak-free: every imputer/encoder/transform is fit on CV-train folds and applied to fold-val + test.

## 1) GPU GBDT Trio + OOF Ridge Blender (recommended)
- Features/encodings (leak-free): numeric median impute + missing indicators; categoricals as string with explicit missing token; for XGB/LGB use one-hot (handle_unknown=ignore) or ordinal + safe missing code; for CatBoost pass categorical indices and fill missing categoricals as "Missing".
- Models + key hyperparameters:  
  - LightGBM: `learning_rate~0.01-0.03`, `num_leaves~31-255`, `feature_fraction`, `bagging_fraction`, `min_data_in_leaf`, `lambda_l1/l2`, `n_estimators` large + early stopping.  
  - XGBoost: `tree_method=gpu_hist`, `max_depth~3-6`, `min_child_weight`, `subsample/colsample_bytree`, `reg_alpha/reg_lambda`, `n_estimators` large + early stopping.  
  - CatBoost: `task_type=GPU`, `depth~6-10`, `learning_rate~0.02-0.08`, `l2_leaf_reg`, `loss_function=RMSE`, early stopping.
- Expected runtime/memory: minutes; sparse one-hot may increase RAM but still manageable at this dataset size.
- Leakage risk: low if fold-wise preprocessing is enforced; avoid fitting encoders on train+test.
- Fallbacks: if CatBoost missing → drop it; if GPU unavailable → switch to CPU params; if LightGBM missing → sklearn HistGradientBoostingRegressor.

## 2) Serigne-inspired Stacked Regressions (max-accuracy toggle)
- Features/encodings: log-target; skew correction on numeric features (Yeo-Johnson/Box-Cox-like); one-hot encoding for categoricals; missing indicators.
- Models: Lasso + ElasticNet + KernelRidge + GradientBoosting + (XGB, LGB); stack with ridge meta-model using OOF preds.
- Runtime/memory: moderate (more base learners); still feasible.
- Leakage risk: medium if stacking is implemented incorrectly; must generate OOF strictly and fit meta-model only on OOF.
- Fallback: replace KernelRidge with Ridge if needed; drop the slowest base learners.

## 3) OneHot Ridge/ENet + LGB/XGB Blend (fast + stable)
- Features: sklearn ColumnTransformer with median/most_frequent impute + OneHotEncoder(handle_unknown=ignore).
- Models: Ridge/ElasticNet + LightGBM and/or XGBoost; blend weights chosen by CV (simple grid or ridge on OOF).
- Runtime: very fast; minimal complexity.
- Leakage risk: very low.
- Fallback: if LGB/XGB unavailable → ExtraTreesRegressor / HGBR.
