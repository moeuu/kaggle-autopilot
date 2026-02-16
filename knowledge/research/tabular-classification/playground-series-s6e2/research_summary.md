# Ranked candidate pipelines (playground-series-s6e2)

## 1) CatBoost GPU (primary)
- Leak-free features/encodings:
  - Use original features + arithmetic interactions (products/ratios).
  - Treat low-cardinality integer-coded columns as categorical (consistent across folds).
  - Optional: quantile binning for a small subset (fit bin edges on train-fold only, apply to val/test).
- Model + key hyperparameters:
  - `CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", iterations=12000, learning_rate=0.03, depth=8, l2_leaf_reg=5, random_strength=1.0, rsm=0.8, od_type="Iter", od_wait=400, task_type="GPU", devices="0")`
- Runtime/memory:
  - Fast on GPU; expect minutes per fold; memory modest (few features).
- Leakage risk:
  - Low if CV split is correct and any binning/encoders are fit per fold.
- Fallback:
  - If GPU unavailable: set `task_type="CPU"` and reduce iterations / increase learning_rate.

## 2) XGBoost GPU (diversity)
- Leak-free features/encodings:
  - One-hot encode low-cardinality columns with an encoder fit on train-fold only (or keep numeric-only as fallback).
  - Include same interaction features as CatBoost.
- Model + key hyperparameters:
  - `XGBClassifier(tree_method="hist", device="cuda", max_depth=8, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, reg_alpha=0.5, learning_rate=0.03, n_estimators=10000)`
- Runtime/memory:
  - Very fast on GPU; one-hot increases columns but still manageable.
- Leakage risk:
  - Medium if encoding is fit globally; keep strictly fold-fit.
- Fallback:
  - If `device` unsupported, try legacy GPU params; else CPU with fewer estimators.

## 3) LightGBM (optional third model)
- Leak-free features/encodings:
  - Same interactions; categorical handled via one-hot (fold-fit) or `category` dtype if stable.
- Model + key hyperparameters:
  - `LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, n_estimators=20000, num_leaves=256, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.5, lambda_l2=5)`
- Runtime/memory:
  - GPU can be fast; CPU may be slower—reduce iterations/seeds if CPU-only.
- Leakage risk:
  - Similar to XGB: encoding/bins must be fold-fit.
- Fallback:
  - Skip LightGBM entirely if GPU build is unavailable or unstable.

## Ensemble choice
- Preferred: ridge/logit blend on OOF (fit blend on OOF only) or weighted mean tuned on OOF.
- Safe fallback: simple mean of model probabilities (often competitive and robust).
