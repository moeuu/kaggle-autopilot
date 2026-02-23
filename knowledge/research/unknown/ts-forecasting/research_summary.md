# Research Summary — ts-forecasting (shortlist)

## 1) LGBM-GPU per-horizon “Notebook++” (Rank 1)
- Leak-free features/encodings: shifted lags/rolling/trend on selected `feature_*` within `(code,sub_code,sub_category,horizon)`; cross-sectional z-scores by `ts_index` (features-only); mean target encoding for `sub_code`/`sub_category` fit on fold-train only (unseen→global mean).
- Models + key hyperparameters: `LGBMRegressor` with `device_type=gpu`, `learning_rate≈0.01–0.02`, `n_estimators≈6000–12000`, `num_leaves≈64–128`, subsampling/feature_fraction, early stopping; train 3–5 seeds.
- Runtime/memory: fits 4 horizons sequentially; expect multi-hour full run on GPU; FAST_DEV trims to 1–2 folds + fewer trees.
- Leakage risk: low if (a) all target encodings are fit only on fold-train and (b) all temporal ops are shifted (no centered windows).
- Fallback: CPU LightGBM (same params minus GPU), reduce bins/trees.

## 2) LGBM + XGB GPU blend (Rank 2, higher ceiling)
- Leak-free features/encodings: identical feature backbone as #1 to keep train→test transform consistent.
- Models + key hyperparameters: XGBoost `tree_method=gpu_hist`, depth 6–10, `eta≈0.01–0.03`, strong regularization; blend weights learned from OOF (per horizon).
- Runtime/memory: +30–70% over LGB-only; still feasible if trained per horizon.
- Leakage risk: same as #1; blending must use OOF only.
- Fallback: XGB CPU `hist`, or drop XGB entirely.

## 3) CatBoost GPU categorical specialist (Rank 3, optional)
- Leak-free features/encodings: minimal extra FE; pass categoricals directly; optional residual modeling on top of #1 predictions.
- Models + key hyperparameters: `CatBoostRegressor(task_type="GPU", loss_function="RMSE")`, depth 8–10, iterations 3k–10k, early stopping.
- Runtime/memory: can be heavy at 5M rows; safest as residual/limited-feature model.
- Leakage risk: low if no target-derived stats computed outside fold-train.
- Fallback: skip CatBoost if OOM/slow.
