# research_summary.md

## Ranked shortlist (practical, leak-free)

### 1) Ensemble: TabICL + CatBoost(GPU) + XGBoost(GPU) (recommended)
- Features/encodings:
  - `df.columns.str.strip()`; numeric coercion with `pd.to_numeric(errors="coerce")`
  - Deterministic FE: pledge ratios (`pledge_total`, limited/unlimited shares), valuation signed-log (`pe_slog`, `pb_slog`, mixes/ratios), market stress interactions (`abs_ret_1y`, `vol_x_absret`, `turnover_x_vol`)
  - Per-fold winsorization (clip to 1%/99% quantiles) fitted on fold-train only, applied to val/test
- Models + key hyperparameters:
  - TabICLClassifier: `n_estimators` 16–64, `norm_methods=["quantile"]`, `outlier_threshold≈4`, `use_amp=True`, `device="cuda"`
  - CatBoostClassifier GPU: depth 6–10, `learning_rate` 0.02–0.08, `l2_leaf_reg` 3–20, `loss_function="Logloss"`, `eval_metric="AUC"`, early stopping
  - XGBoost: `max_depth` 3–6, `eta` 0.02–0.08, strong reg (`min_child_weight`, `subsample`, `colsample_bytree`), `tree_method="gpu_hist"`, early stopping
- Runtime/memory: minutes total; TabICL checkpoint download ~100MB once; GPU RAM moderate.
- Leakage risk: low if all fitted transforms are fold-train only; no test-stat fitting.
- Fallbacks:
  - If `tabicl` install/checkpoint fails: drop TabICL and use CatBoost+XGB blend.
  - If GPU unavailable: run CatBoost CPU + XGB CPU (slower but fine on small data).

### 2) TabICL-only (mandatory reference baseline, improved)
- Features: same deterministic FE + column-strip + optional per-fold clipping.
- Model: TabICLClassifier with conservative `n_estimators` and fewer folds if CV variance explodes.
- Runtime/memory: very fast after checkpoint cached.
- Leakage risk: low.
- Fallback: replace with CatBoost if `tabicl` unavailable.

### 3) CatBoost-only strong baseline (fast, stable)
- Features: deterministic FE + per-fold clipping.
- Model: CatBoost GPU with AUC eval + early stopping; optionally multiple seeds and average.
- Runtime/memory: very fast.
- Leakage risk: low.
- Fallback: XGBoost GPU/CPU.

Notes:
- Prefer **rank-mean blending** of model probabilities to stabilize against distribution shift and reduce sensitivity to calibration.
- Avoid target encoding unless implemented strictly fold-wise (fit on fold-train, apply to val/test); this dataset appears mostly numeric so it’s optional.
