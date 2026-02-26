# research_summary.md (content)

## Ranked candidate pipelines (shortlist)

### 1) `catboost_gpu_rich_features_calibrated_blend` (Recommended)
- **Leak-free features/encodings**
  - Pre-tourney team-season aggregates from `M/WRegularSeason*Results` only (apply `DayNum <= DAY_CUTOFF`)
  - Elo features (end-of-regular-season rating, recent-form deltas), seed parsing (`W01a` → numeric seed + play-in flag)
  - `MMasseyOrdinals` snapshot at/before cutoff day: last-rank per system, then team-level aggregates (mean/median/quantiles); restrict to top-k systems by coverage
  - Matchup features: diffs/ratios + interaction features; optional categorical `TeamID_low`, `TeamID_high`, `Season`, `League`
- **Models + key hyperparameters**
  - `CatBoostClassifier(task_type='GPU', depth=6-8, learning_rate=0.02-0.05, iterations=3000-10000, l2_leaf_reg=5-15, subsample≈0.8, random_strength≈0.5)`
  - Monitor `eval_metric='BrierScore'`, early stop with `od_wait`
  - Optional Platt (sigmoid) calibration fit on OOF preds only
- **Expected runtime/memory**
  - GPU: ~30–120 min full (depends on Massey filtering); FAST_DEV: 5–15 min
- **Leakage risk**
  - Medium if cutoff logic is wrong; low if all feature sources are strictly pre-tourney and snapshot-based
- **Fallback if dependency unavailable**
  - CatBoost CPU; or switch primary to XGBoost + sklearn calibration

### 2) `xgb_gpu_hist_numeric_diffs_calibrated`
- **Leak-free features/encodings**
  - Same team-season table, but only numeric diffs/ratios; no raw categorical IDs
  - Strict `align_features(train_df, test_df, feature_cols)` to add missing columns safely
- **Models + key hyperparameters**
  - `xgboost.train(tree_method='gpu_hist', max_depth=5-7, eta=0.03-0.06, subsample=0.7-0.9, colsample_bytree=0.7-0.9, min_child_weight=10-30, reg_lambda=1-5, early_stopping_rounds=300-800)`
  - Optional sigmoid calibration on OOF
- **Expected runtime/memory**
  - GPU: ~20–60 min full; FAST_DEV: 5–10 min
- **Leakage risk**
  - Low (mostly numeric; main risk is using post-tourney rows by mistake)
- **Fallback**
  - `tree_method='hist'` on CPU; reduce boosting rounds

### 3) `elo_only_or_blend_component`
- **Leak-free features**
  - Elo built from regular season compact results only with cutoff; convert rating diff → probability via logistic transform
- **Model**
  - Deterministic Elo + optional linear seed adjustment; used as a blend input and as a “safe submit” fallback
- **Runtime**
  - Minutes
- **Leakage risk**
  - Low if tourney games are excluded and cutoff enforced
- **Fallback**
  - Unadjusted Elo (no MOV), fixed K-factor

Ensembling rule: use simple blending/stacking **only** across OOF predictions (no peeking), choose the **final** method by lowest CV Brier; tie-breaker: simpler/faster.
