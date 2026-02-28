# research_summary.md (ranked pipelines)

## 1) Blend: CatBoost(GPU) + Hashed Linear (recommended)
- Features (leak-free):
  - Summary cats: `job_id, contractor_id, primary_location, job_category_description`
  - Time: `bid_date` → year/month/dow/doy
  - Raw aggregates per `row_id` from `raw_*` (no target): `n_items`, `sum_qty`, `mean_qty`, `std_qty`, `n_unique_pay_item_id`, `n_unique_category_id`, `n_unique_unit_english_id`, `num_pay_items` sanity stats
  - Optional pricebook (fold-fitted): mean `amount` by `(contractor_id, pay_item_id)` with fallbacks to `(pay_item_id)`, `(contractor_id, category_id)`, global; then `est_total = sum(quantity * mean_amount_lookup)` per `row_id`
  - Sparse hashed tokens: aggregated tokens from `pay_item_id`, words in `pay_item_description`, `unit_english_id`, `category_description`
- Models / hypers:
  - CatBoostRegressor: `task_type='GPU'`, depth 8–10, lr ~0.03, iterations 5k–20k, early stopping; train on `log1p(total_bid)`
  - Ridge/SGDRegressor on hashed sparse matrix: alpha tuned (e.g., 1e-3..1e1), fit on `log1p(total_bid)`
  - Blend in log space with CV-chosen weight (start 0.7 CatBoost / 0.3 linear)
- Runtime/memory: feature build dominates (chunked groupby); CatBoost GPU minutes; hashed linear ~seconds-minutes; fits scale with folds*seeds.
- Leakage risk: low if pricebook and any encoders are fold-fitted; do not use row-level `amount` directly.
- Fallback: if CatBoost unavailable, swap to LightGBM (CPU) with target-encoding; keep hashed linear.

## 2) CatBoost(GPU) only (strong baseline)
- Same dense features (no sparse hashing).
- Expected faster and simpler; slightly lower ceiling.
- Very low leakage risk if fold splits respected.

## 3) Hashed Linear only (fast, robust)
- Focus on pay-item composition via hashed tokens + basic numeric aggregates.
- Useful FAST_DEV and as a blend component.
- Lowest install risk; works with scikit-learn only.
