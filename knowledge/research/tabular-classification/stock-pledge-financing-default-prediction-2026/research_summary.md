# research_summary.md (content)
## Ranked shortlist (2–4 pipelines)

### 1) GPU Boosting Blend (Default / Rules-safe)
- Features/encodings (leak-free): comma-clean + numeric cast; deterministic FE (abs/downside return, volatility/turnover interactions, signed-log PE/PB, pledge ratio interactions). No target encoding by default.
- Models + key hyperparameters:
  - CatBoostClassifier (GPU): `loss_function=Logloss`, `eval_metric=AUC`, `iterations=20000`, `depth=6–10`, `learning_rate=0.01–0.05`, `l2_leaf_reg=3–20`, `random_strength`, `subsample`, `early_stopping_rounds=300`, `task_type="GPU"`.
  - XGBoost (GPU): `tree_method="gpu_hist"`, `max_depth=3–8`, `eta=0.01–0.05`, `subsample/colsample=0.6–1.0`, `min_child_weight`, `lambda/alpha`, `scale_pos_weight`, early stopping on fold.
- Expected runtime/memory: moderate; fits easily on Kaggle GPU; multi-seed CV in ~30–120 min depending on iteration caps.
- Leakage risk: low (no target-based encodings).
- Fallback: if GPU unavailable, run CPU CatBoost/XGBoost with reduced iterations.

### 2) TabICL CV (Reference-aligned, optional)
- Features/encodings: same deterministic numeric FE; optional quantile normalization inside TabICL.
- Model: TabICLClassifier with small `n_estimators` and AMP; CV folds 2–5 to reduce overfit; repeated seeds.
- Runtime/memory: moderate; GPU preferred; dependency `tabicl` + `torch`.
- Leakage risk: low if only using features; policy risk if pretrained checkpoints count as external assets.
- Fallback: disable TabICL and rely on Pipeline 1.

### 3) TabPFN (Optional ensemble booster)
- Features/encodings: numeric-only matrix; optional standardization fit per fold.
- Model: TabPFN classifier producing probabilities per fold; blend with boosting outputs.
- Runtime/memory: variable; usually fine for this dataset size.
- Leakage risk: low (fold fit); policy risk similar to other pretrained models depending on rules.
- Fallback: exclude TabPFN and keep boosting-only blend.

Ensembling guidance: start with simple probability-mean (or rank-mean) across models; only add stacking if CV is stable across ≥3 seeds.
