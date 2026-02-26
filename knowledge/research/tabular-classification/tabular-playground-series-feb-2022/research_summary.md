# research_summary.md (content)

## Ranked shortlist (2–4 pipelines)

### 1) DAE pretrain → Tabular ResNet classifier (recommended)
- Leak-free features/encodings:
  - `feature_cols` = all numeric columns except `row_id`/`target`.
  - Per-fold `StandardScaler` (fit on train fold; apply to val/test).
  - DAE corruption: Gaussian noise + masking (+ optional swap noise). Pretrain label-free on train-only by default; optional toggle for train+test pretrain (transductive) with clear warning.
- Models + key hyperparameters:
  - DAE: encoder dims `[512, 256, 128]`, latent 64–128; dropout 0.1–0.3; reconstruction loss MSE; 50–200 epochs with cosine LR.
  - Classifier: residual MLP/ResNet (blocks 4–8, width 256–1024), BN/LayerNorm, dropout 0.2–0.5, label smoothing 0.05–0.1; AdamW; cosine schedule; early stopping on val accuracy.
- Expected runtime/memory:
  - Moderate; fits easily on GPU; batch size 2048–8192 depending on VRAM; total 5-fold × 3-seed typically manageable (<1–3h).
- Leakage risk:
  - Low if scaler is per-fold and DAE is train-only (default). Medium if transductive pretrain toggle enabled (still label-free, but changes evaluation meaning).
- Fallback if dependency unavailable:
  - Implement DAE + ResNet directly in PyTorch (no extra libs).

### 2) FT-Transformer (numeric tokenization) with strong regularization
- Leak-free features/encodings:
  - Per-fold scaler or rank-gauss/quantile transform (fit on train fold only).
  - Numeric feature tokenization inside model (learned per-feature embeddings).
- Models + key hyperparameters:
  - d_token 64–192, n_layers 3–6, n_heads 4–8, dropout 0.1–0.3, AdamW, cosine LR, label smoothing.
- Expected runtime/memory:
  - Higher than ResNet; still feasible on GPU; reduce layers/heads if slow.
- Leakage risk:
  - Low with per-fold preprocessing.
- Fallback:
  - Use tabular ResNet only (Pipeline 1 without DAE) or a plain MLP with residual blocks.

### 3) Diversity add-on: GPU tree model (XGBoost GPU) + probability blend
- Leak-free features/encodings:
  - No target encoders needed (all numeric). Optionally standardize for consistency (not required for trees).
- Models + key hyperparameters:
  - XGBoost `multi:softprob`, depth 6–10, eta 0.03–0.1, subsample/colsample 0.6–0.9, 2000–10000 trees with early stopping.
- Expected runtime/memory:
  - Fast; low incremental cost; good diversity.
- Leakage risk:
  - Low with proper CV.
- Fallback:
  - If GPU XGBoost missing, skip and rely on deep ensemble.

Ensembling:
- Default: simple mean of per-model probabilities across folds/seeds.
- Optional: weighted blend where weights are proportional to OOF accuracy (only if it improves OOF).
