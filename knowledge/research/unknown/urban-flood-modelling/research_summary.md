# research_summary.md

Ranked shortlist for `urban-flood-modelling`. All pipelines below are leak-free by construction: every scaler, standardizer, aggregate, encoder, graph summary, and clipping threshold is fit on CV-train only and then applied to fold-val and test. Never use future unavailable hydraulic states after the 10-step warm start; only rainfall that is actually present in test may be used as future exogenous input.

## 1) Heterograph Graph WaveNet + auxiliary hydraulics (Primary)
- Leak-free features/encodings: 10-step lag stacks of `water_level`; observed warm-start `inlet_flow`, `water_volume`, edge `flow/velocity`; future rainfall windows and cumulative rain; static node/edge geometry; graph degree/elevation-delta/coupling summaries.
- Models + key hyperparameters: PyTorch Graph-WaveNet-style backbone with heterograph supports, hidden_dim `192`, blocks `8`, dropout `0.10`, AdamW `1e-3`, cosine decay, delta target, auxiliary heads for 2D `water_volume` and edge states.
- Expected runtime/memory: `FAST_DEV` 20-40 min on 1 fold / 1 seed; full run 8-16 GPU hours; VRAM about 10-14 GB.
- Leakage risk: low if train masks future non-rainfall hydraulics exactly as test and validation is full autoregressive rollout.
- Fallback if dependency unavailable: pure PyTorch implementation without specialized graph libs; if needed, use fixed sparse adjacency matmul blocks.

## 2) HydroGraphNet-style encode-process-decode MPNN (Challenger)
- Leak-free features/encodings: same causal state inputs plus node/edge residual-flow summaries and 1D rainfall exposure projected from connected 2D cells.
- Models + key hyperparameters: hidden_dim `160`, message-passing steps `6`, rollout curriculum `1->4->8->16`, Huber + standardized loss, physics penalty `0.03`, AMP.
- Expected runtime/memory: about 10-20% slower than Graph WaveNet, similar VRAM, more implementation complexity.
- Leakage risk: low to medium; biggest failure mode is over-regularizing with noisy pseudo-physics or accidentally consuming future auxiliary states.
- Fallback if dependency unavailable: reduce to fixed-support message passing in vanilla PyTorch and disable physics penalty.

## 3) Directed DCRNN / graph-GRU (Secondary challenger)
- Leak-free features/encodings: directed 1D/2D supports, bipartite 1D-2D coupling, lagged water-level histories, rainfall forcing, static geometry.
- Models + key hyperparameters: hidden_dim `128`, diffusion_steps `3`, seq2seq GRU, scheduled sampling ratio `0.15`, grad_clip `1.0`, AMP.
- Expected runtime/memory: slower than WaveNet for long horizons; typically 10-14 GB VRAM in full mode.
- Leakage risk: medium because teacher-forced validation and bad scheduled sampling setup will overstate quality.
- Fallback if dependency unavailable: replace diffusion convolution with normalized adjacency mixing and keep GRU rollout.

## 4) LightGBM residual expert + graph summaries (Blend insurance)
- Leak-free features/encodings: persistence baseline, lag/rolling/slope features, rainfall summaries, static geometry, neighbor means from previous timestep only, cross-domain connection aggregates, safe `align_features(train_df, test_df, feature_cols)`.
- Models + key hyperparameters: `num_leaves=127`, `learning_rate=0.03`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `min_data_in_leaf=64`, `n_estimators=1800`, residual-to-persistence target.
- Expected runtime/memory: 30-90 min, low RAM, works on CPU or GPU.
- Leakage risk: low if all windows are causal and every feature stat is fit on train only.
- Fallback if dependency unavailable: `xgboost` with the same causal feature frame or sklearn HistGBR.

Recommendation: implement pipeline 1 first, keep pipeline 2 behind a toggle as the only serious neural challenger, and only allow pipeline 4 into the final blend if CV improves the competition metric materially.
