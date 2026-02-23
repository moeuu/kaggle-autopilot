# research_summary.md

## Ranked candidate pipelines (shortlist)

### 1) Physics-informed coupled-graph multi-task forecaster (Primary)
- Leak-free features/encodings:
  - Fit scalers on train events only (per model_id + variable); apply to val/test.
  - Node static features + dynamic warmup (first K timesteps) + known future rainfall (2D).
  - Optional per-node normalization stats computed only from train split; fallback to global stats for unseen nodes.
- Model:
  - Pure-PyTorch message passing (COO edges + `index_add_`) over a unified heterograph (1D, 2D, 1D↔2D).
  - Temporal block: GRU/TCN encoder for warmup + conditional multi-horizon decoder (predict all H steps).
  - Multi-task heads: `water_level` (main), `water_volume` (aux, 2D), edge `flow` (aux).
  - Physics losses: continuity/smoothness regularizers; multi-step curriculum (increase horizon during training).
- Key hyperparameters:
  - Hidden 128–256, 4–8 message-passing layers, dropout 0.1, AdamW lr 1e-3→1e-4 cosine, horizon curriculum.
- Runtime/memory:
  - GPU-friendly; batch by sampled time windows/events; mixed precision recommended.
- Leakage risk: Low if warmup-only + rainfall-only future inputs; ensure no use of future `water_level`.
- Fallback if dependency unavailable:
  - No special deps beyond torch/pandas/pyarrow; if too slow, switch to Pipeline 2.

### 2) Graph WaveNet-style STGNN (Strong fallback / speed baseline)
- Leak-free features/encodings:
  - Same fit-on-train scalers; align feature columns safely.
  - Inputs: past K timesteps of node states + exogenous rainfall channels.
- Model:
  - Dilated causal 1D temporal conv stacks + graph mixing with (a) fixed adjacency from edge_index and (b) optional adaptive adjacency.
  - Direct multi-horizon output head.
- Key hyperparameters:
  - Channels 64–128, layers 6–10, kernel 2–3, receptive field ≥ K, adaptive_adj on/off.
- Runtime/memory:
  - Very fast on GPU; stable training; good for large sweeps/ablations.
- Leakage risk: Low.
- Fallback:
  - Disable adaptive adjacency; use fixed adjacency only.

### 3) DCRNN-style diffusion-conv seq2seq (Long-horizon specialist)
- Leak-free features/encodings:
  - Same event-level CV; scheduled sampling only uses model predictions, not ground-truth future.
- Model:
  - Diffusion convolution on directed adjacency + GRU encoder–decoder + scheduled sampling.
- Key hyperparameters:
  - Hidden 64–128, diffusion steps 2–3, teacher forcing decay schedule, horizon H.
- Runtime/memory:
  - Slower than TCN; can still fit on GPU with windowed batching.
- Leakage risk: Medium if scheduled sampling is implemented incorrectly; must be strict.
- Fallback:
  - Replace seq2seq with direct multi-horizon head or switch to Pipeline 2.
