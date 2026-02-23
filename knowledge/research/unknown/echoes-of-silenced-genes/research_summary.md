# research_summary.md (ranked shortlist: 3 candidate pipelines)

## 1) Metric-calibrated low-rank ridge on gene embeddings (DEFAULT “final”)
- Leak-free features/encodings:
  - Build a fixed embedding per gene (`E[g]`) from **unsupervised** sources only (preferred: PCA/ICA on `training_cells.h5ad`; fallback: correlation/PCA on `training_data_means.csv` + gene statistics).
  - For each perturbation, use the embedding of the **perturbed gene** as `X`.
  - Fit `StandardScaler` on train folds only; apply to val/test.
- Models + key hyperparameters:
  - `Y` compression: `TruncatedSVD(n_components=K)` with `K=128..512`.
  - Per-component `Ridge(alpha=1e-2..1e3)` or a small MLP with strong weight decay.
  - Post-hoc calibration: global scale `alpha_scale` grid + optional clipping to maximize `W * max(0, Wcos)` on CV.
- Expected runtime/memory: ~5–30 minutes CPU; GPU optional; memory small (arrays ~80×5127).
- Leakage risk: low if all fit steps are fold-scoped; embedding from h5ad is unsupervised (safe).
- Fallback if dependency unavailable: if h5ad parsing fails, use a lightweight embedding from CSV-only statistics + keep model linear/low-rank.

## 2) GEARS-lite / diffusion-on-gene-graph (UPGRADE if Pipeline 1 plateaus)
- Leak-free features/encodings:
  - Construct a gene graph from **training_cells** co-expression (or an explicitly allowed external network, with a toggle).
  - Node features: gene embeddings (same as Pipeline 1) + a perturbation indicator channel.
  - Fit graph normalization/statistics on train folds only when learned; for fixed graphs, keep deterministic.
- Models + key hyperparameters:
  - Pure PyTorch message passing (avoid PyG): 2–4 layers, hidden 128–256, dropout 0.2–0.5, strong weight decay.
  - Output head predicts 5,127 deltas; train with weighted MAE/Huber proxy + strong shrinkage.
  - Same metric calibration step as Pipeline 1.
- Expected runtime/memory: 30–180 minutes on GPU depending on graph size (limit to ~5k–8k genes).
- Leakage risk: moderate if you accidentally use test-derived graph stats; keep everything fold-scoped or fixed-from-train-only.
- Fallback: revert to Pipeline 1 if training unstable or graph build fails.

## 3) External pretrain → fine-tune (MAJOR upgrade, only if time allows)
- Leak-free features/encodings:
  - Download a compact external perturb-seq resource (prefer pseudo-bulk signatures/embeddings from Replogle K562 CRISPRi).
  - Pretrain gene embeddings and/or the low-rank decoder; fine-tune only on the 80 competition perturbations.
- Models + key hyperparameters:
  - Same architecture as Pipeline 1 or 2, but initialized from pretraining; fine-tune with small LR, early stopping, and metric calibration.
- Expected runtime/memory: variable; can be heavy if raw AnnData is used—prefer cached, reduced representations.
- Leakage risk: low if external data is independent; ensure no access to competition test labels.
- Fallback: disable external pretrain and run Pipeline 1.
