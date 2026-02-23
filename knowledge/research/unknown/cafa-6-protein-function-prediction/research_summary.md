# research_summary.md

## Ranked shortlist (2–4 pipelines)

### 1) ESM2 frozen embeddings + taxonomy-aware MLP (Fold ensemble) — Recommended
- Leak-free features/encodings:
  - Protein embedding from ESM2 on sequence only; optional two-crop pooling for long proteins.
  - Taxonomy: fit taxon vocabulary on train-fold only; map unknown taxa to `Unknown`.
  - Normalization/PCA: fit on train-fold only, apply to val/test.
- Models + key hyperparameters:
  - Backbone: `facebook/esm2_t30_150M_UR50D` (fallback: `esm2_t12_35M_UR50D`)
  - Head: 2-layer MLP (hidden 1024, dropout 0.2), BCEWithLogitsLoss with pos_weight; AMP fp16.
  - 3 folds (GroupKFold by taxon), average probabilities; per-namespace threshold tuned on holdout via weighted Fmax.
- Expected runtime/memory:
  - GPU: embedding extraction is the main cost (hours); head training per fold is fast (minutes).
  - Memory: embeddings cached to disk; batch size tuned to GPU RAM.
- Leakage risk: low (uses only provided train labels + sequences; no external annotations).
- Fallback if dependency unavailable:
  - If transformers download blocked: use CNN-only (DeepGOPlus-style) on one-hot/k-mer.
  - If no GPU: reduce model to 35M, smaller max_len, single-crop.

### 2) Light fine-tune (last layers or LoRA) ESM2 + multi-label head (Fold ensemble)
- Leak-free: same split/fit protocol; only training labels used.
- Models + hyperparameters:
  - Unfreeze last 2–4 transformer layers; small LR for backbone, larger LR for head; gradient accumulation; early stopping on weighted Fmax.
- Runtime/memory: heavier; may need max_len 512 and small batch; still Kaggle-feasible with AMP.
- Leakage risk: low; rule risk: pretrained weights could be considered “external” (gate behind a toggle).
- Fallback: freeze backbone (Pipeline 1).

### 3) DeepGOPlus-style CNN motifs + similarity component (no pretrained weights)
- Leak-free features:
  - CNN over k-mer/motif channels trained on train-fold only.
  - Similarity: retrieval within train embeddings (e.g., k-mer TF-IDF cosine) fit on train-fold only.
- Runtime/memory: fast and CPU-friendly; similarity search must be chunked.
- Leakage risk: low; accuracy likely below pretrained-LM pipelines.
- Fallback: CNN-only without similarity.
