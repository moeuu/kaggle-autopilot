# CAFA-6 research_summary.md (shortlist)

## Rank 1 — Hybrid DeepGOPlus-style (recommended)
- Leak-free features/encodings: amino-acid tokenization; optional k-mer (k=3/4) frequency embeddings; taxonomy prior P(term|taxon) fit on train-fold only.
- Models + key hyperparameters: small CNN/Transformer encoder (6 layers, d_model~256, dropout~0.1) predicting top M frequent GO terms (M=4000–8000) with BCEWithLogits + class-balanced loss; tail terms via kNN label-transfer (K=50–200) using cosine similarity over k-mer projections; blend weights tuned on val.
- Runtime/memory: training 1–3h on single GPU (mixed precision); retrieval search chunked for 224k test proteins; memory dominated by embedding matrices and FAISS index.
- Leakage risk: moderate if random split; mitigate with taxonomy-based GroupKFold and fold-specific retrieval index.
- Fallback if dependency unavailable: if FAISS missing, use CPU approximate NN (sklearn NearestNeighbors on reduced dims) on smaller K, or pure taxonomy prior + global frequency.

## Rank 2 — Retrieval + calibration (fast strong baseline)
- Leak-free features/encodings: k-mer frequency vectors projected to 128–512 dims (fit projection on train-fold only); optional length/AA composition.
- Models + key hyperparameters: kNN label-transfer with similarity-weighted term aggregation; global logistic calibration (a,b) to map raw scores to probabilities; ontology propagation (max) after scoring.
- Runtime/memory: no GPU training required; retrieval dominates; works well in FAST_DEV.
- Leakage risk: low if folds respected (index built on train-fold only).
- Fallback: CPU-only brute over smaller K or fewer dims.

## Rank 3 — Taxonomy prior ensemble backstop
- Leak-free features/encodings: term frequencies conditioned on taxon with Laplace smoothing + global prior.
- Models + key hyperparameters: no ML; blend into other pipelines at 5–20% weight.
- Runtime/memory: minutes, tiny.
- Leakage risk: low.
- Fallback: global prior only.
