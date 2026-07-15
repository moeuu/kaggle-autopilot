# Ranked candidate shortlist

## 1. Criterion-graph Qwen QLoRA

**Leak-free features/encodings:** Parse source documents into sections and individual criteria. Split original trials before expanding examples. For each CV fold, fit retrieval vocabularies, root-style logic, count priors, and alignment diagnostics only on that fold’s training trials. Represent the current criterion subject as `SELF` and retain source text, parent heading, section type, and at most two fold-local retrieved examples.

**Model and key hyperparameters:** Prefer `Qwen/Qwen3.5-9B` only after a successful 4-bit language-only PEFT smoke test; otherwise use `Qwen/Qwen3-8B`. NF4 double quantization, fp16 compute, LoRA `r=32`, alpha `64`, dropout `0.05`, all language-model linear layers, learning rate `1e-4`, five epochs, batch `1`, gradient accumulation `16`, maximum sequence length `2048`, completion-only loss, gradient checkpointing, cosine schedule, and one full seed.

**Runtime/memory:** Approximately 600–780 training minutes plus generation on RTX3060. Batch 1 and criterion decomposition should keep VRAM under 12GB. Load only one fold model at a time.

**Leakage risk:** Low when trial IDs are split before criterion expansion. High if criterion rows from one trial cross folds, so assert this invariant.

**Fallback:** Qwen3-8B with the identical data and training path; then reduce sequence length to 1,536 and LoRA rank to 16 before considering Qwen3-4B.

## 2. Hybrid criterion plus section-residual graph

**Leak-free features/encodings:** Reuse the primary fold adapter. Trigger the section task from source-only structural signals such as nested bullets, definitions, named cohorts, and “one of the following.” Residual targets are constructed only from the training fold’s target graph.

**Model and key hyperparameters:** Same adapter as Pipeline 1; section input limit `3072`, generation limit `1024`, one retrieved section example, deterministic decoding plus one low-entropy variant.

**Runtime/memory:** No second full model training. Adds roughly 60–150 minutes of generation depending on the number of flagged documents.

**Leakage risk:** Low, but residual target assignment can be wrong. Save assignment coverage and manually inspect train-only diagnostics, never test labels.

**Fallback:** Disable residual generation and retain criterion-only assembly when it lowers OOF FM3S or inflates count.

## 3. Fold-local retrieval and graph scaffolding

**Leak-free features/encodings:** Word TF-IDF `(1,2)` and character TF-IDF `(3,5)` fitted on fold-train criteria only; same-section retrieval; root-style multinomial logistic regression; Ridge regression on `log1p` detail-triple count.

**Models and key hyperparameters:** TF-IDF retrieval with `k=3`, LogisticRegression `C=2.0`, Ridge `alpha=10.0`, deterministic section and bullet parser, strict JSON/triple repair.

**Runtime/memory:** CPU minutes; less than 2GB RAM once sparse matrices are released fold by fold.

**Leakage risk:** Medium if a validation target is accidentally indexed. Assert that every retrieved example’s trial ID belongs to the current training index.

**Fallback:** Character/word TF-IDF without the learned root/count components.

## 4. Cross-fitted semantic consensus reranker

**Leak-free features/encodings:** Build document candidates only from OOF fold predictions. Fit the candidate-quality Ridge model on OOF documents from other folds and apply it to the held-out fold. Refit on all OOF candidates for test selection.

**Models and key hyperparameters:** Maximum eight candidates per document, semantic cluster threshold `0.84`, minimum two-fold support for consensus triples, singleton source coverage at least `0.75`, Ridge `alpha=10.0`.

**Runtime/memory:** CPU minutes; negligible GPU use.

**Leakage risk:** Medium if the reranker is fitted and evaluated on the same OOF rows. Enforce leave-one-fold-out reranker fitting.

**Fallback:** Select the single candidate with the best mean OOF competition metric, with simpler/faster as the tie-breaker.
