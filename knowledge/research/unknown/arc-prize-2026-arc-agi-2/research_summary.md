# Ranked shortlist

## 1. Hybrid NVARC + object DSL + real TRM + cross-fitted pass@2 ranker

**Leak-free features/encodings:** Reversible raw-grid serialization; object scene graphs; D4 and palette-permutation views; exact demonstration replay; leave-one-demo-out program stability; Qwen Product-of-Experts likelihood statistics; TRM confidence; candidate structural descriptors; DiARC near-miss margins; cross-family support. Candidate generation sees only each task’s demonstrations. Candidate labels, category maps, source priors, calibration, and pair-selection weights are fit on training folds and applied to validation/test.

**Models and key hyperparameters:** Qwen3-4B ARC SFT in 4-bit BF16, LoRA `r=64`, `alpha=128`, learning rate `5e-5`, max length 4096, adaptive 12/24/40 TTT steps, 32/48/64 views, 192 decoded candidates, 64 reranked candidates; object DSL beam width 256, depth 4, 5,000-program cap; actual TRM checkpoint with 256 adaptation steps and 16 recursive supervision steps; LightGBM ranker with 31 leaves, depth 6, learning rate 0.03, 600 trees. The design combines the reference NVARC components with the selection and neuro-symbolic lessons supported by current research. ([GitHub][2])

**Expected runtime/memory:** About 1,100–1,380 minutes for the first full local cache-and-CV build on one RTX3060; 10–12GB VRAM with sequential model residency. Hidden-test inference is budgeted near 650 minutes for 240 tasks.

**Leakage risk:** Medium unless task IDs/signatures, public evaluation, and candidate-ranker folds are tightly controlled. The supplied placeholder is a training duplicate and is forbidden for scoring.

**Fallback:** Preserve Qwen3-4B, reduce batch/views/TTT/candidate caps in that order; skip TRM explicitly if its real checkpoint is unavailable; use deterministic PoE+replay ranking if the learned ranker fails.

## 2. NVARC Qwen3 TTT + constrained multi-view PoE

**Leak-free features/encodings:** Task-local demonstrations only, reversible geometric/color views, assistant-only loss, shape-first grid grammar, and cross-view likelihood aggregation.

**Models and key hyperparameters:** Qwen3-4B ARC SFT primary, Qwen3-2B ARC SFT fallback; 4-bit load, BF16, LoRA rank 64, max length 4096, 40 hard-task TTT steps, 64 hard-task views, beam width 24, 192 candidates. NVARC’s repository identifies Qwen3-4B as a core component, and independent work reports 21.7% evaluation accuracy from TTT plus Product-of-Experts. ([GitHub][2])

**Expected runtime/memory:** 9.5–11.8GB VRAM; roughly 2–4 minutes per hard task after caching and adaptive scheduling.

**Leakage risk:** Low at inference, medium during validation if training-solution grids leak into prompts or task adapters. Assert that solution files are used only after generation.

**Fallback:** Bounded iterative beam instead of recursive DFS; derive tokenizer IDs; reduce views 64→48→32 and TTT 40→24→12 before switching to the 2B checkpoint.

## 3. Object-centric symbolic DSL/MDL program search

**Leak-free features/encodings:** Components, bounding boxes, topology, palette, symmetry, separators, repetition, relational predicates, exact demo replay, and leave-one-demo-out stability. All rules are induced per task, then applied to test inputs.

**Models and key hyperparameters:** Deterministic DSL beam search, 256 beam width, depth 4, 5,000 generated programs, 20-second easy and 60-second hard budget, 48 accepted candidates. Current neuro-symbolic evidence supports separating object perception, neural proposal, and symbolic verification; structural research also warns that DSL coverage matters more than merely increasing search budget. ([arXiv][5])

**Expected runtime/memory:** CPU-heavy but generally under one minute per task; negligible GPU memory.

**Leakage risk:** Low, with the main risk being spurious demonstration fit rather than label leakage.

**Fallback:** Never trust a single exact-fitting program as a guaranteed answer. Keep it as a candidate, expand primitive coverage, and route unresolved tasks to Qwen.

## 4. Actual TRM checkpoint adaptation as a diversity branch

**Leak-free features/encodings:** Grid-native embeddings and task-local augmented demonstrations only; no hidden solutions and no ARC-AGI-1/evaluation overlap introduced.

**Models and key hyperparameters:** Public 7M TRM checkpoint, 256 task-adaptation steps, 16 recursive supervision steps, `H_cycles=3`, `L_cycles=4`, batch 32, learning rate `1e-4`, eight candidates, maximum 64 hard tasks. TRM is reported at 8% ARC-AGI-2, while full ARC pretraining takes about three days on four H100s; therefore only checkpoint adaptation is practical here. ([GitHub][8])

**Expected runtime/memory:** Under 3GB VRAM after Qwen is unloaded; moderate additional runtime.

**Leakage risk:** Medium if checkpoint provenance or training composition is unclear. Require a source/license/checksum manifest.

**Fallback:** Skip the branch and record `dependency_blocked`; never emit a heuristic reranker under the TRM name.
