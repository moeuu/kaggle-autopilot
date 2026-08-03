# Ranked candidate pipeline shortlist

## 1. Paired Pareto progressive optimizer — recommended

**Leak-free features/encodings.** Freeze a grouped 25% holdout before authoring. Use only participant-visible instructions and safe metadata; keep verifier/oracle bytes opaque. Quarantine issue-flagged tasks from every primary stage. Fit word 1–2 gram and character 3–5 gram TF-IDF on outer-train prompts plus immutable candidate text. Render every skill as `name | description | body`; use the pinned R3 embedding query path and R3 cross-encoder reranker with body-only truncation. Add fold-local dense projections, lexical/full-body collision, request action/no-action and authority features, token cost, set compatibility, and actual invocation/paired/safety/latency traces. All statistics are train-fit then validation/holdout-apply.

**Models + key hyperparameters.** Exact-pinned `tencent/R3-embedding-0.6b` and `tencent/R3-rerank-0.6b`, frozen and sequential. RTX3060 defaults: FP16, sequence 1,792, OOM floor 1,024, embedding batch 2, rerank batch 1, recall 20, rerank top 8. Train a 256-dimensional dual projection head for 2 epochs (`lr=1e-4`, temperature 0.05, six hard negatives, weight decay 0.01), 32-hidden fusion and compatibility MLPs for 20 epochs (`lr=1e-3`, dropout 0.10), balanced logistic calibrators (`C=1`, 2,000 iterations), and Ridge patch value (`alpha=10`). Three optimizer iterations, two patch candidates each, at most four atomic edits, 12 lines, or 8% of the target skill. Optional Qwen3-8B-AWQ proposer is 4-bit, batch 1, temperature 0.2, top-p 0.9, 768 new tokens.

**Expected runtime/memory.** Roughly 18–22 hours with shared caches; 6–10 GB VRAM for sequential R3 stages, with the optional proposer near the 12 GB ceiling.

**Leakage risk.** High if holdout isolation, sanitized no-skill controls, quarantine, or evaluator-view separation is weakened. Routing scores are never acceptance evidence. Promotion requires real paired gain, non-negative bootstrap lower-bound change, zero critical safety failures, and bounded domain/timeout regression.

**Fallback.** Use pinned SkillRouter, then pinned MiniLM, then outer-train lexical features. If `sentence-transformers` is absent, install it through the locked project or implement an audited local adapter; do not silently fall back to generic mean pooling. If the proposer is unavailable, use deterministic failure-category patches. If official paired execution is unavailable, emit `readiness=false` and no lift claim. The official R3 code/model cards define the correct two-stage contract. ([GitHub][3])

## 2. Compact three-skill core — low-interference challenger

**Leak-free features/encodings.** Reuse the exact split, quarantine, authoring/evaluator separation, no-skill cache, generated hard negatives, R3/lexical transforms, request-risk gate, and paired traces. Merge into three roots only after semantic-parity checks against the eight-skill source portfolio.

**Models + key hyperparameters.** Same pinned R3 embedding path; 256-dimensional projection head for 2 epochs at `1e-4`; 32-hidden fusion head for 20 epochs at `1e-3`; balanced logistic calibration; six description variants per root; preferred one active skill, hard cap two; target 950 tokens and maximum 1,400; body editing disabled.

**Expected runtime/memory.** Four to six incremental hours after shared caches; 6–9 GB VRAM.

**Leakage risk.** Lower task-memorization risk, but broad-root over-activation and omitted specialist procedure are material. A strong routing score can still mask weak paired execution.

**Fallback.** Word-plus-character TF-IDF with calibrated thresholds. Split a root only after repeated paired failures identify a reusable cluster. SkillsBench reports that focused bundles can outperform exhaustive ones. ([arXiv][2])

## 3. Six domain specialists with embedded safety — breadth challenger

**Leak-free features/encodings.** Six specialist roots share the same concise authority, injection, confidentiality, least-privilege, and post-state contract. Use outer-train R3/lexical features, body collision, compatibility, token cost, request risk, invocation outcomes, and identical paired controls. No answer, verifier phrase, oracle content, or final-holdout outcome enters authoring.

**Models + key hyperparameters.** Pinned R3 embedding and reranker; 256-dimensional projection head for 2 epochs at `1e-4`; 32-hidden fusion head for 20 epochs at `1e-3`; balanced logistic calibration; preferred one and maximum two active skills; two optimizer iterations, one patch each, no more than two atomic edits, eight lines, or 5% of the target; target 750 tokens and maximum 1,200.

**Expected runtime/memory.** Six to eight incremental hours after shared caches; 6–9 GB VRAM.

**Leakage risk.** Medium: duplicated safety text can create semantic collision, context bloat, over-refusal, and harmful composition.

**Fallback.** Merge only an empirically colliding pair after semantic-parity and paired checks. If reranking is unavailable, retain dense embedding plus lexical collision. ClawsBench supports separate capability and unsafe-action accounting. ([arXiv][7])

## Promotion and late ablations

Run one full seed and three full folds for pipeline 1; share immutable embeddings, task split, no-skill baselines, and model hashes with challengers. Advance at most two libraries to expensive paired evaluation. Select by fail-fast-adjusted mean paired lift, then bootstrap lower bound, per-domain breadth, negative-delta count, unsafe-action count, over-refusal, timeout, token cost, and simplicity. After the winner is frozen, test SkillReducer-style compression and accept it only if semantic parity, mandatory safety clauses, source-task replay, and paired evidence are non-decreasing. ([arXiv][8])
