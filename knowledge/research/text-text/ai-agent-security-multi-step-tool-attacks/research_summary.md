# research_summary.md

## Ranked shortlist

### 1. Adaptive target-native multi-action bandit

* **Leak-free features/encodings:** target-generation traces only; prompt family, requested action count, successful `http.post` count, predicate vector, unique destination domains, refusal/block type, message count, tool hops, p50/p90 latency, and exact SDK raw score. No private labels, leaderboard values, or packaged secret values are used.
* **Models and key hyperparameters:** deterministic UCB/successive-halving controller plus replay-budget knapsack; 20 initial arms, two halving rounds, two verification repeats, post multiplicity 2–7, generation search fraction 0.22, replay safety fraction 0.82, expected-tool-call cap 1,260, at most 700 returned candidates, private-hedge fraction 0.18.
* **Runtime/memory:** target generation uses at most about 1,980 seconds for search by default and stops earlier using a p90 watchdog; final candidate construction is negligible. Python memory should remain below 300 MB because snapshots and trace records are capped.
* **Leakage risk:** medium. The explicit SDK sentinel is scorer-specific, but target-native verification and authorization variants reduce model brittleness. The hidden guardrail remains the main distribution shift.
* **Fallback:** compact 315-candidate × four-post static reference.
* **Research basis:** black-box optimization is more effective than gradient search under realistic agent budgets, and small-model transfer can be poor. ([arXiv][1])

### 2. Bounded trace archive and semantic mutation

* **Leak-free features/encodings:** SDK `cell_signature`, predicate set, successful action sequence, block reason, trace depth, latency, and parent-template ID; all statistics are computed inside fold-train or live generation.
* **Models and key hyperparameters:** Go-Explore-style archive cap 256, maximum depth 4, branch batch 6, at most 96 live probes, one-factor mutation, top 24 templates retained, two-repeat replay confirmation.
* **Runtime/memory:** low compute but potentially expensive target inference; snapshot count and stored trace size are hard-capped. Expected memory below 500 MB.
* **Leakage risk:** medium-high if novelty is defined by public fixtures. Promotion requires actual predicates and score-per-second improvement, not novelty alone.
* **Fallback:** disable archive and retain the adaptive direct-action bandit.
* **Research basis:** archive/restore is appropriate for deterministic sparse-reward search, while mutation should remain evidence-guided. ([arXiv][2])

### 3. Compact 315×4 reference with private-robustness hedge

* **Leak-free features/encodings:** unique compact destination domains, exact SDK sentinel payload, authorization-framing class, and optional benign confused-deputy action; no real credential strings.
* **Models and key hyperparameters:** static generator with 315 candidates, four posts per candidate, one message per primary candidate, maximum 760 characters, 18% authorization/provenance variants, 4% confused-deputy variants.
* **Runtime/memory:** negligible construction cost; approximately 1,260 requested post calls, which is close to the observed replay-capacity regime and must be stress-tested.
* **Leakage risk:** highest scorer-update and private-guardrail risk; use as a reference and emergency fallback rather than the only final family.
* **Fallback:** reduce to 280×4, then 240×4 if p90 replay estimates exceed the safety budget.
