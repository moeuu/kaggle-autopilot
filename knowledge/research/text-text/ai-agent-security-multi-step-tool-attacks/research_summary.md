# research_summary.md

## Ranked shortlist

### 1. Adaptive K1/K2/K3 target-native bandit with replay-budget knapsack

**Leak-free features/encodings:** prompt-family ID, packed-versus-chained action shape, requested and successful post count, exact predicate-weight vector, SDK score-cell hash, unique simulated `.invalid` destinations, refusal/block category, tool hops, message count, p50/p90 latency, and raw score per second. All arm statistics are fit on fold-train or live-probe traces and then frozen; no private labels, leaderboard values, fixture secrets, or test proxies are used.

**Models and key hyperparameters:** deterministic successive-halving/UCB controller; 16 initial arms, top 6 in round two, top 3 in round three, at most 25 live probe chains; K in {1,2,3} as fixed arm definitions; 75-second generation margin; 600-second replay margin; 700 candidates; 1,260 expected-tool-call cap; K2 promotion threshold 1.05× K1 lower-confidence raw/second; K3 threshold 1.10×; 12% reference floor and 8% genuine read-to-post floor. The final integer knapsack maximizes verified lower-confidence raw score under p90 replay seconds and quota constraints.

**Expected runtime/memory:** target inference dominates. Generation probing is bounded to roughly 5–15% of 9,000 seconds and exits early; candidate construction is negligible. Python RSS should remain below 500 MB. **Leakage risk:** medium, driven by public-guardrail calibration and scorer-specific sentinel use; private-provenance quotas and exact fresh replay reduce but do not eliminate it. **Fallback:** reference K1 644, then 600/560/500 if the p90 watchdog predicts deadline risk. Research supports target-native adaptive black-box optimization over static or small-surrogate transfer. ([arXiv][1])

### 2. Bounded trace archive and deterministic one-factor mutation

**Leak-free features/encodings:** canonical score-cell signature, predicate set, successful tool sequence, guardrail outcome, archive depth, parent template ID, latency bucket, and score-per-second delta. Statistics are computed only from supplied sandbox traces. **Models and key hyperparameters:** Go-Explore-style archive cap 64, maximum depth 3, branch batch 4, maximum 24 live probes, one mutation dimension per child, top 12 templates frozen, two-repeat verification, and promotion only for a successful predicate plus positive lower-confidence score-per-second gain.

**Expected runtime/memory:** below 500 MB with compact trace summaries; inference cost is bounded by 24 probes. **Leakage risk:** medium-high if public fixture novelty is overvalued, so novelty alone never promotes a chain. **Fallback:** disable the archive and retain the adaptive direct-action bandit. Go-Explore and GPTFuzzer justify archive/restore and controlled mutation, but the competition requires much tighter bounds. ([arXiv][2])

### 3. Reference K1 boundary anchor

**Leak-free features/encodings:** unique compact simulated destination domain, one direct `http.post`, prompt-template ID, contract status, and replay latency; no learned encoding. **Models and key hyperparameters:** deterministic static generator, 644 candidates, one message and one post per candidate, maximum 220 characters, reserved `.invalid` destinations, zero live search, and lower-count fallback profiles at 600, 560, and 500.

**Expected runtime/memory:** negligible construction memory; replay dominates. **Leakage risk:** high private-guardrail/scorer-update risk and low upside, but lowest engineering risk. **Fallback if a dependency is unavailable:** this route uses only the bundled SDK and Python standard library. It remains the mandatory non-regression baseline and emergency rollback, not the winner-level final pipeline.
