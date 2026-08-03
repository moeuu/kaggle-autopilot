# Leaderboard Implementation Anomaly Repair

- skill_id: `leaderboard_implementation_anomaly_repair`
- status: `candidate`
- version: `509`
- problem_types: submission, runtime_fidelity, leaderboard
- tags: leaderboard_implementation_anomaly, self_improvement

## Summary
Treat last-place-like outcomes as execution/submission defects and prove runtime fidelity before resuming model search.

## Procedure
1. Detect `leaderboard_implementation_anomaly` from run status, submit attempts, metrics, diagnostics, and top1 gap signals.
2. Hypothesis: Near-last rank or collapsed online score usually indicates that the evaluated candidate and executed submission path are not equivalent.
3. Experiment: Trace hidden-test inputs, loaded assets, fallbacks, prediction distribution, row/ID alignment, output selection, and metric scale before allowing another model-search submission.
4. Success metric: The repaired run emits runtime-fidelity evidence and leaves the bottom-decile anomaly band without resubmitting the same artifact hash.
5. Preserve Kaggle guardrails: no rule acceptance, no secret writes, no unguarded submit side effects.
6. Record outcome back into skill_evaluations so promotion/demotion can use observed fitness.
