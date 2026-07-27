# Iteration Metrics Recovery

- skill_id: `iteration_metrics_recovery`
- status: `candidate`
- version: `19`
- problem_types: runtime, metrics
- tags: no_iteration_metrics, self_improvement

## Summary
Ensure every iteration emits metrics or an explicit failure context.

## Procedure
1. Detect `no_iteration_metrics` from run status, submit attempts, metrics, diagnostics, and top1 gap signals.
2. Hypothesis: The runtime is losing metrics before the supervisor can make informed decisions.
3. Experiment: Harden kernel/runtime exit handling so metrics.json and diagnostics.md are emitted on every path.
4. Success metric: Every iter-* directory has metrics.json or an explicit failure_context artifact.
5. Preserve Kaggle guardrails: no rule acceptance, no secret writes, no unguarded submit side effects.
6. Record outcome back into skill_evaluations so promotion/demotion can use observed fitness.
