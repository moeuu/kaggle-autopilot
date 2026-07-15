# Preflight Failure Promotion

- skill_id: `preflight_failure_promotion`
- status: `candidate`
- version: `2`
- problem_types: orchestration, preflight
- tags: orchestration_preflight_failure, self_improvement

## Summary
Promote failures before run creation into typed incidents, autofix evidence, and tests.

## Procedure
1. Detect `orchestration_preflight_failure` from run status, submit attempts, metrics, diagnostics, and top1 gap signals.
2. Hypothesis: Failures before run creation bypass run-level autofix and disappear from run-only reports.
3. Experiment: Add a typed preflight failure boundary with durable evidence and a reusable regression test.
4. Success metric: Future preflight failures enter self-improvement and auto-repair or stop with a typed blocker.
5. Preserve Kaggle guardrails: no rule acceptance, no secret writes, no unguarded submit side effects.
6. Record outcome back into skill_evaluations so promotion/demotion can use observed fitness.
