# Repair Pipeline Reload

- skill_id: `repair_pipeline_reload`
- status: `candidate`
- version: `503`
- problem_types: orchestration, autofix
- tags: repair_pipeline_failure, self_improvement

## Summary
Verify repository repairs and reload the exact changed source before retrying a run.

## Procedure
1. Detect `repair_pipeline_failure` from run status, submit attempts, metrics, diagnostics, and top1 gap signals.
2. Hypothesis: The repair controller, verification, or source-reload path failed before the fix became active.
3. Experiment: Harden repair verification and reload the exact repository source after a verified change.
4. Success metric: Verified source repairs are loaded by the next process and do not repeat the same fingerprint.
5. Preserve Kaggle guardrails: no rule acceptance, no secret writes, no unguarded submit side effects.
6. Record outcome back into skill_evaluations so promotion/demotion can use observed fitness.
