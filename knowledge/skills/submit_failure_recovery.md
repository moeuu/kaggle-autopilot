# Submit Failure Recovery

- skill_id: `submit_failure_recovery`
- status: `candidate`
- version: `19`
- problem_types: submission, guardrails
- tags: submit_failed, self_improvement

## Summary
Classify submit failures, preserve artifacts, and choose file/notebook retry mode safely.

## Procedure
1. Detect `submit_failed` from run status, submit attempts, metrics, diagnostics, and top1 gap signals.
2. Hypothesis: Submit failures contain recoverable mode, path, or API classifications.
3. Experiment: Improve failure classification and fallback selection from submit_attempts.jsonl and diagnostics.
4. Success metric: Submit-failed runs produce a classified retry or a non-retryable reason with artifact links.
5. Preserve Kaggle guardrails: no rule acceptance, no secret writes, no unguarded submit side effects.
6. Record outcome back into skill_evaluations so promotion/demotion can use observed fitness.
