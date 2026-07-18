# Metric and Validation Contract Repair

- skill_id: `metric_validation_contract_repair`
- status: `candidate`
- version: `12`
- problem_types: metric, validation
- tags: metric_or_validation_error, self_improvement

## Summary
Tighten metric parsing and sample alignment before expensive candidate training.

## Procedure
1. Detect `metric_or_validation_error` from run status, submit attempts, metrics, diagnostics, and top1 gap signals.
2. Hypothesis: Metric/schema ambiguity is creating invalid confidence in candidate submissions.
3. Experiment: Strengthen metric contract parsing, sample-submission alignment, and early scoring checks.
4. Success metric: Invalid metric/schema runs fail before training expensive candidates.
5. Preserve Kaggle guardrails: no rule acceptance, no secret writes, no unguarded submit side effects.
6. Record outcome back into skill_evaluations so promotion/demotion can use observed fitness.
