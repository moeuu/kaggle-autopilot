# Resource Capacity Fallback

- skill_id: `resource_capacity_fallback`
- status: `candidate`
- version: `514`
- problem_types: runtime, resource
- tags: resource_or_capacity, self_improvement

## Summary
Use smoke tests and smaller schedules when GPU/session/memory capacity is unreliable.

## Procedure
1. Detect `resource_or_capacity` from run status, submit attempts, metrics, diagnostics, and top1 gap signals.
2. Hypothesis: Resource failures are consuming iterations before useful model evidence is generated.
3. Experiment: Schedule cheap smoke tests and capacity-aware model choices before expensive training.
4. Success metric: Runs with capacity signals emit a smaller retry plan instead of repeating the same failure.
5. Preserve Kaggle guardrails: no rule acceptance, no secret writes, no unguarded submit side effects.
6. Record outcome back into skill_evaluations so promotion/demotion can use observed fitness.
