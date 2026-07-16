# Top1 Gap Expansion

- skill_id: `top1_gap_expansion`
- status: `candidate`
- version: `8`
- problem_types: model_search, leaderboard
- tags: online_far_from_top1, self_improvement

## Summary
Broaden model family, validation, data-source, and ensemble search when public gap is large.

## Procedure
1. Detect `online_far_from_top1` from run status, submit attempts, metrics, diagnostics, and top1 gap signals.
2. Hypothesis: The current search space is too narrow for competitions with a visible public top score gap.
3. Experiment: Broaden the first-plan model family, ensemble, data-source, or public-LB proxy schedule.
4. Success metric: Median top1_gap decreases across the next comparable runs.
5. Preserve Kaggle guardrails: no rule acceptance, no secret writes, no unguarded submit side effects.
6. Record outcome back into skill_evaluations so promotion/demotion can use observed fitness.
