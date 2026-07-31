# First Valid Submission Path

- skill_id: `first_valid_submission_path`
- status: `candidate`
- version: `508`
- problem_types: submission, validation
- tags: no_successful_submission, self_improvement

## Summary
Prioritize format validation and artifact discovery until the run has one successful submission.

## Procedure
1. Detect `no_successful_submission` from run status, submit attempts, metrics, diagnostics, and top1 gap signals.
2. Hypothesis: Submission validation/submission-mode defects are blocking learning from the leaderboard.
3. Experiment: Add focused validation or recovery that turns one failed submission class into an actionable retry.
4. Success metric: A future run with this cause reaches at least one successful submission outcome.
5. Preserve Kaggle guardrails: no rule acceptance, no secret writes, no unguarded submit side effects.
6. Record outcome back into skill_evaluations so promotion/demotion can use observed fitness.
