# Kagglebot Global Playbook

Use this playbook before planning or improving a competition run.

## Current Priorities
- no_successful_submission: Prioritize submission-mode and artifact validation fixes before model search.
- submit_failed: Improve submit fallback diagnostics, notebook/file mode inference, and retry classification.
- metric_or_validation_error: Tighten metric contract validation and fail earlier when scoring is untrusted.
- resource_or_capacity: Add cheaper smoke tests and resource-aware model schedules before expensive runs.
- online_far_from_top1: Force broader model-family search, ensembling, public-LB validation, and data-source review.

## Guardrails
- Keep submissions validated against the required sample/format.
- Do not automate joining competitions or accepting rules.
- Do not write secrets, datasets, or large artifacts to git.
- Prefer structural or architectural improvements over one-off competition hacks.
- Refactor core boundaries when repeated top1 blockers show the current architecture is the bottleneck.
