# Guardrails Checklist

Use this checklist when changing autopilot behavior.

## 1. Submission Safety

- [ ] Rules acceptance is checked before submission.
- [ ] Submission artifact is validated against `sample_submission.*` and context hints (`submission_format.md` / `overview.md`) when sample is placeholder/header-only.
- [ ] Duplicate hash check is enforced (`submissions/ledger.jsonl`).
- [ ] Submission rate limiting is enforced.
- [ ] Repeated submit-error fingerprint abort is enforced.
- [ ] `--force-submit` behavior is explicit and logged.

## 2. Planning Pipeline Safety

- [ ] Planning order remains `codex -> oracle(latest-pro) -> codex`; Oracle failure, invalid output, or unverified archival blocks implementation without a Codex fallback.
- [ ] GPT output sections are validated (`STRATEGY`, `RESEARCH_SOURCES_JSONL`, `RESEARCH_SUMMARY_MD`, `PLAN_JSON`, `CODEX_INSTRUCTIONS`).
- [ ] Research artifacts are persisted to `context/research_*.{jsonl,md}`.
- [ ] Plan validation rejects malformed or underspecified pipeline configs.

## 3. Data Leakage and Evaluation

- [ ] Feature statistics are fit on train folds and only applied to val/test.
- [ ] No target leakage is introduced in encoders/features.
- [ ] Evaluation direction (min/max) is consistent with metric.
- [ ] CV/holdout/test behavior matches resolved plan settings.

## 4. Operational Safety

- [ ] Non-interactive behavior preserved.
- [ ] No browser automation or rules auto-accept added.
- [ ] Secrets are never logged or persisted in repo files.
- [ ] Error messages remain actionable.

## 5. Resource and Retry Controls

- [ ] Max iteration behavior remains bounded and explicit.
- [ ] Kernel/GPU capacity failures have bounded retries.
- [ ] Autofix loops are bounded.
- [ ] Long-running stages emit progress logs.

## 6. Docs and Tests

- [ ] Update docs when behavior changes (`README.md`, `docs/*`).
- [ ] Update/extend tests for new planning-output contracts.
- [ ] Run `uv run pytest -q` and `uv run ruff check .`.
- [ ] When runner/orchestration behavior changes, also run `uv run pytest -q -m "slow and not competition_artifact"`.
