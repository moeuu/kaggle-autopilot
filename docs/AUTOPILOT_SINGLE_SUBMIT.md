# Autopilot Submission Loop Policy

This document explains the current submission policy in autopilot.

## Policy

Autopilot uses readiness score (SRS) as the primary loop decision.
Submission is controlled by `submission_gate` (default: `always`).

Decision rule:
1. Iterate training/evaluation up to `max_iterations`
2. Compute SRS from metric mean/std/CI (+ optional drift penalty)
3. Submit only when gate allows
4. Use submission score/rank as secondary guardrails (e.g. top1-tier stop, rank-based major-overhaul trigger)

## Why This Exists

- keeps loop control deterministic and reproducible from offline evaluation reports
- still leverages online leaderboard feedback as a safety/strategy signal
- reduces noisy iteration decisions

## Related Guards

Before submission, autopilot applies:
- rules acceptance check
- file format validation against `sample_submission.csv` (with `submission_format.md` / `overview.md` fallback hints when sample is placeholder/header-only)
- duplicate submission hash check
- repeated error fingerprint abort on submit failures

## Current CLI

```bash
uv run kagglebot autopilot <competition> --compute local_gpu
```

Useful options:
- `--max-iterations N`
- `--force-submit`
- `--score-source auto|holdout|cv|test`

## Notes

- `--agent` is not used in autopilot CLI.
- `--submit` flag is not required; autopilot submits by default.
- Rules acceptance is always manual in browser.
