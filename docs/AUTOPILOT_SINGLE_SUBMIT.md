# Autopilot Submission Loop Policy

This document explains the current submission policy in autopilot.

## Policy

Autopilot uses readiness score (SRS) as the primary loop decision.
By default it submits every iteration.
`--submit-policy improved` disables the initial contract-probe submit and waits for an improvement over a submitted checkpoint.
`submission_gate` is only used when rules indicate submission-count limits.

Decision rule:
1. Iterate training/evaluation up to `max_iterations`
2. Compute SRS from metric mean/std/CI (+ optional drift penalty)
3. Submit after each iteration (unless a submission-limit gate is active)
4. Use submission score/rank as secondary guardrails (e.g. top1-tier stop, rank-based major-overhaul trigger)

## Why This Exists

- keeps loop control deterministic and reproducible from offline evaluation reports
- still leverages online leaderboard feedback as a safety/strategy signal
- reduces noisy iteration decisions

## Related Guards

Before submission, autopilot applies:
- rules acceptance check
- file format validation against the published sample submission file, such as `sample_submission.csv`, `.tsv`, `.jsonl`, `.parquet`, `.avro`, `.feather`, or compressed tabular variants like `.csv.zst` (with `submission_format.md` / `overview.md` fallback hints when sample is placeholder/header-only)
- duplicate submission hash check
- local submission cooldown/rate limit (`KAGGLEBOT_SUBMISSION_MIN_HOURS_BETWEEN`, default 5 minutes)
- bounded Kaggle submit CLI timeout (`KAGGLEBOT_SUBMIT_TIMEOUT_SEC`, default 300s)
- repeated error fingerprint abort on submit failures

For Code Competitions, the final send path is deliberately split into three stages:

1. Codex reviews the completed Notebook source, model/reference configuration, exact output contract, metrics, and runtime logs. The decision is bound to hashes of the reviewed evidence and must explicitly approve every check.
2. A deterministic guard re-hashes that evidence and rechecks the expected filename, known daily/rolling quota, exact Notebook/version/output duplicate identity, and local ledger. Restored scores with zero current examples and no full model-backed runtime evidence, fallback-only predictions, repeated runtime exceptions, and dependency/cache trees in Notebook Output are hard failures regardless of the Codex answer.
3. The guarded executor invokes the Kaggle API once. On API success it immediately records the exact kernel/version/output identity in the ledger before outcome polling.

When the rules do not expose a numeric daily limit, autopilot does not invent a one-submission-per-day restriction. If a numeric limit is known, Kaggle submission history is fetched again immediately before execution and failure to verify the quota fails closed.

## Current CLI

```bash
uv run kagglebot --force autopilot <competition> --compute local_gpu
```

Useful options:
- `--max-iterations N`
- `--force-submit`
- `--score-source holdout|cv`

## Notes

- `--agent` is not used in autopilot CLI.
- `--submit` flag is not required; autopilot submits by default.
- Rules acceptance is always manual in browser.
