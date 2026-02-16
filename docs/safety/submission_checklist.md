# Submission Safety Checklist

Use this checklist before real Kaggle submissions.

## 1. Rules Acceptance

- [ ] Competition rules accepted manually in browser.
- [ ] Rules URL is reachable: `https://www.kaggle.com/competitions/<slug>/rules`.

Tool behavior:
- `kagglebot` checks rules before submit.
- If not accepted, it exits with actionable guidance.

## 2. Authentication

- [ ] Kaggle CLI works: `uv run kaggle competitions list`.
- [ ] Credentials are configured via `~/.kaggle/kaggle.json` or env vars.
- [ ] Credentials are not committed into repository.

## 3. File Validation

- [ ] Submission columns match required format from `sample_submission.csv` (or `submission_format.md` / `overview.md` when sample is placeholder/header-only).
- [ ] Row count matches sample submission.
- [ ] ID alignment is correct when ID column exists.
- [ ] No NaN/inf in prediction columns.

Tool behavior:
- invalid files are rejected before CLI submit call.

## 4. Duplicate and Rate Limits

- [ ] Submission hash is not duplicate unless intentionally overridden.
- [ ] Local submit cooldown/rate checks pass.

Tool behavior:
- duplicate detection uses local ledger
- retries are bounded and repeated fingerprints are aborted

## 5. Recommended Execution

Autopilot (default submit behavior):

```bash
uv run kagglebot autopilot <competition-url-or-slug> --compute local_gpu
```

Direct submit command:

```bash
uv run kagglebot submit <competition-url-or-slug> -f <submission.csv> -m "message" --force
```

## 6. Security Constraints

- [ ] No secrets in prompts, kernel code, or logs.
- [ ] No automated rules acceptance.
- [ ] No external data usage unless competition rules explicitly allow it.
