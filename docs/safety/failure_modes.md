# Failure Modes

This document lists common operational failures and current mitigations.

## 1. Kaggle Authentication / Rules

### Missing or invalid credentials
Symptoms:
- `401 Unauthorized`
- Kaggle CLI fails immediately

Mitigation:
- configure `~/.kaggle/kaggle.json` with correct permissions (`600`)
- or use `KAGGLE_USERNAME` / `KAGGLE_KEY`

### Rules not accepted
Symptoms:
- submit blocked with rules-related error

Mitigation:
- manually accept rules in browser
- rerun command

## 2. Network / Kaggle API Instability

Symptoms:
- transient CLI failures
- timeout / connection reset

Mitigation:
- bounded retries on transient cases
- clear error surfaced when retries exhausted

## 3. Kernel Execution Failures (`kaggle_gpu` / `kaggle_tpu`)

Symptoms:
- kernel status becomes failed/cancelled
- output files missing

Mitigation:
- kernel polling with timeout
- kernel output validation
- bounded retry and autofix path

## 4. Local GPU Availability (`local_gpu`)

Symptoms:
- GPU unavailable error or fallback pressure

Mitigation:
- use `--strict-accelerator` to fail fast when GPU is required
- otherwise allow controlled fallback path where applicable

## 5. Submission Guard Failures

### Duplicate submission
Symptoms:
- duplicate hash detected

Mitigation:
- change predictions or use `--force-submit` only when intentional

### Rate limit / cooldown
Symptoms:
- submission delayed or blocked

Mitigation:
- wait for cooldown window
- avoid repeated manual trigger loops

### Repeated submit error fingerprint
Symptoms:
- same submission error repeats

Mitigation:
- autopilot aborts that submit path to avoid infinite loops
- use autofix and rerun with changed code/output
- inspect `artifacts/<slug>/runs/<run-id>/submit_failure_context.json` to see whether the next repair should target the submission artifact, submit mode/kernel path, or a manual blocker

## 6. Planning Contract Failures

Symptoms:
- strategy stage fails quality gate
- missing structured sections (`PLAN_JSON`, research outputs)

Mitigation:
- automatic retry with stricter prompt
- fallback strategy generation when GPT rate-limited

## 7. Data / Schema Mismatch

Symptoms:
- missing expected columns
- train/test mismatch

Mitigation:
- enforce alignment helpers in kernel
- robust feature-column reconciliation before inference

## 8. Autofix Loop Risks

Symptoms:
- repeated runtime failures

Mitigation:
- capped autofix attempts
- explicit final error surfacing when budget exhausted
