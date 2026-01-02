# kagglebot

Safe, non-interactive automation for Kaggle competitions:
bootstrap → implement → train → submit, plus a score-gated autopilot loop.

## Prerequisites

- Python 3.11+
- `uv` installed
- Kaggle CLI on PATH
- Kaggle credentials (`~/.kaggle/kaggle.json` or env vars)
- Rules accepted manually in Kaggle UI

## Install

```bash
uv sync
```

## Core Commands

```bash
uv run kagglebot bootstrap <competition>
uv run kagglebot implement <competition> --agent codex
uv run kagglebot train <competition> --compute local_cpu
uv run kagglebot submit <competition> -f <csv> -m "message" --force-submit
uv run kagglebot autopilot https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques \
  --agent codex \
  --compute kaggle_gpu \
  --submit
```

## Autopilot Plan

Autopilot lets the agent decide target metric/score/direction and stores them in
`artifacts/<slug>/plan.json`. Edit that file to override targets or evaluation settings
before re-running.

## Compute Targets

- `local_cpu`
- `local_gpu` (CUDA/MPS if available)
- `kaggle_gpu` (Kaggle notebook GPU)
- `kaggle_tpu` (Kaggle notebook TPU)

Use `--accelerator auto|cpu|gpu|tpu` to keep compute/accelerator consistent.

## Safety Defaults

- `--dry-run` skips external commands (Kaggle CLI, Codex).
- Submission is gated by target score in autopilot unless the agent sets a submit-at-final policy in `plan.json`.
- Dedupe ledger stored in `artifacts/<slug>/submissions/ledger.jsonl`.

## Artifacts Layout

```
artifacts/<slug>/
  meta.json
  context/
    rules_url.txt
    dataset_profile.json
    sample_submission.csv
    top1_public.json
  prompts/
    codex_plan_and_baseline.md
    codex_improve.md
  runs/<runid>/
    run.json
    iter-<k>/
      metrics.json
      diagnostics.md
      submission.csv
      agent/
  submissions/
    ledger.jsonl
```

## Docs

- `docs/autopilot.md`
- `docs/knowledge.md`
- `docs/taxonomy.md`

## Testing

```bash
uv run pytest -q
```
