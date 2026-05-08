# Kagglebot Autopilot Specification

**Version**: 2.0
**Status**: Active
**Last Updated**: 2026-02-15

This spec reflects the current implementation.

## 1. CLI Contract

Primary command:

```bash
uv run kagglebot autopilot <competition-url-or-slug> --compute <local_gpu|kaggle_gpu|kaggle_tpu>
```

Key points:
- `--agent` is not part of autopilot CLI
- submission is enabled by default in autopilot
- `--force-submit` exists for submit guard override cases

Common options:
- `--accelerator auto|gpu|tpu`
- `--score-source auto|holdout|cv|test`
- `--holdout-frac`, `--cv-folds`, `--seed`
- `--max-iterations`, `--patience`, `--min-improvement`, `--max-total-min`
- `--internet auto|off|on`

## 2. Planning Contract (`gpt -> gpt -> gpt`)

Autopilot planning is fixed to:

1. GPT brief (`gpt-5.5`, xhigh)
2. GPT strategy (`gpt-5.5`, xhigh)
3. GPT kernel implementation (`gpt-5.5`, xhigh)

GPT strategy output must include:
- `===STRATEGY===`
- `===RESEARCH_SOURCES_JSONL===`
- `===RESEARCH_SUMMARY_MD===`
- `===PLAN_JSON===`
- `===CODEX_INSTRUCTIONS===`

Persisted outputs:
- `artifacts/<slug>/context/research_sources.jsonl`
- `artifacts/<slug>/context/research_summary.md`
- `artifacts/<slug>/context/research_storage.json`
- `knowledge/research/<problem_type>/<slug>/research_sources.jsonl` (persistent copy)
- `knowledge/research/<problem_type>/<slug>/research_summary.md` (persistent copy)
- `artifacts/<slug>/plan.json`
- `artifacts/<slug>/context/agent/*.md|*.txt`

## 3. `PLAN_JSON` Minimum Fields

Required core fields:
- `target_metric`
- `target_direction`
- `target_score`
- `score_source`
- `holdout_frac`
- `cv_folds`
- `seed`
- `max_iterations`
- `patience`
- `min_improvement`

Required planning-extended fields:
- `pipelines` (2-4)
- `toggles`
- `evaluation_protocol` (task/metric-appropriate CV, default 5 folds, >=3 seeds, primary metric from plan/rules)
- `stop_policy` (must include `max_iterations` and exact key `error_fingerprint_abort`)

## 4. Runtime Evaluation and Stop Rules

Per iteration runtime:
1. Train model/kernel
2. Evaluate offline metric(s)
3. Persist metrics and diagnostics
4. Decide continue vs stop

Stop behavior:
- early stop when submission score reaches top1-tier
- otherwise stop at `max_iterations`

CV strategy selection:
- prefers time-aware split (`TimeSeriesSplit`) when reliable time columns are detected
- otherwise uses `GroupKFold` when group-like keys are detected
- otherwise uses `StratifiedKFold` for classification or `KFold` for other tasks

Submission behavior:
- default submit every iteration
- activate `submission_gate` only when rules indicate submission-count limits
- wait for submission outcome when submission is attempted
- use readiness score (SRS) as primary loop-decision signal; submission score/rank are secondary guardrails

## 5. Submission Guardrails

Guarded checks before submit:
- rules accepted
- submission format validation vs sample submission
- supports both ID-aligned and row-order-aligned submissions
- supports multi-target submission columns (schema-matching enforced)
- dedupe by SHA256
- submission rate limiting and retry policy
- repeated error fingerprint abort in same run

## 6. Error Fixing

Autofix path uses:
1. GPT analysis step for error strategy
2. Codex fix step for code edits

Autofix can patch `src/` when framework-level fixes are required.

## 6.1 Metric + Model Extensibility

- Built-in metrics support classification and regression aliases (AUC/logloss/F1/precision/recall/AP/RMSE/MAE/MAPE/R2).
- Custom metric plugin hook is supported via metric format:
  - `custom:<module_or_py_path>:<function>`
- Model-family preference can be configured with:
  - `KAGGLEBOT_MODEL_CANDIDATES` (comma-separated families, e.g. `catboost,xgboost,lightgbm,torch`)

## 7. Artifacts Contract

```text
artifacts/<slug>/
  plan.json
  context/
    research_sources.jsonl
    research_summary.md
    research_storage.json
    agent/
      brief_for_strategy.md
      strategy_plan.md
      codex_instructions.md
      strategy_transcript.txt
  kernel/
    kernel.py
  runs/<run-id>/iter-<k>/
    metrics.json
    diagnostics.md
    submission.csv

knowledge/
  kb.sqlite
  taxonomy.yml
  research/<problem_type>/<slug>/
    research_sources.jsonl
    research_summary.md
```

## 8. Non-Goals

- No automated Kaggle rules acceptance
- No secret storage in repository
- No browser automation
