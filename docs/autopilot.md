# Autopilot

Autopilot is a non-interactive Kaggle loop with readiness-score iteration control.
It always follows this high-level path:

1. Bootstrap competition context
2. Plan and implement initial kernel via `codex -> gpt -> codex`
3. Train/evaluate per iteration
4. Improve if needed
5. Submit and use submission outcomes as secondary guardrails

## Quick Start

```bash
uv run kagglebot autopilot https://www.kaggle.com/competitions/<slug> \
  --compute local_gpu
```

Supported compute values:
- `local_gpu`
- `kaggle_gpu`
- `kaggle_tpu`

## Planning Flow (codex -> gpt -> codex)

Autopilot planning is fixed to:

1. Codex (`gpt-5.3-codex`, extra high): reads local context and writes a brief.
2. GPT (`gpt-5.2`, extra high): performs strategy planning with live web search when available.
3. Codex (`gpt-5.3-codex`, extra high): implements kernel code from frozen instructions.

The GPT stage now produces and the pipeline persists:
- `artifacts/<slug>/context/research_sources.jsonl`
- `artifacts/<slug>/context/research_summary.md`
- `knowledge/research/<problem_type>/<slug>/research_sources.jsonl` (persistent copy)
- `knowledge/research/<problem_type>/<slug>/research_summary.md` (persistent copy)
- `artifacts/<slug>/plan.json`

`research_sources.jsonl` stores per-source research metadata (query, top URLs, publish dates, takeaway, and extracted technique).

## Iteration and Submission Policy

Per iteration, autopilot does:
1. Train and evaluate
2. Write metrics and diagnostics
3. Check top1-tier condition (direction-aware)
4. If not top1-tier and iterations remain, run improvement

Submission behavior:
- Default: submit every iteration
- `submission_gate` is activated only when rules indicate submission-count limits
- Loop decision uses readiness score (SRS); submission score/rank are secondary guardrails
- Repeated submit-error fingerprints are aborted safely

## Important Defaults

- `--max-iterations`: default runtime behavior is 3 unless overridden by CLI
- `--internet`: default `on` for autopilot
- Submission in autopilot is enabled by default
- `--agent` and `--submit` are not part of autopilot CLI
- Submission schema handling is flexible:
  - supports ID-based alignment when an ID column exists
  - falls back to row-order alignment when no reliable ID column exists
  - supports multi-target submission columns at I/O/validation layer
- CV strategy auto-selects from:
  - `TimeSeriesSplit` when reliable time columns exist
  - `GroupKFold` when group-like columns are detected
  - `StratifiedKFold` for classification
  - `KFold` otherwise
- Model-family selection can be customized by `KAGGLEBOT_MODEL_CANDIDATES`

## Main Flags

```text
--compute local_gpu|kaggle_gpu|kaggle_tpu   (required)
--accelerator auto|gpu|tpu
--score-source auto|holdout|cv|test
--holdout-frac FLOAT
--cv-folds INT
--max-iterations INT
--max-total-min INT
--patience INT
--min-improvement FLOAT
--force-submit
--internet auto|off|on
--verify-cmd "..."
--strict-accelerator
--resume-run-id RUN_ID
--resume-latest
```

## Artifacts

Key files:

- `artifacts/<slug>/plan.json`
- `artifacts/<slug>/context/research_sources.jsonl`
- `artifacts/<slug>/context/research_summary.md`
- `artifacts/<slug>/context/research_storage.json`
- `artifacts/<slug>/context/agent/brief_for_strategy.md`
- `artifacts/<slug>/context/agent/strategy_plan.md`
- `artifacts/<slug>/context/agent/codex_instructions.md`
- `artifacts/<slug>/runs/<run-id>/iter-<k>/metrics.json`
- `artifacts/<slug>/runs/<run-id>/iter-<k>/diagnostics.md`
- `knowledge/research/<problem_type>/<slug>/research_sources.jsonl` (authoritative persistence)
- `knowledge/research/<problem_type>/<slug>/research_summary.md` (authoritative persistence)

## Notes

- Rules acceptance is always manual in browser.
- For local kernel training (`local_gpu`), terminal logs show elapsed/ETA and stage progress (`seed i/N`, `fold j/K`, `step s/T`) when patterns are detectable from kernel output.
- For Kaggle kernel training, execution and logs are tracked through kernel run artifacts.
- If autopilot crashes, restart with `--resume-run-id <run-id>` or `--resume-latest`.
