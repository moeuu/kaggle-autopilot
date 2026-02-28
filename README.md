# kagglebot

Safe, non-interactive automation for Kaggle competitions with readiness-score-driven autopilot.

## Prerequisites

- Python 3.11+
- `uv` installed ([install](https://github.com/astral-sh/uv))
- Kaggle CLI on PATH
- Kaggle credentials (`~/.kaggle/kaggle.json` or env vars)
- **Competition rules accepted manually in browser** (required once per competition)
- Rules/overview/data are fetched from Kaggle during download; you can override rules with `--rules-file` (md/txt/html).

## Install

```bash
uv sync
```

## Quick Start (Minimal Args)

Run autopilot with a single command:

```bash
uv run kagglebot autopilot https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques \
  --compute kaggle_gpu
```
Optional: add `--rules-file /path/to/rules.md` (md/txt/html) to override fetched rules text.

This will:
1. **Bootstrap**: Download data, profile dataset, query Knowledge Base for similar competitions
2. **Plan**: Codex (gpt-5.3-codex, extra high) summarizes context, GPT (gpt-5.2, extra high) runs research + frozen plan, Codex (gpt-5.3-codex, extra high) implements it
3. **Initial model**: Agent implements an initial solution in `artifacts/<slug>/kernel/kernel.py` (all compute modes)
4. **Iterate**: Train → evaluate → diagnose → improve (default 3 iterations; override with `--max-iterations`)
5. **Submit + Decide**: Submit each iteration, wait for Kaggle score, then decide continue/stop
6. **Log**: Print Top1 public score and agent prompt/response to the terminal

Schema/method flexibility:
- Target/schema inference is file-name agnostic (not fixed to exact `train.csv`/`test.csv` naming only)
- Multi-target sample submissions are supported at I/O/validation level
- ID-based and row-order-based submission alignment are both supported
- Metric handling supports a broader set (e.g. AUC, logloss, F1, precision/recall, AP, RMSE/MAE/MAPE/R2)
- CV strategy auto-selection supports `TimeSeriesSplit` / `GroupKFold` / `StratifiedKFold` / `KFold`
- Model family selection is plugin-like with optional families (XGBoost/LightGBM if installed)

**Safe defaults**:
- Default max iterations: 3 (`--max-iterations` to override)
- No training time limit (accuracy-first)
- Internet ON by default (disable with `--internet off`)
- Each iteration submits and waits for Kaggle score before stop/continue decision

## Manual Commands

For more control, use individual commands:

```bash
# Bootstrap competition
uv run kagglebot bootstrap <competition>

# Implement solution
uv run kagglebot implement <competition>

# Train locally
uv run kagglebot train <competition> --compute local_gpu

# Submit manually
uv run kagglebot --force submit <competition> -f submission.csv -m "message"
```

## Plan Configuration

Autopilot creates `artifacts/<slug>/plan.json` with agent-defined targets:

```json
{
  "target_metric": "rmse",
  "target_score": 0.13,
  "target_direction": "minimize",
  "score_source": "holdout",
  "holdout_frac": 0.2,
  "cv_folds": 5,
  "seed": 42,
  "internet": "on",
  "submit_policy": "always"
}
```

**Edit `plan.json` to override** targets or evaluation settings before re-running autopilot.

## Compute Targets

- `local_gpu` - Local training (GPU preferred; falls back to CPU when unavailable)
- `kaggle_gpu` - Kaggle notebook with GPU (T4)
- `kaggle_tpu` - Kaggle notebook with TPU (v3-8)

Use `--accelerator auto|gpu|tpu` to force specific accelerator.

Optional environment knobs:
- `KAGGLEBOT_MODEL_CANDIDATES="catboost,xgboost,lightgbm,torch,extra_trees"` to prioritize/limit model families
- Submission sample stage selection (for competitions that publish Stage 1/2 sample files):
  - Default preference is earlier stage samples (`Stage1` before `Stage2`) when multiple staged files exist.
  - `KAGGLEBOT_SUBMISSION_STAGE=<int>` or `KAGGLEBOT_SAMPLE_SUBMISSION_STAGE=<int>` forces preferred stage.
- Large competition download strategy (auto-enabled):
  - Kagglebot now probes competition file sizes first.
  - If total size is large (default threshold: `8 GiB`), downloads switch to per-file mode with retry/backoff.
  - When per-file mode hits `429 Too Many Requests`, Kagglebot retries per-file mode (no single-shot fallback).
  - `KAGGLEBOT_DOWNLOAD_SPLIT_THRESHOLD_BYTES=<int>` override split threshold bytes.
  - `KAGGLEBOT_DOWNLOAD_RETRY_ATTEMPTS=<int>` retry attempts per command (`0` = unlimited; default `8`).
  - `KAGGLEBOT_DOWNLOAD_RATE_LIMIT_RETRY_ATTEMPTS=<int>` retry attempts for `429` in per-file mode (`0` = unlimited; default `8`).
  - `KAGGLEBOT_DOWNLOAD_RETRY_BACKOFF_SEC=<float>` base retry backoff seconds (default `2.0`).
  - `KAGGLEBOT_DOWNLOAD_RETRY_MAX_BACKOFF_SEC=<float>` max retry backoff seconds (default `120.0`).
  - `KAGGLEBOT_DOWNLOAD_MIN_INTERVAL_SEC=<float>` minimum delay between split-download API requests (default `0.0`).
- Training progress logging (applies even when kernel code is quiet):
  - `KAGGLEBOT_TRAIN_PROGRESS=1|0` (default `1`) enable/disable forced periodic training logs
  - `KAGGLEBOT_PROGRESS_INTERVAL_SEC=<float>` watchdog silence threshold before emitting "no new logs" status (default `45`)
  - `KAGGLEBOT_MODEL_PROGRESS_INTERVAL_SEC=<float>` baseline per-model `fit()` tick interval (auto-adjusted by method/data size; default `12`)
  - `KAGGLEBOT_BOOSTING_LOG_EVERY=<int>` boosting eval-log period in iterations (`0` = auto, default)
- Custom metric hook: use metric string `custom:<module_or_py_path>:<function>`
- Vision YOLO routing: if sample columns are `filename,right_place,prediction_string` and YOLO folders exist
  (`images/train`, `images/test`, `labels/train`), Kagglebot uses a detection pipeline instead of tabular models.
- Vision training knobs:
  - `KAGGLEBOT_YOLO_PRETRAIN=1|0` (default `1`) toggles pretrained detector initialization.
  - `KAGGLEBOT_YOLO_EPOCHS=<int>` overrides detector epochs (time-budget caps still apply).

## Safety Guardrails

- ✅ **Readiness-score-driven loop**: stop/continue uses SRS (offline metric + uncertainty), with optional submission/rank guardrails
- ✅ **Strict local validation before submit**: Column order, row count, ID integrity, numeric prediction checks
- ✅ **Duplicate prevention**: SHA256 hash check against `submissions/ledger.jsonl`
- ✅ **Rate limiting**: 5-min cooldown between submissions
- ✅ **No infinite submit loop**: Same submit-error fingerprint aborts the run immediately
- ✅ **Controlled retries**: Transient submit errors retry up to 3 times with backoff; permanent errors abort immediately
- ✅ **No rule automation**: Must accept rules manually in browser
- ✅ **Dry-run mode**: `--dry-run` skips external API calls (Kaggle CLI, Codex)

## Top1 Public Leaderboard (Reference)

The public leaderboard leader's score is fetched and stored in `context/top1_public.json`.
Autopilot uses this as a reference signal (for diagnostics and rank-based major-overhaul guardrails), while primary loop control uses readiness score (SRS).

## Artifacts Layout

```
artifacts/<slug>/
  meta.json                      # Competition metadata
  plan.json                      # Agent-defined targets (editable)
  context/
    dataset_profile.json         # Dataset statistics
    research_sources.jsonl       # Strategy web-research log (working copy)
    research_summary.md          # Ranked research shortlist (working copy)
    research_storage.json        # Mapping to persisted knowledge paths
    sample_submission.csv        # Required submission format
    sample_submission_head.csv   # Head of sample submission
    top1_public.json             # Leaderboard leader snapshot
    rules_url.txt                # Competition rules URL
    rules.md                     # Rules markdown (fetched or from --rules-file)
    rules.html                   # Rules HTML (if provided)
    overview.md                  # Competition overview (if available)
    data.md                      # Data description (if available)
    submission_format.md         # Submission format (if available)
    knowledge_hints.txt          # Similar competitions + hints
    agent/
      brief_for_strategy.md      # Codex brief
      strategy_plan.md           # GPT strategy section
      codex_instructions.md      # GPT implementation instructions
      strategy_transcript.txt    # Raw GPT stage output
  kernel/
    kernel.py                    # Authoritative kernel entrypoint (all compute modes)
  prompts/
    codex_plan_and_implement.md   # Initial plan + implementation prompt
    codex_improve.md             # Improvement iteration prompt
  kernels/
    <run-id>/                    # Kaggle kernel workspace
  runs/<run-id>/
    run.json                     # Run configuration and status
    run_state.json               # Submit stage state (attempted/ok/last fingerprint)
    submit_attempts.jsonl        # Submit attempts (success/failure/retry/abort/skip)
    iter-<k>/
      metrics.json               # Offline evaluation results
      diagnostics.md             # Agent-readable performance analysis
      submission.csv             # Predictions for this iteration
      agent/
        prompt.md                # Codex input
        codex_last_message.txt   # Codex output summary
  submissions/
    ledger.jsonl                 # Deduplication log (append-only)
```

Knowledge Base lives in:
- `knowledge/kb.sqlite`
- `knowledge/taxonomy.yml`
- `knowledge/research/<problem_type>/<slug>/research_sources.jsonl` (persistent)
- `knowledge/research/<problem_type>/<slug>/research_summary.md` (persistent)

## Documentation

For detailed guides, see:
- [docs/autopilot.md](docs/autopilot.md) - Autopilot usage, configuration, troubleshooting
- [docs/knowledge.md](docs/knowledge.md) - Knowledge Base system for cross-competition learning
- [docs/taxonomy.md](docs/taxonomy.md) - Tag taxonomy for competition similarity
- [docs/architecture.md](docs/architecture.md) - Control flow and safety gates (developer-focused)

## Testing

```bash
uv run pytest -q
```

## Example Workflow

```bash
# 1. Run autopilot (will bootstrap, plan, iterate, and submit when target met)
uv run kagglebot autopilot titanic --compute kaggle_gpu

# 2. If needed, edit plan.json to adjust target_score or evaluation strategy
nano artifacts/titanic/plan.json

# 3. Re-run autopilot with adjusted target
uv run kagglebot autopilot titanic --compute kaggle_gpu

# 4. Resume a crashed run (same competition)
uv run kagglebot autopilot titanic --compute kaggle_gpu --resume-run-id <run-id>
# or resume the latest run automatically
uv run kagglebot autopilot titanic --compute kaggle_gpu --resume-latest

# 5. Check Knowledge Base for learnings
uv run kagglebot knowledge show titanic

# 6. Find similar competitions
uv run kagglebot knowledge search --tag tabular --tag binary --limit 5
```

## Notes

- **Non-interactive**: No prompts for input. All decisions via CLI flags or `plan.json`.
- **Crash recovery**: use `--resume-run-id <run-id>` (from `artifacts/<slug>/runs/<run-id>/`) or `--resume-latest` to continue a prior run.
- **Submit resume behavior**: resume can continue submitting new iteration outputs in the same run; exact same submission path is skipped unless forced.
