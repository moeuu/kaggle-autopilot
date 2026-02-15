# kagglebot

Safe, non-interactive automation for Kaggle competitions with top1-gated autopilot.

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
2. **Plan**: Codex (gpt-5.3-codex, extra high) summarizes context, GPT (gpt-5.2, extra high) designs strategy, Codex (gpt-5.3-codex, extra high) implements it
3. **Initial model**: Agent implements an initial solution (local compute in `kagglebot/solver/`, kaggle_gpu/kaggle_tpu in `artifacts/<slug>/kernel.py`)
4. **Iterate**: Train → evaluate → diagnose → improve (default 1 iteration; override with `--max-iterations`)
5. **Submit**: Auto-submit when top1-tier (or at final iteration)
6. **Log**: Print Top1 public score and agent prompt/response to the terminal

**Safe defaults**:
- Default max iterations: 1 (`--max-iterations` to override)
- No training time limit (accuracy-first)
- Internet ON by default (disable with `--internet off`)
- Submit only when top1-tier or at final iteration

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
uv run kagglebot submit <competition> -f submission.csv -m "message"
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
  "submit_policy": "on_target_only"
}
```

**Edit `plan.json` to override** targets or evaluation settings before re-running autopilot.

## Compute Targets

- `local_gpu` - Local training (GPU preferred; falls back to CPU when unavailable)
- `kaggle_gpu` - Kaggle notebook with GPU (T4)
- `kaggle_tpu` - Kaggle notebook with TPU (v3-8)

Use `--accelerator auto|gpu|tpu` to force specific accelerator.

## Safety Guardrails

- ✅ **Top1-gated submission**: Submit only when offline score meets/exceeds top1 (direction-aware)
- ✅ **Strict local validation before submit**: Column order, row count, ID integrity, numeric prediction checks
- ✅ **Duplicate prevention**: SHA256 hash check against `submissions/ledger.jsonl`
- ✅ **Rate limiting**: 5-min cooldown between submissions
- ✅ **No infinite submit loop**: Same submit-error fingerprint aborts the run immediately
- ✅ **Controlled retries**: Transient submit errors retry up to 3 times with backoff; permanent errors abort immediately
- ✅ **No rule automation**: Must accept rules manually in browser
- ✅ **Dry-run mode**: `--dry-run` skips external API calls (Kaggle CLI, Codex)

## Top1 Public Leaderboard (Heuristic Gate)

The public leaderboard leader's score is fetched and stored in `context/top1_public.json`.
Autopilot uses it as a heuristic gate:
- Minimize metrics: submit when offline score <= top1
- Maximize metrics: submit when offline score >= top1

Offline evaluation is a pseudo-test (holdout/CV) and is **not directly comparable** to the public leaderboard; the heuristic gate is a safety check, not a guarantee.
If top1-tier is not reached, autopilot iterates up to `max_iterations` (default 1) and submits the best offline candidate on the final iteration.

## Artifacts Layout

```
artifacts/<slug>/
  meta.json                      # Competition metadata
  plan.json                      # Agent-defined targets (editable)
  context/
    dataset_profile.json         # Dataset statistics
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
  kernel.py                      # Kaggle kernel entrypoint (for kaggle_gpu/kaggle_tpu)
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

# 4. Check Knowledge Base for learnings
uv run kagglebot knowledge show titanic

# 5. Find similar competitions
uv run kagglebot knowledge search --tag tabular --tag binary --limit 5
```

## Notes

- **Non-interactive**: No prompts for input. All decisions via CLI flags or `plan.json`.
- **Submit resume behavior**: If a run already attempted submit once, resume will skip submit in that same run. Override only with `--force-submit` (or `KAGGLEBOT_FORCE_RESUBMIT=1`).
