# kagglebot

Safe, non-interactive automation for Kaggle competitions with top1-gated autopilot.

## Prerequisites

- Python 3.11+
- `uv` installed ([install](https://github.com/astral-sh/uv))
- Kaggle CLI on PATH
- Kaggle credentials (`~/.kaggle/kaggle.json` or env vars)
- **Competition rules accepted manually in browser** (required once per competition)

## Install

```bash
uv sync
```

## Quick Start (Minimal Args)

Run autopilot with a single command:

```bash
uv run kagglebot autopilot https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques \
  --agent codex \
  --compute kaggle_gpu \
  --submit
```

This will:
1. **Bootstrap**: Download data, profile dataset, query Knowledge Base for similar competitions
2. **Plan**: Agent generates `plan.json` with target metric/score/direction
3. **Baseline**: Agent implements initial solution in `kagglebot/solver/`
4. **Iterate**: Train → evaluate → diagnose → improve (up to 5 iterations)
5. **Submit**: Auto-submit when top1-tier (or at iteration 5 if `--submit`)

**Safe defaults**:
- Max 5 iterations
- No training time limit (accuracy-first)
- Internet OFF for Kaggle kernels
- Submit only when top1-tier or at iteration 5 (with `--submit`)

## Manual Commands

For more control, use individual commands:

```bash
# Bootstrap competition
uv run kagglebot bootstrap <competition>

# Implement solution
uv run kagglebot implement <competition> --agent codex

# Train locally
uv run kagglebot train <competition> --compute local_cpu

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
  "submit_policy": "on_target_only"
}
```

**Edit `plan.json` to override** targets or evaluation settings before re-running autopilot.

## Compute Targets

- `local_cpu` - Local CPU training
- `local_gpu` - Local GPU (CUDA/MPS if available)
- `kaggle_gpu` - Kaggle notebook with GPU (T4)
- `kaggle_tpu` - Kaggle notebook with TPU (v3-8)

Use `--accelerator auto|cpu|gpu|tpu` to force specific accelerator.

## Safety Guardrails

- ✅ **Top1-gated submission**: Submit only when offline score meets/exceeds top1 (direction-aware)
- ✅ **Duplicate prevention**: SHA256 hash check against `submissions/ledger.jsonl`
- ✅ **Rate limiting**: 5-min cooldown between submissions
- ✅ **Max submissions**: One submission per run
- ✅ **No rule automation**: Must accept rules manually in browser
- ✅ **Dry-run mode**: `--dry-run` skips external API calls (Kaggle CLI, Codex)

## Top1 Public Leaderboard (Heuristic Gate)

The public leaderboard leader's score is fetched and stored in `context/top1_public.json`.
Autopilot uses it as a heuristic gate:
- Minimize metrics: submit when offline score <= top1
- Maximize metrics: submit when offline score >= top1

Offline evaluation is a pseudo-test (holdout/CV) and is **not directly comparable** to the public leaderboard; the heuristic gate is a safety check, not a guarantee.
If top1-tier is not reached, autopilot iterates up to 5 times and submits the best offline candidate on iteration 5 (when `--submit` is set).

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
    rules.html                   # Rules HTML (best effort)
    knowledge_hints.txt          # Similar competitions + hints
  prompts/
    codex_plan_and_baseline.md   # Baseline generation prompt
    codex_improve.md             # Improvement iteration prompt
  kernels/
    <run-id>/                    # Kaggle kernel workspace
  runs/<run-id>/
    run.json                     # Run configuration and status
    iter-<k>/
      metrics.json               # Offline evaluation results
      diagnostics.md             # Agent-readable performance analysis
      submission.csv             # Predictions for this iteration
      code.diff                  # Git diff snapshot
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
uv run kagglebot autopilot titanic --agent codex --compute kaggle_gpu --submit

# 2. If needed, edit plan.json to adjust target_score or evaluation strategy
nano artifacts/titanic/plan.json

# 3. Re-run autopilot with adjusted target
uv run kagglebot autopilot titanic --agent codex --compute kaggle_gpu --submit

# 4. Check Knowledge Base for learnings
uv run kagglebot knowledge show titanic

# 5. Find similar competitions
uv run kagglebot knowledge search --tag tabular --tag binary --limit 5
```

## Notes

- **Non-interactive**: No prompts for input. All decisions via CLI flags or `plan.json`.
- **Git policy**: Autopilot stashes dirty state and ensures the repo is on `main`. No branches or commits are created; diffs are saved in artifacts.
