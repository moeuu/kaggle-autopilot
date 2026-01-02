# Autopilot

Autopilot runs a non-interactive loop: verify → train → evaluate → diagnose → (if needed) improve → repeat.
Submissions happen **only** when the target score is met (unless the agent sets a submit-at-final policy in `plan.json`).

## Usage

```bash
uv run kagglebot autopilot https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques \
  --agent codex \
  --compute kaggle_gpu \
  --submit
```

## Scoring Strategy

Autopilot evaluates as follows:

- Default (`auto`):
  - If labeled test data exists, evaluate on test.
  - Otherwise, use holdout by default (CV if explicitly requested in plan).
- `--score-source holdout`: fixed train/val split (with stratification for classification).
- `--score-source cv`: KFold/StratifiedKFold.
- `--score-source test`: requires labeled test; errors otherwise.

Kaggle public Top1 is fetched as context only and **never** used as a submit condition.

## Agent-Defined Targets

The agent fills `artifacts/<slug>/plan.json` with:
- `target_metric`, `target_score`, `target_direction`
- `score_source`, `holdout_frac`/`cv_folds`, `seed`

You can edit the plan file to override these choices for future runs.

## Safety

- No rule acceptance automation.
- No interactive prompts.
- Dedupe + cooldown enforced before submission.
- `--dry-run` avoids Kaggle CLI and Codex calls.
