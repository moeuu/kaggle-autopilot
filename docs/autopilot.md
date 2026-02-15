# Autopilot

Autopilot runs a top1-gated, non-interactive improvement loop that iterates: **verify → train → evaluate → diagnose → improve → repeat** until top1-tier is reached or `max_iterations` (default 1) complete.
Iteration 0 is expected to implement a **strong initial model** (web‑researched), not a simplistic initial model.

**Key principle**: Submission happens **only** when offline score is top1-tier (direction-aware) or at the final iteration.

## Usage

```bash
uv run kagglebot autopilot https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques \
  --compute kaggle_gpu
```

Rules/overview/data/submission format are fetched from Kaggle during download. Use `--rules-file` (md/txt/html) to override rules.

## How It Works

### 1. Bootstrap & Planning (Iteration 0)

When autopilot starts, it:
1. Downloads competition data
2. Profiles the dataset (rows, columns, missing values, etc.)
3. Queries the Knowledge Base for similar competitions
4. Generates competition-specific code under `artifacts/<slug>/kernel/`. Runner/validation fixes may update `src/` during auto-fix.
5. Runs a three-stage planning pipeline:
   - Codex (`gpt-5.3-codex`, extra high): context brief
   - GPT (`gpt-5.2`, extra high): strategy + `PLAN_JSON`
   - Codex (`gpt-5.3-codex`, extra high): implementation
6. Runs initial verification (tests)

**Output**: `plan.json` defines your success criteria.

Example `plan.json`:
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

### 2. Iteration Loop (Iterations 1-N)

For each iteration:

1. **Verify**: Run `pytest` to ensure code quality
2. **Train**: Run local solver or Kaggle kernel (kaggle_gpu/kaggle_tpu uses `kernel.py`)
   - Before kernel push, sources are validated for syntax and required outputs (submission/metrics paths).
   - After kernel push, kernel registration is verified; if not found, a single re-push is attempted.
   - Feature engineering must be **fit on train and applied to test**; do not recompute stats/encoders on test.
3. **Evaluate**: Compute offline score using the evaluation strategy (holdout/CV/test)
4. **Check Top1**: Compare offline score to public Top1 (direction-aware)
5. **Diagnose**: Generate `diagnostics.md` with actionable improvement hints
6. **Improve** (if not top1-tier):
   - Call agent with diagnostics
  - Improvement mode is selected by top1 gap:
     - `major_overhaul`: large redesign if far from top1
     - `moderate_update`: meaningful changes if mid-gap
     - `minor_tuning`: small adjustments if close to top1
  - If iteration 1 is tree-only and far from top1, iteration 2 must add an NN family (MLP or FT-Transformer) alongside the tree baseline.
   - Agent modifies code in `kagglebot/solver/` (or `artifacts/<slug>/kernel/kernel.py` on kaggle_gpu)
   - Repeat from step 1

The loop stops when:
- **Top1-tier reached** → Submit
- **Max iterations reached** → Submit best offline candidate at the final iteration
- **Kernel failure** → GPT proposes fix strategy, then Codex applies edits and re-pushes; repeated identical kernel errors are capped to avoid infinite loops
- **Kaggle GPU capacity limit** → Autopilot waits and retries a few times, then exits with a message to stop running GPU sessions
- **Kernel registration delay** → If the kernel isn’t visible after push, autopilot waits and retries the push before giving up
- **Other runtime errors** → GPT proposes fix strategy, then Codex auto-fix runs up to 2 attempts before surfacing the error

### 3. Submission

If top1-tier is reached:
1. Check competition rules are accepted (manual browser action required)
2. Check submission is not a duplicate (SHA256 hash check)
3. Enforce rate limit (default: 5 min cooldown between submissions)
5. Submit via Kaggle CLI

**Important**: Submission is guarded. You must:
- Reach top1-tier or finish the final iteration (best candidate)
- Have accepted competition rules manually in your browser

## Scoring Strategy

Autopilot uses **offline evaluation** to avoid leaking the public leaderboard:

### Score Sources

1. **`auto` (default)**:
   - Use labeled test data if available (rare)
   - Otherwise, use holdout split on train data
   - CV only if explicitly requested in `plan.json`

2. **`holdout`**:
   - Fixed train/val split (default: 80/20)
   - Stratified for classification (preserves class balance)
   - Reproducible with fixed `seed`

3. **`cv` (Cross-Validation)**:
   - KFold for regression
   - StratifiedKFold for classification
   - Configurable `cv_folds` (default: 5)
   - Reports mean and std across folds

4. **`test` (Labeled Test)**:
   - Only if `test.csv` contains the target column
   - Errors if not available
   - Rare in Kaggle competitions

### Top1 Public Leaderboard

The public leaderboard leader's score is fetched and recorded in `context/top1_public.json` and used as a **heuristic gate**:
- Minimize metrics: submit when offline score <= top1
- Maximize metrics: submit when offline score >= top1

Offline evaluation is a pseudo-test and not directly comparable to the public leaderboard; the heuristic is for safety, not a guarantee.

## Agent-Defined Targets

The planning pipeline generates `plan.json` during iteration 0 with:

- **`target_metric`**: Metric to optimize (rmse, mae, accuracy, logloss, auc_roc, f1, etc.)
- **`target_score`**: Realistic target based on dataset complexity and Top1 public
- **`target_direction`**: "minimize" or "maximize"
- **`score_source`**: Evaluation strategy (auto, holdout, cv, test)
- **`holdout_frac`**: Validation split fraction (default: 0.2)
- **`cv_folds`**: Number of CV folds (default: 5)
- **`seed`**: Random seed for reproducibility (default: 42)
- **`time_budget_min`**: Max training time per iteration (optional)
- **`submit_policy`**: "on_target_only" (default) or "at_final"

You can **manually edit** `plan.json` to override these choices before re-running autopilot.

## Safety Guardrails

### Hard Caps

- **Max iterations**: Default 1, configurable via `--max-iterations`
- **Max total time**: Disabled by default (no wall-clock limit). You can set `--max-total-min` if you want a cap.
- **Patience**: Default 2 iterations without improvement, configurable via `--patience`
- **Min improvement**: Default 0.0 (any improvement counts), configurable via `--min-improvement`
- **Max submissions**: Default 5 per autopilot run

### Submission Safety

- ✅ **Duplicate prevention**: SHA256 hash check against `submissions/ledger.jsonl`
- ✅ **Rate limiting**: Default 5 min cooldown between submissions
- ✅ **Submission quota**: Max 5 submissions per run
- ✅ **Rules check**: Errors if competition rules not accepted manually
- ✅ **Validation**: Submission format must match `sample_submission.csv` exactly

### Non-Interactive Design

Autopilot runs completely non-interactively:
- ❌ No prompts for user input
- ❌ No automated rule acceptance (you must accept in browser first)
- ✅ All decisions via CLI flags or `plan.json`
- ✅ Errors exit with actionable messages

### Secret Protection

- ❌ Never commits `kaggle.json`, API keys, or secrets
- ❌ Never logs credentials
- ✅ `.gitignore` excludes sensitive files
- ✅ Knowledge Base stores only metadata, never raw data

## Dry Run Mode

Use `--dry-run` to:
- Skip Kaggle CLI submission calls
- Skip Codex API calls (uses dummy responses)
- Test the full autopilot loop without external API usage

Useful for:
- Testing changes to the autopilot logic
- CI/CD pipelines
- Debugging iteration flow

## Artifacts Layout

All autopilot outputs are stored in `artifacts/<slug>/`:

```
artifacts/<slug>/
  meta.json                      # Competition metadata
  plan.json                      # Agent-defined targets (editable)

  context/
    dataset_profile.json         # Dataset stats + file format summary
    sample_submission.csv        # Required submission format
    top1_public.json             # Public leaderboard leader (context only)
    rules_url.txt                # Competition rules URL
    rules.md                     # Rules markdown (fetched or from --rules-file)
    overview.md                  # Competition overview (if available)
    data.md                      # Data description (if available)

  prompts/
    codex_plan_and_implement.md   # Initial plan + implementation prompt
    codex_improve.md             # Improvement iteration prompt

  runs/<run-id>/
    run.json                     # Run configuration and status

    iter-<k>/
      metrics.json               # Offline evaluation results
      diagnostics.md             # Agent-readable performance analysis
      submission.csv             # Predictions for this iteration
      logs/                      # Training logs (if any)
      agent/
        prompt.md                # Codex input for this iteration
        codex_last_message.txt   # Codex output summary

  submissions/
    ledger.jsonl                 # Deduplication log (append-only)

  kernel.py                      # Kaggle kernel entrypoint
```

### Key Files

**`metrics.json`** (per iteration):
```json
{
  "score_source": "holdout",
  "metric": "rmse",
  "direction": "minimize",
  "value": 0.145,
  "train_score": 0.120,
  "val_score": 0.145,
  "target": 0.13,
  "met_target": false,
  "top1_public": 0.11234
}
```

**`diagnostics.md`** (generated for agent):
- Current score vs target
- Train/val gap analysis (overfitting/underfitting)
- Improvement suggestions ranked by priority
- Previous iteration summaries (what was tried)

## Configuration Options

### CLI Flags

```bash
uv run kagglebot autopilot <competition-url> [OPTIONS]

Required:
  competition-url                Kaggle competition URL or slug

Compute:
  --compute local_gpu|kaggle_gpu|kaggle_tpu
                                 Training environment (required)
  --accelerator auto|gpu|tpu     Accelerator override (optional)

Evaluation:
  --score-source auto|holdout|cv|test
                                 Evaluation strategy (default: auto)
  --holdout-frac FLOAT           Validation split fraction (default: 0.2)
  --cv-folds INT                 Number of CV folds (default: 5)
  --seed INT                     Random seed (default: 42)

Iteration Control:
  --max-iterations INT           Max improvement cycles (default: 1). Aliases: --max-iteration, --max-iter, --iter
  --max-total-min INT            Max wall-clock time in minutes (default: none)
  --patience INT                 Early stopping patience (default: 2)
  --min-improvement FLOAT        Minimum score improvement threshold (default: 0.0)

Submission:
  --force-submit                 Bypass deduplication and rate limit (DANGEROUS)
  --message TEXT                 Custom submission message

Other:
  --dry-run                      Skip external API calls (testing mode)
  --verify-cmd TEXT              Test command (default: "uv run pytest -q")
```

### Overriding Plan Values

CLI flags take precedence over `plan.json`:

```bash
# Override target metric from command line
uv run kagglebot autopilot titanic --target-metric mae --target-score 0.15
```

## Examples

### Basic Autopilot Run

```bash
uv run kagglebot autopilot titanic --compute local_gpu
```

Output:
- Generates plan.json and initial model implementation
- Runs up to `max_iterations` (default 1)
- Stops when target met or patience exhausted
- Submits when top1-tier is reached or at final iteration

### Autopilot (Default Submission Flow)

```bash
uv run kagglebot autopilot titanic \
  --compute kaggle_gpu
```

Requirements:
- Competition rules must be accepted manually first
- Submission only happens if target score is met
- Respects deduplication and rate limits

### Aggressive Tuning (More Iterations)

```bash
uv run kagglebot autopilot titanic \
  --max-iterations 10 \
  --patience 3 \
  --min-improvement 0.01
```

Settings:
- Up to 10 iterations
- Requires 0.01 minimum improvement to count
- Stops if no improvement for 3 consecutive iterations

### Quick Prototyping (Holdout + Fast Compute)

```bash
uv run kagglebot autopilot titanic \
  --compute local_gpu \
  --score-source holdout \
  --max-iterations 3 \
  --max-total-min 30
```

Settings:
- Fast local CPU training
- Holdout validation (faster than CV)
- Max 3 iterations or 30 minutes

### Robust Evaluation (Cross-Validation)

```bash
uv run kagglebot autopilot house-prices \
  --score-source cv \
  --cv-folds 10 \
  --compute kaggle_gpu
```

Settings:
- 10-fold cross-validation (more robust than holdout)
- Kaggle GPU for faster training
- Submit when target met

## Troubleshooting

### "Rules not accepted" Error

**Problem**: Autopilot errors with `RulesNotAcceptedError`.

**Solution**:
1. Visit the competition rules URL (printed in error message)
2. Click "I Understand and Accept" in your browser
3. Re-run autopilot

Kagglebot **never** automates rule acceptance for safety.

### "Duplicate submission" Error

**Problem**: Autopilot errors with `DuplicateSubmissionError`.

**Cause**: You're trying to submit the same predictions twice (hash collision).

**Solution**:
- If this is intentional, use `--force-submit` (not recommended)
- If this is a mistake, make changes to your solution and re-run

### Target Score Never Met

**Problem**: Autopilot runs all iterations but never meets target.

**Possible causes**:
1. Target score is too ambitious (edit `plan.json` to relax target)
2. Dataset is too difficult for simplistic models
3. Agent improvements aren't effective

**Solutions**:
- Lower `target_score` in `plan.json`
- Increase `--max-iterations` for more improvement attempts
- Reduce `--min-improvement` to accept smaller gains
- Manually improve the solver code and re-run

### Autopilot Stops Early (Patience)

**Problem**: Autopilot stops after 2-3 iterations without meeting target.

**Cause**: No improvement detected for `--patience` consecutive iterations.

**Solution**:
- Increase `--patience` (e.g., `--patience 5`)
- Reduce `--min-improvement` to count smaller gains
- Check diagnostics to see if agent is stuck (e.g., overfitting loop)

### GPU Not Available

**Problem**: `GPUNotAvailableError` when using `--compute local_gpu`.

**Solution**:
- Use `--compute local_gpu` without `--strict-accelerator` to allow CPU fallback
- Install CUDA/cuDNN for GPU support
- Use `--compute kaggle_gpu` to run on Kaggle's infrastructure

## Best Practices

1. **Start with holdout**: Use `--score-source holdout` for faster iterations, switch to CV later for robust evaluation.

2. **Trust offline evaluation**: Don't chase public leaderboard scores. A robust offline score generalizes better.

3. **Set realistic targets**: Use Top1 public score as a reference, but account for private leaderboard shift.

4. **Edit plan.json between runs**: If autopilot fails to meet target, manually adjust and re-run.

5. **Use --dry-run for testing**: Test changes to autopilot logic without consuming Kaggle submissions.

6. **Review diagnostics.md**: The agent-generated diagnostics often reveal root causes (overfitting, bad features, etc.).

7. **Don't force-submit duplicates**: Duplicate submissions waste quota and provide no new information.

8. **Accept rules once**: You only need to accept competition rules once in your browser. Autopilot will detect this.

## See Also

- [docs/architecture.md](architecture.md) - Control flow and safety gates (developer-focused)
- [docs/knowledge.md](knowledge.md) - Knowledge Base system
- [docs/taxonomy.md](taxonomy.md) - Tag taxonomy for competition similarity
