# Autopilot Feature: Executive Summary

## Overview

The **Autopilot Single-Submit** feature (Method 1) enables kagglebot to iteratively improve models offline, compare against Kaggle's public leaderboard Top1 score, and submit only once when a confidence heuristic is met.

## Key Features

### 1. Offline Iteration Loop
- **5 iterations maximum** (hard-coded for safety)
- Train → Evaluate offline → Diagnose → Improve → Repeat
- No submissions during loop (offline-only evaluation)

### 2. Top1 Heuristic
- Fetch Kaggle public leaderboard Top1 score (no submission required)
- Compare offline score to Top1 using dual-margin heuristic:
  - **Absolute margin** (default: 0.05)
  - **Relative margin** (default: 2%)
  - Must meet **BOTH** margins (AND condition)

### 3. Submission Modes
- **Early Submit** (optional, `--early-submit`): Submit immediately when heuristic met
- **Final Submit** (default): Complete all 5 iterations, submit best
- **No Submit** (`--no-final-submit`): Exploration mode, no submission

### 4. Guardrails
- **MAX_SUBMISSIONS = 1** (hard-coded, cannot be overridden)
- Enhanced deduplication (autopilot_run_id tracking)
- Time cap (default: 8 hours)
- Consecutive failure cap (abort after 3 training failures)
- Secret scanning (pre-submission safety check)

### 5. Knowledge Base Integration
- Captures what worked (features, hyperparameters, diagnostics)
- Retrieves similar improvements via Jaccard similarity
- Informs Codex prompts with relevant patterns

## Quick Start

```bash
# Basic autopilot run (5 iterations, submit best)
uv run kagglebot run titanic \
  --agent codex \
  --autopilot \
  --target-metric accuracy \
  --force

# Early submit (submit first iteration that meets heuristic)
uv run kagglebot run titanic \
  --autopilot \
  --early-submit \
  --force

# Exploration mode (no submission)
uv run kagglebot run titanic \
  --autopilot \
  --no-final-submit
```

## How It Works

```
Iteration 1: Baseline
├─ Train model (LogisticRegression)
├─ Evaluate offline (holdout or CV)
├─ Fetch Top1 from Kaggle leaderboard
├─ Check heuristic: 0.9567 >= 0.931? ✓
└─ Generate diagnostics

Iteration 2: Improvement
├─ Read diagnostics from iteration 1
├─ Retrieve similar KB entries
├─ Generate improvement prompt for Codex
├─ Train improved model
├─ Evaluate offline: 0.9601
└─ Check heuristic: meets? ✓

... (iterations 3-5)

Final Submit
├─ Select best iteration (by offline score)
├─ Validate submission format
├─ Run safety checks (dedup, secrets)
└─ Submit to Kaggle (MAX_SUBMISSIONS enforced)
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--autopilot` | False | Enable autopilot mode |
| `--target-metric` | auto | Metric name (accuracy, rmse, etc.) |
| `--target-direction` | auto | "maximize" or "minimize" |
| `--top1-margin-abs` | 0.05 | Absolute margin for heuristic |
| `--top1-margin-rel` | 0.02 | Relative margin (2%) |
| `--score-source` | auto | Offline evaluation: holdout/cv |
| `--early-submit` | False | Submit when heuristic met |
| `--no-final-submit` | False | Skip final submission |
| `--max-time` | 480 | Time cap in minutes |

## Key Design Decisions

### Why Single-Submit?

- **Safety first**: Prevents accidental submission spam
- **Aligns with competition limits**: Many competitions cap at 5 submissions/day
- **Forces quality over quantity**: Encourages thoughtful iteration

### Why Both Margins?

- **Absolute margin**: Prevents bad submissions (e.g., 0.5 vs Top1 0.95)
- **Relative margin**: Scales with metric range (e.g., RMSE 0.01 vs 100)
- **AND condition**: More conservative (stricter margin wins)

### Why Offline Evaluation?

- **No submission required**: Can iterate freely without quota
- **Faster feedback**: No waiting for Kaggle kernel execution
- **Distribution shift awareness**: Offline ≠ public LB (by design)

### Why Knowledge Base?

- **Learn from history**: What worked before (similar competitions)
- **Pattern recognition**: Common improvements (feature engineering, etc.)
- **Codex guidance**: Informs agent prompts with proven strategies

## Example Heuristic Calculation

**Scenario:**
- Top1 public score: 0.95 (accuracy)
- Offline score: 0.9567
- margin_abs: 0.05
- margin_rel: 0.02 (2%)

**Calculation:**
```python
# Maximize (higher is better)
abs_threshold = 0.95 - 0.05 = 0.90
rel_threshold = 0.95 * (1 - 0.02) = 0.931

# Check both
0.9567 >= 0.90?   ✓ (meets absolute)
0.9567 >= 0.931?  ✓ (meets relative)

# Result: TRUE (meets heuristic, ready to submit)
```

## Metrics.json Schema

Three new blocks added to metrics.json:

### 1. offline_score
```json
{
  "source": "holdout",
  "metric": "accuracy",
  "value": 0.9567,
  "n_train": 800,
  "n_eval": 200,
  "random_seed": 42
}
```

### 2. top1_comparison
```json
{
  "top1_score": 0.95,
  "direction": "maximize",
  "margin_abs": 0.05,
  "margin_rel": 0.02,
  "threshold_used": 0.931,
  "offline_meets_heuristic": true
}
```

### 3. chosen_best
```json
{
  "iteration": 3,
  "reason": "best_offline_score",
  "submitted": true,
  "submission_path": "artifacts/titanic/submissions/..."
}
```

## Codex Prompt Templates

### Baseline Prompt (~250 lines)
- Competition context (rules, Top1, target)
- Retrieved KB entries (top 3 by Jaccard similarity)
- Dataset profile (features, missing data, distributions)
- Acceptance criteria (validation, heuristic)

### Improvement Prompt (~200 lines)
- Previous iteration results (offline score, model, features)
- Diagnostics summary (error analysis, feature importance)
- Similar KB improvements (what worked before)
- Overfitting warnings (holdout/CV specific)
- Suggested improvements (auto-generated from diagnostics)

## Implementation Status

**Current State:** Specification complete (1687 lines)

**Next Steps:**
1. Implement autopilot loop (Phase A1-A4)
2. Add guardrails (Phase B1-B5)
3. Integrate Knowledge Base (Phase C1-C3)
4. Create prompt templates (Phase D1-D3)
5. Extend CLI (Phase E1-E3)

See [AUTOPILOT_SINGLE_SUBMIT.md](./AUTOPILOT_SINGLE_SUBMIT.md) for full specification.

## Safety Guarantees

✅ **MAX_SUBMISSIONS = 1** (hard-coded, cannot be bypassed)
✅ **Secret scanning** (pre-submission check)
✅ **Enhanced deduplication** (SHA256 + autopilot_run_id)
✅ **Time cap** (default: 8 hours)
✅ **Consecutive failure cap** (abort after 3 training failures)
✅ **Validation** (columns, row count, ID alignment)
✅ **Rate limiting** (existing ledger checks)

## Documentation Links

- **Full Specification:** [AUTOPILOT_SINGLE_SUBMIT.md](./AUTOPILOT_SINGLE_SUBMIT.md)
- **Architecture Review:** [../ARCHITECTURE_REVIEW.md](../ARCHITECTURE_REVIEW.md)
- **Failure Modes:** [../FAILURE_MODES.md](../FAILURE_MODES.md)
- **Submission Checklist:** [../CHECKLIST_SUBMIT.md](../CHECKLIST_SUBMIT.md)

---

**Status:** Ready for implementation (all 7 deliverables complete)

**Contact:** See [CLAUDE.md](../CLAUDE.md) for development guidelines
