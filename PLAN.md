# Kaggle Autopilot - Architecture Plan

## Overview

Kaggle Autopilot is a CLI tool that automates Kaggle competition workflows with safety-first design principles. The tool prioritizes preventing accidents (invalid submissions, duplicate submissions, unauthorized actions) over performance.

## Design Principles

1. **Safe by default**: All potentially destructive operations require explicit flags
2. **Fail loudly**: Validation errors should be clear and actionable
3. **Audit trail**: All submissions tracked with hashes and timestamps
4. **No automation of human decisions**: Never auto-accept rules or bypass Kaggle safeguards
5. **Deterministic and reproducible**: Same inputs should produce same outputs

## CLI Commands (Typer-based)

### 1. `kagglebot bootstrap <slug>`

**Purpose**: Download and prepare competition data

**Inputs**:
- `slug` (positional): Competition identifier (e.g., "titanic")
- `--force` (flag): Re-download even if data exists

**Outputs**:
- Downloads competition files to `data/<slug>/raw/`
- Extracts ZIP files automatically
- Creates directory structure for artifacts

**Safety Guardrails**:
- Checks if competition rules are accepted (via Kaggle CLI probe)
- If rules not accepted: prints rules URL and exits with code 2
- Never attempts to accept rules programmatically

**Error Conditions**:
- Kaggle CLI not installed → RuntimeError with installation instructions
- Kaggle API credentials missing → RuntimeError with setup instructions
- Rules not accepted → SystemExit(2) with manual instructions
- Network/download failure → RuntimeError with error details

---

### 2. `kagglebot run <slug>`

**Purpose**: End-to-end pipeline (download → train → validate → optionally submit)

**Inputs**:
- `slug` (positional): Competition identifier
- `--submit` (flag, default=False): Actually submit to Kaggle
- `--message` (str, default="auto baseline"): Submission message
- `--force-submit` (flag, default=False): Allow duplicate submission hash
- `--force-download` (flag, default=False): Force re-download data

**Outputs**:
- Trained model: `artifacts/<slug>/models/baseline.joblib`
- Submission CSV: `artifacts/<slug>/submissions/submission.csv`
- History record: `artifacts/<slug>/submissions/history.jsonl` (if submitted)

**Pipeline Steps**:
1. Bootstrap (download/unzip) - uses `bootstrap_competition()`
2. Train baseline model - uses `train_and_make_submission()`
3. Validate submission - uses `validate_submission()`
4. **DRY RUN by default** - prints warning and exits
5. (If `--submit`) Check duplicate hash - uses `SubmissionHistory.is_duplicate()`
6. (If `--submit` and not duplicate) Submit via Kaggle CLI
7. (If submitted) Record in history with hash + timestamp

**Safety Guardrails**:
- Default behavior is DRY RUN (no submission)
- Duplicate detection prevents re-submitting identical files
- Strict validation before any submission attempt
- Requires explicit `--force-submit` to override duplicate check

**Error Conditions**:
- Missing required files (train.csv, test.csv, sample_submission.csv) → FileNotFoundError
- Multi-target competitions → NotImplementedError (MVP limitation)
- Validation failures → ValueError with detailed mismatch info
- Duplicate submission without force flag → Exit code 2

---

### 3. `kagglebot train <slug>` (Future)

**Purpose**: Train model only (no submission generation)

**Status**: Not yet implemented

**Planned Inputs**:
- `slug`: Competition identifier
- `--model-type`: Override auto-detection (ridge, logistic, etc.)

---

### 4. `kagglebot predict <slug>` (Future)

**Purpose**: Generate submission from pre-trained model

**Status**: Not yet implemented

---

## Module Architecture

```
kagglebot/
├── cli.py                    # CLI entry point (Typer app)
├── paths.py                  # Path management (CompetitionPaths dataclass)
├── kaggle_cli.py             # Kaggle CLI subprocess wrapper
├── bootstrap.py              # Download & rule-check logic
├── tabular_baseline.py       # Baseline model training
├── validation.py             # Submission validation
├── history.py                # Submission tracking
└── hashing.py                # File hashing utilities
```

### Module Boundaries

#### `cli.py`
- **Responsibility**: User interface, command orchestration
- **Dependencies**: All other modules
- **Exports**: Typer app
- **Key functions**: `bootstrap()`, `run()`

#### `paths.py`
- **Responsibility**: Centralized path construction
- **Dependencies**: None (stdlib only)
- **Exports**: `CompetitionPaths` dataclass, `repo_root()` function
- **Invariant**: All paths constructed through `CompetitionPaths` for consistency

#### `kaggle_cli.py`
- **Responsibility**: Kaggle CLI interaction
- **Dependencies**: subprocess (stdlib)
- **Exports**: `run_kaggle()`, `kaggle_submit()`, `CmdResult`
- **Design**: Thin wrapper around subprocess for testability

#### `bootstrap.py`
- **Responsibility**: Download data, check rule acceptance
- **Dependencies**: kaggle_cli, paths
- **Exports**: `bootstrap_competition()`
- **Key Safety**: Detects unaccepted rules and exits gracefully

#### `tabular_baseline.py`
- **Responsibility**: Train baseline model, generate predictions
- **Dependencies**: paths, sklearn, pandas, joblib
- **Exports**: `train_and_make_submission()`, `RunOutputs`
- **Design Decisions**:
  - Auto-detect classification vs regression (heuristic: nunique <= 20)
  - Single-target only (MVP limitation)
  - Fixed preprocessing pipeline (median imputation + OHE)
  - 80/20 holdout for quick validation score

#### `validation.py`
- **Responsibility**: Strict submission validation
- **Dependencies**: pandas
- **Exports**: `validate_submission()`
- **Validation Rules**:
  1. Column names must match exactly (order matters)
  2. Row count must match exactly
  3. ID column must have no missing values
  4. Target columns must not be all-NaN

#### `history.py`
- **Responsibility**: Track submission history
- **Dependencies**: hashing, paths
- **Exports**: `SubmissionHistory` class
- **Storage Format**: JSONL (one JSON object per line)
- **Record Schema**: `{ts, sha256, submission_path, message}`

#### `hashing.py`
- **Responsibility**: File hashing for duplicate detection
- **Dependencies**: hashlib (stdlib)
- **Exports**: `sha256_file()`
- **Implementation**: Chunked reading (1MB chunks) for large files

---

## Safety Guardrail Implementations

### 1. No Auto-Accept Rules

**Implementation**: `bootstrap.py::_looks_like_rules_not_accepted()`
- Probes with `kaggle competitions files -c <slug>`
- Parses stderr/stdout for keywords: "accept the rules", "must accept", etc.
- On detection: prints rules URL and exits

**Rationale**: Kaggle API doesn't expose rule acceptance, so we probe and fail safely

---

### 2. Dry-Run by Default

**Implementation**: `cli.py::run()` command
- `--submit` flag defaults to False
- Without flag: prints `[yellow]DRY RUN[/yellow]` and exits after validation
- Forces explicit user action for submissions

**Rationale**: Prevents accidental submissions during development/testing

---

### 3. Duplicate Submission Prevention

**Implementation**: `history.py::SubmissionHistory`
- Computes SHA256 hash of submission.csv
- Compares against history.jsonl
- Rejects if duplicate found (unless `--force-submit`)

**Rationale**: Prevents wasting submission quota on identical files

---

### 4. Strict Validation

**Implementation**: `validation.py::validate_submission()`
- Checks: columns, row count, missing IDs, all-NaN targets
- Raises ValueError with detailed error messages
- Runs before any submission attempt

**Rationale**: Kaggle will reject malformed submissions; catch early

---

## Logging and Artifacts Structure

### Directory Layout

```
kaggle-autopilot/
├── data/                          # gitignored
│   └── <slug>/
│       └── raw/
│           ├── train.csv
│           ├── test.csv
│           ├── sample_submission.csv
│           └── <slug>.zip         # downloaded zip
│
├── artifacts/                     # gitignored
│   └── <slug>/
│       ├── models/
│       │   └── baseline.joblib    # trained sklearn pipeline
│       └── submissions/
│           ├── submission.csv     # latest submission
│           └── history.jsonl      # submission audit log
│
└── kagglebot/
    └── (source code)
```

### Logging Strategy

**Current (MVP)**:
- Rich library for colored console output
- `print()` statements with `[green]`, `[yellow]`, `[red]` tags
- No file-based logging

**Future Enhancements**:
- Structured logging to `artifacts/<slug>/logs/run.log`
- Log levels: DEBUG, INFO, WARNING, ERROR
- Log rotation for long-running experiments

### History Format (JSONL)

Each submission creates one line in `history.jsonl`:

```json
{"ts": "2026-01-01T00:15:30.123456+00:00", "sha256": "abc123...", "submission_path": "artifacts/titanic/submissions/submission.csv", "message": "auto baseline"}
```

**Benefits**:
- Append-only (no file corruption on partial writes)
- Easy to grep/parse
- Human-readable

---

## Testing Strategy

### Current Coverage

- `tests/test_hashing.py`: File hashing correctness
- `tests/test_validation.py`: Submission validation rules

### Future Test Needs

1. **Integration tests**: Full `run` command with mock Kaggle CLI
2. **Fixture-based tests**: Small synthetic competitions for end-to-end testing
3. **Property tests**: Validation invariants (e.g., submission always matches sample shape)

---

## Future Enhancements

### Near-term (Post-MVP)

1. **Multi-target support**: Relax single-target constraint in `tabular_baseline.py`
2. **Model selection**: Add `--model` flag to choose Ridge/Logistic/XGBoost/etc.
3. **Cross-validation**: Replace 80/20 holdout with k-fold CV
4. **Feature engineering**: Add basic feature interactions, polynomial features
5. **Hyperparameter tuning**: GridSearchCV or Optuna integration

### Medium-term

1. **Competition type detection**: Auto-detect tabular/CV/NLP/time-series
2. **CV baselines**: Image classification with pretrained models
3. **NLP baselines**: Transformer-based models for text competitions
4. **Ensemble support**: Combine multiple models
5. **Submission comparison**: Diff tool for comparing submission files

### Long-term

1. **Automated feature selection**: Recursive feature elimination
2. **AutoML integration**: H2O, AutoGluon, or FLAML
3. **Leaderboard tracking**: Store and visualize submission scores
4. **Collaboration**: Multi-user submission history
5. **Kaggle Notebooks integration**: Generate notebook from CLI runs

---

## Security Considerations

### Secrets Management

- **Kaggle API credentials**: OAuth token in `~/.kaggle/access_token` (never committed)
- **Gitignore**: Explicit blocks for `.kaggle/`, `data/`, `artifacts/`

### Subprocess Injection

- **Current risk**: Low (slug is passed directly to Kaggle CLI, which validates it)
- **Mitigation**: All `subprocess.run()` calls use list-form args (no shell=True)

### Dependency Vulnerabilities

- **Strategy**: Pin major versions in `pyproject.toml`
- **Tooling**: Use `pip-audit` or `safety` for vulnerability scanning

---

## Open Questions / Design Decisions

1. **Repo root detection**: Currently uses `Path.cwd()`. Should we use git root detection?
   - **Decision**: Keep simple for MVP; add git root in v0.2

2. **Model versioning**: Should we timestamp model files or overwrite?
   - **Decision**: Currently overwrites `baseline.joblib`; future versions should timestamp

3. **Submission limits**: Should we query Kaggle API for daily limits?
   - **Decision**: Not in MVP; rely on Kaggle CLI error messages

4. **Parallel competitions**: Support multiple competitions in one workspace?
   - **Decision**: Already supported via `<slug>` namespacing

5. **Configuration file**: Should users have a `.kagglebot.toml` for defaults?
   - **Decision**: Not in MVP; all config via CLI flags

---

## Success Metrics (MVP)

- [ ] Successfully downloads and trains on Titanic competition
- [ ] Produces valid submission.csv (passes Kaggle's upload check)
- [ ] Prevents duplicate submission when run twice
- [ ] Detects unaccepted rules and exits gracefully
- [ ] DRY RUN mode works (no accidental submissions)
- [ ] All tests pass with pytest
- [ ] Lints cleanly with ruff
- [ ] Type checks pass with pyright

---

## Maintainer Notes

- **Primary language**: Python 3.11+
- **Package manager**: uv (recommended) or pip
- **CLI framework**: Typer (type-safe, auto-generated help)
- **ML framework**: scikit-learn (simple, stable, widely understood)
- **Code style**: Ruff (fast, opinionated)
- **Type checking**: Pyright (strict mode recommended)

**When adding new commands**:
1. Add command function to `cli.py` with `@app.command()`
2. Update this PLAN.md with command specification
3. Add integration test in `tests/test_cli.py`
4. Update README.md with usage example

**When adding new competition types**:
1. Create new module (e.g., `kagglebot/cv_baseline.py`)
2. Add detection logic in `kagglebot/detect.py` (future module)
3. Update `cli.py` to dispatch based on competition type
4. Add type-specific validation in `validation.py`
