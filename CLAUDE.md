# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role
You are the **Staff Engineer / Architect** for this repository.
- **Planning first**: Review docs/architecture_final.md, docs/architecture.md, and docs/spec_autopilot.md before implementing
- **Risk analysis**: Check SECURITY.md for safety requirements
- **Phased implementation**: Follow docs/agents/codex_implementation_plan.md phases - don't skip ahead
- **Small, testable chunks**: Each PR should be a complete, tested unit of work

## Project Overview

Kaggle Autopilot is a **production-grade, fully automated** CLI tool for Kaggle competitions:

### Vision
- User provides competition URL: `kagglebot autopilot https://www.kaggle.com/c/titanic --agent codex --compute local_cpu --submit`
- Tool automatically: downloads data → analyzes competition → trains models → generates predictions → validates → submits
- **Zero prompts** - completely non-interactive (except manual rules acceptance once)
- **Production-grade models** - serious GBDT, stacking, CV (not toy baselines)
- **Safe by default** - dedup, rate limits, validation, reproducibility

### Current State (MVP)
- ✅ Data download and validation
- ✅ Basic tabular training (Ridge, LogisticRegression)
- ✅ Submission validation and local ledger
- ⏳ **Next**: Competition analyzer, orchestrator, production models

See docs/agents/codex_implementation_plan.md for detailed roadmap.

## Critical Constraints (NEVER violate)

1. **Do NOT automate rule acceptance**: Never automate clicking "Join"/"I Agree" or accepting competition rules. Users must manually accept rules in browser at least once per competition. If rules aren't accepted, detect this and print URL + instructions, then exit.

2. **Do NOT scrape or bypass**: Never scrape Kaggle webpages, bypass rate limits, or circumvent submission limits.

3. **Do NOT enable abuse**: Never implement multi-account behavior, submission spamming, or rule circumvention.

4. **Do NOT commit secrets**: Never commit API credentials, tokens, or large datasets.

## Design Principles (Production-Grade Automation)

### Non-Interactive Operation
- **No prompts**: All decisions automated or configured via CLI flags/config files
- **Single command**: `kagglebot autopilot <competition> --agent codex --compute local_cpu --submit` runs end-to-end
- **Safe defaults**: Conservative choices when ambiguous (no external data, etc.)
- **Clear errors**: Exit codes and messages guide user on failures (see docs/spec_autopilot.md)

### Submission Safety Guardrails (must implement)

- **Require explicit flag**: Submissions require `--submit` flag (default: validate only)
- **Duplicate detection**: Hash submission.csv and check local ledger before submitting
- **Rate limiting**: Max 5 submissions/day, min 1 hour between (configurable)
- **Strict validation**: Before any submission:
  - Columns must be identical to sample_submission.csv
  - Row count must match exactly
  - ID column alignment verified
  - Value ranges validated (e.g., probabilities in [0,1])

## Tooling (uv package manager)

This project uses **uv** for fast, reliable Python package management.

### Installation
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Common uv Commands
```bash
# Install project in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Add a new dependency
uv add pandas

# Add a dev dependency
uv add --dev pytest

# Remove a dependency
uv remove <package>

# Run commands in the uv environment
uv run pytest
uv run kagglebot train titanic --compute local_cpu

# Update all dependencies
uv pip compile pyproject.toml -o requirements.txt --upgrade
```

### uv Best Practices
- Always use `uv run` for CLI commands to ensure correct environment
- Keep `uv.lock` committed for reproducible installs
- Use `uv add` instead of manually editing pyproject.toml
- Run `uv sync` after pulling changes to update environment

## Development Commands

```bash
# Sync deps (uv)
uv sync

# Run tests
uv run pytest -q

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking (optional)
uv run pyright  # or mypy
```

## CLI Interface

### Compute Switching (Primary Feature)

**Single flag for execution mode**: `--compute {local_cpu, local_gpu, kaggle_gpu, kaggle_tpu}`

```bash
# Local CPU (default)
kagglebot train titanic --compute local_cpu

# Local GPU (auto-detect CUDA/MPS, fallback to CPU)
kagglebot train titanic --compute local_gpu

# Local GPU (strict mode - fail if no GPU)
kagglebot train titanic --compute local_gpu --strict-accelerator

# Kaggle GPU kernel
kagglebot train titanic --compute kaggle_gpu --force

# Kaggle TPU kernel
kagglebot train titanic --compute kaggle_tpu --internet on --force

# Dry-run (preview without execution)
kagglebot train titanic --compute kaggle_gpu --dry-run
```

**Key Principles**:
- **Non-interactive**: All decisions via flags/config, zero prompts
- **GPU auto-detection**: Automatically detect CUDA/MPS, fallback to CPU by default
- **Strict mode opt-in**: Use `--strict` to fail if requested compute unavailable
- **Kaggle kernels**: Push to Kaggle for GPU/TPU, but submission always local (for safety)
- **Kaggle username**: Auto-detect from `KAGGLE_USERNAME` or `~/.kaggle/kaggle.json` (override with `--kaggle-username`)
- **Rules acceptance**: ALWAYS manual (NEVER automated)

### Other Commands

```bash
kagglebot bootstrap <competition_slug>                 # Prepare workspace dirs + context
kagglebot implement <competition_slug> --agent codex    # Generate baseline solution
kagglebot train <slug> --compute local_cpu              # Train baseline model
kagglebot submit <slug> -f <path> -m "<message>" --force # Submit with guardrails
kagglebot autopilot <slug> --agent codex --compute local_cpu --submit
```

## Target Architecture

```
src/kagglebot/
  cli.py           # Typer/Rich CLI entry point
  kaggle_cli.py    # Wrapper around Kaggle CLI
  detect.py        # Competition type detection
  tabular/         # Baseline tabular pipeline
  rules/           # Optional: parse/summarize rules
tests/             # pytest tests
data/              # Downloaded datasets (gitignored)
artifacts/         # Models, submissions (gitignored)
```

## Coding Guidelines

- **Non-interactive execution**: No prompts, all decisions via flags/config
- **uv package manager**: Use `uv add/remove` for dependencies, `uv run` for commands
- **Small, composable functions**: Clear inputs/outputs, easy to test
- **User-friendly failures**: Actionable error messages with remediation hints
- **Kaggle CLI**: Use subprocess wrappers in `kaggle_cli.py`
- **Authentication**: Use `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`
- **Minimal dependencies**: Keep dependency tree small and auditable
- **Deterministic runs**: Control random seeds, document non-deterministic behavior
- **Python 3.11+**: Target modern Python with type hints
- **GPU support**: Auto-detect CUDA/MPS, graceful fallback to CPU
- **Resource limits**: Current MVP assumes datasets fit in memory
  - For large competitions (>1GB CSV), add chunked processing
  - Model training has configurable time budgets
  - Consider memory-mapped files or Dask for huge datasets

## Kaggle CLI Integration

**Primary Method**: Use Kaggle CLI via `kaggle_cli.py` (subprocess wrapper)

```bash
# Download competition data
kaggle competitions download -c <slug> -p <path>

# Submit to competition
kaggle competitions submit -c <slug> -f <file> -m "<message>"

# Check submission status
kaggle competitions submissions -c <slug>
```

**Authentication**:
- Use `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`
- If auth fails, print clear instructions and exit (exit code 2)

**Rules Acceptance**:
- Check rules accepted BEFORE any submission
- If not accepted: print URL and exit with code 2
- NEVER automate acceptance (user must click in browser)

**Subprocess Usage** (only for Kaggle kernels):
- Kernel push/poll/download still uses Kaggle CLI subprocess
- Use list args (NEVER `shell=True`)
- Example: `subprocess.run(["kaggle", "kernels", "push", "-p", kernel_dir], check=True)`

## ZIP File Handling

- bootstrap.py extracts all ZIPs in `data/<slug>/raw/`
- **No zip bomb protection** (acceptable: Kaggle CLI is trusted source)
- If adding support for user-uploaded ZIPs, add size/ratio checks
- Uses `zipfile.ZipFile.extractall()` with fixed destination (no path traversal risk)

## Architect/Reviewer Role

When operating as architect/reviewer:

1. **Before Implementation**: Review docs/architecture_final.md and docs/spec_autopilot.md for design coherence
2. **Interface Design**: Ensure CLI commands have clear contracts (inputs/outputs/errors)
3. **Safety Analysis**: Check all code paths for accidental submission or rule bypass
4. **Testability**: Verify new code has clear test boundaries (dependency injection where needed)
5. **Documentation**: Update docs/architecture_final.md, docs/spec_autopilot.md, and README.md for any architectural changes

## Pre-Commit Safety Checklist

Before committing any change, verify:

- [ ] No API credentials, tokens, or secrets in diff
- [ ] No large CSV/zip files committed (check .gitignore)
- [ ] All submission code paths guarded by `--submit` flag
- [ ] Duplicate submission check cannot be bypassed accidentally
- [ ] Validation runs before any Kaggle CLI submit call
- [ ] Error messages include actionable next steps
- [ ] All new functions have type hints
- [ ] Tests pass: `uv run pytest -q`
- [ ] Linting passes: `uv run ruff check .`

## Code Review Checklist

When reviewing PRs or changes:

### Safety
- [ ] No infinite submit loops or retry logic without bounds
- [ ] No repeated submissions of same file (duplicate check working)
- [ ] No automation of rule acceptance or browser actions
- [ ] Submission history growth is bounded (JSONL append-only)
  - Typical: <1000 submissions = ~100KB
  - If supporting high-volume use: add rotation or cleanup tool

### Reproducibility
- [ ] Random seeds set where applicable (train/test splits, model init)
- [ ] Deterministic runs for same input data
- [ ] Clear logging of validation scores and model choices

### Testing
- [ ] Critical validators have test coverage
- [ ] New commands have integration tests
- [ ] Edge cases tested (empty files, missing columns, etc.)

### User Experience
- [ ] Error messages are actionable (not just stack traces)
- [ ] Help text is clear (`kagglebot <command> --help`)
- [ ] Progress indicators for long operations
- [ ] Dry-run mode clearly announces no submission occurred

## MVP Success Criteria

- `kagglebot autopilot <slug> --agent codex --compute local_cpu` downloads data, trains baseline, produces valid submission.csv in artifacts/
- `kagglebot autopilot <slug> --agent codex --compute local_cpu --submit --message "baseline"` submits once with all guardrails enforced
- Works on common tabular competitions (e.g., Titanic-like: train.csv, test.csv, sample_submission.csv)
- Focus: "always produce valid submission" over leaderboard score

---

## Architectural Documentation

Before implementing features, review these documents:

### Core Architecture
1. **docs/architecture_final.md** - System design, modules, data flow, extension points
2. **docs/spec_autopilot.md** - CLI commands, artifacts, exit codes, autopilot contract
3. **docs/agents/codex_implementation_plan.md** - Phased implementation roadmap
4. **SECURITY.md** - Security guidelines, credential handling, safety guardrails
5. **AGENTS.md** - Instructions for implementer/tester agents (Codex)

### Compute Switching (Current Priority)
6. **docs/compute/spec.md** - CLI flags, ComputePlan, exit codes, artifact layout
7. **docs/compute/architecture.md** - Module boundaries, runner interface, GPU detection
8. **docs/compute/plan.md** - 7-week implementation plan (140 tasks)
9. **docs/notebook_runner/design.md** - Kaggle kernel execution design
10. **docs/notebook_runner/tasks.md** - Granular notebook runner tasks

### Implementation Workflow

1. **Plan**: Review docs/agents/codex_implementation_plan.md phase tasks
2. **Design**: Check docs/architecture_final.md for interfaces and patterns
3. **Spec**: Verify behavior matches docs/spec_autopilot.md contracts
4. **Security**: Ensure compliance with SECURITY.md
5. **Test**: Write tests before implementation
6. **Implement**: Small, testable chunks
7. **Review**: Self-review against all docs

### Key Architectural Decisions

- **Non-interactive**: No prompts during execution
- **Plugin architecture**: Extensible for different competition types
- **Checkpointing**: Resume from failures
- **Structured logging**: JSON logs for automation
- **Time budgets**: Configurable training time limits
- **Resource limits**: Memory and CPU constraints
- **Model registry**: Extensible model system (see docs/architecture_final.md)

Follow the critical path in docs/agents/codex_implementation_plan.md:
```
Phase 0 (Foundation) → Phase 1 (Analyzer) + Phase 2 (Orchestrator) →
Phase 3 (Training) → Phase 4 (Submission) → Phase 5 (Polish) →
Phase N (Notebook Runner - optional)
```

---

## Compute Switching Architecture (Current Implementation Priority)

**See docs/compute/spec.md, docs/compute/architecture.md, and docs/compute/plan.md for full details**

### Overview

Compute switching enables seamless execution across 4 modes:
- **local_cpu**: Train on local CPU (default, safe, works everywhere)
- **local_gpu**: Train on local GPU with CUDA/MPS auto-detection (2-5x faster)
- **kaggle_gpu**: Execute on Kaggle GPU kernel (free cloud GPU)
- **kaggle_tpu**: Execute on Kaggle TPU kernel (free cloud TPU)

### Key Components

```
src/kagglebot/
  compute/
    planner.py         # ComputePlan generation, GPU detection fallback
    gpu_detector.py    # CUDA/MPS detection via PyTorch
    exceptions.py      # GPUNotAvailableError, etc.
  runners/
    base.py            # Runner ABC, RunContext, RunResult
    local.py           # LocalRunner (CPU/GPU)
    kaggle_notebook.py # KaggleNotebookRunner (kernel execution)
  kernel/
    packager.py        # Generate kernel packages
    manager.py         # Push, poll, download kernels
    metadata.py        # kernel-metadata.json generation
    templates/         # Jinja2 templates for kernel scripts
  training/
    tabular_engine.py  # UPDATED: GPU params for LightGBM/CatBoost/XGBoost
```

### Critical Implementation Rules

1. **Non-interactive execution**:
   - All compute decisions via `--compute` flag
   - No prompts for GPU availability
   - Fallback to CPU by default (strict mode opt-in via `--strict`)

2. **GPU detection and fallback**:
   - Auto-detect CUDA (NVIDIA) or MPS (Apple Silicon)
   - If GPU requested but not available: fall back to CPU (unless `--strict`)
   - Log detection results and fallback decisions
   - Clear error messages if strict mode fails

3. **Kaggle kernel execution**:
   - Submission always happens locally (NEVER from kernel)
   - Check rules accepted BEFORE pushing kernel
   - No secrets in kernel code (validate before push)
   - `enable_internet=false` by default (require explicit flag)
   - Kernel slug includes run_id for uniqueness
   - Timeout enforcement on kernel polling

4. **uv package manager**:
   - Use `uv add jinja2` for new dependencies
   - Use `uv run pytest` for testing
   - Keep `uv.lock` committed
   - No pip or requirements.txt

5. **Exit codes**:
   - 2: Rules not accepted
   - 10: GPU not available (strict mode)
   - 11: Kernel timeout
   - 12: Kernel execution failed

### Implementation Checklist

Before merging compute switching code:
- [ ] GPU detection works (CUDA and MPS)
- [ ] Fallback logic works (local_gpu → local_cpu when no GPU)
- [ ] Strict mode fails correctly (raises GPUNotAvailableError)
- [ ] LocalRunner trains with GPU params (LightGBM, CatBoost, XGBoost)
- [ ] KaggleNotebookRunner generates valid kernel packages
- [ ] Kernel push/poll/download works (mocked in tests)
- [ ] Rules acceptance check enforced
- [ ] No secrets in kernel code (validation works)
- [ ] Submission happens locally (not from kernel)
- [ ] All tests pass (>80% coverage)
- [ ] Documentation complete

---

## Kaggle Notebook Runner (Phase N)

**See docs/notebook_runner/design.md for full design**

### Critical Security Rules

When implementing notebook runner:

1. **NEVER embed secrets in kernels**:
   - ❌ No Kaggle API keys in kernel code
   - ❌ No credentials in kernel-metadata.json
   - ❌ No tokens in generated main.py
   - ✅ Submission happens locally (not from kernel)

2. **Internet access OFF by default**:
   - `enable_internet: false` in kernel-metadata.json
   - Require explicit `--enable-internet` flag
   - Log warning when enabled
   - Only allow if competition permits external data

3. **Rules acceptance required**:
   - Check rules accepted BEFORE pushing kernel
   - Exit code 2 if not accepted
   - NEVER automate acceptance

4. **Kernel package validation**:
   - Scan for secret patterns before push
   - Fail if potential credentials detected
   - Log what would be pushed in dry-run

5. **Competition sources format**:
   - Use slug only: `["titanic"]`
   - NOT: `["c/titanic"]` or `["competitions/titanic"]`
   - Kaggle CLI requirement

### Implementation Checklist

Before merging notebook runner code:
- [ ] No secrets in kernel templates
- [ ] enable_internet defaults to false
- [ ] Rules check implemented (no automation)
- [ ] Kernel package validated before push
- [ ] Dry-run mode works (no push/execute)
- [ ] All subprocess calls secure (no shell=True)
- [ ] Timeout enforced on kernel polling
- [ ] Submission happens locally (not in kernel)
- [ ] Ledger records kernel_id for audit
- [ ] Documentation complete (usage + security)

### Testing Requirements

- [ ] Unit tests with mocked Kaggle CLI
- [ ] Integration tests (no actual push)
- [ ] Manual test on Kaggle (private kernel)
- [ ] Verify no secrets in pushed kernel
- [ ] Test all error paths (timeout, failure, etc.)
- [ ] Backward compatibility (local runner still works)
