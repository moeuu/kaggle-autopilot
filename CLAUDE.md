# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role
You are the **Staff Engineer / Architect** for this repository.
- **Planning first**: Review ARCHITECTURE.md, SPEC.md, and PLAN.md before implementing
- **Risk analysis**: Check SECURITY.md for safety requirements
- **Phased implementation**: Follow PLAN.md phases - don't skip ahead
- **Small, testable chunks**: Each PR should be a complete, tested unit of work

## Project Overview

Kaggle Autopilot is a **production-grade, fully automated** CLI tool for Kaggle competitions:

### Vision
- User provides competition URL: `kagglebot run https://www.kaggle.com/c/titanic --submit`
- Tool automatically: downloads data → analyzes competition → trains models → generates predictions → validates → submits
- **Zero prompts** - completely non-interactive (except manual rules acceptance once)
- **Production-grade models** - serious GBDT, stacking, CV (not toy baselines)
- **Safe by default** - dedup, rate limits, validation, reproducibility

### Current State (MVP)
- ✅ Data download and validation
- ✅ Basic tabular training (Ridge, LogisticRegression)
- ✅ Submission validation and local ledger
- ⏳ **Next**: Competition analyzer, orchestrator, production models

See PLAN.md for detailed roadmap.

## Critical Constraints (NEVER violate)

1. **Do NOT automate rule acceptance**: Never automate clicking "Join"/"I Agree" or accepting competition rules. Users must manually accept rules in browser at least once per competition. If rules aren't accepted, detect this and print URL + instructions, then exit.

2. **Do NOT scrape or bypass**: Never scrape Kaggle webpages, bypass rate limits, or circumvent submission limits.

3. **Do NOT enable abuse**: Never implement multi-account behavior, submission spamming, or rule circumvention.

4. **Do NOT commit secrets**: Never commit API credentials, tokens, or large datasets.

## Design Principles (Production-Grade Automation)

### Non-Interactive Operation
- **No prompts**: All decisions automated or configured via CLI flags/config files
- **Single command**: `kagglebot run <competition> --submit` runs end-to-end
- **Safe defaults**: Conservative choices when ambiguous (no external data, etc.)
- **Clear errors**: Exit codes and messages guide user on failures (see SPEC.md)

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
uv run kagglebot run titanic

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

## Planned CLI Interface

```bash
kagglebot bootstrap <competition_slug>                 # Prepare workspace dirs + config (no network)
kagglebot run <slug> --submission <path>               # Validate + ledger (dry-run default)
kagglebot run <slug> --no-dry-run --force --submission <path> -m "<message>"
kagglebot train <slug>                                 # Train baseline model (future)
kagglebot predict <slug>                               # Generate submission.csv (future)
kagglebot submit <slug> -m "<message>"                 # Submit with guardrails (future)
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

- **Small, composable functions**: Clear inputs/outputs, easy to test
- **User-friendly failures**: Actionable error messages
- **Kaggle CLI first**: Use Kaggle CLI with `~/.kaggle/kaggle.json` or env vars
- **Minimal dependencies**: pandas + scikit-learn + typer + rich for MVP
- **Deterministic runs**: Control random seeds when feasible
- **Python 3.11+**: Target modern Python
- **Resource limits**: Current MVP assumes datasets fit in memory
  - For large competitions (>1GB CSV), add chunked processing
  - LogisticRegression has max_iter=2000 to prevent hangs
  - Consider memory-mapped files or Dask for huge datasets

## Kaggle CLI Integration

- Uses Kaggle CLI (`kaggle competitions ...`)
- Authentication via `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`
- Download: `kaggle competitions download -c <slug>`
- Submit: `kaggle competitions submit -c <slug> -f <file> -m <message>`
- If CLI fails due to missing rule acceptance, print the competition rules URL and exit gracefully

## ZIP File Handling

- bootstrap.py extracts all ZIPs in `data/<slug>/raw/`
- **No zip bomb protection** (acceptable: Kaggle CLI is trusted source)
- If adding support for user-uploaded ZIPs, add size/ratio checks
- Uses `zipfile.ZipFile.extractall()` with fixed destination (no path traversal risk)

## Architect/Reviewer Role

When operating as architect/reviewer:

1. **Before Implementation**: Review PLAN.md for design coherence
2. **Interface Design**: Ensure CLI commands have clear contracts (inputs/outputs/errors)
3. **Safety Analysis**: Check all code paths for accidental submission or rule bypass
4. **Testability**: Verify new code has clear test boundaries (dependency injection where needed)
5. **Documentation**: Update PLAN.md and README.md for any architectural changes

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

- `kagglebot run <slug>` downloads data, trains baseline, produces valid submission.csv in artifacts/
- `kagglebot run <slug> --submit -m "baseline"` submits once with all guardrails enforced
- Works on common tabular competitions (e.g., Titanic-like: train.csv, test.csv, sample_submission.csv)
- Focus: "always produce valid submission" over leaderboard score

---

## Architectural Documentation

Before implementing features, review these documents:

1. **ARCHITECTURE.md** - System design, modules, data flow, extension points
2. **SPEC.md** - CLI commands, config schema, exit codes, artifact layout
3. **PLAN.md** - Phased implementation roadmap (7 phases to production)
4. **SECURITY.md** - Security guidelines, credential handling, safety guardrails
5. **AGENTS.md** - Instructions for implementer/tester agents (Codex)

### Implementation Workflow

1. **Plan**: Review PLAN.md phase tasks
2. **Design**: Check ARCHITECTURE.md for interfaces and patterns
3. **Spec**: Verify behavior matches SPEC.md contracts
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
- **Model registry**: Extensible model system (see ARCHITECTURE.md)

Follow the critical path in PLAN.md:
```
Phase 0 (Foundation) → Phase 1 (Analyzer) + Phase 2 (Orchestrator) →
Phase 3 (Training) → Phase 4 (Submission) → Phase 5 (Polish)
```
