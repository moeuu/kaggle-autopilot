# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle Autopilot is a CLI tool that automates Kaggle competition workflows:
- Download competition data via Kaggle CLI
- Build baseline models (MVP: tabular CSV competitions)
- Generate valid submission.csv matching sample_submission.csv
- Submit to Kaggle via CLI with safety guardrails

## Critical Constraints (NEVER violate)

1. **Do NOT automate rule acceptance**: Never automate clicking "Join"/"I Agree" or accepting competition rules. Users must manually accept rules in browser at least once per competition. If rules aren't accepted, detect this and print URL + instructions, then exit.

2. **Do NOT scrape or bypass**: Never scrape Kaggle webpages, bypass rate limits, or circumvent submission limits.

3. **Do NOT enable abuse**: Never implement multi-account behavior, submission spamming, or rule circumvention.

4. **Do NOT commit secrets**: Never commit `kaggle.json`, API keys, tokens, or large datasets.

## Submission Safety Guardrails (must implement)

- **Default to DRY RUN**: End-to-end commands must NOT submit by default
- **Require explicit flag**: Submissions require `--submit` flag AND human-readable message
- **Duplicate detection**: Hash submission.csv and maintain local history to prevent duplicate submissions
- **Strict validation**: Before any submission:
  - Columns must be identical to sample_submission.csv
  - Row count must match exactly
  - If id column exists, align rows by id

## Development Commands

```bash
# Run tests
pytest -q

# Lint and format
ruff check .
ruff format .

# Type checking (optional)
pyright  # or mypy
```

## Planned CLI Interface

```bash
kagglebot bootstrap <competition_slug>  # Download data, setup workspace
kagglebot train <slug>                  # Train baseline model
kagglebot predict <slug>                # Generate submission.csv
kagglebot submit <slug> -m "<message>"  # Submit with guardrails
kagglebot run <slug> [--submit]         # End-to-end (default: dry-run)
```

## Target Architecture

```
kagglebot/
  cli.py           # Typer/Rich CLI entry point
  kaggle_cli.py    # Wrapper around `kaggle` command (subprocess)
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
- **Subprocess wrappers**: Use `subprocess.run(..., check=True)` for Kaggle CLI
- **Minimal dependencies**: pandas + scikit-learn + typer + rich for MVP
- **Deterministic runs**: Control random seeds when feasible
- **Python 3.11+**: Target modern Python

## Kaggle CLI Integration

- Download: `kaggle competitions download -c <slug>`
- Submit: `kaggle competitions submit -c <slug> -f submission.csv -m "<message>"`
- If Kaggle CLI fails due to missing rule acceptance, print the competition rules URL and exit gracefully

## MVP Success Criteria

- `kagglebot run <slug>` downloads data, trains baseline, produces valid submission.csv in artifacts/
- `kagglebot run <slug> --submit -m "baseline"` submits once with all guardrails enforced
- Works on common tabular competitions (e.g., Titanic-like: train.csv, test.csv, sample_submission.csv)
- Focus: "always produce valid submission" over leaderboard score
