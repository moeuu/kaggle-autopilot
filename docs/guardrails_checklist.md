# Kagglebot Guardrails Checklist

This document provides a comprehensive checklist for ensuring safety, correctness, and ethical operation of kagglebot autopilot. Review this checklist before every code change, deployment, and autopilot run.

Note: Git integration has been removed from the implementation; section 4 is historical.

---

## 1. Safety Guardrails

### 1.1 Submission Safety

- [ ] **Default to dry-run**: All commands that could submit default to dry-run mode
- [ ] **Explicit submit flag**: Submissions require `--submit` flag explicitly set by user
- [ ] **Deduplication enforced**: SHA256 hash checked against `ledger.jsonl` before submission
- [ ] **Rate limiting active**: Cooldown period (default 5 min) enforced between submissions
- [ ] **Max submissions cap**: Hard limit per autopilot run (default 5) cannot be bypassed
- [ ] **Validation before submit**: submission.csv validated against sample_submission.csv (columns, rows, types)
- [ ] **No auto-submit loops**: No infinite submission loops or unbounded retry logic
- [ ] **Clear user messaging**: Submit commands print clear messages about what will happen

### 1.2 Rules and Ethics

- [ ] **No automated rules acceptance**: Never automate clicking "Join" or "I Agree" on Kaggle
- [ ] **Rules check enforced**: `check_rules_accepted()` runs before first submission attempt
- [ ] **Clear error messages**: If rules not accepted, print URL and instructions, then exit with code 2
- [ ] **No scraping**: Never scrape Kaggle webpages or bypass rate limits
- [ ] **No circumvention**: Never bypass submission limits or Kaggle API restrictions
- [ ] **No multi-account**: Never implement multi-account behavior or abuse detection evasion
- [ ] **Respect competition deadlines**: Detect and prevent submissions after competition closes

### 1.3 Secret Protection

- [ ] **No secrets in code**: kaggle.json, API keys, tokens never hardcoded
- [ ] **gitignore enforced**: `.env`, `kaggle.json`, `data/`, `artifacts/` in .gitignore
- [ ] **No secret logging**: Credentials never echoed in subprocess output or logs
- [ ] **No secret prompts**: Never prompt user for credentials interactively (must use kaggle.json)
- [ ] **No secrets in KB**: Knowledge base never stores credentials or sensitive data
- [ ] **No secrets in prompts**: Agent prompts never include API keys or credentials
- [ ] **Artifact privacy**: No competition data or predictions committed to git

### 1.4 Non-Interactive Design

- [ ] **No user prompts**: Zero interactive prompts during autopilot run
- [ ] **No browser automation**: No automated browser actions (Selenium, Puppeteer, etc.)
- [ ] **All decisions pre-configured**: All choices encoded in plan.json or CLI flags
- [ ] **Actionable error messages**: Errors exit with clear next steps and non-zero exit codes
- [ ] **Unattended operation**: Can run in cron jobs, CI/CD, or background without human intervention

---

## 2. Correctness Guardrails

### 2.1 Submission Validation

- [ ] **Column match**: submission.csv columns identical to sample_submission.csv
- [ ] **Row count match**: submission.csv row count exactly matches sample
- [ ] **ID alignment**: If ID column exists, rows aligned by ID (not just index)
- [ ] **Type validation**: Column types match expected (numeric vs categorical)
- [ ] **No NaN values**: All predictions are valid (no NaN, inf, or missing values)
- [ ] **Format validation**: CSV format correct (encoding, delimiters, quotes)
- [ ] **Range validation**: Predictions within valid range (e.g., probabilities in [0, 1])

### 2.2 Evaluation Correctness

- [ ] **Offline only**: Never use test.csv for validation (only train.csv splits)
- [ ] **Direction-aware**: metric comparison respects direction (minimize vs maximize)
- [ ] **Reproducible**: Fixed random seeds (42) for train/test splits and model initialization
- [ ] **Stratified splits**: Classification uses stratified splits to preserve class balance
- [ ] **No data leakage**: Test set never seen during training or validation
- [ ] **Correct metric**: Metric matches competition (check plan.json and metadata)
- [ ] **CV robust**: Cross-validation folds properly stratified and seeded

### 2.3 Score-Based Gating

- [ ] **meets_target implemented**: `_meets_target()` function direction-aware (minimize/maximize)
- [ ] **Target from plan.json**: Target score read from plan.json, not hardcoded
- [ ] **Submission gated**: Submission only when `met_target=true` OR iteration 5 (final policy)
- [ ] **Top1 context only**: Top1 public leaderboard used for context, NOT submission decision
- [ ] **Heuristic logged**: Top1 comparison logged with disclaimer about offline-online mismatch

### 2.4 Patience and Early Stopping

- [ ] **Patience counter implemented**: Stops after N iterations without improvement (default: 2)
- [ ] **Improvement tracking**: `_update_best_score()` correctly detects improvement
- [ ] **Min improvement threshold**: Configurable min_improvement (default: 0.0)
- [ ] **Best score tracking**: best_score persisted across iterations
- [ ] **Early stop logged**: Clear message when patience exceeded

---

## 3. Resource Guardrails

### 3.1 Hard Caps

- [ ] **Max iterations**: Default 1, configurable, enforced
- [ ] **Max total time**: Default 120 min (2 hours), enforced with wall-clock check
- [ ] **Max submissions per run**: Default 5, enforced in submission counter
- [ ] **Timeout per iteration**: Reasonable timeout (e.g., 60 min per training run)
- [ ] **Memory limits**: Respect system memory, don't assume unlimited RAM
- [ ] **Disk space checks**: Verify sufficient space before downloading data

### 3.2 Compute Modes

- [ ] **local_cpu safe default**: Default to local_cpu if no compute specified
- [ ] **GPU availability check**: local_gpu errors gracefully if no GPU available
- [ ] **Kaggle kernel timeout**: Handle Kaggle kernel timeouts (exit code 11)
- [ ] **Kaggle kernel failure**: Handle kernel failures (exit code 12)
- [ ] **Internet default OFF**: Kaggle kernels default to `--internet off`
- [ ] **Internet override explicit**: `--internet on` requires explicit user flag

### 3.3 GPU/TPU Utilization

- [ ] **Utilization logged**: GPU/TPU utilization recorded in metrics.json
- [ ] **Target utilization**: Aim for >80% GPU, >70% TPU MXU
- [ ] **Low util warning**: Warn if utilization <50% (resource waste)
- [ ] **OOM handling**: Out-of-memory errors caught and logged
- [ ] **Batch size tuning**: Suggestions for increasing batch size if util low

---

## 4. Git Guardrails

### 4.1 Main-Only Workflow

- [ ] **Ensure on main**: Check current branch, switch to main if needed
- [ ] **Auto-stash implemented**: Dirty state stashed (including untracked) before run
- [ ] **No branch creation**: Never create feature branches during autopilot
- [ ] **Diff saved**: git diff saved to artifacts/diffs/{run_id}.diff after run
- [ ] **No auto-commit**: Do NOT commit by default (user decision)
- [ ] **Stash restoration**: Offer to restore stash after run completes
- [ ] **Test verification**: If tests fail, revert changes and warn user

### 4.2 Git Safety

- [ ] **No force push**: Never run git push --force
- [ ] **No history rewrite**: Never run git rebase, git commit --amend, or git reset --hard
- [ ] **No submodule automation**: Don't automatically init/update submodules
- [ ] **No gitignore changes**: Never modify .gitignore programmatically
- [ ] **Clean working dir check**: Warn if working directory has uncommitted critical files

---

## 5. Agent Prompts Guardrails

### 5.1 Prompt Content

- [ ] **Context files included**: dataset_profile, sample_submission, top1, rules_url, KB hints
- [ ] **Compute mode specified**: Prompt includes compute mode and constraints
- [ ] **Acceptance criteria clear**: Tests pass, offline score improves, GPU/TPU utilized
- [ ] **Safety rules included**: All CRITICAL SAFETY RULES listed in prompt
- [ ] **No secrets in prompts**: Never include kaggle.json, API keys, tokens
- [ ] **Web search allowed**: Explicitly state web search allowed for docs (not secrets)
- [ ] **Paths absolute**: All file paths in prompts are absolute (not relative)

### 5.2 Agent Behavior

- [ ] **Agent timeout**: Agent has reasonable timeout (e.g., 30 min per call)
- [ ] **Agent error handling**: Agent errors logged and gracefully handled
- [ ] **Agent output validation**: Agent output parsed and validated before use
- [ ] **No infinite loops**: Agent calls have max retry limit
- [ ] **Agent cost tracking**: Track API costs if using paid agent services

---

## 6. Error Handling Guardrails

### 6.1 Exit Codes

- [ ] **Exit codes defined**: All errors have unique exit codes (see exceptions.py)
- [ ] **Actionable messages**: Every error includes what went wrong, why, and how to fix
- [ ] **Non-zero on error**: Always exit with non-zero code on failure
- [ ] **Clean shutdown**: Resources cleaned up (temp files, processes) before exit
- [ ] **Log preservation**: Errors don't delete logs (keep for debugging)

### 6.2 Specific Error Handling

- [ ] **RulesNotAcceptedError (2)**: Print rules URL, exit cleanly
- [ ] **KaggleCliError (4)**: Log Kaggle CLI stderr, suggest fixes
- [ ] **ValidationError (6)**: Show submission vs sample diff, suggest fixes
- [ ] **DuplicateSubmissionError (8)**: Show ledger entry, explain deduplication
- [ ] **SubmissionRateLimitError (9)**: Show cooldown time remaining
- [ ] **GPUNotAvailableError (10)**: Suggest switching to local_cpu or kaggle_gpu
- [ ] **KernelTimeoutError (11)**: Suggest reducing dataset size or model complexity
- [ ] **MaxSubmissionsError (14)**: Show submission count, explain quota

---

## 7. Knowledge Base Guardrails

### 7.1 Data Storage

- [ ] **No raw data**: KB never stores raw train.csv, test.csv, or predictions
- [ ] **Metadata only**: Only store competition metadata, tags, run outcomes
- [ ] **Summaries only**: Store improvement summaries, not full model weights
- [ ] **Privacy respected**: No personally identifiable information in KB
- [ ] **Controlled taxonomy**: Tags use controlled vocabulary (taxonomy.yml)

### 7.2 KB Operations

- [ ] **SQL injection safe**: All queries parameterized (no string concatenation)
- [ ] **Schema validated**: DB schema matches expected tables/columns
- [ ] **Transaction safety**: Use transactions for multi-table updates
- [ ] **Backup before update**: KB backed up before major schema changes
- [ ] **Query efficiency**: Retrieval queries indexed and fast (<100ms)

---

## 8. Testing Guardrails

### 8.1 Test Coverage

- [ ] **Unit tests**: Core functions (_meets_target, _update_best_score) have tests
- [ ] **Integration tests**: End-to-end autopilot flow tested (with mocks)
- [ ] **Safety tests**: Submission guardrails tested (deduplication, validation, rate limit)
- [ ] **Edge cases**: Empty files, missing columns, type mismatches tested
- [ ] **No real submissions**: Tests never submit to Kaggle (use monkeypatching/mocks)

### 8.2 CI/CD

- [ ] **Tests run on PR**: GitHub Actions runs pytest on every PR
- [ ] **Linting enforced**: ruff check and ruff format run on every commit
- [ ] **Type checking**: pyright or mypy runs on critical modules
- [ ] **Coverage threshold**: Minimum test coverage (e.g., >70%)
- [ ] **No secrets in CI**: CI environment variables never contain real credentials

---

## 9. Documentation Guardrails

### 9.1 User-Facing Docs

- [ ] **README minimal**: README has single command example, links to docs/
- [ ] **Help text complete**: Every CLI command has `--help` with examples
- [ ] **Safety warnings**: Submit commands warn about real submission
- [ ] **Defaults documented**: All default values documented in help text
- [ ] **Error messages actionable**: Every error in docs/errors.md with examples

### 9.2 Developer Docs

- [ ] **Architecture doc**: docs/architecture.md up to date
- [ ] **CLAUDE.md current**: Project instructions for Claude Code up to date
- [ ] **Changelog maintained**: CHANGELOG.md has all notable changes
- [ ] **Decision log**: Major architectural decisions documented in docs/spec_autopilot.md
- [ ] **API docs**: Public functions have docstrings with examples

---

## 10. Pre-Commit Checklist

Before every commit, verify:

- [ ] No `kaggle.json`, `.env`, API keys, or secrets in diff
- [ ] No large CSV/zip files committed (check .gitignore)
- [ ] All submission code paths guarded by `--submit` flag
- [ ] Duplicate submission check cannot be bypassed
- [ ] Validation runs before any Kaggle CLI submit call
- [ ] Error messages include actionable next steps
- [ ] No `shell=True` in subprocess calls (injection risk)
- [ ] All new functions have type hints
- [ ] Tests pass: `uv run pytest -q`
- [ ] Linting passes: `uv run ruff check . && ruff format .`
- [ ] Docstrings added for public functions
- [ ] CHANGELOG.md updated if user-facing change

---

## 11. Pre-Release Checklist

Before every release, verify:

- [ ] All items in Pre-Commit Checklist pass
- [ ] Integration tests pass end-to-end
- [ ] Dry-run mode tested and verified (no real submissions)
- [ ] Submit mode tested with test competition (if safe)
- [ ] Documentation reviewed and updated
- [ ] Version bumped in pyproject.toml
- [ ] CHANGELOG.md has release notes
- [ ] Git tags created for version
- [ ] No breaking changes without major version bump
- [ ] Backward compatibility tested for existing users

---

## 12. Incident Response Checklist

If an incident occurs (e.g., unintended submission, secret leak):

### 12.1 Immediate Actions

- [ ] **Stop all runs**: Kill any running autopilot processes
- [ ] **Revoke credentials**: If secrets leaked, revoke Kaggle API key immediately
- [ ] **Assess impact**: Check ledger.jsonl for unintended submissions
- [ ] **Document incident**: Write detailed timeline of what happened
- [ ] **Notify users**: If public, notify affected users via GitHub issue

### 12.2 Remediation

- [ ] **Fix root cause**: Patch code to prevent recurrence
- [ ] **Add regression test**: Ensure incident can't happen again (test coverage)
- [ ] **Update guardrails**: Add new checklist items if gap identified
- [ ] **Review related code**: Check for similar vulnerabilities elsewhere
- [ ] **Postmortem**: Write postmortem with timeline, root cause, prevention

### 12.3 Prevention

- [ ] **Audit logs**: Review all logs for similar issues
- [ ] **Security review**: External review of code for vulnerabilities
- [ ] **Enhanced monitoring**: Add logging/alerting for risky operations
- [ ] **User education**: Update docs with incident learnings

---

## Quick Reference: Mandatory Checks

### Every Code Change

1. No secrets in diff
2. Tests pass
3. Linting passes
4. Submission guardrails intact

### Every Autopilot Run

1. Rules accepted (manual browser)
2. `--submit` flag explicitly set by user
3. Deduplication check passes
4. Validation check passes
5. Rate limit respected

### Every Submission

1. SHA256 not in ledger
2. Format matches sample_submission.csv
3. Cooldown period elapsed
4. Max submissions not exceeded
5. User explicitly set `--submit`

---

## Audit Trail

Every autopilot run should produce:

- [ ] `run.json` with full configuration and outcomes
- [ ] `ledger.jsonl` with submission hashes and timestamps
- [ ] `metrics.json` for each iteration
- [ ] `diagnostics.md` for each iteration
- [ ] `diffs/{run_id}.diff` with code changes
- [ ] `logs/` with full stdout/stderr

These artifacts enable:
- Reproducibility (what was run, when, with what config)
- Accountability (who submitted what)
- Debugging (what went wrong)
- Learning (what worked, what didn't)

---

## Summary

This checklist ensures kagglebot operates:
- **Safely**: No unintended submissions, no secret leaks
- **Ethically**: Respects Kaggle rules, no circumvention
- **Correctly**: Valid submissions, accurate evaluation
- **Reliably**: Graceful errors, clean shutdown, reproducible
- **Transparently**: Full audit trail, clear messaging

Review this checklist regularly and update as new risks are identified.

**Last Updated**: 2026-01-02
**Next Review**: Every major release or incident
