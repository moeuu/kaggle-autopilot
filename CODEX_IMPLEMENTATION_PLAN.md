# Codex Implementation Plan

**Purpose**: Prioritized task list for completing kagglebot implementation

**Last Updated**: 2026-01-02

**Current Status**: Core functionality complete, autopilot implemented

---

## Overview

This document provides a concrete implementation plan for Codex (or any AI coding agent) to complete remaining kagglebot features.

**Completed (Phase 1):**
- ✅ Core CLI commands (bootstrap, implement, train, submit, run, autopilot)
- ✅ Kaggle notebook runner (GPU/TPU support)
- ✅ Autopilot with Top1 heuristic (single-submit method)
- ✅ Comprehensive validation + deduplication
- ✅ Test coverage (72 tests)

**Remaining (Phase 2):**
- 🔨 Codex improvement prompt generation (autopilot iterations)
- 🔨 Knowledge Base integration (SQLite + tags)
- 🔨 Diagnostics auto-generation (error analysis, feature importance)
- 🔨 End-to-end autopilot test (real Kaggle integration)

---

## Critical Safety Rules

**MUST DO:**
- ✅ Validate all inputs (competition slug, file paths, etc.)
- ✅ Never log or commit secrets (API keys, credentials)
- ✅ Never automate rules acceptance
- ✅ Use subprocess with list args (NEVER shell=True)
- ✅ Test all error paths
- ✅ Add docstrings + type hints

**MUST NOT DO:**
- ❌ Skip validation before submission
- ❌ Embed credentials in code or logs
- ❌ Automate browser actions (rules acceptance, etc.)
- ❌ Use unsafe subprocess calls (shell=True)
- ❌ Commit secrets, large files, or artifacts
- ❌ Override MAX_SUBMISSIONS = 1 in autopilot

---

## Priority 1: Critical Gaps (Complete These First)

### Task P1.1: Diagnostics Auto-Generation

**File**: `src/kagglebot/diagnostics.py` (NEW)

**Purpose**: Generate diagnostics.md with error analysis and recommendations

**Requirements:**
```python
def generate_diagnostics(
    model,
    X_train,
    y_train,
    X_eval,
    y_eval,
    y_pred,
    metrics: dict,
    output_path: Path,
) -> None:
    """
    Generate diagnostics.md with:
    1. Summary (model, score, target)
    2. Performance breakdown (by quantile, by feature)
    3. Error analysis (misclassifications, residuals)
    4. Feature importance (top 10)
    5. Recommendations (based on errors)
    6. Change log (what changed from previous iteration)
    7. Next steps (suggested improvements)
    """
```

**Acceptance Criteria:**
- [ ] Generates valid markdown file
- [ ] Includes all 7 required sections
- [ ] Works for classification + regression
- [ ] Handles missing features (e.g., no feature importance for NN)
- [ ] Tests: pytest with mock model + data

**Example Output:**
```markdown
# Diagnostics: Iteration 2

## Summary
- Model: RandomForestClassifier
- Offline Score: 0.9567 (accuracy, holdout 20%)
- Target: Survived (binary classification)

## Performance Breakdown
- Overall Accuracy: 0.9567
- Precision: 0.94
- Recall: 0.96
- F1: 0.95

## Error Analysis
- Total errors: 17 / 200 (8.5%)
- False Positives: 8 (predicted 1, actual 0)
- False Negatives: 9 (predicted 0, actual 1)

Most common errors:
- Young 3rd class passengers (60% of FN)
- Elderly 1st class passengers (50% of FP)

## Feature Importance (Top 10)
1. age_pclass_interaction: 0.23
2. Sex: 0.19
3. Age: 0.15
...

## Recommendations
1. Add age/class interaction features (addresses 60% of errors)
2. Consider fare/pclass interaction (hypothesis: 1st class fares vary)
3. Handle missing Age values better (imputation vs dropping)

## Change Log (from Iteration 1)
- Added: age_pclass_interaction, cabin_null_flag
- Removed: Ticket (low importance)
- Tuned: max_depth 6 → 8

## Next Steps
- Try gradient boosting (may capture nonlinear patterns)
- Add feature crosses for rare cases (elderly 3rd class)
- Consider ensemble with LogisticRegression
```

---

### Task P1.2: Codex Improvement Prompt Generation

**File**: `src/kagglebot/prompts.py` (NEW)

**Purpose**: Generate improvement prompts for Codex between autopilot iterations

**Requirements:**
```python
def generate_improvement_prompt(
    iteration: int,
    prev_metrics: dict,
    diagnostics_path: Path,
    kb_entries: list[dict],
    top1_score: float,
    output_path: Path,
) -> None:
    """
    Generate improve_codex.md with:
    1. Task description (iteration N of 5)
    2. Previous iteration results (offline score, model, features)
    3. Diagnostics summary (from diagnostics.md)
    4. Top1 gap analysis (how far from Top1)
    5. Retrieved KB patterns (similar improvements that worked)
    6. Overfitting warnings (holdout vs CV considerations)
    7. Acceptance criteria (beat previous or meet heuristic)
    """
```

**Template**: See `docs/AUTOPILOT_SINGLE_SUBMIT.md` Section 7.2

**Acceptance Criteria:**
- [ ] Uses Jinja2 or similar templating
- [ ] Includes all prev_metrics fields
- [ ] Parses diagnostics.md correctly
- [ ] Injects top 3 KB entries
- [ ] Warns about overfitting
- [ ] Tests: Mock inputs → valid prompt

**Integration Point:**
```python
# In autopilot_runner.py, after iteration completes:
if iteration < max_iterations:
    generate_improvement_prompt(
        iteration=iteration + 1,
        prev_metrics=load_metrics(iter_dir / "metrics.json"),
        diagnostics_path=iter_dir / "diagnostics.md",
        kb_entries=retrieve_kb_entries(config.kb_tags),
        top1_score=top1_info.top1_score,
        output_path=iter_dir / "improve_codex.md",
    )
    # Then invoke Codex with this prompt
```

---

### Task P1.3: Codex Invocation in Autopilot Loop

**File**: `src/kagglebot/autopilot_runner.py` (MODIFY)

**Purpose**: Automatically invoke Codex between iterations

**Requirements:**
```python
def invoke_codex_improvement(
    prompt_path: Path,
    agent_dir: Path,
    verify_cmd: str,
    dry_run: bool,
) -> bool:
    """
    Invoke Codex with improvement prompt.

    Args:
        prompt_path: Path to improve_codex.md
        agent_dir: Directory for agent transcript
        verify_cmd: Verification command (e.g., "uv run pytest -q")
        dry_run: Skip actual execution

    Returns:
        True if Codex succeeded and verification passed

    Behavior:
        1. Run: codex exec -a never --sandbox workspace-write \
                 --json --output-last-message <agent_dir>/last_message.json \
                 - < <prompt_path>
        2. Run verification command
        3. If pass: commit changes to dedicated branch
        4. If fail: rollback changes, record failure
    """
```

**Acceptance Criteria:**
- [ ] Uses codex CLI (not API)
- [ ] Reads prompt from file
- [ ] Writes transcript to agent_dir
- [ ] Runs verification command
- [ ] Commits on success
- [ ] Rollbacks on failure
- [ ] Respects dry-run mode
- [ ] Tests: Mock subprocess calls

**Integration:**
```python
# In autopilot_runner.py loop:
if iteration < max_iterations:
    prompt_path = iter_dir / "improve_codex.md"
    generate_improvement_prompt(...)

    success = invoke_codex_improvement(
        prompt_path=prompt_path,
        agent_dir=paths.runs_dir / run_id / f"iter-{iteration+1}" / "codex",
        verify_cmd=config.verify_cmd,
        dry_run=config.dry_run,
    )

    if not success:
        print(f"[yellow]Codex improvement failed, keeping iteration {iteration} code[/yellow]")
```

---

## Priority 2: Knowledge Base (Nice-to-Have)

### Task P2.1: Knowledge Base Schema

**File**: `src/kagglebot/kb.py` (NEW)

**Purpose**: SQLite-based knowledge base for improvement patterns

**Schema:**
```sql
CREATE TABLE kb_entries (
    id TEXT PRIMARY KEY,  -- kb_<timestamp>_<hash>
    competition_slug TEXT,
    tags TEXT,  -- JSON array: ["tabular", "classification"]
    created_at TEXT,  -- ISO timestamp

    -- Improvement record
    offline_score_before REAL,
    offline_score_after REAL,
    delta REAL,
    iterations_taken INTEGER,
    top1_score REAL,
    meets_heuristic BOOLEAN,
    metric TEXT,
    direction TEXT,  -- "minimize" or "maximize"

    -- What changed (JSON)
    what_changed TEXT,

    -- Why it worked (JSON)
    why_it_worked TEXT,

    -- Paths
    diagnostics_file TEXT,
    submission_path TEXT
);

CREATE INDEX idx_tags ON kb_entries(tags);
CREATE INDEX idx_slug ON kb_entries(competition_slug);
```

**Python Interface:**
```python
class KnowledgeBase:
    def __init__(self, db_path: Path):
        """Initialize KB with SQLite connection."""

    def create_entry(self, entry: KBEntry) -> str:
        """Create new KB entry, return ID."""

    def retrieve_by_tags(self, tags: set[str], limit: int = 10) -> list[KBEntry]:
        """Retrieve entries by Jaccard similarity on tags."""

    def retrieve_by_slug(self, slug: str, limit: int = 10) -> list[KBEntry]:
        """Retrieve entries for same competition."""
```

**Acceptance Criteria:**
- [ ] Creates SQLite database
- [ ] Implements CRUD operations
- [ ] Jaccard similarity ranking
- [ ] Tests: In-memory SQLite with mock data

---

### Task P2.2: KB Auto-Capture (Autopilot End)

**File**: `src/kagglebot/autopilot_runner.py` (MODIFY)

**Purpose**: Automatically create KB entry when autopilot completes

**Requirements:**
```python
def capture_kb_entry(
    slug: str,
    run_id: str,
    iterations: list[IterationResult],
    best_iteration: IterationResult,
    kb: KnowledgeBase,
) -> str:
    """
    Create KB entry from autopilot run.

    Captures:
    - offline_score_before: Iteration 1 score
    - offline_score_after: Best iteration score
    - delta: Improvement
    - top1_score: Fetched Top1
    - what_changed: Diff between iter 1 and best
    - why_it_worked: Parse diagnostics.md for hypothesis
    """
```

**Integration:**
```python
# At end of run_autopilot():
if improvement_observed:
    kb = KnowledgeBase(paths.repo_root / "knowledge_base.db")
    entry_id = capture_kb_entry(
        slug=slug,
        run_id=run_record.run_id,
        iterations=state.iteration_history,
        best_iteration=best,
        kb=kb,
    )
    print(f"[green]KB entry created:[/green] {entry_id}")
```

**Acceptance Criteria:**
- [ ] Only creates entry if delta > 0 (for maximize) or delta < 0 (for minimize)
- [ ] Parses diagnostics.md correctly
- [ ] Stores all required fields
- [ ] Tests: Mock iterations + KB

---

## Priority 3: Polishing (Final Touches)

### Task P3.1: End-to-End Autopilot Test (Real Kaggle)

**File**: `tests/test_autopilot_integration.py` (NEW)

**Purpose**: Verify autopilot works with real Kaggle competition

**Requirements:**
```python
@pytest.mark.integration
@pytest.mark.skipif(not has_kaggle_credentials(), reason="Requires Kaggle credentials")
def test_autopilot_titanic_end_to_end():
    """
    Test autopilot on Titanic competition.

    Flow:
    1. Bootstrap titanic (download data)
    2. Run autopilot (1 iteration only, --no-submit-at-final)
    3. Verify:
       - Iteration 1 completes
       - metrics.json created
       - submission.csv created and valid
       - Top1 score fetched
       - Heuristic evaluated
    """
```

**Acceptance Criteria:**
- [ ] Uses pytest.mark.integration
- [ ] Skips if no Kaggle credentials
- [ ] Uses real Kaggle CLI (not mocked)
- [ ] Cleans up artifacts after test
- [ ] Runs in < 5 minutes (1 iteration only)
- [ ] Documents manual test results in comments

**Manual Test Checklist:**
```bash
# 1. Bootstrap
uv run kagglebot bootstrap titanic --download --force

# 2. Autopilot (1 iteration, no submit)
uv run kagglebot autopilot titanic \
  --max-iterations 1 \
  --no-submit-at-final \
  --force

# 3. Verify
ls artifacts/titanic/runs/*/iter-1/metrics.json
ls artifacts/titanic/submissions/*_iter1_submission.csv
cat artifacts/titanic/context/top1_public.json

# 4. Check logs for:
# - "Fetching Kaggle Top1 public score..."
# - "Leaderboard] Top1 public score: X.XX"
# - "Heuristic] Offline X.XX vs Top1 X.XX"
```

---

### Task P3.2: CLI Help Documentation

**File**: All CLI commands (MODIFY docstrings)

**Purpose**: Ensure --help output is comprehensive and accurate

**Requirements:**
- Each command has clear docstring
- All flags documented with descriptions
- Examples in help text
- Exit codes documented

**Example:**
```python
@app.command()
def autopilot(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    ...
) -> None:
    """
    Autopilot: Iterative offline improvement with Top1 heuristic gating.

    Iterates offline evaluation (holdout/CV) up to max-iterations,
    compares to Kaggle Public Top1 score, and submits at most once.

    Examples:
        # Basic autopilot (5 iterations, submit best)
        uv run kagglebot autopilot titanic --force

        # Early submit when heuristic met
        uv run kagglebot autopilot titanic --submit-on-heuristic --force

        # Exploration mode (no submission)
        uv run kagglebot autopilot titanic --no-submit-at-final

    Exit Codes:
        0: Success
        1: General error
        2: Rules not accepted
        6: Validation error
        14: Max submissions exceeded
    """
```

**Acceptance Criteria:**
- [ ] All 6 commands have comprehensive docstrings
- [ ] Examples included
- [ ] Exit codes documented
- [ ] Run `uv run kagglebot --help` and verify output
- [ ] Run `uv run kagglebot <command> --help` for each command

---

### Task P3.3: Security Audit

**File**: All files (REVIEW)

**Purpose**: Ensure no security vulnerabilities

**Checklist:**
- [ ] No secrets in code or logs
- [ ] No `shell=True` in subprocess calls
- [ ] No hardcoded credentials
- [ ] Input validation on all user inputs (slug, paths, etc.)
- [ ] Path traversal prevention (validate_slug)
- [ ] Secret scanning patterns comprehensive
- [ ] Rules acceptance never automated

**Tools:**
```bash
# 1. Search for shell=True
rg "shell\s*=\s*True" src/

# 2. Search for hardcoded credentials
rg "kaggle.*key|api.*key|password" src/

# 3. Search for rule automation keywords
rg "accept.*rules|join.*competition" src/

# 4. Check .gitignore
cat .gitignore | grep -E "kaggle.json|artifacts|data"
```

**Acceptance Criteria:**
- [ ] No security issues found
- [ ] All subprocess calls use list args
- [ ] .gitignore excludes secrets + artifacts
- [ ] README warns about credential handling

---

## Priority 4: Documentation Updates

### Task P4.1: Update README with Autopilot Examples

**File**: README.md (MODIFY)

**Status**: ✅ COMPLETED

**Verification:**
- [x] Autopilot section exists
- [x] Examples provided
- [x] Disclaimers included (offline vs public LB)
- [x] Key flags documented

---

### Task P4.2: Update CLAUDE.md with Autopilot Context

**File**: CLAUDE.md (MODIFY)

**Add Section:**
```markdown
## Autopilot Mode

Autopilot (Method 1: Single-Submit) is designed to:
- Iterate improvements offline (no submissions during loop)
- Compare offline score to Kaggle Top1 using heuristic
- Submit at most once per run

Key design decisions:
- MAX_SUBMISSIONS = 1 (hard-coded, cannot override)
- Heuristic uses BOTH absolute and relative margins (AND condition)
- Top1 fetched from public leaderboard (no submission required)
- Offline scores NOT directly comparable to public LB (distribution shift)

When reviewing autopilot code:
- Ensure MAX_SUBMISSIONS enforcement
- Verify heuristic logic (AND not OR)
- Check Top1 caching (60 min TTL)
- Confirm submission happens from local machine (not kernel)
```

**Acceptance Criteria:**
- [ ] CLAUDE.md includes autopilot section
- [ ] Design decisions documented
- [ ] Review guidelines provided

---

### Task P4.3: Create TESTING.md

**File**: TESTING.md (NEW)

**Purpose**: Testing strategy and guidelines

**Content:**
```markdown
# Testing Strategy

## Unit Tests (72 tests)

Run all tests:
```bash
uv run pytest -q
```

Run specific test file:
```bash
uv run pytest tests/test_autopilot.py -v
```

## Test Coverage

Target: >80% coverage for core logic

Check coverage:
```bash
uv run pytest --cov=src/kagglebot --cov-report=html
open htmlcov/index.html
```

## Integration Tests

Real Kaggle tests (manual, once):
```bash
# Mark as @pytest.mark.integration
# Run with: pytest -m integration
```

## Mocking Strategy

- Mock Kaggle CLI calls (subprocess)
- Mock Kaggle Python API (KaggleApi)
- Mock file system operations (Path.exists, etc.)
- Use pytest fixtures for common setups

## Manual Testing Checklist

Before release:
- [ ] Bootstrap works (download data)
- [ ] Implement works (codex runner)
- [ ] Train works (local CPU)
- [ ] Train works (local GPU, if available)
- [ ] Submit works (validation + dedupe)
- [ ] Autopilot works (1 iteration, no submit)
- [ ] All --help outputs correct
- [ ] No secrets logged
```

**Acceptance Criteria:**
- [ ] TESTING.md created
- [ ] Covers all test types
- [ ] Provides examples
- [ ] Documents manual testing

---

## Acceptance Criteria (Overall)

### Functionality
- [ ] All 6 CLI commands work end-to-end
- [ ] Autopilot completes 5 iterations (local CPU)
- [ ] Autopilot fetches Top1 and evaluates heuristic
- [ ] Submission validation prevents invalid CSVs
- [ ] Deduplication prevents duplicate submissions
- [ ] MAX_SUBMISSIONS = 1 enforced in autopilot

### Testing
- [ ] All tests pass: `uv run pytest -q` (72+ tests)
- [ ] Code coverage > 80% (core modules)
- [ ] Manual integration test documented (Titanic)

### Documentation
- [ ] README accurate and comprehensive
- [ ] ARCHITECTURE_FINAL.md complete
- [ ] CHECKLIST_SUBMIT.md reviewed
- [ ] FAILURE_MODES.md covers all scenarios
- [ ] All CLI --help outputs correct

### Security
- [ ] No secrets in code or logs
- [ ] No shell=True in subprocess
- [ ] .gitignore excludes secrets + artifacts
- [ ] Secret scanning patterns comprehensive
- [ ] Rules acceptance never automated

### Code Quality
- [ ] Passes: `uv run ruff check .`
- [ ] Passes: `uv run ruff format .` (no changes)
- [ ] All functions have type hints
- [ ] All public functions have docstrings
- [ ] No TODOs or FIXMEs in main branch

---

## Implementation Order

**Week 1: Diagnostics + Prompts**
1. Task P1.1: Diagnostics auto-generation
2. Task P1.2: Improvement prompt generation
3. Tests for both

**Week 2: Codex Integration**
1. Task P1.3: Codex invocation in loop
2. End-to-end test (1 iteration)
3. Manual testing

**Week 3: Knowledge Base (Optional)**
1. Task P2.1: KB schema + CRUD
2. Task P2.2: Auto-capture
3. Retrieval integration

**Week 4: Polishing**
1. Task P3.1: Integration test
2. Task P3.2: CLI help review
3. Task P3.3: Security audit
4. Task P4.1-P4.3: Docs updates

---

## Notes for Implementer

**Before Starting Each Task:**
1. Read task requirements carefully
2. Check existing code for similar patterns
3. Write tests FIRST (TDD approach)
4. Implement minimal solution
5. Run tests frequently

**While Implementing:**
- Commit after each sub-task (small commits)
- Update docstrings as you go
- Add type hints to all new functions
- Handle errors gracefully (user-friendly messages)

**Before Marking Task Complete:**
- [ ] All tests pass
- [ ] Linting passes (ruff)
- [ ] Documentation updated
- [ ] Manual testing done (if applicable)
- [ ] No TODOs left in code

**Common Pitfalls:**
- Don't skip validation (always validate inputs)
- Don't use shell=True (security risk)
- Don't commit secrets or large files
- Don't assume Kaggle CLI format won't change (parse defensively)

---

## Questions / Ambiguities

**Q1**: Should KB be per-competition or global?
**A**: Start with global (easier), add per-competition filtering later.

**Q2**: Should Codex invocation be automatic or require flag?
**A**: Automatic in autopilot mode (opt-out with --no-codex-improve).

**Q3**: What if diagnostics generation fails?
**A**: Log error, continue autopilot without diagnostics.md, don't fail entire run.

**Q4**: Should we support GPT-4 for diagnostics analysis?
**A**: Future enhancement, not MVP. Keep diagnostics rule-based for now.

**Q5**: How to handle Codex rate limits?
**A**: Implement exponential backoff, skip iteration if rate limited after 3 retries.

---

**Last Updated**: 2026-01-02
**Status**: Ready for implementation
**Estimated Effort**: 2-4 weeks (for Priority 1 + 2)
