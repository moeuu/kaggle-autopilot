# Codex Implementation Prompt

**For**: AI coding assistants (Codex, Claude Code agents, GitHub Copilot)
**Purpose**: Implement kagglebot CLI automation with safety guardrails
**Estimated Time**: 7 weeks (140 tasks)

---

## Critical Safety Rules

**YOU MUST NEVER**:
1. ❌ Automate Kaggle competition rules acceptance (user must click in browser)
2. ❌ Embed secrets in code or logs (kaggle.json, API keys, tokens)
3. ❌ Use `shell=True` in subprocess calls (command injection risk)
4. ❌ Skip validation before submission (format, dedup, rules check)
5. ❌ Commit credentials, datasets, or large artifacts

**YOU MUST ALWAYS**:
1. ✅ Check rules accepted before submission (exit code 2 if not)
2. ✅ Validate submission format against sample_submission.csv
3. ✅ Check for duplicate submissions (SHA256 hash in ledger)
4. ✅ Scan kernel packages for secrets before push
5. ✅ Use list args for subprocess: `["kaggle", "competitions", "download"]`

---

## Implementation Order (Critical Path)

Follow this exact order (dependencies enforced):

### **Phase C1: Foundation** (Week 1) - MUST complete before C2

**Goal**: Safety infrastructure and input validation

**Tasks**:
1. Create `src/kagglebot/exceptions.py`:
   ```python
   class KaggleBotError(Exception):
       exit_code = 1

   class RulesNotAcceptedError(KaggleBotError):
       exit_code = 2

   class GPUNotAvailableError(KaggleBotError):
       exit_code = 10
   # ... (see ../architecture_final.md section 5.4)
   ```

2. Create `src/kagglebot/competition.py`:
   ```python
   def validate_slug(slug: str) -> str:
       """Validate slug format, reject path traversal."""
       if not re.match(r"^[a-z0-9-]+$", slug):
           raise ValueError(f"Invalid slug: {slug}")
       return slug

   def parse_competition_slug(competition: str) -> str:
       """Extract slug from URL or return as-is."""
       # Parse https://www.kaggle.com/competitions/<slug>
       # Or return slug directly
       pass
   ```

3. Setup logging in `src/kagglebot/logging_setup.py`

4. Create `src/kagglebot/compute/` package (GPU detection)

**Acceptance Criteria**:
- ✅ `validate_slug("../etc/passwd")` raises ValueError
- ✅ `validate_slug("titanic")` returns "titanic"
- ✅ `parse_competition_slug("https://www.kaggle.com/competitions/titanic")` returns "titanic"
- ✅ All exceptions have exit_code attribute
- ✅ Tests pass: `uv run pytest tests/test_competition.py tests/test_exceptions.py -v`

---

### **Phase C2: Critical Safety Checks** (Week 2) - MUST complete before C6

**Goal**: Implement rules check and duplicate detection

**Tasks**:
1. Implement `check_rules_accepted()` in `src/kagglebot/kaggle_cli.py`:
   ```python
   def check_rules_accepted(slug: str) -> bool:
       """
       Check if user accepted competition rules.

       Uses: kaggle competitions list
       Checks if slug in user's competitions.
       """
       result = subprocess.run(
           ["kaggle", "competitions", "list", "--csv"],
           capture_output=True,
           text=True,
           check=True,
       )
       # Parse CSV output
       # Return True if slug in list
       pass
   ```

2. Implement `SubmissionLedger` in `src/kagglebot/history.py`:
   ```python
   class SubmissionLedger:
       def __init__(self, artifacts_dir: Path):
           self.ledger_path = artifacts_dir / "submissions" / "history.jsonl"

       def is_duplicate(self, submission_path: Path) -> bool:
           """Check if hash already in ledger."""
           hash_val = compute_hash(submission_path)
           # Read ledger, check if hash exists
           pass

       def record(self, submission_path: Path, message: str) -> None:
           """Append to ledger (JSONL format)."""
           entry = {
               "hash": compute_hash(submission_path),
               "timestamp": datetime.now().isoformat(),
               "message": message,
               "file": str(submission_path),
           }
           # Append to ledger
           pass
   ```

3. Implement `compute_hash()` in `src/kagglebot/hashing.py`:
   ```python
   def compute_hash(file_path: Path) -> str:
       """Compute SHA256 hash of file."""
       hasher = hashlib.sha256()
       with open(file_path, "rb") as f:
           for chunk in iter(lambda: f.read(4096), b""):
               hasher.update(chunk)
       return hasher.hexdigest()
   ```

**Acceptance Criteria**:
- ✅ `check_rules_accepted("titanic")` returns True (if accepted) or False
- ✅ `is_duplicate()` detects identical files by hash
- ✅ `record()` appends to JSONL (not overwrites)
- ✅ Tests pass: `uv run pytest tests/test_history.py tests/test_kaggle_cli.py -v`
- ✅ Manual test: Accept Titanic rules, verify check returns True

---

### **Phase C3: Kernel Package Generation** (Week 3)

**Goal**: Generate valid kernel packages with secret scanning

**Tasks**:
1. Implement `generate_kernel_metadata()` in `src/kagglebot/kernel/metadata.py`

2. **CRITICAL**: Implement secret scanning in `src/kagglebot/kernel/packager.py`:
   ```python
   SECRET_PATTERNS = [
       r"kaggle\.json",
       r"KAGGLE_KEY",
       r"api_key",
       r"password",
       r"token",
       r"secret",
   ]

   def validate_kernel_package(package_dir: Path) -> None:
       """Scan for secrets before push."""
       main_py = (package_dir / "main.py").read_text()
       metadata = (package_dir / "kernel-metadata.json").read_text()

       for pattern in SECRET_PATTERNS:
           if re.search(pattern, main_py, re.IGNORECASE):
               raise ValueError(f"Secret pattern detected: {pattern}")
           if re.search(pattern, metadata, re.IGNORECASE):
               raise ValueError(f"Secret in metadata: {pattern}")
   ```

3. Create Jinja2 template in `src/kagglebot/kernel/templates/tabular_script.py.j2`

**Acceptance Criteria**:
- ✅ Generated metadata has correct format (JSON schema)
- ✅ `validate_kernel_package()` catches "kaggle.json" in code
- ✅ `validate_kernel_package()` catches "KAGGLE_KEY" in code
- ✅ Template renders without syntax errors
- ✅ Tests pass: `uv run pytest tests/test_kernel_packager.py -v`

---

### **Phase C4-C5: Runner Implementation** (Week 4-5)

**Goal**: Implement LocalRunner and KaggleNotebookRunner

**Follow**: ../compute/tasks.md tasks C021-C105

**Key checks**:
- LocalRunner validates GPU before training
- KaggleNotebookRunner checks rules before kernel push
- KaggleNotebookRunner validates kernel package before push

**Acceptance Criteria**:
- ✅ LocalRunner works on CPU
- ✅ LocalRunner detects GPU (mocked in tests)
- ✅ KaggleNotebookRunner generates kernel (mocked push)
- ✅ Tests pass: `uv run pytest tests/test_local_runner.py tests/test_kaggle_notebook_runner.py -v`

---

### **Phase C6: CLI Integration** (Week 6)

**Goal**: Wire everything together with safety checks

**Tasks**:
1. Update `cli.py` with all flags
2. Call `validate_slug()` on competition argument
3. Call `check_rules_accepted()` before submission
4. Call `is_duplicate()` before submission
5. Add Kaggle CLI version check

**CRITICAL Integration Points**:
```python
# In orchestrator.py Pipeline.execute()

# 1. Validate slug
slug = validate_slug(self.slug)

# 2. Check rules (before submission)
if self.submit:
    if not check_rules_accepted(slug):
        raise RulesNotAcceptedError(
            f"Rules not accepted for {slug}. "
            f"Visit https://www.kaggle.com/competitions/{slug}/rules"
        )

# 3. Check duplicate (before submission)
ledger = SubmissionLedger(self.artifacts_dir)
if self.submit and ledger.is_duplicate(submission_path):
    logger.warning("Duplicate submission detected, skipping")
    return  # Or raise

# 4. Submit
kaggle_submit(slug, submission_path, self.message)

# 5. Record in ledger
ledger.record(submission_path, self.message)
```

**Acceptance Criteria**:
- ✅ `kagglebot train ../etc/passwd` fails with exit code 1 (invalid slug)
- ✅ `kagglebot train titanic --compute kaggle_gpu --force` fails with exit code 2 if rules not accepted
- ✅ Submitting same file twice is blocked by duplicate detection
- ✅ Help text clear: `uv run kagglebot --help`
- ✅ Tests pass: `uv run pytest tests/test_cli.py -v`

---

### **Phase C7: Documentation and Testing** (Week 7)

**Goal**: Production-ready release

**Tasks**:
1. Update README.md (already done)
2. Manual test on Titanic:
   ```bash
   # Accept rules in browser first
   uv run kagglebot train titanic --compute local_cpu --dry-run
   uv run kagglebot submit titanic -f <submission.csv> -m "test" --force
   ```

3. Security audit checklist:
   - [ ] No `shell=True` in subprocess calls
   - [ ] No secrets in logs
   - [ ] No secrets in kernel packages
   - [ ] Rules check enforced
   - [ ] Duplicate detection works
   - [ ] Validation runs before submission

**Acceptance Criteria**:
- ✅ Manual Titanic run succeeds end-to-end
- ✅ Security audit passes
- ✅ Test coverage >80%: `uv run pytest --cov=kagglebot --cov-report=term-missing`
- ✅ Linting passes: `uv run ruff check .`
- ✅ All docs updated

---

## Acceptance Tests (End-to-End)

### Test 1: Invalid Slug Rejected
```bash
uv run kagglebot train "../etc/passwd"
# Expected: Exit code 1, error message about invalid slug
```

### Test 2: Rules Not Accepted
```bash
# Don't accept rules for test-competition
uv run kagglebot train test-competition --compute kaggle_gpu --force
# Expected: Exit code 2, message with rules URL
```

### Test 3: Duplicate Detection
```bash
# Submit same file twice
uv run kagglebot submit titanic -f <submission.csv> -m "v1" --force
uv run kagglebot submit titanic -f <submission.csv> -m "v1" --force
# Expected: Second submission skipped with warning
```

### Test 4: Secret Scanning
```bash
# Create kernel with "kaggle.json" in code
# Try to push
# Expected: Exit code 9, error about secret detected
```

### Test 5: Validation Failure
```bash
# Create submission.csv with wrong columns
# Try to submit
# Expected: Exit code 6, detailed error about column mismatch
```

---

## Code Quality Standards

### Every Function Must Have:
1. Type hints for all parameters and return value
2. Docstring with description, args, returns, raises
3. Input validation (reject invalid inputs early)
4. Error handling (specific exceptions, not bare except)
5. Logging at INFO level (for user-visible operations)

### Example:
```python
def check_rules_accepted(slug: str) -> bool:
    """
    Check if user has accepted competition rules.

    Args:
        slug: Competition slug (e.g., "titanic")

    Returns:
        True if rules accepted, False otherwise

    Raises:
        subprocess.CalledProcessError: If Kaggle CLI fails
        ValueError: If slug is invalid

    Example:
        >>> check_rules_accepted("titanic")
        True
    """
    slug = validate_slug(slug)  # Input validation
    logger.info(f"Checking rules acceptance for {slug}")  # Logging

    # Implementation...
```

---

## Testing Requirements

### Unit Tests:
- ✅ Test all public functions
- ✅ Test all error paths
- ✅ Mock external dependencies (Kaggle API, filesystem)
- ✅ Use pytest fixtures for common setup

### Integration Tests:
- ✅ Test full flow with mocks
- ✅ Test error recovery
- ✅ Test edge cases (empty files, missing files)

### Manual Tests:
- ✅ One real Kaggle submission (Titanic)
- ✅ Verify kernel push (if implementing notebook runner)
- ✅ Verify error messages user-friendly

---

## Forbidden Patterns

**NEVER write code like this**:

```python
# ❌ BAD: Shell injection risk
os.system(f"kaggle competitions download -c {slug}")

# ❌ BAD: Logging credentials
logger.info(f"API key: {api_key}")

# ❌ BAD: Automating rule acceptance
requests.post("https://www.kaggle.com/accept-rules", ...)

# ❌ BAD: Silencing all errors
try:
    dangerous_operation()
except:  # Bare except
    pass

# ❌ BAD: Path traversal
data_dir = Path("data") / user_input  # No validation!
```

**ALWAYS write code like this**:

```python
# ✅ GOOD: Safe subprocess
subprocess.run(["kaggle", "competitions", "download", "-c", slug], check=True)

# ✅ GOOD: Never log credentials
logger.info("Credentials validated successfully")  # No values!

# ✅ GOOD: Manual rule acceptance
if not check_rules_accepted(slug):
    print(f"Please accept rules: https://www.kaggle.com/competitions/{slug}/rules")
    sys.exit(2)

# ✅ GOOD: Specific error handling
try:
    risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise

# ✅ GOOD: Input validation
slug = validate_slug(user_input)  # Raises if invalid
data_dir = Path("data") / slug
```

---

## Debugging Checklist

If tests fail:
1. Check exit codes match ../compute/spec.md
2. Check error messages are actionable
3. Check no secrets in logs (grep for "key", "token", "password")
4. Check subprocess uses list args (not strings)
5. Check paths use Path() (not string concatenation)

If manual tests fail:
1. Check Kaggle CLI version: `kaggle --version`
2. Check credentials: `kaggle competitions list`
3. Check rules accepted: Visit competition rules page
4. Check logs: `cat artifacts/<slug>/kagglebot.log`
5. Check ledger: `cat artifacts/<slug>/submissions/history.jsonl`

---

## Success Criteria

**Definition of Done** (all must be true):
- ✅ All 140 tasks in ../compute/tasks.md complete
- ✅ All acceptance tests pass
- ✅ Manual Titanic test succeeds
- ✅ Security audit passes (no secrets, no shell=True, etc.)
- ✅ Test coverage >80%
- ✅ Linting passes
- ✅ Documentation complete

**Ship Criteria**:
- ✅ All "Definition of Done" items complete
- ✅ Code reviewed by human
- ✅ Manual test on second competition (not Titanic)
- ✅ Beta testing by 2+ users
- ✅ No open P0 bugs

---

## Quick Reference

### Key Files to Implement:
1. `src/kagglebot/exceptions.py` - Exception hierarchy
2. `src/kagglebot/competition.py` - Slug validation
3. `src/kagglebot/history.py` - Submission ledger
4. `src/kagglebot/hashing.py` - SHA256 hashing
5. `src/kagglebot/kernel/packager.py` - Secret scanning
6. `src/kagglebot/compute/planner.py` - GPU detection
7. `src/kagglebot/runners/local.py` - Local execution
8. `src/kagglebot/runners/kaggle_notebook.py` - Kaggle execution

### Exit Codes:
- 0: Success
- 1: General error
- 2: Rules not accepted / Auth failure
- 6: Validation error
- 7: Missing submission
- 8: Download error
- 9: Secret detected
- 10: GPU not available (strict)
- 11: Kernel timeout
- 12: Kernel failed

### Essential Commands:
```bash
# Run tests
uv run pytest -q

# Run specific test
uv run pytest tests/test_history.py -v

# Check coverage
uv run pytest --cov=kagglebot --cov-report=term-missing

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Manual test
uv run kagglebot train titanic --compute local_cpu --dry-run
```

---

## Contact / Help

- **Documentation**: See ../compute/spec.md, ../compute/architecture.md, ../compute/plan.md
- **Safety Rules**: See ../safety/submission_checklist.md
- **Failure Modes**: See ../safety/failure_modes.md
- **Detailed Tasks**: See ../compute/tasks.md

**Remember**: Safety first. When in doubt, fail safely (exit with error) rather than proceeding unsafely.
