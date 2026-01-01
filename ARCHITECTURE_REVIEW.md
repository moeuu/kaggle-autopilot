# Architecture Review: kagglebot

**Date**: 2026-01-02
**Reviewer**: Claude Code (Architect)
**Status**: Pre-Implementation Review

---

## Executive Summary

**Overall Assessment**: Architecture is well-designed with strong safety guarantees, but has **9 critical gaps** that must be addressed before production use.

**Strengths**:
- ✅ Non-interactive design (all decisions via flags)
- ✅ Strong safety guardrails (dry-run, dedup, validation)
- ✅ Clear module boundaries (compute/, runners/, kernel/)
- ✅ Comprehensive documentation (SPEC, ARCHITECTURE, PLAN)

**Critical Gaps**:
1. ❌ Rules acceptance check not implemented
2. ❌ Submission deduplication not implemented
3. ❌ Kernel secret scanning not implemented
4. ❌ Rate limiting not implemented
5. ❌ Path traversal protection not implemented
6. ❌ Disk space checks not implemented
7. ❌ Training timeout enforcement not implemented
8. ❌ Kaggle CLI version check not implemented
9. ❌ Error recovery mechanisms not designed

**Recommendation**: **Address critical gaps before Phase C6 (CLI Integration)**

---

## 1. Safety Guardrails (Critical)

### 1.1 Rules Acceptance Check ❌

**Status**: Documented but NOT implemented

**Risk**: HIGH
- User submits without accepting rules
- Violates Kaggle Terms of Service
- Potential account suspension

**Implementation gap**:
```python
# MISSING: src/kagglebot/kaggle_cli.py
def check_rules_accepted(slug: str) -> bool:
    """
    Check if user has accepted competition rules.

    Uses Kaggle API to query user's competition status.
    Returns True if rules accepted, False otherwise.
    """
    # TODO: Implement via Kaggle API
    #   api.competition_list_cli()
    #   Check if slug in user's competitions
    pass
```

**Where needed**:
- `orchestrator.py` before submission
- `KaggleNotebookRunner.validate_preconditions()`

**Priority**: P0 (MUST fix before first release)

---

### 1.2 Duplicate Submission Detection ❌

**Status**: Designed but NOT implemented

**Risk**: MEDIUM
- Wastes submission quota
- Pollutes submission history
- User frustration

**Implementation gap**:
```python
# MISSING: src/kagglebot/history.py
class SubmissionLedger:
    def is_duplicate(self, submission_path: Path) -> bool:
        """
        Check if submission.csv hash already submitted.

        Computes SHA256 hash and checks against history.jsonl.
        Returns True if hash exists, False otherwise.
        """
        # TODO: Implement
        pass

    def record(self, submission_path: Path, message: str) -> None:
        """
        Record submission in history.jsonl.

        Stores: hash, timestamp, message, file path.
        """
        # TODO: Implement
        pass
```

**Where needed**:
- `orchestrator.py` before submission

**Priority**: P0 (MUST fix before first release)

---

### 1.3 Kernel Secret Scanning ❌

**Status**: Designed but NOT implemented

**Risk**: HIGH
- Credentials leaked in public kernel
- Security incident
- Account compromise

**Implementation gap**:
```python
# MISSING: src/kagglebot/kernel/packager.py
def validate_kernel_package(package_dir: Path) -> None:
    """
    Scan kernel package for secret patterns.

    Checks:
    - "kaggle.json"
    - "api_key", "KAGGLE_KEY"
    - "password", "token", "secret"

    Raises ValueError if secrets detected.
    """
    # TODO: Implement
    # Read main.py and kernel-metadata.json
    # Regex scan for secret patterns
    # Fail if found
    pass
```

**Where needed**:
- `KernelPackager.generate_package()` before returning

**Priority**: P0 (MUST fix before Phase C3)

---

### 1.4 Rate Limiting ❌

**Status**: Not designed or implemented

**Risk**: LOW (user responsibility)
- Exceeds Kaggle limits
- Account warnings

**Design decision**: Log warnings, don't enforce

**Recommended implementation**:
```python
# OPTIONAL: src/kagglebot/history.py
class SubmissionLedger:
    def check_rate_limit(self) -> tuple[bool, str]:
        """
        Check if user is approaching rate limits.

        Returns:
            (is_safe, warning_message)
        """
        # Count submissions in last 24 hours
        # Warn if > 4 (leave 1 for safety)
        # Check time since last submission
        pass
```

**Priority**: P2 (Nice to have)

---

## 2. Input Validation (Critical)

### 2.1 Path Traversal Protection ❌

**Status**: Not implemented

**Risk**: HIGH
- Malicious slug escapes repo directory
- Writes to arbitrary locations
- Security vulnerability

**Implementation gap**:
```python
# MISSING: src/kagglebot/competition.py
def validate_slug(slug: str) -> str:
    """
    Validate competition slug format.

    Allowed: lowercase letters, numbers, hyphens
    Rejects: .., /, \, and other path separators

    Raises ValueError if invalid.
    """
    if not re.match(r"^[a-z0-9-]+$", slug):
        raise ValueError(f"Invalid competition slug: {slug}")
    return slug
```

**Where needed**:
- `cli.py` when parsing competition argument
- All path construction

**Priority**: P0 (MUST fix before Phase C6)

---

### 2.2 Kernel Metadata Validation ❌

**Status**: Partially designed, not implemented

**Risk**: MEDIUM
- Invalid metadata rejected by Kaggle
- Kernel push fails

**Implementation gap**:
```python
# MISSING: src/kagglebot/kernel/metadata.py
def validate_metadata(metadata: dict) -> None:
    """
    Validate kernel-metadata.json against schema.

    Checks:
    - Required fields present
    - enable_gpu and enable_tpu not both true
    - competition_sources format (no "c/" prefix)
    - JSON types correct (lowercase booleans)
    """
    # TODO: Implement validation
    # Check required fields: id, title, code_file, language, kernel_type
    # Check accelerator conflict
    # Check competition_sources format
    pass
```

**Where needed**:
- `generate_kernel_metadata()` before returning

**Priority**: P1 (Should fix in Phase C3)

---

## 3. Resource Management

### 3.1 Disk Space Checks ❌

**Status**: Not designed or implemented

**Risk**: MEDIUM
- Download fails mid-way
- Extraction fails
- Partial files corrupt repo

**Recommended implementation**:
```python
# MISSING: src/kagglebot/bootstrap.py
def check_disk_space(path: Path, required_gb: int) -> None:
    """
    Check available disk space before download.

    Raises OSError if insufficient space.
    """
    import shutil
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024**3)
    if available_gb < required_gb * 1.5:  # 1.5x for extraction
        raise OSError(
            f"Insufficient disk space: {available_gb:.1f}GB available, "
            f"{required_gb * 1.5:.1f}GB required (download + extraction)"
        )
```

**Where needed**:
- Before `kaggle competitions download`

**Priority**: P2 (Nice to have)

---

### 3.2 Training Timeout Enforcement ❌

**Status**: Designed but not implemented

**Risk**: LOW
- Infinite training loops
- Wasted compute

**Implementation gap**:
```python
# MISSING: src/kagglebot/training/tabular_engine.py
class TabularTrainingEngine:
    def __init__(self, ..., max_training_minutes: int = 60):
        self.max_training_minutes = max_training_minutes

    def train(self) -> dict:
        """Train with timeout enforcement."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Training exceeded {self.max_training_minutes} minutes")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.max_training_minutes * 60)

        try:
            # Training logic
            pass
        finally:
            signal.alarm(0)  # Cancel timeout
```

**Where needed**:
- All training engines

**Priority**: P2 (Nice to have)

---

## 4. Error Handling

### 4.1 Kaggle CLI Version Check ❌

**Status**: Not designed or implemented

**Risk**: MEDIUM
- Untested Kaggle CLI version
- Parsing breaks on format changes
- Silent failures

**Recommended implementation**:
```python
# MISSING: src/kagglebot/kaggle_cli.py
TESTED_KAGGLE_VERSIONS = ["1.6.0", "1.6.1", "1.6.12", "1.6.14"]

def check_kaggle_cli_version() -> None:
    """
    Warn if Kaggle CLI version is untested.
    """
    result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
    version = result.stdout.strip()

    # Extract version number
    match = re.search(r"(\d+\.\d+\.\d+)", version)
    if not match:
        logger.warning(f"Could not parse Kaggle CLI version: {version}")
        return

    version_num = match.group(1)
    if version_num not in TESTED_KAGGLE_VERSIONS:
        logger.warning(
            f"Untested Kaggle CLI version: {version_num}. "
            f"Tested versions: {TESTED_KAGGLE_VERSIONS}. "
            "Tool may not work correctly."
        )
```

**Where needed**:
- CLI startup (once per invocation)

**Priority**: P1 (Should fix in Phase C6)

---

### 4.2 Error Recovery Mechanisms ❌

**Status**: Not designed

**Risk**: LOW
- Failed operations leave corrupt state
- User must manually clean up

**Recommended design**:
```python
# MISSING: Error recovery patterns

# 1. Atomic file writes
def write_submission(path: Path, content: str) -> None:
    """Write atomically (tmp file + rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.rename(path)  # Atomic on POSIX

# 2. Transaction log
class TransactionLog:
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def record(self, action: str, status: str, metadata: dict):
        """Record action in append-only log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "status": status,  # "started" | "completed" | "failed"
            "metadata": metadata,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_incomplete_actions(self) -> list[dict]:
        """Find actions that started but didn't complete."""
        # Parse log
        # Find "started" without matching "completed"
        pass

# 3. Cleanup on exit
import atexit

def cleanup_temp_files():
    """Clean up temporary files on exit."""
    # Remove *.tmp files
    # Remove partial downloads
    pass

atexit.register(cleanup_temp_files)
```

**Priority**: P3 (Future enhancement)

---

## 5. Module Design Review

### 5.1 compute/ Package ✅

**Status**: Well-designed

**Strengths**:
- Clear separation of concerns
- GPU detection abstracted
- Fallback logic clean

**Minor issues**:
- Mixing detection and planning in same module
- Could split into `compute/detector.py` and `compute/planner.py`

**Recommendation**: Keep current design (good enough)

---

### 5.2 runners/ Package ✅

**Status**: Well-designed

**Strengths**:
- Runner ABC is clean
- LocalRunner and KaggleNotebookRunner follow same interface
- Easy to add new runners (e.g., AWSRunner, ColabRunner)

**Minor issues**:
- RunContext has many fields (could be split)
- RunResult.summary is untyped dict (could be dataclass)

**Recommendation**: Keep current design

---

### 5.3 kernel/ Package ⚠️

**Status**: Mostly well-designed, one gap

**Strengths**:
- Packager generates valid kernels
- Manager handles lifecycle cleanly
- Templates are maintainable

**Critical gap**: Secret scanning not implemented (see 1.3)

**Recommendation**: Add secret scanning in Phase C3

---

### 5.4 Missing Modules ❌

**What's missing**:

1. **src/kagglebot/competition.py**: Competition metadata handling
   - Parse competition URL
   - Validate slug
   - Fetch competition info (rules status, deadline, etc.)

2. **src/kagglebot/exceptions.py**: Centralized exception hierarchy
   - All custom exceptions in one place
   - Exit code mapping
   - Error message templates

3. **src/kagglebot/config.py**: Configuration management
   - Load config from TOML
   - Environment variable overrides
   - Config validation

**Priority**: P1 (Should create in Phase C1-C2)

---

## 6. Data Flow Review

### 6.1 Artifact Layout ✅

**Status**: Well-designed

```
artifacts/<slug>/
  runs/<run_id>/
    plan.json         # Modeling strategy
    summary.json      # CV scores, model info
    submission.csv    # Final submission
    kernel/           # Kernel package (if Kaggle)
  submissions/
    history.jsonl     # Submission ledger
  reports/
    analysis.json     # Competition analysis
```

**Strengths**:
- Clear separation by run
- Easy to find artifacts
- JSONL for append-only history

**Recommendation**: Keep current design

---

### 6.2 Data Download Flow ⚠️

**Current design**:
```
1. kaggle competitions download -c <slug>
2. Extract ZIP to data/<slug>/extracted/
3. Delete ZIP (optional)
```

**Issues**:
- No disk space check before download
- No resume for partial downloads
- No verification of extracted files

**Recommendation**:
- Add disk space check (P2)
- Keep ZIP for now (simplicity)
- Add extraction verification (P2)

---

### 6.3 Submission Flow ✅

**Current design**:
```
1. Train → submission.csv
2. Validate format
3. Check deduplication
4. Submit to Kaggle
5. Record in ledger
```

**Strengths**:
- Clear linear flow
- Validation before submission
- Audit trail

**Recommendation**: Keep current design

---

## 7. Testing Strategy Review

### 7.1 Unit Test Coverage ⚠️

**Current plan**: >80% coverage

**Gaps**:
- No tests for error paths (404, timeout, etc.)
- No tests for edge cases (empty files, huge files)
- No tests for concurrent access (ledger writes)

**Recommendation**:
- Add error path tests in each phase
- Add edge case tests in Phase C7
- Document test coverage requirements

---

### 7.2 Integration Test Strategy ✅

**Current plan**: Mock all Kaggle API calls

**Strengths**:
- Fast tests
- No network dependency
- Reproducible

**Gaps**:
- Need at least one real Kaggle test (manual)
- Need end-to-end test with actual competition (Titanic)

**Recommendation**:
- Manual test checklist in Phase C7
- Document manual test procedure

---

## 8. Security Review

### 8.1 Credential Handling ✅

**Current design**: kaggle.json or env vars

**Strengths**:
- Follows Kaggle's official approach
- No custom credential storage
- .gitignore prevents commits

**Recommendation**: Keep current design

---

### 8.2 Subprocess Safety ⚠️

**Current design**: Use `subprocess.run()` with list args

**Issue**: No explicit ban on `shell=True`

**Recommendation**:
```python
# Add to coding standards
# NEVER use shell=True (command injection risk)

# ❌ BAD
subprocess.run(f"kaggle competitions download -c {slug}", shell=True)

# ✅ GOOD
subprocess.run(["kaggle", "competitions", "download", "-c", slug])
```

**Priority**: P1 (Document in Phase C6)

---

### 8.3 Kernel Security ⚠️

**Current design**: No secrets in kernels

**Gaps**:
- Secret scanning not implemented (see 1.3)
- No automated review of generated code
- No sandboxing of kernel execution

**Recommendation**:
- Implement secret scanning (P0)
- Manual review of templates (P1)
- Sandboxing not feasible (Kaggle's responsibility)

---

## 9. Missing Features

### 9.1 Configuration System ❌

**Status**: Documented but not implemented

**What's needed**:
```python
# src/kagglebot/config.py
from dataclasses import dataclass
import tomllib

@dataclass
class Config:
    compute: str = "local_cpu"
    max_kernel_runtime: int = 120
    enable_internet: bool = False
    # ... other config

def load_config(config_path: Path) -> Config:
    """Load config from TOML file."""
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    return Config(**data.get("kagglebot", {}))
```

**Priority**: P2 (Nice to have)

---

### 9.2 Logging System ⚠️

**Status**: Ad-hoc logging, not structured

**What's needed**:
```python
# src/kagglebot/logging_setup.py
import logging
import sys

def setup_logging(verbose: bool = False):
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler("artifacts/kagglebot.log"),
        ]
    )
```

**Priority**: P1 (Should add in Phase C1)

---

### 9.3 Progress Indicators ❌

**Status**: Not designed

**What's needed**:
- Rich progress bars for downloads
- Spinner for kernel polling
- Status messages during training

**Example**:
```python
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("[cyan]Downloading...", total=100)
    for i in range(100):
        progress.update(task, advance=1)
```

**Priority**: P3 (Nice to have)

---

## 10. Recommendations

### Critical (MUST Fix Before Release)

1. **Implement rules acceptance check** (Phase C2)
   - Add `check_rules_accepted()` to kaggle_cli.py
   - Call in orchestrator before submission
   - Exit code 2 if not accepted

2. **Implement duplicate detection** (Phase C2)
   - Add `SubmissionLedger.is_duplicate()`
   - Add `SubmissionLedger.record()`
   - Use in orchestrator

3. **Implement secret scanning** (Phase C3)
   - Add `validate_kernel_package()` to packager
   - Scan for credential patterns
   - Fail if detected

4. **Add path traversal protection** (Phase C1)
   - Add `validate_slug()` to competition.py
   - Call in CLI parsing
   - Use for all path construction

5. **Create exception hierarchy** (Phase C1)
   - Centralize in src/kagglebot/exceptions.py
   - Map exit codes to exceptions
   - Use throughout codebase

### High Priority (Should Fix)

6. **Add Kaggle CLI version check** (Phase C6)
   - Warn on untested versions
   - Log version in debug output

7. **Add kernel metadata validation** (Phase C3)
   - Validate schema before push
   - Check required fields
   - Check accelerator conflicts

8. **Add structured logging** (Phase C1)
   - Configure logging at startup
   - Log to file and stderr
   - Support --verbose flag

### Nice to Have

9. **Add disk space checks** (Phase C2)
   - Check before download
   - Estimate extraction size
   - Warn if low

10. **Add configuration system** (Phase C2)
    - Load from config/default.toml
    - Support environment overrides
    - Validate config schema

---

## 11. Risk Assessment

### High Risk (P0 - Address Immediately)

1. ❌ Rules acceptance not checked → **Kaggle ToS violation**
2. ❌ Secret scanning not implemented → **Credential leak**
3. ❌ Path traversal not protected → **Security vulnerability**

### Medium Risk (P1 - Address Before Release)

4. ⚠️ Duplicate detection not implemented → **Wasted quotas**
5. ⚠️ Metadata validation incomplete → **Kernel push failures**
6. ⚠️ Error messages not actionable → **Poor UX**

### Low Risk (P2 - Nice to Have)

7. ⚠️ Disk space not checked → **Failed downloads**
8. ⚠️ Rate limiting not enforced → **Account warnings**
9. ⚠️ No progress indicators → **Poor UX**

---

## 12. Conclusion

**Overall**: Architecture is **production-ready** with critical gaps addressed.

**Action Plan**:

**Phase C1** (Week 1):
- ✅ Create exceptions.py
- ✅ Add path validation
- ✅ Setup logging

**Phase C2** (Week 2):
- ✅ Implement rules check
- ✅ Implement duplicate detection
- ✅ Add disk space checks

**Phase C3** (Week 3):
- ✅ Implement secret scanning
- ✅ Add metadata validation

**Phase C6** (Week 6):
- ✅ Add CLI version check
- ✅ Improve error messages

**Sign-off Criteria**:
- All P0 items resolved
- All P1 items resolved or documented as "won't fix"
- Manual test on Titanic successful
- Security audit passed

**Estimated Additional Effort**: +2 weeks to roadmap (critical gaps)

---

## Appendix: Quick Reference

### Exit Codes
- 0: Success
- 1: General error
- 2: Rules not accepted / Auth failure
- 6: Validation error
- 7: Missing submission
- 8: Download/network error
- 9: Secret detected
- 10: GPU not available (strict mode)
- 11: Kernel timeout
- 12: Kernel execution failed

### Key Files
- `CHECKLIST_SUBMIT.md`: Pre-submission checklist
- `FAILURE_MODES.md`: Known failure modes and mitigations
- `SPEC_COMPUTE.md`: CLI specification
- `ARCHITECTURE_COMPUTE.md`: Module design
- `PLAN_COMPUTE.md`: Implementation roadmap
- `TASKS_COMPUTE.md`: Task breakdown
