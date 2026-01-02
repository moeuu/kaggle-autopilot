# Failure Modes and Ambiguous Items

**Purpose**: Catalog known failure modes, edge cases, and ambiguous behavior in kagglebot

**Audience**: Implementers, debuggers, architects

---

## 1. Kaggle Authentication Failures

### Failure Mode: Credentials Not Found

**Symptom**:
```
Error: Kaggle credentials not found
Exit code: 2
```

**Causes**:
- `~/.kaggle/kaggle.json` missing
- `KAGGLE_USERNAME` / `KAGGLE_KEY` not set
- Incorrect file permissions on kaggle.json (not readable)

**Remediation**:
```bash
# Create kaggle.json
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<EOF
{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json

# Or use environment variables
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Test
kaggle competitions list
```

**Tool behavior**:
- Print clear error message with remediation steps
- Exit code 2 (user action required)
- Never log credential values

---

### Failure Mode: Invalid Credentials

**Symptom**:
```
401 Unauthorized
Exit code: 2
```

**Causes**:
- API key expired or revoked
- Incorrect username/key
- Kaggle account suspended

**Remediation**:
1. Visit: https://www.kaggle.com/<username>/account
2. Click "Create New API Token"
3. Replace kaggle.json with new credentials
4. Retry

**Tool behavior**:
- Distinguish between "not found" and "invalid"
- Print API token regeneration URL
- Exit code 2

---

### Failure Mode: Network Errors

**Symptom**:
```
Connection timeout / Network unreachable
Exit code: 8
```

**Causes**:
- No internet connection
- Kaggle API down
- Firewall blocking requests
- VPN issues

**Remediation**:
```bash
# Test connectivity
curl -I https://www.kaggle.com

# Check Kaggle status
# https://status.kaggle.com
```

**Tool behavior**:
- Retry with exponential backoff (max 3 retries)
- Print connectivity test instructions
- Exit code 8 (network error)

---

## 2. Kaggle Kernel Failures

### Failure Mode: Kernel Push Rejected

**Symptom**:
```
Error pushing kernel: Invalid metadata
Exit code: 12
```

**Causes**:
- Invalid kernel-metadata.json format
- Both `enable_gpu` and `enable_tpu` set to true
- Invalid competition slug in `competition_sources`
- Missing required fields (id, language, code_file)

**Remediation**:
1. Check kernel-metadata.json format
2. Validate with JSON schema
3. Ensure only one accelerator enabled
4. Use competition slug without "c/" prefix

**Tool behavior**:
- Validate metadata before push
- Print validation errors with line numbers
- Suggest corrections
- Exit code 12 (kernel failed)

**Prevention**:
- Use `generate_kernel_metadata()` function
- Add unit tests for metadata generation
- Validate against Kaggle's JSON schema

---

### Failure Mode: Kernel Timeout

**Symptom**:
```
Kernel timed out after 120 minutes
Exit code: 11
```

**Causes**:
- Model training too slow
- Dataset too large
- Infinite loop in kernel code
- Exceeded Kaggle runtime limit

**Remediation**:
```bash
# Increase timeout
uv run kagglebot run <slug> --compute kaggle_gpu --max-kernel-runtime 180

# Or optimize kernel code:
# - Reduce dataset size
# - Simplify model
# - Use GPU-optimized libraries
```

**Ambiguity**: What counts as "timeout"?
- Tool timeout (default: 120 min)
- Kaggle hard limit (GPU: 120 min, CPU: 540 min, TPU: 180 min)
- Competition-specific limit

**Resolution**:
- Tool timeout should be <= Kaggle limit
- Warn if user sets timeout > Kaggle limit
- Poll Kaggle for actual status (may complete after tool timeout)

---

### Failure Mode: Kernel Execution Error

**Symptom**:
```
Kernel failed with error state
Exit code: 12
```

**Causes**:
- Python exception in kernel code
- Out of memory (OOM)
- Missing dependencies
- Invalid file paths
- Dataset not found

**Remediation**:
1. Visit kernel URL: `https://www.kaggle.com/code/<kernel_id>`
2. Check logs for error message
3. Fix kernel code
4. Regenerate and push

**Common errors**:
```python
# Wrong path (kernel environment)
# ❌ train = pd.read_csv("data/train.csv")
# ✅ train = pd.read_csv("/kaggle/input/<slug>/train.csv")

# Missing package
# ❌ import obscure_package  # Not in Kaggle environment
# ✅ Use Kaggle's pre-installed packages

# Out of memory
# ❌ X = huge_dataset.values  # Loads entire dataset
# ✅ Use chunked processing or Dask
```

**Tool behavior**:
- Print kernel URL for manual review
- Download kernel logs (if available)
- Preserve kernel metadata in artifacts
- Exit code 12

---

### Failure Mode: Missing Submission in Kernel Outputs

**Symptom**:
```
No submission.csv found in kernel outputs
Exit code: 7
```

**Causes**:
- Kernel code didn't generate submission.csv
- Saved to wrong path (not /kaggle/working/)
- Kernel failed before generating output
- Typo in filename

**Remediation**:
1. Check kernel logs for errors
2. Verify kernel saves to `/kaggle/working/submission.csv`
3. Check for typos in filename
4. Ensure kernel completes successfully

**Tool behavior**:
- List files in kernel outputs
- Print expected path: `/kaggle/working/submission.csv`
- Exit code 7 (missing submission)

**Prevention**:
- Use template with correct path
- Add assertion in kernel code:
  ```python
  assert (Path("/kaggle/working/submission.csv").exists()), "submission.csv not created"
  ```

---

## 3. Dataset and File Issues

### Failure Mode: sample_submission.csv Missing

**Symptom**:
```
Error: sample_submission.csv not found in competition data
Exit code: 8
```

**Causes**:
- Competition doesn't provide sample_submission.csv
- Data download incomplete
- File in unexpected location

**Remediation**:
```bash
# Manual check
ls data/<slug>/extracted/

# Re-download data
uv run kagglebot run <slug> --force-download
```

**Ambiguity**: How to validate without sample_submission.csv?
- Some competitions only provide description (e.g., "submit CSV with columns A, B")
- No programmatic validation possible

**Resolution**:
- **MVP**: Require sample_submission.csv (fail if missing)
- **Future**: Parse competition description for format (ambitious)
- **Workaround**: User creates sample_submission.csv manually

**Tool behavior**:
- Check for sample_submission.csv in data
- Exit code 8 if missing
- Print message: "This competition requires manual submission format creation"

---

### Failure Mode: Schema Mismatch

**Symptom**:
```
Validation error: Columns don't match
Expected: ['id', 'target']
Got: ['id', 'prediction']
Exit code: 6
```

**Causes**:
- Incorrect column names in submission
- Column order wrong
- Missing columns
- Extra columns

**Remediation**:
- Check sample_submission.csv for expected format
- Ensure column names match exactly (case-sensitive)
- Ensure column order matches

**Tool behavior**:
- Print detailed diff (expected vs actual)
- Exit code 6 (validation error)

**Edge case**: Column order matters?
- **Observation**: Most competitions accept any order (as long as columns match)
- **But**: Some competitions may be strict
- **Resolution**: Match order exactly to be safe

---

### Failure Mode: Row Count Mismatch

**Symptom**:
```
Validation error: Row count mismatch
Expected: 28000
Got: 27999
Exit code: 6
```

**Causes**:
- Missing rows in submission
- Duplicate ID rows
- Test data filtering error

**Remediation**:
- Check test.csv row count
- Ensure all test IDs present in submission
- Check for filtering logic errors

**Tool behavior**:
- Print expected vs actual row count
- List missing IDs (if ID column present)
- Exit code 6

---

### Failure Mode: Data Type Errors

**Symptom**:
```
Validation warning: Data type mismatch
Column 'target': Expected int, got float
```

**Causes**:
- Regression predictions submitted as classification
- Classification probabilities submitted as int
- Missing value handling

**Ambiguity**: Should tool enforce data types?
- **Observation**: Kaggle accepts various types (coerces on server)
- **But**: Incorrect types may indicate logic error

**Resolution**:
- **Warning** (not error) for type mismatches
- User can override with --ignore-type-warnings
- Log warning in submission ledger

---

## 4. Kaggle CLI Behavior Changes

### Risk: Kaggle CLI Output Format Change

**Scenario**: Kaggle CLI changes output format, breaking parsing

**Example**:
```bash
# Current output
$ kaggle kernels push -p kernel/
Kernel version 1 successfully pushed. Please check progress at https://www.kaggle.com/code/user/kernel

# Hypothetical change
$ kaggle kernels push -p kernel/
{"status": "success", "version": 1, "url": "..."}
```

**Impact**:
- Tool can't parse kernel_id
- Submission fails silently
- Polling breaks

**Mitigation**:
- Use Kaggle Python API where possible (more stable than CLI)
- Parse CLI output defensively (regex fallbacks)
- Version-check Kaggle CLI at startup
- Warn if Kaggle CLI version unknown/untested

**Tool behavior**:
```bash
# At startup
kaggle_version=$(kaggle --version)
if [[ ! "$kaggle_version" =~ ^1\.[6-9] ]]; then
  echo "Warning: Untested Kaggle CLI version: $kaggle_version"
  echo "Tested versions: 1.6.0-1.9.0"
fi
```

---

### Risk: Kernel Metadata Schema Change

**Scenario**: Kaggle changes kernel-metadata.json schema

**Example**:
```json
// Current schema
{
  "id": "user/kernel",
  "enable_gpu": true
}

// Hypothetical change (new required field)
{
  "id": "user/kernel",
  "enable_gpu": true,
  "runtime_version": "2.0"  // NEW
}
```

**Impact**:
- Kernel push fails
- Validation breaks

**Mitigation**:
- Use `kaggle kernels init` to generate metadata (Kaggle's canonical source)
- Template metadata from Kaggle's examples
- Validate before push
- Version metadata schema in code

**Tool behavior**:
```python
# Validate against known schema version
KERNEL_METADATA_SCHEMA_VERSION = "2024-01"
metadata["schema_version"] = KERNEL_METADATA_SCHEMA_VERSION
```

---

## 5. File System and Path Issues

### Failure Mode: Path Traversal in Competition Slug

**Scenario**: Malicious/malformed slug breaks path handling

**Example**:
```bash
# Malicious slug
uv run kagglebot run "../../../etc/passwd"

# Tool creates:
data/../../../etc/passwd/  # Escapes repo
```

**Impact**:
- Writes outside repo
- Potential security vulnerability

**Mitigation**:
- Validate slug format (regex: `^[a-z0-9-]+$`)
- Reject slugs with "..", "/", or other path separators
- Use `pathlib` for path joining (auto-sanitizes)

**Tool behavior**:
```python
def validate_slug(slug: str) -> str:
    if not re.match(r"^[a-z0-9-]+$", slug):
        raise ValueError(f"Invalid competition slug: {slug}")
    return slug
```

---

### Failure Mode: Disk Space Exhaustion

**Scenario**: Large datasets fill disk during download/extraction

**Example**:
```bash
# 50GB dataset + 50GB extraction = 100GB
uv run kagglebot run large-competition
```

**Impact**:
- Download fails mid-way
- Extraction fails
- Partial files corrupt repo

**Mitigation**:
- Check available disk space before download
- Stream extraction (delete ZIP after extracting)
- Configurable data directory (external drive)

**Tool behavior**:
```python
import shutil

def check_disk_space(path: Path, required_gb: int):
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024**3)
    if available_gb < required_gb:
        raise OSError(f"Insufficient disk space: {available_gb:.1f}GB available, {required_gb}GB required")
```

---

## 6. Model Training Failures

### Failure Mode: OOM (Out of Memory)

**Scenario**: Dataset too large for available RAM

**Causes**:
- Large CSV loaded entirely into memory
- Model too complex
- Memory leak in training loop

**Remediation**:
```python
# Chunked loading
for chunk in pd.read_csv("train.csv", chunksize=10000):
    process(chunk)

# Reduce batch size
model.fit(X, y, batch_size=32)  # Instead of 128

# Use memory-efficient libraries
import dask.dataframe as dd
df = dd.read_csv("train.csv")
```

**Tool behavior**:
- Catch MemoryError
- Print memory usage statistics
- Suggest chunked processing
- Exit code 9 (training error)

---

### Failure Mode: Infinite Training Loop

**Scenario**: Model training doesn't converge

**Causes**:
- Missing early stopping
- Learning rate too high/low
- Incorrect loss function

**Mitigation**:
- Enforce max training time (configurable)
- Add early stopping to all models
- Log training progress

**Tool behavior**:
```python
from multiprocessing import Process, Queue

def train_with_timeout(timeout_minutes=60):
    queue = Queue()
    process = Process(target=train_model, args=(queue,))
    process.start()
    process.join(timeout=timeout_minutes * 60)
    if process.is_alive():
        process.terminate()
        raise TimeoutError(f"Training exceeded {timeout_minutes} minutes")
```

---

## 7. Ambiguous Design Decisions

### Ambiguity: When to fallback vs fail?

**Scenario**: GPU not available with `--compute local_gpu`

**Options**:
1. **Fail immediately** (strict mode): Exit code 10
2. **Fallback to CPU** (lenient mode): Continue with CPU
3. **Prompt user** (interactive): Ask user to choose

**Current design**: Fallback by default, fail with `--strict-accelerator`

**Rationale**:
- Most users want "just work" behavior
- Advanced users can use strict mode
- Never prompt (non-interactive principle)

---

### Ambiguity: Local vs Kaggle submission?

**Scenario**: Kernel completes, submission.csv ready

**Options**:
1. **Submit from kernel** (Kaggle notebook): Kernel calls Kaggle API
2. **Download and submit locally** (current design): Tool submits

**Current design**: Always submit locally

**Rationale**:
- **Security**: No credentials in kernel
- **Audit**: Local ledger records all submissions
- **Validation**: Can validate before submit
- **Deduplication**: Local hash check

**Trade-off**: Extra step (download outputs)

---

### Ambiguity: Competition slug from URL?

**Scenario**: User provides full URL or slug

**Examples**:
```bash
# Full URL
uv run kagglebot run https://www.kaggle.com/competitions/titanic

# Slug only
uv run kagglebot run titanic
```

**Current design**: Accept both, extract slug

**Implementation**:
```python
def parse_competition_slug(competition: str) -> str:
    # Extract slug from URL or return as-is
    if "kaggle.com" in competition:
        # Parse: https://www.kaggle.com/competitions/titanic
        match = re.search(r"/competitions/([a-z0-9-]+)", competition)
        if match:
            return match.group(1)
    return competition  # Assume it's already a slug
```

---

### Ambiguity: Kaggle username detection?

**Scenario**: Need username for kernel metadata

**Options**:
1. **Require flag**: `--kaggle-username USER`
2. **Auto-detect**: From `KAGGLE_USERNAME` or `kaggle.json`
3. **Kaggle API**: Call API to get current user

**Current design**: Auto-detect with fallback

**Priority**:
1. `--kaggle-username` flag (explicit)
2. `KAGGLE_USERNAME` env var
3. Parse `~/.kaggle/kaggle.json`
4. Fail if none found (exit code 2)

---

## 8. Recommendations

### Immediate Fixes

1. **Add input validation**:
   - Competition slug format
   - File path sanitization
   - Kernel metadata schema

2. **Improve error messages**:
   - Include remediation steps
   - Link to relevant docs
   - Provide kernel URL for failures

3. **Add defensive parsing**:
   - Kaggle CLI output (regex fallbacks)
   - JSON with schema validation
   - CSV with error recovery

4. **Implement retries**:
   - Network requests (exponential backoff)
   - Kernel polling (with timeout)
   - File downloads (resume partial)

### Future Enhancements

1. **Telemetry** (opt-in):
   - Log failure modes to improve tool
   - Aggregate common errors
   - Version compatibility matrix

2. **Recovery mechanisms**:
   - Resume failed kernel downloads
   - Retry failed submissions
   - Checkpoint training progress

3. **Better validation**:
   - Parse competition description for format
   - Validate before kernel push (not after)
   - Pre-flight checks (disk space, credentials, rules)

---

## Testing Strategy

### Unit Tests

- ✅ Test all error paths
- ✅ Mock Kaggle CLI failures
- ✅ Test input validation
- ✅ Test path sanitization

### Integration Tests

- ✅ Test end-to-end with mocks
- ✅ Test error recovery
- ✅ Test timeout handling

### Manual Tests (Real Kaggle)

- ⚠️ Test kernel push/poll/download (once)
- ⚠️ Test submission (on Titanic)
- ⚠️ Verify error messages accurate

---

## Monitoring

### Metrics to Track

- Submission success rate
- Common error codes (frequency)
- Kernel timeout rate
- Average training time

### Logs to Preserve

- All Kaggle API calls (sanitize credentials)
- Kernel metadata (for debugging)
- Validation errors (detailed)
- Submission history (JSONL)

### Alerts

- Exit code 2: User action required
- Exit code 11: Timeout (may need tuning)
- Exit code 12: Kernel failure (investigate logs)
