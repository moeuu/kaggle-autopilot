# Security Guidelines

## Critical Rules

### NEVER Commit
- ❌ `~/.kaggle/kaggle.json` or any Kaggle credentials
- ❌ API keys, passwords, secrets of any kind
- ❌ Large datasets (CSV, images, etc.)
- ❌ Model artifacts (`.pkl`, `.joblib`, `.model` files)
- ❌ User-specific paths or credentials

### ALWAYS Use
- ✅ `.gitignore` for sensitive directories (`.kaggle/`, `data/`, `artifacts/`)
- ✅ Environment variables for optional secrets
- ✅ Kaggle CLI authentication via `~/.kaggle/kaggle.json` or env vars
- ✅ Input validation (URLs, paths, slugs)
- ✅ Safe defaults (no external data, no pretrained models unless verified)

## Credential Handling

### Kaggle Authentication

**Correct (kaggle.json or env)**:
```bash
# Uses ~/.kaggle/kaggle.json
kaggle competitions list

# Or use env vars
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
kaggle competitions list
```

**What NOT to do**:
- ❌ Don't check in `kaggle.json` (keep it in `~/.kaggle/`)
- ❌ Don't hardcode API credentials
- ❌ Don't automate rules acceptance
- ❌ Don't bypass authentication

### Token Storage

Credentials live in:
- `~/.kaggle/kaggle.json` (legacy API key file)
- `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars
- **NEVER** in the repository
- **NEVER** in config files
- **NEVER** in code

## Input Validation

### Competition URLs
```python
def parse_competition_url(url: str) -> str:
    """
    Extract slug from URL and validate.

    Allowed:
      - titanic
      - https://www.kaggle.com/competitions/titanic
      - https://www.kaggle.com/c/titanic

    Returns:
      Sanitized slug (alphanumeric + hyphens only)

    Raises:
      ValueError: Invalid URL or slug
    """
    # Extract slug
    if url.startswith("http"):
        match = re.match(r"https://www\.kaggle\.com/(c|competitions)/([a-z0-9-]+)", url)
        if not match:
            raise ValueError(f"Invalid Kaggle URL: {url}")
        slug = match.group(2)
    else:
        slug = url

    # Validate slug (prevent path traversal)
    if not re.match(r"^[a-z0-9-]+$", slug):
        raise ValueError(f"Invalid competition slug: {slug}")

    return slug
```

### File Paths
```python
def safe_path(base: Path, user_input: Path) -> Path:
    """
    Resolve path and ensure it's within base directory.

    Prevents:
      - Path traversal (../)
      - Symlink attacks
      - Absolute paths outside base

    Raises:
      ValueError: Path outside allowed directory
    """
    resolved = (base / user_input).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise ValueError(f"Path escapes base directory: {user_input}")
    return resolved
```

## Safe Defaults

### External Data
```python
# Default: Assume external data is NOT allowed
allows_external_data: bool = False  # Conservative default

# Only set to True if explicitly mentioned in rules
if "external data" in rules_text and "allowed" in rules_text:
    allows_external_data = True
```

### Pretrained Models
```python
# Default: Assume pretrained models are NOT allowed
allows_pretrained: bool = False

# Only enable if rules explicitly permit
if "pretrained" in rules_text and "allowed" in rules_text:
    allows_pretrained = True
```

## Resource Limits

### Prevent Resource Exhaustion

```python
# In config/default.toml
[resources]
max_memory_gb = 16  # Fail if training exceeds this
max_training_time_minutes = 240  # Hard timeout
max_cv_folds = 10  # Prevent excessive CV
max_dataset_rows = 10_000_000  # Warn on huge datasets
```

### Implementation
```python
def train_with_limits(model, X, y, config: Config):
    """Train model with resource limits."""
    import resource
    import signal

    # Memory limit (soft)
    max_bytes = config.max_memory_gb * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

    # Time limit
    def timeout_handler(signum, frame):
        raise TimeoutError("Training exceeded time limit")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(config.max_training_time_minutes * 60)

    try:
        model.fit(X, y)
    finally:
        signal.alarm(0)  # Cancel alarm
```

## Submission Safety

### Duplicate Detection
```python
def compute_submission_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of submission file content.

    Note: Hash only the CSV content, not metadata (timestamps, etc.)
    """
    import hashlib

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
```

### Rate Limiting
```python
def check_rate_limit(ledger: SubmissionLedger, config: Config) -> tuple[bool, str]:
    """
    Check if submission is allowed based on rate limits.

    Returns:
      (allowed: bool, reason: str)
    """
    recent = ledger.get_recent_submissions(hours=24)

    # Check daily limit
    if len(recent) >= config.max_submissions_per_day:
        return False, f"Daily limit reached ({config.max_submissions_per_day})"

    # Check time since last submission
    if recent:
        last = recent[-1]
        elapsed = (datetime.now() - last.timestamp).total_seconds() / 3600
        if elapsed < config.min_hours_between_submissions:
            wait_minutes = int((config.min_hours_between_submissions - elapsed) * 60)
            return False, f"Wait {wait_minutes} minutes before next submission"

    return True, ""
```

## Code Execution Safety

### Sandbox Training (Future)

For production use, consider sandboxing model training:

```python
# Run training in subprocess with resource limits
import multiprocessing as mp

def train_in_sandbox(model_class, X, y, config):
    """Train model in separate process with timeout."""
    def worker(queue):
        try:
            model = model_class(config)
            model.fit(X, y)
            queue.put(("success", model))
        except Exception as e:
            queue.put(("error", str(e)))

    queue = mp.Queue()
    process = mp.Process(target=worker, args=(queue,))
    process.start()
    process.join(timeout=config.max_training_time_minutes * 60)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Training timeout")

    status, result = queue.get()
    if status == "error":
        raise RuntimeError(f"Training failed: {result}")

    return result
```

## Logging and Audit Trail

### What to Log
- ✅ All API calls (timestamp, endpoint, status)
- ✅ All file operations (create, read, write paths)
- ✅ All submissions (hash, message, response)
- ✅ All errors and exceptions
- ✅ Config snapshots for each run

### What NOT to Log
- ❌ API tokens or credentials
- ❌ Full dataset contents
- ❌ Sensitive user data
- ❌ File content (just paths and hashes)

### Log Sanitization
```python
def sanitize_for_logging(data: dict) -> dict:
    """Remove sensitive fields from log data."""
    sensitive_keys = ["token", "password", "key", "secret", "credential"]
    return {
        k: "***REDACTED***" if any(s in k.lower() for s in sensitive_keys) else v
        for k, v in data.items()
    }
```

## Dependency Security

### Pin Versions
```toml
# pyproject.toml
dependencies = [
    "kaggle>=1.8.0,<2.0",  # Pin major version
    "pandas>=2.0.0,<3.0",
    "scikit-learn>=1.3.0,<2.0",
]
```

### Regular Updates
- Review dependencies monthly
- Check for security advisories
- Update `uv.lock` regularly

### Avoid Untrusted Sources
- Only install from PyPI
- Verify package names (typosquatting)
- Check package popularity and maintainers

## Kaggle Terms of Service Compliance

### Required
- ✅ User manually accepts rules (once per competition)
- ✅ Respect submission limits (5/day typical)
- ✅ No automation of rule acceptance
- ✅ No scraping if API exists
- ✅ Proper attribution if using others' code

### Forbidden
- ❌ Multi-accounting
- ❌ Submission spamming
- ❌ Bypassing rate limits
- ❌ Reverse engineering Kaggle infrastructure
- ❌ Automated rule acceptance

## Security Checklist

Before every commit:
- [ ] No secrets in diff (`git diff | grep -i "token\|password\|key"`)
- [ ] No large files (`git diff --stat` - all files < 100KB)
- [ ] `.gitignore` covers all sensitive dirs
- [ ] All inputs validated
- [ ] No hardcoded paths or credentials
- [ ] Resource limits in place
- [ ] Logging sanitized

Before every release:
- [ ] Security review of all user inputs
- [ ] Dependency audit (`uv pip check`)
- [ ] Review all subprocess calls
- [ ] Test rate limiting
- [ ] Test duplicate detection
- [ ] Review error messages (no info leakage)
- [ ] Documentation updated

## Incident Response

If credentials are accidentally committed:

1. **Immediately revoke** the token/key on Kaggle
2. **Rewrite git history** to remove the commit:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch <file>" \
     --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push** to remote (if already pushed)
4. **Rotate all credentials**
5. **Document the incident**

## Security Contact

For security issues, contact:
- GitHub Issues (for non-sensitive bugs)
- Email (for sensitive vulnerabilities): [create security contact]
- Do NOT publicly disclose security vulnerabilities before patching
