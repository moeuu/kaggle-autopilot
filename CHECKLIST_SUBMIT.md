# Submission Safety Checklist

**Purpose**: Pre-submission verification to prevent unsafe or invalid Kaggle submissions

**Audience**: Users, CI/CD pipelines, review processes

---

## Before Every Submission

### 1. Competition Rules Acceptance ✅

**Requirement**: User MUST manually accept rules in browser

**Verification**:
```bash
# Tool checks this automatically before submission
# Exit code 2 if not accepted
```

**Manual steps**:
1. Visit: `https://www.kaggle.com/competitions/<slug>/rules`
2. Read the rules carefully
3. Click "I Understand and Accept"
4. Wait for confirmation page

**Why manual?**: Automating rule acceptance violates Kaggle Terms of Service

**Tool behavior**:
- ✅ Checks rules accepted before submission
- ✅ Prints rules URL if not accepted
- ❌ NEVER automates acceptance
- ✅ Exit code 2 for human action required

---

### 2. Authentication ✅

**Requirement**: Valid Kaggle credentials configured

**Verification**:
```bash
# Test Kaggle CLI works
kaggle competitions list

# Credentials stored at:
# - ~/.kaggle/kaggle.json (username + key)
# - Or: KAGGLE_USERNAME + KAGGLE_KEY env vars
```

**Security requirements**:
- ✅ Keep `kaggle.json` out of git (in .gitignore)
- ✅ Never log credentials
- ✅ Never commit API keys
- ✅ Use environment variables in CI/CD

**Tool behavior**:
- ✅ Checks credentials before submission
- ✅ Clear error message if credentials missing
- ❌ NEVER logs credential values
- ✅ Exit code 2 for auth failure

---

### 3. Submission Validation ✅

**Requirement**: submission.csv MUST match sample_submission.csv format

**Validation checks**:
```python
# Automatic validation before submission:
# 1. Column names match exactly (order matters)
# 2. Row count matches exactly
# 3. ID column alignment (if present)
# 4. Value ranges valid (e.g., probabilities in [0,1])
# 5. No missing values (unless sample has them)
# 6. Data types match
```

**Tool behavior**:
- ✅ Validates before submission
- ✅ Prints detailed error if mismatch
- ❌ NEVER submits invalid file
- ✅ Exit code 6 for format errors

---

### 4. Duplicate Detection ✅

**Requirement**: Don't submit identical file twice

**How it works**:
```bash
# SHA256 hash of submission.csv
# Check against local ledger: artifacts/<slug>/submissions/history.jsonl
```

**Tool behavior**:
- ✅ Computes hash before submission
- ✅ Checks ledger for duplicate hash
- ✅ Warns user if duplicate detected
- ✅ Skips submission (safe default)
- ✅ User can override with --force-duplicate (not implemented yet)

**Ledger format**:
```jsonl
{"hash": "abc123...", "timestamp": "2026-01-02T10:30:00Z", "message": "baseline v1", "file": "submission.csv"}
```

---

### 5. Rate Limiting ✅

**Requirement**: Respect Kaggle submission limits

**Limits** (as of 2025):
- Max 5 submissions per day (most competitions)
- Min 1 hour between submissions (some competitions)
- Check competition-specific limits

**Tool behavior**:
- ✅ Checks local ledger for recent submissions
- ✅ Warns if approaching daily limit
- ⚠️ Does NOT enforce rate limits (user responsibility)
- ✅ Logs all submissions with timestamps

**User responsibility**:
- Check competition rules for specific limits
- Monitor submission count manually
- Use dry-run to test before submitting

---

### 6. Dry-Run Mode ✅

**Requirement**: Test before actual submission

**Usage**:
```bash
# Dry-run (no Kaggle API calls)
uv run kagglebot run titanic --submit --message "test" --dry-run

# Actual submission (requires explicit flags)
uv run kagglebot run titanic --submit --message "baseline v1" --force
```

**Dry-run behavior**:
- ✅ Downloads data (if needed)
- ✅ Trains models
- ✅ Generates submission.csv
- ✅ Validates format
- ❌ Does NOT submit to Kaggle
- ✅ Prints "DRY RUN - would have submitted: <file>"

**Production behavior** (with --force):
- ✅ All dry-run steps
- ✅ Submits to Kaggle
- ✅ Records in ledger
- ✅ Prints submission ID

---

## Kaggle Notebook Submissions

### 7. Kernel Package Validation ✅

**Requirement**: No secrets in kernel code

**Validation checks**:
```python
# Scan kernel package for secret patterns:
# - "kaggle.json"
# - "api_key"
# - "KAGGLE_KEY"
# - "password"
# - "token"
# - "secret"
```

**Tool behavior**:
- ✅ Scans before push
- ✅ Fails if secrets detected
- ❌ NEVER pushes kernel with secrets
- ✅ Exit code 9 for secret detected

**Safe patterns**:
- ✅ Read from `/kaggle/input/<slug>/` (competition data)
- ✅ Write to `/kaggle/working/` (outputs)
- ❌ No API calls in kernel code
- ❌ No credential files

---

### 8. Internet Access Policy ✅

**Requirement**: Minimize internet access in kernels

**Default**: `enable_internet: false` in kernel-metadata.json

**When to enable**:
- Competition explicitly allows external data
- Using pre-trained models from public URLs
- Installing packages not in Kaggle environment

**Usage**:
```bash
# Enable internet (logs security warning)
uv run kagglebot run titanic --compute kaggle_gpu --enable-internet
```

**Tool behavior**:
- ✅ Defaults to internet OFF
- ✅ Logs warning if enabled
- ✅ Checks competition rules (manual user verification)
- ⚠️ User responsible for rule compliance

**Security risks**:
- ⚠️ Kernel could download malicious code
- ⚠️ Could leak data to external servers
- ⚠️ Violates competition rules if external data prohibited

---

### 9. Dataset License Constraints ✅

**Requirement**: Respect Kaggle dataset licenses

**Common licenses**:
- **CC0** (Public Domain): No restrictions
- **CC BY** (Attribution): Cite source
- **CC BY-SA** (Share-Alike): Derivatives under same license
- **Competition-specific**: Check rules

**User responsibility**:
- ✅ Read competition rules for data usage
- ✅ Check external dataset licenses
- ✅ Don't use prohibited data sources
- ✅ Cite sources in submission message

**Tool behavior**:
- ⚠️ Tool does NOT check licenses (user responsibility)
- ✅ Competition rules link provided in error messages
- ✅ Encourages manual review

---

### 10. Kernel Execution Monitoring ✅

**Requirement**: Monitor kernel status and handle failures

**Polling behavior**:
```bash
# Polls Kaggle for kernel status every 30s
# Max runtime: 120 min (GPU), 540 min (CPU)
# Exponential backoff: 10s → 60s
```

**Possible states**:
- `queued`: Waiting to start
- `running`: Executing
- `complete`: Success
- `error`: Failed (check logs)
- `cancelled`: User cancelled

**Tool behavior**:
- ✅ Polls until complete/error/cancelled
- ✅ Enforces timeout (exit code 11)
- ✅ Downloads outputs on success
- ✅ Prints kernel URL for manual review
- ✅ Preserves kernel metadata in artifacts

**Manual review**:
- Visit: `https://www.kaggle.com/code/<kernel_id>`
- Check logs for errors
- Verify outputs downloaded correctly

---

## Pre-Submission Checklist (Human)

Before running `--submit --force`, verify:

- [ ] Competition rules accepted in browser
- [ ] Kaggle credentials configured (`kaggle competitions list` works)
- [ ] Submission message is descriptive (required)
- [ ] Dry-run completed successfully
- [ ] Validation passed (columns, rows, types)
- [ ] No duplicate submission (check ledger)
- [ ] Submission count under daily limit
- [ ] (Kaggle kernels) No secrets in kernel code
- [ ] (Kaggle kernels) Internet access policy correct
- [ ] (Kaggle kernels) Dataset licenses respected

---

## Automated Checks (Tool)

The tool automatically enforces:

1. ✅ Rules acceptance check (exit code 2)
2. ✅ Credentials validation (exit code 2)
3. ✅ Format validation (exit code 6)
4. ✅ Duplicate detection (warning)
5. ✅ Secret scanning (exit code 9)
6. ✅ Kernel timeout enforcement (exit code 11)
7. ✅ Ledger recording (all submissions)

The tool CANNOT enforce (user responsibility):

1. ⚠️ Rate limits (competition-specific)
2. ⚠️ Dataset licenses
3. ⚠️ External data rules
4. ⚠️ Code plagiarism
5. ⚠️ Multi-account abuse

---

## Emergency: Undo Submission

**Kaggle does NOT support submission deletion.**

If you submitted by mistake:
1. Don't panic - submissions are private until selected
2. Submit a corrected version
3. Select the correct submission for scoring
4. Old submissions remain visible in your history

**Prevention**:
- Always use dry-run first
- Double-check submission message
- Review validation output
- Test on toy competition (e.g., Titanic)

---

## Security Audit

Before first use, verify:

- [ ] `.gitignore` includes `kaggle.json`, `data/`, `artifacts/`
- [ ] No credentials in environment variables (in shell history)
- [ ] Pre-commit hooks prevent secret commits (optional)
- [ ] Team members trained on submission checklist
- [ ] CI/CD uses secure credential storage (GitHub Secrets, etc.)

---

## References

- **Kaggle Terms**: https://www.kaggle.com/terms
- **Kaggle API Docs**: https://github.com/Kaggle/kaggle-api
- **Competition Rules**: https://www.kaggle.com/competitions/<slug>/rules
- **SECURITY.md**: Security guidelines for this repo
- **SPEC_COMPUTE.md**: Exit codes and error messages
