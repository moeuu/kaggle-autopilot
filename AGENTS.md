# Agent Instructions (Codex / coding agents)

## Role
You are the implementer/tester for this repo.
- Prefer small, testable changes with clear diffs.
- If asked for architecture or design review, ask for guidance or route to Claude Code (see CLAUDE.md).

## Repository purpose
Build a Kaggle competition automation CLI:
- download data via Kaggle CLI
- train a robust baseline (MVP: tabular CSV)
- generate a valid submission.csv (must match sample_submission.csv)
- optionally submit via Kaggle CLI with strong guardrails

## Hard constraints (must follow)
- Do NOT automate accepting rules / joining competitions in the browser.
- Do NOT scrape Kaggle pages.
- Do NOT bypass limits, spam submissions, or encourage multi-account behavior.
- Do NOT write or commit secrets:
  - API credentials, tokens
- Do NOT commit large datasets or artifacts.

## Operational safety defaults
- Default to NO DRY RUN for end-to-end command; use `--dry-run` for previews.
- Submissions require an explicit flag (e.g., `--submit`) AND a human-readable message.
- Require `--force` to allow side effects beyond local validation/ledgers.
- Implement duplicate submission detection (hash + local history).
- Implement strict submission validation:
  - identical columns to sample_submission.csv
  - matching row count
  - align by id column when present

## Development workflow
1) Before large changes, produce a short plan and list touched files.
2) Keep changes minimal and well-tested.
3) Run unit tests (`uv run pytest -q`) and linters (`uv run ruff check .`, `uv run ruff format .`) before concluding.
4) Update docs (README/CLAUDE.md) if behavior changes.

## Tooling
- Use `uv` for dependency management and command execution only (no pip/poetry).
- Use `uv add/remove` for dependencies.
- Keep `uv.lock` committed.
- Standard test run: `uv sync` then `uv run pytest -q`.

## Coding standards
- Python 3.11+ recommended
- Use Kaggle CLI via subprocess; authenticate via `~/.kaggle/kaggle.json` or env vars
- Clear exceptions + actionable error messages
- Deterministic runs (seed control) when feasible

## Notes on Kaggle CLI integration
- Use Kaggle CLI: `kaggle competitions download -c <slug>`, `kaggle competitions submit -c <slug> ...`
- If a CLI call fails due to missing rule acceptance, print the Rules URL and exit.

## What "done" looks like for MVP
- `kagglebot autopilot <slug> --agent codex --compute local_cpu` downloads, trains, and produces a valid submission.csv in artifacts/
- `kagglebot autopilot <slug> --agent codex --compute local_cpu --submit` submits once (with guardrails)
- Works on a common tabular competition (e.g., Titanic-like structure)

---

## Kaggle Notebook Runner (Optional Phase N)

**See docs/notebook_runner/design.md and docs/notebook_runner/tasks.md**

### Implementation Guidelines

When implementing notebook runner:

1. **Runner Interface First**:
   - Implement base Runner abstraction before notebook-specific code
   - Refactor existing training into LocalRunner
   - Test runner abstraction thoroughly
   - Then add KaggleNotebookRunner

2. **Security-First Development**:
   - Validate kernel package BEFORE push (secret detection)
   - Never bypass rules acceptance check
   - Test dry-run mode extensively (no side effects)
   - Log all security decisions (internet, accelerator, etc.)

3. **Template Best Practices**:
   - Use Jinja2 for kernel script generation
   - Keep templates simple and maintainable
   - Test template rendering with various strategies
   - Include error handling in generated code

4. **Kernel Lifecycle**:
   - Mock Kaggle CLI calls in unit tests (no actual push)
   - Use subprocess with list args (NEVER shell=True)
   - Implement robust polling with backoff
   - Handle all Kaggle CLI error states

5. **Testing Strategy**:
   - Unit tests: Mock all Kaggle CLI calls
   - Integration tests: Mock subprocess, test full flow
   - Manual tests: One real kernel push to verify (use private kernel)
   - Verify backward compatibility: LocalRunner still works

### Critical Implementation Rules

**MUST DO**:
- ✅ Check rules accepted before push
- ✅ Validate kernel package for secrets
- ✅ Default enable_internet to false
- ✅ Include run_id in kernel slug (uniqueness)
- ✅ Enforce timeout on kernel polling
- ✅ Submit from local machine (not kernel)
- ✅ Record kernel_id in ledger
- ✅ Use competition slug only in competition_sources (no "c/" prefix)

**MUST NOT DO**:
- ❌ Embed API keys in kernel code
- ❌ Automate rules acceptance
- ❌ Use shell=True in subprocess
- ❌ Push kernel in dry-run mode
- ❌ Set enable_internet without explicit flag
- ❌ Overwrite existing kernels without warning

### Testing Checklist

Before committing notebook runner code:
- [ ] All unit tests pass (mocked CLI)
- [ ] Integration tests pass (no actual push)
- [ ] Dry-run shows correct metadata preview
- [ ] Secret detection catches test secrets
- [ ] Rules check works (tested with mock)
- [ ] Kernel slug includes run_id
- [ ] Timeout enforced in polling
- [ ] LocalRunner still works (no regressions)
- [ ] Code coverage > 80%
- [ ] Documentation updated

### Manual Verification (Once)

After implementation, verify manually:
- [ ] Push private kernel to Kaggle
- [ ] Verify no secrets in pushed code
- [ ] Kernel runs successfully
- [ ] Outputs downloaded correctly
- [ ] Submission validated locally
- [ ] Submission submitted from local (not kernel)
- [ ] Ledger records kernel_id
- [ ] Kernel can be found at generated URL

### Common Pitfalls

Avoid these mistakes:

1. **Competition sources format**:
   - ❌ `"competition_sources": ["c/titanic"]`
   - ✅ `"competition_sources": ["titanic"]`

2. **JSON booleans**:
   - ❌ `"enable_gpu": True` (Python)
   - ✅ `"enable_gpu": true` (JSON)

3. **Accelerator conflicts**:
   - ❌ `enable_gpu: true, enable_tpu: true` (both)
   - ✅ Only one accelerator enabled

4. **Secrets in templates**:
   - ❌ Hardcode credentials
   - ✅ No API calls in kernel code

5. **Kernel paths**:
   - ❌ Assume local paths work in kernel
   - ✅ Use `/kaggle/input/` and `/kaggle/working/`
