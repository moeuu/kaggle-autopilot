# Design: Kaggle Notebook Runner

## Overview

This design adds a **remote execution mode** to kagglebot that leverages Kaggle's free GPU/TPU quotas while maintaining local control over submissions, validation, and audit logging.

**Key Principle**: Computation runs remotely (Kaggle Kernels), but safety guardrails remain local (validation, deduplication, submission control).

---

## User Interface

### New CLI Flags

Add to existing `kagglebot run` command:

```bash
kagglebot run <competition> [OPTIONS]

New Options:
  --runner {local,kaggle_notebook}
      Execution mode (default: local)

  --accelerator {none,gpu,tpu,auto}
      Hardware accelerator (default: auto when runner=kaggle_notebook, else none)

  --enable-internet
      Allow internet access in notebook (default: false)
      SECURITY: Only set if competition explicitly requires external data

  --kaggle-username TEXT
      Kaggle username for kernel ownership (default: auto-detect from ~/.kaggle/)

  --kernel-slug TEXT
      Custom kernel slug (default: auto-generated: kb-<competition>-<run_id>)

  --max-kernel-runtime MINUTES
      Maximum kernel execution time (default: 120, max: 540 for GPU)

Existing flags still work:
  --submit / --no-submit
  --dry-run
  --message TEXT
  --time-budget MINUTES (ignored for kaggle_notebook; use --max-kernel-runtime)
```

### Usage Examples

```bash
# Dry-run: see what would be executed
kagglebot run titanic --runner kaggle_notebook --accelerator gpu --dry-run

# Run on Kaggle GPU, download results, validate locally (no submit)
kagglebot run titanic --runner kaggle_notebook --accelerator gpu

# Full pipeline: run on Kaggle TPU, validate, submit
kagglebot run titanic \
  --runner kaggle_notebook \
  --accelerator tpu \
  --submit \
  --message "TPU baseline v1"

# Use internet (ONLY if competition allows external data)
kagglebot run my-nlp-comp \
  --runner kaggle_notebook \
  --accelerator gpu \
  --enable-internet \
  --submit
```

---

## Architecture

### Runner Interface

Introduce a plugin-style runner abstraction:

```python
# src/kagglebot/runners/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunContext:
    """Input context for runner execution."""
    slug: str
    competition_metadata: CompetitionMetadata
    modeling_strategy: ModelingStrategy
    config: Config
    run_id: str
    artifacts_dir: Path
    data_dir: Path

@dataclass
class RunResult:
    """Output from runner execution."""
    success: bool
    submission_path: Path | None
    artifacts_dir: Path
    runner_metadata: dict  # Runner-specific info (kernel_id, status, etc.)
    summary: dict  # Machine-readable summary (CV scores, timing, etc.)
    error_message: str | None = None

class Runner(ABC):
    """Base runner interface."""

    @abstractmethod
    def validate_preconditions(self, ctx: RunContext) -> None:
        """Check if runner can execute (credentials, quotas, etc.)."""

    @abstractmethod
    def run(self, ctx: RunContext) -> RunResult:
        """Execute training/inference and return results."""

    @abstractmethod
    def cleanup(self, ctx: RunContext) -> None:
        """Clean up temporary resources (optional)."""
```

### Runner Registry

```python
# src/kagglebot/runners/__init__.py

from .base import Runner, RunContext, RunResult
from .local import LocalRunner
from .kaggle_notebook import KaggleNotebookRunner

RUNNER_REGISTRY = {
    "local": LocalRunner,
    "kaggle_notebook": KaggleNotebookRunner,
}

def get_runner(name: str, config: Config) -> Runner:
    """Factory function to get runner instance."""
    if name not in RUNNER_REGISTRY:
        raise ValueError(f"Unknown runner: {name}")
    return RUNNER_REGISTRY[name](config)
```

### Module Structure

```
src/kagglebot/
├── runners/
│   ├── __init__.py          # Runner registry + factory
│   ├── base.py              # Runner interface + data classes
│   ├── local.py             # LocalRunner (existing training engine)
│   └── kaggle_notebook.py   # KaggleNotebookRunner (new)
├── notebook_templates/      # Templates for kernel generation
│   ├── tabular_script.py.j2      # Jinja2 template for tabular script
│   ├── text_script.py.j2         # Template for text (future)
│   └── image_script.py.j2        # Template for image (future)
└── kernel_manager.py        # Kernel lifecycle management
```

---

## KaggleNotebookRunner Design

### Execution Flow

```
1. PREPARATION (Local)
   ├─ Validate user accepted competition rules
   ├─ Detect Kaggle username (from ~/.kaggle/ or --kaggle-username)
   ├─ Generate kernel package directory
   │  ├─ kernel-metadata.json
   │  ├─ main.py (from template + strategy)
   │  └─ plan.json (modeling strategy serialized)
   └─ Dry-run: print metadata + exit

2. PUSH (Local → Kaggle)
   └─ `kaggle kernels push -p <kernel_dir>`

3. POLL (Local)
   ├─ `kaggle kernels status <username>/<kernel_slug>` (loop with backoff)
   ├─ Timeout: max_kernel_runtime minutes
   └─ Success states: "complete", "completeWithErrors" (exit 0)
   └─ Failure states: "error", "cancelled", "timeout"

4. DOWNLOAD (Kaggle → Local)
   └─ `kaggle kernels output <username>/<kernel_slug> -p <dest>`

5. VALIDATE (Local)
   ├─ Locate submission.csv in downloaded outputs
   ├─ Run existing validation.validate_submission()
   ├─ Check format, rows, columns, ID alignment
   └─ Compute hash for deduplication

6. SUBMIT (Local, if --submit)
   ├─ Check rate limits + duplicates (existing logic)
   ├─ `kaggle competitions submit -c <slug> -f <submission.csv> -m <message>`
   └─ Record in ledger with kernel_id

7. CLEANUP (Local)
   └─ Optionally delete kernel (configurable)
```

### Kernel Package Structure

Generated in `artifacts/<slug>/kernels/<run_id>/`:

```
kernel_package/
├── kernel-metadata.json    # Kaggle kernel configuration
├── main.py                 # Generated training script
└── plan.json              # Modeling strategy (optional, for debugging)
```

#### kernel-metadata.json Format

```json
{
  "id": "moritaeiji/kb-titanic-7f8e9d2a",
  "title": "kagglebot: titanic (7f8e9d2a)",
  "code_file": "main.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "kernel_sources": [],
  "model_sources": [],
  "competition_sources": ["titanic"]
}
```

**Critical Rules**:
- `competition_sources`: Array of competition slugs (NO "c/" prefix)
- `enable_gpu` and `enable_tpu`: Never both true
- `enable_internet`: Default false (security)
- `is_private`: Always true (don't expose strategy publicly)
- Use lowercase `true`/`false` (JSON booleans, not Python)

#### main.py Template (Tabular Example)

```python
#!/usr/bin/env python3
"""
Auto-generated by kagglebot
Run ID: {run_id}
Competition: {slug}
Strategy: {strategy_summary}
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import catboost as cb

# Paths
INPUT_DIR = Path("/kaggle/input/{slug}")
OUTPUT_DIR = Path("/kaggle/working")

# Load plan
with open("/kaggle/input/__package__/plan.json") as f:
    plan = json.load(f)

# Load data
train_df = pd.read_csv(INPUT_DIR / "train.csv")
test_df = pd.read_csv(INPUT_DIR / "test.csv")
sample_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv")

# Feature engineering (from plan)
features = plan["features"]
target = plan["target"]

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

# Preprocessing
# ... (generated from strategy)

# Train models with CV
cv_scores = []
predictions = []

for model_name in plan["models"]:
    print(f"Training {model_name}...")
    # ... CV loop ...
    # ... collect OOF predictions ...
    cv_scores.append({{"model": model_name, "score": cv_mean}})

# Generate submission
submission = sample_sub.copy()
submission[target] = final_predictions

# Save outputs
submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump({{
        "cv_scores": cv_scores,
        "run_id": "{run_id}",
        "strategy": plan["strategy_name"]
    }}, f, indent=2)

print("✓ Submission saved to /kaggle/working/submission.csv")
print(f"✓ CV Score: {{cv_scores}}")
```

**Key Points**:
- Reads competition data from `/kaggle/input/<slug>/`
- Writes submission to `/kaggle/working/submission.csv`
- No Kaggle API calls (no secrets needed)
- Logs CV scores to metrics.json for debugging
- Deterministic (controlled by plan.json)

---

## Kernel Lifecycle Management

### KernelManager Class

```python
# src/kagglebot/kernel_manager.py

from dataclasses import dataclass
import subprocess
import time
from pathlib import Path

@dataclass
class KernelStatus:
    """Kernel execution status."""
    kernel_id: str
    status: str  # "queued", "running", "complete", "error", "cancelled"
    last_update: str
    failure_message: str | None = None

class KernelManager:
    """Manage Kaggle kernel lifecycle via CLI."""

    def __init__(self, username: str, config: Config):
        self.username = username
        self.config = config
        self.logger = get_logger(__name__)

    def push_kernel(self, kernel_dir: Path) -> str:
        """
        Push kernel to Kaggle.

        Returns:
            kernel_id (e.g., "moritaeiji/kb-titanic-7f8e9d2a")
        """
        cmd = ["kaggle", "kernels", "push", "-p", str(kernel_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"Kernel push failed: {result.stderr}")

        # Parse kernel ID from output
        kernel_id = self._parse_kernel_id_from_output(result.stdout)
        self.logger.info("kernel_pushed", kernel_id=kernel_id)
        return kernel_id

    def poll_until_complete(
        self,
        kernel_id: str,
        timeout_minutes: int = 120,
        poll_interval_seconds: int = 30,
    ) -> KernelStatus:
        """
        Poll kernel status until complete or timeout.

        Returns:
            Final KernelStatus

        Raises:
            TimeoutError: If kernel doesn't complete in time
            RuntimeError: If kernel fails
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            status = self.get_status(kernel_id)

            if status.status in ["complete", "completeWithErrors"]:
                self.logger.info("kernel_complete", kernel_id=kernel_id, status=status.status)
                return status

            if status.status in ["error", "cancelled"]:
                raise RuntimeError(f"Kernel failed: {status.failure_message}")

            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Kernel timeout after {timeout_minutes} min")

            self.logger.debug(
                "kernel_polling",
                kernel_id=kernel_id,
                status=status.status,
                elapsed_min=int(elapsed / 60),
            )

            time.sleep(poll_interval_seconds)

    def get_status(self, kernel_id: str) -> KernelStatus:
        """Get current kernel status."""
        cmd = ["kaggle", "kernels", "status", kernel_id]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse status from output
        return self._parse_status_output(result.stdout)

    def download_outputs(self, kernel_id: str, dest: Path) -> Path:
        """
        Download kernel outputs.

        Returns:
            Path to downloaded output directory
        """
        cmd = ["kaggle", "kernels", "output", kernel_id, "-p", str(dest)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        self.logger.info("kernel_outputs_downloaded", kernel_id=kernel_id, dest=str(dest))
        return dest

    def delete_kernel(self, kernel_id: str) -> None:
        """Delete kernel (cleanup)."""
        # Note: Kaggle CLI doesn't have a delete command
        # Kernels auto-delete after ~6 months or can be deleted via web UI
        self.logger.warning(
            "kernel_cleanup_manual",
            kernel_id=kernel_id,
            message="Kernel must be deleted manually via Kaggle web UI if desired"
        )
```

---

## Accelerator Selection

### Auto-Detection Heuristic

```python
# src/kagglebot/runners/accelerator.py

def select_accelerator(
    metadata: CompetitionMetadata,
    strategy: ModelingStrategy,
    requested: Literal["auto", "gpu", "tpu", "none"],
) -> Literal["gpu", "tpu", "none"]:
    """
    Select hardware accelerator based on competition type and strategy.

    Logic:
    1. If requested != "auto", return requested
    2. If competition is image/video/text AND strategy uses DL → GPU
    3. If strategy explicitly requests TPU-compatible models → TPU
    4. If tabular with GBDT only → none (CPU fine)
    5. Default for DL → GPU
    """
    if requested != "auto":
        return requested

    # Tabular competitions with traditional ML → CPU is fine
    if metadata.type == "tabular":
        has_deep_learning = any(
            m in ["mlp", "tabnet", "neural_network"]
            for m in strategy.models
        )
        if not has_deep_learning:
            return "none"

    # Image/video → GPU (most common)
    if metadata.type in ["image", "video"]:
        return "gpu"

    # Text → GPU (transformers, etc.)
    if metadata.type == "text":
        return "gpu"

    # TPU only if explicitly using TPU-compatible stack
    if any(m in ["tpu_mlp", "jax_model"] for m in strategy.models):
        return "tpu"

    # Default: GPU for any DL, none for traditional ML
    if has_deep_learning:
        return "gpu"

    return "none"
```

**Logged Decision**:
```json
{
  "accelerator_decision": {
    "requested": "auto",
    "selected": "gpu",
    "reason": "Image competition detected",
    "competition_type": "image",
    "models": ["resnet50", "efficientnet"]
  }
}
```

---

## Guardrails and Safety

### 1. Dry-Run Mode

```python
if dry_run:
    # Print what WOULD happen
    print("[DRY RUN] Would execute:")
    print(f"  1. Generate kernel package: {kernel_dir}")
    print(f"  2. Push kernel: {kernel_id}")
    print(f"  3. Poll status (max {max_runtime} min)")
    print(f"  4. Download outputs")
    print(f"  5. Validate submission")
    if submit:
        print(f"  6. Submit to Kaggle: {slug}")

    # Show metadata preview
    print("\nGenerated kernel-metadata.json:")
    print(json.dumps(metadata, indent=2))

    # Exit without executing
    return
```

### 2. Rules Acceptance Check

```python
def check_rules_accepted(slug: str) -> bool:
    """Verify user already accepted competition rules."""
    try:
        # Try to list competition files (requires rules acceptance)
        api = get_kaggle_api()
        api.competition_list_files(slug)
        return True
    except Exception as e:
        if "403" in str(e) or "must accept" in str(e).lower():
            return False
        raise

if not check_rules_accepted(slug):
    print(f"❌ Competition rules not accepted")
    print(f"\nYou must manually accept rules:")
    print(f"  https://www.kaggle.com/competitions/{slug}/rules")
    print(f"\nAfter accepting, re-run this command.")
    sys.exit(2)  # Exit code 2: rules not accepted
```

### 3. Kernel Runtime Limits

```python
# In config/default.toml
[runners.kaggle_notebook]
max_kernel_runtime_minutes = 120  # Default
max_kernel_runtime_gpu_minutes = 540  # Kaggle limit: 9 hours
max_kernel_runtime_tpu_minutes = 180  # Conservative for TPU
poll_interval_seconds = 30
enable_internet_default = false

# Enforce limits
if accelerator == "gpu":
    max_runtime = min(max_runtime, config.max_kernel_runtime_gpu_minutes)
elif accelerator == "tpu":
    max_runtime = min(max_runtime, config.max_kernel_runtime_tpu_minutes)
```

### 4. Kernel Slug Uniqueness

```python
def generate_kernel_slug(slug: str, run_id: str, overwrite: bool = False) -> str:
    """
    Generate unique kernel slug to avoid collisions.

    Format: kb-<competition>-<short_run_id>
    """
    if overwrite:
        # Use competition slug only (will overwrite existing)
        return f"kb-{slug}"

    # Include run_id for uniqueness
    short_id = run_id[:8]  # First 8 chars of UUID
    return f"kb-{slug}-{short_id}"
```

### 5. Internet Access Warning

```python
if enable_internet:
    logger.warning(
        "internet_enabled",
        message="Kernel has internet access. Ensure competition allows external data."
    )

    # Require explicit confirmation in logs
    if not config.allow_internet_without_confirmation:
        # Log this decision clearly
        logger.warning(
            "internet_confirmation",
            message="Set config.allow_internet_without_confirmation=true to suppress this check"
        )
```

### 6. No Secrets in Kernel

```python
def validate_kernel_package(kernel_dir: Path) -> None:
    """Ensure no secrets are in kernel package."""

    # Check for common secret patterns
    for file in kernel_dir.rglob("*"):
        if file.is_file():
            content = file.read_text()

            # Pattern matching for secrets
            secret_patterns = [
                r"kaggle.*json",
                r"api.*key",
                r"KAGGLE_USERNAME",
                r"KAGGLE_KEY",
                r"password\s*=",
            ]

            for pattern in secret_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    raise ValueError(
                        f"Potential secret detected in {file}: {pattern}\n"
                        "Remove all credentials before kernel push."
                    )
```

---

## Artifacts Layout

### Directory Structure

```
artifacts/<slug>/
├── kernels/
│   └── <run_id>/
│       ├── kernel_package/
│       │   ├── kernel-metadata.json
│       │   ├── main.py
│       │   └── plan.json
│       ├── kernel_outputs/           # Downloaded from Kaggle
│       │   ├── submission.csv
│       │   ├── metrics.json
│       │   └── *.log
│       └── metadata.json             # Local run metadata
│           {
│             "run_id": "...",
│             "runner": "kaggle_notebook",
│             "kernel_id": "moritaeiji/kb-titanic-7f8e9d2a",
│             "accelerator": "gpu",
│             "status": "complete",
│             "kernel_url": "https://www.kaggle.com/code/...",
│             "push_time": "...",
│             "complete_time": "...",
│             "duration_minutes": 45,
│             "cv_score": 0.854
│           }
├── runs/<run_id>/                    # Existing structure
│   └── submission.csv                # Validated submission (from kernel)
└── submissions/ledger.jsonl          # Existing ledger
```

### Ledger Entry Format

```jsonl
{
  "timestamp": "2026-01-01T12:00:00Z",
  "run_id": "7f8e9d2a",
  "runner": "kaggle_notebook",
  "kernel_id": "moritaeiji/kb-titanic-7f8e9d2a",
  "kernel_url": "https://www.kaggle.com/code/moritaeiji/kb-titanic-7f8e9d2a",
  "accelerator": "gpu",
  "submission_hash": "a1b2c3d4",
  "file_path": "artifacts/titanic/runs/7f8e9d2a/submission.csv",
  "message": "GPU baseline v1",
  "kaggle_submission_id": "12345678",
  "kaggle_score": 0.77511,
  "cv_score": 0.854,
  "submitted_at": "2026-01-01T12:45:00Z"
}
```

---

## Error Handling

### Failure Modes

| Failure | Exit Code | Action |
|---------|-----------|--------|
| Rules not accepted | 2 | Print rules URL, exit |
| Kaggle username not found | 3 | Prompt for --kaggle-username |
| Kernel push failed | 4 | Show Kaggle CLI error, check credentials |
| Kernel timeout | 5 | Show kernel URL for debugging |
| Kernel error/cancelled | 6 | Download logs, show error message |
| No submission.csv in output | 7 | Show kernel outputs, check main.py |
| Submission validation failed | 8 | Standard validation error (existing) |
| Rate limit exceeded | 9 | Standard rate limit message (existing) |

### Error Messages

**Kernel Timeout**:
```
❌ Kernel timeout (120 minutes exceeded)

Kernel is still running on Kaggle:
  https://www.kaggle.com/code/moritaeiji/kb-titanic-7f8e9d2a

Options:
  1. Wait for completion and check manually
  2. Increase timeout: --max-kernel-runtime 240
  3. Check kernel logs for issues
```

**No Submission Found**:
```
❌ No submission.csv found in kernel outputs

Downloaded outputs: artifacts/titanic/kernels/7f8e9d2a/kernel_outputs/
Expected: submission.csv

Kernel may have failed. Check logs:
  artifacts/titanic/kernels/7f8e9d2a/kernel_outputs/*.log

Kernel URL:
  https://www.kaggle.com/code/moritaeiji/kb-titanic-7f8e9d2a
```

---

## Configuration

### Config Schema Addition

```toml
# config/default.toml

[runners]
default = "local"  # or "kaggle_notebook"

[runners.kaggle_notebook]
# Runtime limits
max_kernel_runtime_minutes = 120
max_kernel_runtime_gpu_minutes = 540  # Kaggle limit
max_kernel_runtime_tpu_minutes = 180
poll_interval_seconds = 30

# Safety
enable_internet_default = false
allow_internet_without_confirmation = false
kernel_slug_include_run_id = true  # Prevent overwrites

# Cleanup
auto_delete_kernel = false  # Manual deletion only for now

# Auto-accelerator
auto_accelerator_for_tabular = "none"  # CPU fine for GBDT
auto_accelerator_for_image = "gpu"
auto_accelerator_for_text = "gpu"
auto_accelerator_for_timeseries = "none"

# Kernel defaults
kernel_is_private = true
kernel_type = "script"  # or "notebook"
kernel_language = "python"
```

---

## Integration with Existing Architecture

### Orchestrator Changes

```python
# src/kagglebot/orchestrator.py

class Pipeline:
    def __init__(self, slug: str, config: Config, runner_name: str = "local"):
        self.slug = slug
        self.config = config
        self.runner = get_runner(runner_name, config)  # NEW

    def _train_models(self, metadata: CompetitionMetadata) -> ModelArtifacts:
        """Train models using configured runner."""

        # Create run context
        ctx = RunContext(
            slug=self.slug,
            competition_metadata=metadata,
            modeling_strategy=self.strategy,
            config=self.config,
            run_id=self.run_id,
            artifacts_dir=self.paths.run_dir,
            data_dir=self.paths.data_raw,
        )

        # Validate runner preconditions
        self.runner.validate_preconditions(ctx)

        # Execute training
        result = self.runner.run(ctx)

        if not result.success:
            raise RuntimeError(f"Runner failed: {result.error_message}")

        # Return model artifacts (submission path for now)
        return ModelArtifacts(
            submission_path=result.submission_path,
            metadata=result.runner_metadata,
        )
```

### LocalRunner (Refactored)

```python
# src/kagglebot/runners/local.py

class LocalRunner(Runner):
    """Existing training engine wrapped as a runner."""

    def validate_preconditions(self, ctx: RunContext) -> None:
        """Check local resources (memory, disk, etc.)."""
        pass  # Always available

    def run(self, ctx: RunContext) -> RunResult:
        """Run training locally (existing logic)."""

        # Use existing TrainingEngine
        engine = TrainingEngine(ctx.config)
        artifacts = engine.train(ctx.competition_metadata, ctx.modeling_strategy)

        # Generate predictions (existing logic)
        submission_path = generate_predictions(
            artifacts.models,
            ctx.data_dir / "test.csv",
            ctx.competition_metadata,
            ctx.artifacts_dir / "submission.csv",
        )

        return RunResult(
            success=True,
            submission_path=submission_path,
            artifacts_dir=ctx.artifacts_dir,
            runner_metadata={
                "runner": "local",
                "models_trained": len(artifacts.models),
            },
            summary={
                "cv_scores": artifacts.cv_scores,
                "best_model": artifacts.best_model,
            },
        )
```

---

## Testing Strategy

### Unit Tests
- [ ] Runner interface implementations
- [ ] Kernel metadata generation (all accelerator combinations)
- [ ] Kernel slug generation (uniqueness)
- [ ] Accelerator selection heuristic
- [ ] Template rendering (tabular, text, image)
- [ ] Status parsing from Kaggle CLI output

### Integration Tests
- [ ] End-to-end with mock Kaggle CLI (no actual push)
- [ ] Dry-run mode (no side effects)
- [ ] Rules acceptance check
- [ ] Secret detection in kernel package

### Manual Tests (Requires Kaggle Account)
- [ ] Push kernel to Kaggle (private)
- [ ] Poll until complete
- [ ] Download outputs
- [ ] Validate submission
- [ ] Submit from local
- [ ] Verify ledger entry

---

## Security Checklist

Before merging:
- [ ] No API keys in kernel code
- [ ] No secrets in kernel package
- [ ] enable_internet defaults to false
- [ ] Dry-run mode doesn't push/execute
- [ ] Rules acceptance required (no automation)
- [ ] Kernel slug includes run_id (no overwrites)
- [ ] Timeout enforced (no infinite loops)
- [ ] All CLI calls use subprocess with list args (no shell injection)
- [ ] Ledger records kernel_id for audit
- [ ] Local validation before any submission

---

## Limitations and Future Work

### Current Limitations
- **No notebook format**: Only script kernels (simpler for MVP)
- **No multi-kernel orchestration**: Single kernel per run
- **No incremental training**: Each run is independent
- **No kernel reuse**: Always creates new kernel
- **Manual kernel deletion**: No auto-cleanup via CLI

### Future Enhancements
- **Notebook templates**: Support .ipynb format
- **Kernel caching**: Reuse kernel if code unchanged
- **Multi-stage kernels**: Separate preprocessing and training
- **GPU quota management**: Track and warn on quota limits
- **Kernel logs streaming**: Real-time progress updates
- **Web UI integration**: Monitor kernels via dashboard

---

## Success Criteria

After implementation:
- [ ] `kagglebot run titanic --runner kaggle_notebook --accelerator gpu` works end-to-end
- [ ] Kernel runs on Kaggle GPU
- [ ] Submission downloaded and validated locally
- [ ] No secrets in pushed kernel
- [ ] Dry-run shows plan without executing
- [ ] Rules acceptance required (no automation)
- [ ] Ledger records kernel_id
- [ ] All safety guardrails enforced
- [ ] Clear error messages for all failure modes
- [ ] Documentation complete

---

## Migration Path

For existing users:
1. **Default behavior unchanged**: `--runner local` is default
2. **Opt-in**: Must explicitly use `--runner kaggle_notebook`
3. **Backward compatible**: All existing flags still work
4. **Config override**: Can set `runners.default = "kaggle_notebook"` in config
5. **Gradual rollout**: Start with tabular, add other types later
