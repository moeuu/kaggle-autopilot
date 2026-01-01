# Compute Switching Architecture

**Version**: 1.0
**Status**: Design
**Last Updated**: 2026-01-01

---

## Overview

This document defines the architecture for compute switching in kagglebot, enabling seamless execution across:
- **local_cpu**: Train on local CPU
- **local_gpu**: Train on local GPU (CUDA/MPS auto-detection)
- **kaggle_gpu**: Train on Kaggle GPU kernel
- **kaggle_tpu**: Train on Kaggle TPU kernel

**Key Principles**:
1. **Non-interactive**: All decisions via flags/config, zero prompts
2. **Safety-first**: Submission control always local (dedup, validation, audit)
3. **Transparent**: Clear logging of compute decisions and fallback behavior
4. **Extensible**: Easy to add new compute backends

---

## System Architecture

### Module Boundary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer (cli.py)                       │
│  - Parse --compute flag                                          │
│  - Create ComputePlan                                            │
│  - Pass to Orchestrator                                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Orchestrator (orchestrator.py)                      │
│  - Coordinate pipeline stages                                   │
│  - Select runner via RunnerFactory                               │
│  - Execute training                                              │
│  - Local submission (always)                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           Compute Planner (compute/planner.py)                   │
│  - Parse --compute flag                                          │
│  - Detect local GPU availability                                 │
│  - Apply fallback strategy if strict=false                       │
│  - Return validated ComputePlan                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            Runner Factory (runners/__init__.py)                  │
│  - Get runner by name                                            │
│  - Initialize with accelerator config                            │
└─────────────┬──────────────────────┬────────────────────────────┘
              │                      │
              ▼                      ▼
┌──────────────────────┐  ┌─────────────────────────────────────┐
│   LocalRunner        │  │   KaggleNotebookRunner              │
│   (runners/local.py) │  │   (runners/kaggle_notebook.py)      │
│                      │  │                                     │
│ - CPU training       │  │ - Generate kernel package           │
│ - GPU training       │  │ - Push to Kaggle                    │
│   (CUDA/MPS)         │  │ - Poll until complete               │
│ - Direct model       │  │ - Download outputs                  │
│   invocation         │  │ - Extract submission.csv            │
└──────────────────────┘  └─────────────────────────────────────┘
```

### Module Responsibilities

#### 1. `src/kagglebot/compute/` (NEW package)

**Purpose**: Compute planning and GPU detection logic

**Files**:
- `compute/planner.py`: ComputePlan generation and validation
- `compute/gpu_detector.py`: Local GPU detection (CUDA/MPS)
- `compute/exceptions.py`: Compute-specific exceptions

**Public API**:
```python
# compute/planner.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class ComputePlan:
    """Validated compute execution plan."""
    compute: Literal["local_cpu", "local_gpu", "kaggle_gpu", "kaggle_tpu"]
    runner: Literal["local", "kaggle_notebook"]
    accelerator: Literal["cpu", "gpu", "tpu"]
    strict: bool
    detected_backend: str | None  # "cuda", "mps", None

def create_compute_plan(
    compute: str,
    strict: bool = False,
    config: dict | None = None,
) -> ComputePlan:
    """
    Create and validate compute plan.

    Args:
        compute: One of {local_cpu, local_gpu, kaggle_gpu, kaggle_tpu}
        strict: If True, fail if requested compute unavailable
                If False, fall back (local_gpu -> local_cpu if no GPU)
        config: Optional config dict for Kaggle username, etc.

    Returns:
        ComputePlan with validated runner and accelerator

    Raises:
        GPUNotAvailableError: If strict=True and GPU requested but not found
        InvalidComputeError: If compute value invalid

    Example:
        >>> plan = create_compute_plan("local_gpu", strict=False)
        >>> print(plan.runner)  # "local"
        >>> print(plan.accelerator)  # "gpu" or "cpu" (fallback)
    """
    pass

# compute/gpu_detector.py
from dataclasses import dataclass

@dataclass
class GPUInfo:
    """GPU availability information."""
    available: bool
    backend: str | None  # "cuda", "mps", None
    device_count: int
    device_name: str | None

def detect_local_gpu() -> GPUInfo:
    """
    Detect local GPU availability and backend.

    Returns:
        GPUInfo with detection results

    Example:
        >>> gpu = detect_local_gpu()
        >>> if gpu.available:
        ...     print(f"Found {gpu.device_count} GPU(s) via {gpu.backend}")
    """
    pass

def get_torch_device(accelerator: str) -> str:
    """
    Get PyTorch device string for accelerator.

    Args:
        accelerator: "cpu", "gpu", or "tpu"

    Returns:
        Device string ("cpu", "cuda", "mps")

    Example:
        >>> device = get_torch_device("gpu")
        >>> print(device)  # "cuda" or "mps" depending on platform
    """
    pass

# compute/exceptions.py
class ComputeError(Exception):
    """Base exception for compute errors."""
    pass

class GPUNotAvailableError(ComputeError):
    """GPU requested but not available."""
    exit_code = 10

class InvalidComputeError(ComputeError):
    """Invalid compute value provided."""
    exit_code = 1
```

#### 2. `src/kagglebot/runners/` (UPDATED package)

**Purpose**: Execution backend abstraction

**Files**:
- `runners/base.py`: Runner interface (ABC)
- `runners/local.py`: LocalRunner implementation
- `runners/kaggle_notebook.py`: KaggleNotebookRunner implementation
- `runners/__init__.py`: Factory and registry

**Public API**:
```python
# runners/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass
class RunContext:
    """Context passed to runner.run()."""
    competition_slug: str
    data_dir: Path
    artifacts_dir: Path
    run_id: str
    strategy: "ModelingStrategy"  # from analyzer
    accelerator: Literal["cpu", "gpu", "tpu"]
    config: dict

@dataclass
class RunResult:
    """Result returned by runner.run()."""
    success: bool
    submission_path: Path | None
    summary: dict  # CV scores, model info, etc.
    error: str | None
    kernel_id: str | None = None  # Only for KaggleNotebookRunner

class Runner(ABC):
    """Abstract base class for execution runners."""

    def __init__(self, accelerator: Literal["cpu", "gpu", "tpu"]):
        self.accelerator = accelerator

    @abstractmethod
    def validate_preconditions(self, ctx: RunContext) -> None:
        """
        Validate runner can execute in current environment.

        Raises:
            GPUNotAvailableError: If GPU required but not available
            KernelCredentialsError: If Kaggle credentials missing
            RulesNotAcceptedError: If competition rules not accepted
        """
        pass

    @abstractmethod
    def run(self, ctx: RunContext) -> RunResult:
        """
        Execute training and generate submission.

        Args:
            ctx: RunContext with all necessary information

        Returns:
            RunResult with submission path and metadata

        Raises:
            TrainingError: If training fails
            KernelTimeoutError: If kernel times out (notebook runner)
        """
        pass

    @abstractmethod
    def cleanup(self, ctx: RunContext) -> None:
        """Clean up temporary files and resources."""
        pass

# runners/__init__.py
from typing import Type
from .base import Runner
from .local import LocalRunner
from .kaggle_notebook import KaggleNotebookRunner

_REGISTRY: dict[str, Type[Runner]] = {
    "local": LocalRunner,
    "kaggle_notebook": KaggleNotebookRunner,
}

def get_runner(
    runner_name: str,
    accelerator: Literal["cpu", "gpu", "tpu"],
) -> Runner:
    """
    Get runner instance by name.

    Args:
        runner_name: "local" or "kaggle_notebook"
        accelerator: "cpu", "gpu", or "tpu"

    Returns:
        Initialized Runner instance

    Raises:
        ValueError: If runner_name not recognized

    Example:
        >>> runner = get_runner("local", "gpu")
        >>> runner.validate_preconditions(ctx)
        >>> result = runner.run(ctx)
    """
    if runner_name not in _REGISTRY:
        raise ValueError(f"Unknown runner: {runner_name}")
    return _REGISTRY[runner_name](accelerator)
```

#### 3. `src/kagglebot/runners/local.py` (NEW)

**Purpose**: Local execution (CPU/GPU)

**Implementation**:
```python
# runners/local.py
import logging
from pathlib import Path
from typing import Literal

from .base import Runner, RunContext, RunResult
from ..compute.gpu_detector import detect_local_gpu, GPUInfo
from ..compute.exceptions import GPUNotAvailableError
from ..training.tabular_engine import TabularTrainingEngine

logger = logging.getLogger(__name__)

class LocalRunner(Runner):
    """Runner for local CPU/GPU training."""

    def __init__(self, accelerator: Literal["cpu", "gpu", "tpu"]):
        super().__init__(accelerator)
        self.gpu_info: GPUInfo | None = None

    def validate_preconditions(self, ctx: RunContext) -> None:
        """
        Validate local environment for training.

        Raises:
            GPUNotAvailableError: If GPU requested but not available
        """
        if self.accelerator == "gpu":
            self.gpu_info = detect_local_gpu()
            if not self.gpu_info.available:
                raise GPUNotAvailableError(
                    "GPU requested but not available. "
                    "Use --compute local_cpu or omit --strict to fall back."
                )
            logger.info(
                f"GPU detected: {self.gpu_info.device_name} "
                f"({self.gpu_info.backend}, {self.gpu_info.device_count} device(s))"
            )
        elif self.accelerator == "tpu":
            raise ValueError("TPU not supported in LocalRunner")

    def run(self, ctx: RunContext) -> RunResult:
        """
        Execute training locally and generate submission.

        Returns:
            RunResult with submission path and CV scores
        """
        logger.info(f"Starting local training on {self.accelerator.upper()}")

        # Initialize training engine
        engine = TabularTrainingEngine(
            data_dir=ctx.data_dir,
            artifacts_dir=ctx.artifacts_dir,
            strategy=ctx.strategy,
            accelerator=self.accelerator,
            gpu_backend=self.gpu_info.backend if self.gpu_info else None,
        )

        # Train models
        try:
            cv_results = engine.train()
            logger.info(f"Training complete. CV scores: {cv_results}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return RunResult(
                success=False,
                submission_path=None,
                summary={},
                error=str(e),
            )

        # Generate predictions
        try:
            submission_path = engine.predict()
            logger.info(f"Predictions saved to {submission_path}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return RunResult(
                success=False,
                submission_path=None,
                summary={"cv_results": cv_results},
                error=str(e),
            )

        # Build summary
        summary = {
            "cv_results": cv_results,
            "models": engine.get_model_info(),
            "accelerator": self.accelerator,
            "gpu_backend": self.gpu_info.backend if self.gpu_info else None,
        }

        return RunResult(
            success=True,
            submission_path=submission_path,
            summary=summary,
            error=None,
        )

    def cleanup(self, ctx: RunContext) -> None:
        """Clean up temporary files (if any)."""
        pass  # LocalRunner doesn't create temp files currently
```

#### 4. `src/kagglebot/runners/kaggle_notebook.py` (NEW)

**Purpose**: Kaggle kernel execution (GPU/TPU)

**Dependencies**:
- `src/kagglebot/kernel/packager.py`: Generate kernel package
- `src/kagglebot/kernel/manager.py`: Lifecycle management
- `src/kagglebot/kernel/templates/`: Jinja2 templates

**Implementation**:
```python
# runners/kaggle_notebook.py
import logging
from pathlib import Path
from typing import Literal

from .base import Runner, RunContext, RunResult
from ..kernel.packager import KernelPackager
from ..kernel.manager import KernelManager
from ..kernel.exceptions import (
    KernelTimeoutError,
    KernelFailedError,
    MissingSubmissionError,
)
from ..kaggle_cli import check_rules_accepted

logger = logging.getLogger(__name__)

class KaggleNotebookRunner(Runner):
    """Runner for Kaggle kernel execution."""

    def __init__(self, accelerator: Literal["cpu", "gpu", "tpu"]):
        super().__init__(accelerator)
        self.packager: KernelPackager | None = None
        self.manager: KernelManager | None = None

    def validate_preconditions(self, ctx: RunContext) -> None:
        """
        Validate Kaggle environment for kernel execution.

        Raises:
            KernelCredentialsError: If Kaggle credentials missing
            RulesNotAcceptedError: If competition rules not accepted
        """
        # Check Kaggle credentials
        from kaggle.api.kaggle_api_extended import KaggleApi
        try:
            api = KaggleApi()
            api.authenticate()
            username = api.get_config_value("username")
            if not username:
                raise ValueError("Kaggle username not configured")
        except Exception as e:
            from ..compute.exceptions import ComputeError
            raise ComputeError(
                f"Kaggle credentials not found or invalid: {e}. "
                "Ensure ~/.kaggle/kaggle.json exists or set KAGGLE_USERNAME/KAGGLE_KEY"
            ) from e

        # Check rules accepted
        if not check_rules_accepted(ctx.competition_slug):
            from ..exceptions import RulesNotAcceptedError
            raise RulesNotAcceptedError(
                f"Competition rules not accepted for {ctx.competition_slug}. "
                f"Visit https://www.kaggle.com/competitions/{ctx.competition_slug}/rules "
                "and accept rules manually in your browser."
            )

        # Initialize packager and manager
        self.packager = KernelPackager(username=username)
        self.manager = KernelManager(
            username=username,
            config=ctx.config,
        )

        logger.info(f"Kaggle credentials validated for user: {username}")

    def run(self, ctx: RunContext) -> RunResult:
        """
        Execute training on Kaggle kernel and retrieve submission.

        Returns:
            RunResult with submission path and kernel metadata
        """
        logger.info(
            f"Starting Kaggle kernel execution on {self.accelerator.upper()}"
        )

        # Generate kernel package
        try:
            kernel_dir = self.packager.generate_package(
                slug=ctx.competition_slug,
                run_id=ctx.run_id,
                strategy=ctx.strategy,
                accelerator=self.accelerator,
                enable_internet=ctx.config.get("enable_internet", False),
            )
            logger.info(f"Kernel package generated at {kernel_dir}")
        except Exception as e:
            logger.error(f"Kernel package generation failed: {e}")
            return RunResult(
                success=False,
                submission_path=None,
                summary={},
                error=str(e),
            )

        # Push kernel to Kaggle
        try:
            kernel_id = self.manager.push_kernel(kernel_dir)
            logger.info(f"Kernel pushed: {kernel_id}")
        except Exception as e:
            logger.error(f"Kernel push failed: {e}")
            return RunResult(
                success=False,
                submission_path=None,
                summary={},
                error=str(e),
            )

        # Poll until complete
        try:
            max_runtime = ctx.config.get("max_kernel_runtime", 120)
            status = self.manager.poll_until_complete(
                kernel_id=kernel_id,
                timeout_minutes=max_runtime,
            )
            logger.info(f"Kernel completed with status: {status.state}")
        except KernelTimeoutError as e:
            logger.error(f"Kernel timed out: {e}")
            return RunResult(
                success=False,
                submission_path=None,
                summary={"kernel_id": kernel_id},
                error=str(e),
                kernel_id=kernel_id,
            )
        except KernelFailedError as e:
            logger.error(f"Kernel failed: {e}")
            return RunResult(
                success=False,
                submission_path=None,
                summary={"kernel_id": kernel_id},
                error=str(e),
                kernel_id=kernel_id,
            )

        # Download outputs
        try:
            outputs_dir = ctx.artifacts_dir / ctx.run_id / "kernel_outputs"
            self.manager.download_outputs(kernel_id, outputs_dir)
            logger.info(f"Kernel outputs downloaded to {outputs_dir}")
        except Exception as e:
            logger.error(f"Output download failed: {e}")
            return RunResult(
                success=False,
                submission_path=None,
                summary={"kernel_id": kernel_id},
                error=str(e),
                kernel_id=kernel_id,
            )

        # Locate submission.csv
        try:
            submission_path = self._locate_submission(outputs_dir)
            logger.info(f"Submission found: {submission_path}")
        except MissingSubmissionError as e:
            logger.error(f"No submission.csv in kernel outputs: {e}")
            return RunResult(
                success=False,
                submission_path=None,
                summary={"kernel_id": kernel_id},
                error=str(e),
                kernel_id=kernel_id,
            )

        # Parse metrics (if available)
        summary = {
            "kernel_id": kernel_id,
            "kernel_url": f"https://www.kaggle.com/code/{kernel_id}",
            "accelerator": self.accelerator,
        }
        metrics_path = outputs_dir / "metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                summary.update(json.load(f))

        return RunResult(
            success=True,
            submission_path=submission_path,
            summary=summary,
            error=None,
            kernel_id=kernel_id,
        )

    def cleanup(self, ctx: RunContext) -> None:
        """Save kernel metadata to artifacts."""
        # Kernel package already in artifacts, nothing to clean up
        pass

    def _locate_submission(self, outputs_dir: Path) -> Path:
        """
        Locate submission.csv in kernel outputs.

        Raises:
            MissingSubmissionError: If submission.csv not found
        """
        submission_path = outputs_dir / "submission.csv"
        if not submission_path.exists():
            raise MissingSubmissionError(
                f"No submission.csv found in {outputs_dir}. "
                "Kernel must write submission to /kaggle/working/submission.csv"
            )
        return submission_path
```

#### 5. `src/kagglebot/kernel/` (NEW package)

**Purpose**: Kaggle kernel package generation and lifecycle management

**Files**:
- `kernel/packager.py`: Generate kernel packages
- `kernel/manager.py`: Push, poll, download kernels
- `kernel/metadata.py`: Generate kernel-metadata.json
- `kernel/templates/tabular_script.py.j2`: Jinja2 template for kernel script
- `kernel/exceptions.py`: Kernel-specific exceptions

**Public API**:
```python
# kernel/packager.py
from pathlib import Path
from typing import Literal
from ..analyzer.strategy import ModelingStrategy

class KernelPackager:
    """Generate Kaggle kernel packages."""

    def __init__(self, username: str):
        self.username = username

    def generate_package(
        self,
        slug: str,
        run_id: str,
        strategy: ModelingStrategy,
        accelerator: Literal["cpu", "gpu", "tpu"],
        enable_internet: bool = False,
    ) -> Path:
        """
        Generate kernel package with metadata and script.

        Args:
            slug: Competition slug
            run_id: Unique run identifier
            strategy: ModelingStrategy from analyzer
            accelerator: "cpu", "gpu", or "tpu"
            enable_internet: Whether to enable internet in kernel

        Returns:
            Path to kernel package directory

        Example:
            >>> packager = KernelPackager("myusername")
            >>> pkg_dir = packager.generate_package(
            ...     slug="titanic",
            ...     run_id="run_20260101_120000",
            ...     strategy=strategy,
            ...     accelerator="gpu",
            ... )
            >>> print(pkg_dir)  # artifacts/titanic/run_20260101_120000/kernel/
        """
        pass

# kernel/manager.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class KernelStatus:
    """Kernel execution status."""
    kernel_id: str
    state: str  # "queued", "running", "complete", "error", "cancelled"
    metadata: dict

class KernelManager:
    """Manage Kaggle kernel lifecycle."""

    def __init__(self, username: str, config: dict):
        self.username = username
        self.config = config

    def push_kernel(self, kernel_dir: Path) -> str:
        """
        Push kernel to Kaggle.

        Args:
            kernel_dir: Path to kernel package directory

        Returns:
            kernel_id (e.g., "username/kernel-slug")

        Raises:
            KernelPushError: If push fails
        """
        pass

    def get_status(self, kernel_id: str) -> KernelStatus:
        """
        Get current kernel status.

        Args:
            kernel_id: Kernel identifier

        Returns:
            KernelStatus with current state
        """
        pass

    def poll_until_complete(
        self,
        kernel_id: str,
        timeout_minutes: int = 120,
    ) -> KernelStatus:
        """
        Poll kernel until complete or timeout.

        Args:
            kernel_id: Kernel identifier
            timeout_minutes: Max wait time in minutes

        Returns:
            KernelStatus when complete

        Raises:
            KernelTimeoutError: If timeout exceeded
            KernelFailedError: If kernel execution failed
        """
        pass

    def download_outputs(self, kernel_id: str, dest: Path) -> Path:
        """
        Download kernel outputs to local directory.

        Args:
            kernel_id: Kernel identifier
            dest: Destination directory

        Returns:
            Path to downloaded outputs

        Raises:
            DownloadError: If download fails
        """
        pass

# kernel/metadata.py
from typing import Literal

def generate_kernel_metadata(
    username: str,
    slug: str,
    run_id: str,
    accelerator: Literal["cpu", "gpu", "tpu"],
    enable_internet: bool = False,
) -> dict:
    """
    Generate kernel-metadata.json content.

    Args:
        username: Kaggle username
        slug: Competition slug
        run_id: Unique run identifier
        accelerator: "cpu", "gpu", or "tpu"
        enable_internet: Whether to enable internet in kernel

    Returns:
        Dict suitable for writing to kernel-metadata.json

    Example:
        >>> metadata = generate_kernel_metadata(
        ...     username="myuser",
        ...     slug="titanic",
        ...     run_id="run_20260101_120000",
        ...     accelerator="gpu",
        ... )
        >>> print(metadata["id"])  # "myuser/titanic-run-20260101-120000"
        >>> print(metadata["enable_gpu"])  # true
    """
    pass

# kernel/exceptions.py
class KernelError(Exception):
    """Base exception for kernel errors."""
    pass

class KernelTimeoutError(KernelError):
    """Kernel execution timed out."""
    exit_code = 11

class KernelFailedError(KernelError):
    """Kernel execution failed."""
    exit_code = 12

class MissingSubmissionError(KernelError):
    """No submission.csv found in kernel outputs."""
    exit_code = 7
```

#### 6. `src/kagglebot/training/tabular_engine.py` (UPDATED)

**Purpose**: Tabular model training with GPU support

**Key Changes**:
```python
# training/tabular_engine.py
import logging
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

logger = logging.getLogger(__name__)

class TabularTrainingEngine:
    """Tabular model training with CPU/GPU support."""

    def __init__(
        self,
        data_dir: Path,
        artifacts_dir: Path,
        strategy: "ModelingStrategy",
        accelerator: Literal["cpu", "gpu", "tpu"] = "cpu",
        gpu_backend: str | None = None,  # "cuda" or "mps"
    ):
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.strategy = strategy
        self.accelerator = accelerator
        self.gpu_backend = gpu_backend

    def _get_lightgbm_params(self) -> dict:
        """Get LightGBM parameters with GPU support."""
        params = {
            "objective": self.strategy.objective,
            "metric": self.strategy.metric,
            "verbosity": -1,
            "seed": 42,
        }

        if self.accelerator == "gpu":
            if self.gpu_backend == "cuda":
                # LightGBM GPU support via CUDA
                params["device"] = "gpu"
                params["gpu_platform_id"] = 0
                params["gpu_device_id"] = 0
                logger.info("LightGBM: Using CUDA GPU acceleration")
            elif self.gpu_backend == "mps":
                # LightGBM doesn't support MPS, fall back to CPU
                logger.warning(
                    "LightGBM: MPS not supported, using CPU. "
                    "For GPU training on macOS, use CatBoost or XGBoost."
                )
                params["device"] = "cpu"

        return params

    def _get_xgboost_params(self) -> dict:
        """Get XGBoost parameters with GPU support."""
        params = {
            "objective": self.strategy.objective,
            "eval_metric": self.strategy.metric,
            "seed": 42,
        }

        if self.accelerator == "gpu":
            if self.gpu_backend == "cuda":
                # XGBoost GPU support via CUDA
                params["device"] = "cuda"
                params["tree_method"] = "hist"  # GPU-compatible
                logger.info("XGBoost: Using CUDA GPU acceleration")
            elif self.gpu_backend == "mps":
                # XGBoost doesn't officially support MPS
                logger.warning(
                    "XGBoost: MPS not officially supported, using CPU. "
                    "For GPU training on macOS, use CatBoost."
                )
                params["device"] = "cpu"

        return params

    def _get_catboost_params(self) -> dict:
        """Get CatBoost parameters with GPU support."""
        params = {
            "loss_function": self.strategy.objective,
            "verbose": False,
            "random_seed": 42,
        }

        if self.accelerator == "gpu":
            # CatBoost supports both CUDA and Apple Silicon GPU
            params["task_type"] = "GPU"
            if self.gpu_backend == "cuda":
                params["devices"] = "0"  # GPU device ID
                logger.info("CatBoost: Using CUDA GPU acceleration")
            elif self.gpu_backend == "mps":
                # CatBoost on macOS uses Metal Performance Shaders
                logger.info("CatBoost: Using Metal GPU acceleration (MPS)")
        else:
            params["task_type"] = "CPU"

        return params

    def train(self) -> dict:
        """
        Train models with cross-validation.

        Returns:
            Dict of CV scores per model
        """
        # Implementation details...
        pass

    def predict(self) -> Path:
        """
        Generate predictions and save submission.csv.

        Returns:
            Path to submission.csv
        """
        # Implementation details...
        pass

    def get_model_info(self) -> dict:
        """Get metadata about trained models."""
        # Implementation details...
        pass
```

---

## GPU Detection Strategy

### Detection Logic

```python
# compute/gpu_detector.py (full implementation)
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """GPU availability information."""
    available: bool
    backend: str | None  # "cuda", "mps", None
    device_count: int
    device_name: str | None

def detect_local_gpu() -> GPUInfo:
    """
    Detect local GPU availability and backend.

    Detection order:
    1. Try CUDA (torch.cuda.is_available())
    2. Try MPS (torch.backends.mps.is_available())
    3. Return None if neither available

    Returns:
        GPUInfo with detection results
    """
    # Try CUDA first (most common)
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else None
            logger.info(
                f"CUDA GPU detected: {device_name} ({device_count} device(s))"
            )
            return GPUInfo(
                available=True,
                backend="cuda",
                device_count=device_count,
                device_name=device_name,
            )
    except ImportError:
        logger.warning("PyTorch not installed, cannot detect CUDA")
    except Exception as e:
        logger.warning(f"CUDA detection failed: {e}")

    # Try MPS (macOS Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS GPU detected (Apple Silicon)")
            return GPUInfo(
                available=True,
                backend="mps",
                device_count=1,
                device_name="Apple Silicon GPU",
            )
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"MPS detection failed: {e}")

    # No GPU detected
    logger.info("No GPU detected, will use CPU")
    return GPUInfo(
        available=False,
        backend=None,
        device_count=0,
        device_name=None,
    )

def get_torch_device(accelerator: str, gpu_backend: str | None = None) -> str:
    """
    Get PyTorch device string for accelerator.

    Args:
        accelerator: "cpu", "gpu", or "tpu"
        gpu_backend: "cuda", "mps", or None (auto-detect)

    Returns:
        Device string ("cpu", "cuda", "mps")

    Example:
        >>> device = get_torch_device("gpu", "cuda")
        >>> print(device)  # "cuda"
    """
    if accelerator == "cpu":
        return "cpu"
    elif accelerator == "gpu":
        if gpu_backend == "cuda":
            return "cuda"
        elif gpu_backend == "mps":
            return "mps"
        else:
            # Auto-detect
            gpu_info = detect_local_gpu()
            if gpu_info.available:
                return gpu_info.backend
            else:
                return "cpu"
    elif accelerator == "tpu":
        # TPU support would require torch_xla
        raise NotImplementedError("TPU support not implemented for local runner")
    else:
        raise ValueError(f"Invalid accelerator: {accelerator}")
```

### Fallback Strategy

```python
# compute/planner.py (full implementation)
import logging
from dataclasses import dataclass
from typing import Literal

from .gpu_detector import detect_local_gpu, GPUInfo
from .exceptions import GPUNotAvailableError, InvalidComputeError

logger = logging.getLogger(__name__)

@dataclass
class ComputePlan:
    """Validated compute execution plan."""
    compute: Literal["local_cpu", "local_gpu", "kaggle_gpu", "kaggle_tpu"]
    runner: Literal["local", "kaggle_notebook"]
    accelerator: Literal["cpu", "gpu", "tpu"]
    strict: bool
    detected_backend: str | None  # "cuda", "mps", None

# Mapping from compute to (runner, accelerator)
_COMPUTE_MAPPING = {
    "local_cpu": ("local", "cpu"),
    "local_gpu": ("local", "gpu"),
    "kaggle_gpu": ("kaggle_notebook", "gpu"),
    "kaggle_tpu": ("kaggle_notebook", "tpu"),
}

def create_compute_plan(
    compute: str,
    strict: bool = False,
    config: dict | None = None,
) -> ComputePlan:
    """
    Create and validate compute plan.

    Args:
        compute: One of {local_cpu, local_gpu, kaggle_gpu, kaggle_tpu}
        strict: If True, fail if requested compute unavailable
                If False, fall back (local_gpu -> local_cpu if no GPU)
        config: Optional config dict for Kaggle username, etc.

    Returns:
        ComputePlan with validated runner and accelerator

    Raises:
        GPUNotAvailableError: If strict=True and GPU requested but not found
        InvalidComputeError: If compute value invalid
    """
    # Validate compute value
    if compute not in _COMPUTE_MAPPING:
        raise InvalidComputeError(
            f"Invalid --compute value: {compute}. "
            f"Must be one of: {list(_COMPUTE_MAPPING.keys())}"
        )

    runner, accelerator = _COMPUTE_MAPPING[compute]
    detected_backend = None

    # Special handling for local_gpu: detect GPU and fallback if needed
    if compute == "local_gpu":
        gpu_info: GPUInfo = detect_local_gpu()
        detected_backend = gpu_info.backend

        if not gpu_info.available:
            if strict:
                raise GPUNotAvailableError(
                    "GPU requested (--compute local_gpu) but not available. "
                    "Checked CUDA and MPS backends. "
                    "Options:\n"
                    "  - Use --compute local_cpu to run on CPU\n"
                    "  - Use --compute kaggle_gpu to run on Kaggle GPU\n"
                    "  - Omit --strict to automatically fall back to CPU"
                )
            else:
                # Fallback to CPU
                logger.warning(
                    "GPU requested but not available. Falling back to CPU. "
                    "Use --strict to prevent fallback."
                )
                compute = "local_cpu"
                runner = "local"
                accelerator = "cpu"

    # Log compute plan
    logger.info(f"Compute plan: {compute} → runner={runner}, accelerator={accelerator}")
    if detected_backend:
        logger.info(f"GPU backend: {detected_backend}")

    return ComputePlan(
        compute=compute,
        runner=runner,
        accelerator=accelerator,
        strict=strict,
        detected_backend=detected_backend,
    )
```

---

## Integration Points

### 1. CLI Layer (`src/kagglebot/cli.py`)

**Changes**:
```python
# cli.py
import typer
from pathlib import Path
from typing import Literal

from .compute.planner import create_compute_plan
from .orchestrator import Pipeline

app = typer.Typer()

@app.command()
def run(
    competition: str,
    compute: Literal["local_cpu", "local_gpu", "kaggle_gpu", "kaggle_tpu"] = "local_cpu",
    strict: bool = False,
    submit: bool = False,
    message: str | None = None,
    dry_run: bool = True,
    force: bool = False,
    kaggle_username: str | None = None,
    enable_internet: bool = False,
    max_kernel_runtime: int = 120,
):
    """
    Run end-to-end pipeline: download → analyze → train → predict → validate → submit.

    Args:
        competition: Competition URL or slug
        compute: Execution mode (local_cpu, local_gpu, kaggle_gpu, kaggle_tpu)
        strict: Fail if requested compute unavailable (no fallback)
        submit: Submit to Kaggle after validation
        message: Submission message (required if --submit)
        dry_run: Dry-run mode (no network actions except download)
        force: Required for network actions
        kaggle_username: Kaggle username (for notebook runner)
        enable_internet: Enable internet in Kaggle kernel
        max_kernel_runtime: Max kernel runtime in minutes
    """
    # Parse competition slug
    slug = parse_competition_slug(competition)

    # Create compute plan
    try:
        compute_plan = create_compute_plan(
            compute=compute,
            strict=strict,
            config={
                "kaggle_username": kaggle_username,
                "enable_internet": enable_internet,
                "max_kernel_runtime": max_kernel_runtime,
            },
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(e.exit_code if hasattr(e, "exit_code") else 1)

    # Create pipeline
    pipeline = Pipeline(
        slug=slug,
        compute_plan=compute_plan,
        submit=submit,
        message=message,
        dry_run=dry_run,
        force=force,
    )

    # Execute
    try:
        result = pipeline.execute()
        if result.success:
            typer.echo(f"Success! Submission: {result.submission_path}")
        else:
            typer.echo(f"Failed: {result.error}", err=True)
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Pipeline failed: {e}", err=True)
        raise typer.Exit(1)
```

### 2. Orchestrator Layer (`src/kagglebot/orchestrator.py`)

**Changes**:
```python
# orchestrator.py
import logging
from dataclasses import dataclass
from pathlib import Path

from .compute.planner import ComputePlan
from .runners import get_runner
from .runners.base import RunContext, RunResult
from .analyzer import CompetitionAnalyzer
from .validation import SubmissionValidator
from .history import SubmissionLedger
from .kaggle_cli import kaggle_submit

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Result of full pipeline execution."""
    success: bool
    submission_path: Path | None
    error: str | None
    metadata: dict

class Pipeline:
    """Orchestrate end-to-end competition workflow."""

    def __init__(
        self,
        slug: str,
        compute_plan: ComputePlan,
        submit: bool = False,
        message: str | None = None,
        dry_run: bool = True,
        force: bool = False,
    ):
        self.slug = slug
        self.compute_plan = compute_plan
        self.submit = submit
        self.message = message
        self.dry_run = dry_run
        self.force = force

        # Initialize components
        self.data_dir = Path("data") / slug
        self.artifacts_dir = Path("artifacts") / slug
        self.run_id = self._generate_run_id()

    def execute(self) -> PipelineResult:
        """Execute full pipeline."""
        logger.info(f"Starting pipeline for {self.slug}")
        logger.info(f"Compute: {self.compute_plan.compute}")
        logger.info(f"Runner: {self.compute_plan.runner}")
        logger.info(f"Accelerator: {self.compute_plan.accelerator}")

        # Stage 1: Download data (if needed)
        if not self.data_dir.exists():
            logger.info("Downloading competition data...")
            # Download implementation...

        # Stage 2: Analyze competition
        logger.info("Analyzing competition...")
        analyzer = CompetitionAnalyzer(self.data_dir)
        strategy = analyzer.analyze()

        # Stage 3: Get runner and validate preconditions
        logger.info(f"Initializing {self.compute_plan.runner} runner...")
        runner = get_runner(
            runner_name=self.compute_plan.runner,
            accelerator=self.compute_plan.accelerator,
        )

        ctx = RunContext(
            competition_slug=self.slug,
            data_dir=self.data_dir,
            artifacts_dir=self.artifacts_dir,
            run_id=self.run_id,
            strategy=strategy,
            accelerator=self.compute_plan.accelerator,
            config={
                "enable_internet": False,  # From CLI
                "max_kernel_runtime": 120,  # From CLI
            },
        )

        try:
            runner.validate_preconditions(ctx)
        except Exception as e:
            logger.error(f"Precondition validation failed: {e}")
            return PipelineResult(
                success=False,
                submission_path=None,
                error=str(e),
                metadata={},
            )

        # Stage 4: Run training
        logger.info("Starting training...")
        try:
            result: RunResult = runner.run(ctx)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return PipelineResult(
                success=False,
                submission_path=None,
                error=str(e),
                metadata={},
            )

        if not result.success:
            return PipelineResult(
                success=False,
                submission_path=result.submission_path,
                error=result.error,
                metadata=result.summary,
            )

        # Stage 5: Validate submission
        logger.info("Validating submission...")
        validator = SubmissionValidator(self.data_dir)
        try:
            validator.validate(result.submission_path)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return PipelineResult(
                success=False,
                submission_path=result.submission_path,
                error=str(e),
                metadata=result.summary,
            )

        # Stage 6: Submit (if requested)
        if self.submit and not self.dry_run and self.force:
            logger.info("Submitting to Kaggle...")
            ledger = SubmissionLedger(self.artifacts_dir)
            if ledger.is_duplicate(result.submission_path):
                logger.warning("Duplicate submission detected, skipping")
            else:
                try:
                    kaggle_submit(self.slug, result.submission_path, self.message)
                    ledger.record(result.submission_path, self.message)
                except Exception as e:
                    logger.error(f"Submission failed: {e}")
                    return PipelineResult(
                        success=False,
                        submission_path=result.submission_path,
                        error=str(e),
                        metadata=result.summary,
                    )

        # Stage 7: Cleanup
        runner.cleanup(ctx)

        return PipelineResult(
            success=True,
            submission_path=result.submission_path,
            error=None,
            metadata=result.summary,
        )

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        from datetime import datetime
        return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

---

## Kernel Template System

### Template Structure

```
src/kagglebot/kernel/templates/
├── tabular_script.py.j2       # Main training script
└── kernel_metadata.json.j2    # Kernel metadata
```

### Example Template: `tabular_script.py.j2`

```python
#!/usr/bin/env python3
"""
Kaggle kernel for {{ competition_slug }}
Generated by kagglebot
Run ID: {{ run_id }}
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kaggle paths
INPUT_DIR = Path("/kaggle/input/{{ competition_slug }}")
WORKING_DIR = Path("/kaggle/working")

def load_data():
    """Load training and test data."""
    train = pd.read_csv(INPUT_DIR / "train.csv")
    test = pd.read_csv(INPUT_DIR / "test.csv")
    sample = pd.read_csv(INPUT_DIR / "sample_submission.csv")
    return train, test, sample

def load_plan():
    """Load modeling plan."""
    # Plan is embedded during kernel generation
    plan = {{ plan_json }}
    return plan

def preprocess(df, plan):
    """Apply preprocessing from plan."""
    # Implementation based on plan.preprocessing
    # ...
    return df

def train_model(X, y, plan):
    """Train model with cross-validation."""
    n_folds = plan.get("cv_folds", 5)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        logger.info(f"Training fold {fold + 1}/{n_folds}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=1000,
            task_type="GPU",  # Use GPU
            verbose=False,
            random_seed=42,
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        score = model.score(X_val, y_val)
        cv_scores.append(score)
        models.append(model)

        logger.info(f"Fold {fold + 1} score: {score:.4f}")

    logger.info(f"Mean CV score: {np.mean(cv_scores):.4f}")
    return models, cv_scores

def predict(models, X_test):
    """Generate predictions from ensemble."""
    predictions = np.zeros(len(X_test))
    for model in models:
        predictions += model.predict_proba(X_test)[:, 1]
    predictions /= len(models)
    return (predictions > 0.5).astype(int)

def main():
    logger.info("Starting kernel execution")

    # Load data
    train, test, sample = load_data()
    plan = load_plan()

    # Prepare features
    y = train[plan["target"]]
    X = train[plan["features"]]
    X_test = test[plan["features"]]

    # Preprocess
    X = preprocess(X, plan)
    X_test = preprocess(X_test, plan)

    # Train
    models, cv_scores = train_model(X, y, plan)

    # Predict
    predictions = predict(models, X_test)

    # Generate submission
    submission = sample.copy()
    submission[plan["target"]] = predictions

    # Save outputs
    submission.to_csv(WORKING_DIR / "submission.csv", index=False)
    logger.info(f"Submission saved to {WORKING_DIR / 'submission.csv'}")

    # Save metrics
    metrics = {
        "cv_scores": cv_scores,
        "mean_cv": float(np.mean(cv_scores)),
        "std_cv": float(np.std(cv_scores)),
    }
    with open(WORKING_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to metrics.json")

    logger.info("Kernel execution complete")

if __name__ == "__main__":
    main()
```

### Template Rendering

```python
# kernel/packager.py (template rendering)
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import json

class KernelPackager:
    def __init__(self, username: str):
        self.username = username
        self.template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=False,  # Python code, not HTML
        )

    def generate_package(
        self,
        slug: str,
        run_id: str,
        strategy: "ModelingStrategy",
        accelerator: str,
        enable_internet: bool = False,
    ) -> Path:
        """Generate kernel package with rendered templates."""
        # Create package directory
        pkg_dir = Path("artifacts") / slug / run_id / "kernel"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # Render main script
        script_template = self.jinja_env.get_template("tabular_script.py.j2")
        script_content = script_template.render(
            competition_slug=slug,
            run_id=run_id,
            plan_json=json.dumps(strategy.to_dict()),
            accelerator=accelerator,
        )

        script_path = pkg_dir / "main.py"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Generate metadata
        from .metadata import generate_kernel_metadata
        metadata = generate_kernel_metadata(
            username=self.username,
            slug=slug,
            run_id=run_id,
            accelerator=accelerator,
            enable_internet=enable_internet,
        )

        metadata_path = pkg_dir / "kernel-metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return pkg_dir
```

---

## Error Handling and Exit Codes

### Exit Code Mapping

```python
# exceptions.py (consolidated)
class KaggleBotError(Exception):
    """Base exception for all kagglebot errors."""
    exit_code = 1

class RulesNotAcceptedError(KaggleBotError):
    """Competition rules not accepted."""
    exit_code = 2

class ValidationError(KaggleBotError):
    """Submission validation failed."""
    exit_code = 3

class DuplicateSubmissionError(KaggleBotError):
    """Duplicate submission detected."""
    exit_code = 4

class RateLimitError(KaggleBotError):
    """Kaggle rate limit exceeded."""
    exit_code = 5

class SubmissionFormatError(KaggleBotError):
    """Submission file format invalid."""
    exit_code = 6

class MissingSubmissionError(KaggleBotError):
    """No submission.csv found."""
    exit_code = 7

class DownloadError(KaggleBotError):
    """Data download failed."""
    exit_code = 8

class TrainingError(KaggleBotError):
    """Model training failed."""
    exit_code = 9

class GPUNotAvailableError(KaggleBotError):
    """GPU requested but not available."""
    exit_code = 10

class KernelTimeoutError(KaggleBotError):
    """Kernel execution timed out."""
    exit_code = 11

class KernelFailedError(KaggleBotError):
    """Kernel execution failed."""
    exit_code = 12
```

### Error Message Templates

```python
# Error messages with remediation hints
ERROR_MESSAGES = {
    2: (
        "Competition rules not accepted. "
        "Visit https://www.kaggle.com/competitions/{slug}/rules "
        "and accept rules manually in your browser. "
        "Then retry this command."
    ),
    10: (
        "GPU requested (--compute local_gpu) but not available. "
        "Checked CUDA and MPS backends. "
        "Options:\n"
        "  - Use --compute local_cpu to run on CPU\n"
        "  - Use --compute kaggle_gpu to run on Kaggle GPU\n"
        "  - Omit --strict to automatically fall back to CPU"
    ),
    11: (
        "Kaggle kernel timed out after {timeout} minutes. "
        "Options:\n"
        "  - Increase timeout: --max-kernel-runtime {new_timeout}\n"
        "  - Simplify model or reduce data size\n"
        "  - Check kernel logs: https://www.kaggle.com/code/{kernel_id}"
    ),
    12: (
        "Kaggle kernel execution failed. "
        "Check kernel logs for details: https://www.kaggle.com/code/{kernel_id}\n"
        "Common causes:\n"
        "  - Out of memory (reduce batch size or model complexity)\n"
        "  - Missing dependencies (check kernel requirements)\n"
        "  - Code errors (check kernel output)"
    ),
}
```

---

## Configuration Schema

### Config File: `config/default.toml`

```toml
# Compute defaults
[compute]
default_runner = "local"
default_accelerator = "cpu"
strict_mode = false  # Allow fallback by default

# Local GPU settings
[compute.local_gpu]
auto_detect = true
prefer_cuda = true  # Prefer CUDA over MPS if both available

# Kaggle kernel settings
[compute.kaggle_kernel]
enable_internet = false  # Default: no internet
kernel_is_private = true  # Default: private kernels
max_runtime_cpu = 540    # 9 hours (Kaggle limit)
max_runtime_gpu = 120    # 2 hours (free tier)
max_runtime_tpu = 180    # 3 hours (free tier)
poll_interval = 30       # Poll every 30 seconds

# Model training
[training]
cv_folds = 5
random_seed = 42

[training.lightgbm]
num_iterations = 1000
early_stopping_rounds = 50

[training.catboost]
iterations = 1000
early_stopping_rounds = 50

[training.xgboost]
num_boost_round = 1000
early_stopping_rounds = 50
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_compute_planner.py
import pytest
from kagglebot.compute.planner import create_compute_plan
from kagglebot.compute.exceptions import GPUNotAvailableError, InvalidComputeError

def test_local_cpu_plan():
    plan = create_compute_plan("local_cpu")
    assert plan.runner == "local"
    assert plan.accelerator == "cpu"
    assert plan.compute == "local_cpu"

def test_local_gpu_fallback(mocker):
    # Mock GPU not available
    mocker.patch(
        "kagglebot.compute.gpu_detector.detect_local_gpu",
        return_value=GPUInfo(available=False, backend=None, device_count=0, device_name=None),
    )

    # Should fall back to CPU when strict=False
    plan = create_compute_plan("local_gpu", strict=False)
    assert plan.runner == "local"
    assert plan.accelerator == "cpu"

def test_local_gpu_strict_fails(mocker):
    # Mock GPU not available
    mocker.patch(
        "kagglebot.compute.gpu_detector.detect_local_gpu",
        return_value=GPUInfo(available=False, backend=None, device_count=0, device_name=None),
    )

    # Should raise when strict=True
    with pytest.raises(GPUNotAvailableError):
        create_compute_plan("local_gpu", strict=True)

def test_invalid_compute():
    with pytest.raises(InvalidComputeError):
        create_compute_plan("invalid_compute")

# tests/test_runners.py
def test_local_runner_cpu(tmp_path, mocker):
    from kagglebot.runners.local import LocalRunner
    from kagglebot.runners.base import RunContext

    runner = LocalRunner(accelerator="cpu")
    ctx = RunContext(
        competition_slug="titanic",
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        run_id="test_run",
        strategy=mocker.Mock(),
        accelerator="cpu",
        config={},
    )

    # Should not raise
    runner.validate_preconditions(ctx)

def test_local_runner_gpu_not_available(mocker):
    from kagglebot.runners.local import LocalRunner
    from kagglebot.runners.base import RunContext
    from kagglebot.compute.exceptions import GPUNotAvailableError

    # Mock GPU not available
    mocker.patch(
        "kagglebot.compute.gpu_detector.detect_local_gpu",
        return_value=GPUInfo(available=False, backend=None, device_count=0, device_name=None),
    )

    runner = LocalRunner(accelerator="gpu")
    ctx = mocker.Mock()

    with pytest.raises(GPUNotAvailableError):
        runner.validate_preconditions(ctx)
```

### Integration Tests

```python
# tests/integration/test_compute_switching.py
def test_local_cpu_end_to_end(tmp_path):
    """Test full pipeline with local CPU."""
    # Setup test data
    # ...

    # Run pipeline
    from kagglebot.compute.planner import create_compute_plan
    from kagglebot.orchestrator import Pipeline

    plan = create_compute_plan("local_cpu")
    pipeline = Pipeline(
        slug="titanic",
        compute_plan=plan,
        submit=False,
        dry_run=True,
        force=False,
    )

    result = pipeline.execute()
    assert result.success
    assert result.submission_path.exists()

def test_kaggle_notebook_dry_run(tmp_path, mocker):
    """Test Kaggle notebook runner in dry-run mode."""
    # Mock all Kaggle API calls
    mocker.patch("kagglebot.kernel.manager.KernelManager.push_kernel")
    mocker.patch("kagglebot.kernel.manager.KernelManager.poll_until_complete")
    mocker.patch("kagglebot.kernel.manager.KernelManager.download_outputs")

    # Run pipeline
    plan = create_compute_plan("kaggle_gpu")
    pipeline = Pipeline(
        slug="titanic",
        compute_plan=plan,
        submit=False,
        dry_run=True,
        force=False,
    )

    result = pipeline.execute()
    # In dry-run, should generate kernel package but not push
    kernel_dir = tmp_path / "artifacts" / "titanic" / "kernel"
    assert kernel_dir.exists()
```

---

## Backward Compatibility

### Migration Path

**Existing users** (using old CLI):
```bash
# Old command
kagglebot run titanic

# New equivalent
kagglebot run titanic --compute local_cpu
```

**Default behavior**:
- `--compute` defaults to `local_cpu` (same as before)
- No breaking changes for users not using new flags

**Deprecation strategy**:
- Keep old `--runner` and `--accelerator` flags for 2 releases
- Print deprecation warning if used
- Translate to `--compute` internally

```python
# cli.py (deprecation handling)
def run(
    competition: str,
    compute: str | None = None,
    runner: str | None = None,  # DEPRECATED
    accelerator: str | None = None,  # DEPRECATED
    **kwargs,
):
    # Handle deprecated flags
    if runner or accelerator:
        import warnings
        warnings.warn(
            "--runner and --accelerator are deprecated. "
            "Use --compute instead. "
            "See docs for migration guide.",
            DeprecationWarning,
        )

        # Translate to compute
        if runner == "local" and accelerator == "cpu":
            compute = "local_cpu"
        elif runner == "local" and accelerator == "gpu":
            compute = "local_gpu"
        elif runner == "kaggle_notebook" and accelerator == "gpu":
            compute = "kaggle_gpu"
        elif runner == "kaggle_notebook" and accelerator == "tpu":
            compute = "kaggle_tpu"

    # Use compute flag
    compute = compute or "local_cpu"
    # ... rest of implementation
```

---

## Summary

### Key Components

1. **`compute/` package**: GPU detection, compute planning, fallback logic
2. **`runners/` package**: LocalRunner (CPU/GPU) and KaggleNotebookRunner
3. **`kernel/` package**: Kernel generation, lifecycle management, templates
4. **`training/` updates**: GPU support for LightGBM, XGBoost, CatBoost
5. **CLI integration**: `--compute` flag with 4 modes
6. **Orchestrator**: Runner selection and execution flow

### Execution Flow

```
User runs: kagglebot run titanic --compute local_gpu
                 ↓
CLI parses flags → create_compute_plan("local_gpu")
                 ↓
ComputePlanner detects GPU → ComputePlan(runner="local", accelerator="gpu")
                 ↓
Orchestrator creates LocalRunner(accelerator="gpu")
                 ↓
LocalRunner.validate_preconditions() → checks GPU availability
                 ↓
LocalRunner.run() → TabularTrainingEngine with GPU params
                 ↓
Models trained on GPU (CUDA/MPS)
                 ↓
submission.csv generated → validation → optional submit
```

### Exit Code Reference

| Code | Error | Meaning |
|------|-------|---------|
| 0 | Success | All operations completed |
| 1 | General | Unspecified error |
| 2 | RulesNotAccepted | Must accept rules in browser |
| 10 | GPUNotAvailable | GPU requested but not found (strict mode) |
| 11 | KernelTimeout | Kaggle kernel timed out |
| 12 | KernelFailed | Kaggle kernel execution failed |

---

## Next Steps

See **PLAN_COMPUTE.md** for phased implementation roadmap.
