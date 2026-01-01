# Compute Switching: Implementation Checklist

**For**: Implementer agents (Codex / coding assistants)
**Status**: Ready for implementation
**Total Tasks**: 140
**Estimated Time**: 7 weeks (1 developer, full-time)

---

## How to Use This Checklist

1. **Work in order**: Tasks are ordered by dependency
2. **Complete each task fully**: Write code + tests before moving on
3. **Check off tasks**: Mark with `[x]` when complete
4. **Run tests frequently**: `uv run pytest -q` after each task
5. **Review docs**: Check SPEC_COMPUTE.md, ARCHITECTURE_COMPUTE.md for details

---

## Phase C1: Foundation (Week 1)

**Goal**: Set up compute planning infrastructure and GPU detection

### Package Setup

- [ ] **C001**: Create `src/kagglebot/compute/` package
  - Create directory: `mkdir -p src/kagglebot/compute`
  - Create `__init__.py`: `touch src/kagglebot/compute/__init__.py`
  - Add docstring to `__init__.py` describing package purpose

### Exceptions

- [ ] **C002**: Create `src/kagglebot/compute/exceptions.py`
  - Define `ComputeError(Exception)` base class
  - Define `GPUNotAvailableError(ComputeError)` with exit_code=10
  - Define `InvalidComputeError(ComputeError)` with exit_code=1
  - Add docstrings for each exception

### GPU Detection

- [ ] **C003**: Create `src/kagglebot/compute/gpu_detector.py`
  - Import required modules (torch, logging, dataclasses)
  - Define `GPUInfo` dataclass with fields: available, backend, device_count, device_name
  - Add module-level docstring

- [ ] **C004**: Implement `detect_local_gpu()` function
  - Try CUDA detection: `torch.cuda.is_available()`
  - If CUDA available: get device count and name
  - Return GPUInfo with backend="cuda"
  - Handle ImportError if PyTorch not installed

- [ ] **C005**: Add MPS detection to `detect_local_gpu()`
  - Try MPS detection: `torch.backends.mps.is_available()`
  - If MPS available: return GPUInfo with backend="mps"
  - Handle cases where torch.backends.mps doesn't exist

- [ ] **C006**: Add logging to `detect_local_gpu()`
  - Log CUDA detection results (device name, count)
  - Log MPS detection results
  - Log if no GPU detected
  - Log warnings for import failures

- [ ] **C007**: Implement `get_torch_device()` function
  - Accept accelerator parameter ("cpu", "gpu", "tpu")
  - Return "cpu" if accelerator=="cpu"
  - For GPU: auto-detect backend or use provided gpu_backend
  - Raise NotImplementedError for TPU (local runner)
  - Add docstring with examples

### Compute Planning

- [ ] **C008**: Create `src/kagglebot/compute/planner.py`
  - Import required modules (dataclasses, typing, logging)
  - Define `ComputePlan` dataclass with fields: compute, runner, accelerator, strict, detected_backend
  - Define `_COMPUTE_MAPPING` dict mapping compute → (runner, accelerator)
  - Add module-level docstring

- [ ] **C009**: Implement `create_compute_plan()` function
  - Accept: compute, strict, config parameters
  - Validate compute value against _COMPUTE_MAPPING
  - Raise InvalidComputeError if invalid
  - Extract runner and accelerator from mapping
  - Return ComputePlan for non-GPU cases

- [ ] **C010**: Add GPU fallback logic to `create_compute_plan()`
  - If compute=="local_gpu": call detect_local_gpu()
  - If GPU not available and strict=True: raise GPUNotAvailableError
  - If GPU not available and strict=False: fall back to local_cpu
  - Log all decisions (GPU detected, fallback, etc.)
  - Return ComputePlan with detected_backend

### Unit Tests

- [ ] **C011**: Create `tests/test_gpu_detector.py`
  - Import pytest, unittest.mock, gpu_detector module
  - Add module-level docstring

- [ ] **C012**: Test CUDA detection
  - Mock `torch.cuda.is_available()` to return True
  - Mock `torch.cuda.device_count()` to return 1
  - Mock `torch.cuda.get_device_name()` to return "NVIDIA GPU"
  - Call detect_local_gpu()
  - Assert backend="cuda", available=True, device_count=1

- [ ] **C013**: Test MPS detection
  - Mock `torch.cuda.is_available()` to return False
  - Mock `torch.backends.mps.is_available()` to return True
  - Call detect_local_gpu()
  - Assert backend="mps", available=True, device_count=1

- [ ] **C014**: Test no GPU available
  - Mock both CUDA and MPS to return False
  - Call detect_local_gpu()
  - Assert available=False, backend=None, device_count=0

- [ ] **C015**: Create `tests/test_compute_planner.py`
  - Import pytest, unittest.mock, planner module
  - Add module-level docstring

- [ ] **C016**: Test all compute values
  - Test create_compute_plan("local_cpu") → runner="local", accelerator="cpu"
  - Test create_compute_plan("local_gpu") → runner="local", accelerator="gpu" (mocked GPU available)
  - Test create_compute_plan("kaggle_gpu") → runner="kaggle_notebook", accelerator="gpu"
  - Test create_compute_plan("kaggle_tpu") → runner="kaggle_notebook", accelerator="tpu"

- [ ] **C017**: Test strict mode
  - Mock GPU not available
  - Call create_compute_plan("local_gpu", strict=True)
  - Assert raises GPUNotAvailableError

- [ ] **C018**: Test fallback mode
  - Mock GPU not available
  - Call create_compute_plan("local_gpu", strict=False)
  - Assert compute="local_cpu", accelerator="cpu"

- [ ] **C019**: Test invalid compute value
  - Call create_compute_plan("invalid_value")
  - Assert raises InvalidComputeError

### Exit Codes

- [ ] **C020**: Update main `src/kagglebot/exceptions.py`
  - Add exit code 10 for GPUNotAvailableError (if not already present)
  - Add exit code 11 for KernelTimeoutError (if not already present)
  - Add exit code 12 for KernelFailedError (if not already present)
  - Document exit codes in docstring

**Checkpoint**: Run `uv run pytest tests/test_gpu_detector.py tests/test_compute_planner.py -v`
All tests should pass.

---

## Phase C2: Runner Interface (Week 2)

**Goal**: Implement Runner abstraction and LocalRunner

### Runner Package Setup

- [ ] **C021**: Create `src/kagglebot/runners/` package
  - Create directory: `mkdir -p src/kagglebot/runners`
  - Create `__init__.py` with package docstring

### Base Runner Interface

- [ ] **C022**: Create `src/kagglebot/runners/base.py`
  - Import required modules (abc, dataclasses, pathlib, typing)
  - Add module-level docstring explaining Runner abstraction

- [ ] **C023**: Define `RunContext` dataclass in `base.py`
  - Fields: competition_slug, data_dir, artifacts_dir, run_id, strategy, accelerator, config
  - Add field docstrings

- [ ] **C024**: Define `RunResult` dataclass in `base.py`
  - Fields: success, submission_path, summary, error, kernel_id
  - Add field docstrings

- [ ] **C025**: Define `Runner` ABC in `base.py`
  - Add `__init__(self, accelerator)` method
  - Add `@abstractmethod validate_preconditions(self, ctx: RunContext) -> None`
  - Add `@abstractmethod run(self, ctx: RunContext) -> RunResult`
  - Add `@abstractmethod cleanup(self, ctx: RunContext) -> None`
  - Add comprehensive class docstring with usage examples

### Runner Registry

- [ ] **C026**: Create runner registry in `runners/__init__.py`
  - Import Runner, LocalRunner (will create next)
  - Define `_REGISTRY: dict[str, Type[Runner]]` with "local" key
  - Add module docstring

- [ ] **C027**: Implement `get_runner()` factory function
  - Accept: runner_name, accelerator
  - Look up runner class in _REGISTRY
  - Raise ValueError if not found
  - Instantiate and return runner
  - Add docstring with examples

### LocalRunner Implementation

- [ ] **C028**: Create `src/kagglebot/runners/local.py`
  - Import required modules (logging, pathlib, typing, base, gpu_detector, exceptions)
  - Define LocalRunner class extending Runner
  - Add class docstring

- [ ] **C029**: Implement `LocalRunner.__init__()`
  - Call super().__init__(accelerator)
  - Initialize self.gpu_info = None
  - Add docstring

- [ ] **C030**: Implement `LocalRunner.validate_preconditions()`
  - If accelerator=="gpu": call detect_local_gpu()
  - If GPU not available: raise GPUNotAvailableError
  - If GPU available: log device name and backend
  - If accelerator=="tpu": raise ValueError("TPU not supported in LocalRunner")
  - Add docstring

- [ ] **C031**: Implement `LocalRunner.run()` (basic structure)
  - Log "Starting local training on {accelerator}"
  - Initialize TabularTrainingEngine with accelerator and gpu_backend
  - Call engine.train() and capture CV results
  - Call engine.predict() and get submission_path
  - Build summary dict with CV results, models, accelerator, gpu_backend
  - Return RunResult with success=True
  - Handle exceptions and return RunResult with success=False, error=str(e)
  - Add comprehensive docstring

- [ ] **C032**: Implement `LocalRunner.cleanup()`
  - For now: just `pass` (no cleanup needed)
  - Add docstring explaining no temp files created

### Training Engine GPU Support

- [ ] **C033**: Update `training/tabular_engine.py` __init__
  - Add `accelerator: Literal["cpu", "gpu", "tpu"] = "cpu"` parameter
  - Add `gpu_backend: str | None = None` parameter
  - Store both in self.accelerator and self.gpu_backend
  - Add docstrings for new parameters

- [ ] **C034**: Add `_get_lightgbm_params()` method
  - Start with base params: objective, metric, verbosity, seed
  - If accelerator=="gpu" and gpu_backend=="cuda": set device="gpu", add gpu_platform_id, gpu_device_id
  - If accelerator=="gpu" and gpu_backend=="mps": log warning "MPS not supported", use device="cpu"
  - Return params dict
  - Add docstring explaining GPU support

- [ ] **C035**: Add `_get_catboost_params()` method
  - Start with base params: loss_function, verbose, random_seed
  - If accelerator=="gpu": set task_type="GPU"
  - If gpu_backend=="cuda": set devices="0"
  - If gpu_backend=="mps": log "Using Metal GPU acceleration"
  - Else: set task_type="CPU"
  - Return params dict
  - Add docstring

- [ ] **C036**: Add `_get_xgboost_params()` method
  - Start with base params: objective, eval_metric, seed
  - If accelerator=="gpu" and gpu_backend=="cuda": set device="cuda", tree_method="hist"
  - If accelerator=="gpu" and gpu_backend=="mps": log warning "MPS not supported", use device="cpu"
  - Return params dict
  - Add docstring

- [ ] **C037**: Update model training to use accelerator-specific params
  - In train() method: call _get_lightgbm_params() for LightGBM models
  - Call _get_catboost_params() for CatBoost models
  - Call _get_xgboost_params() for XGBoost models
  - Pass params to model constructors

### Unit Tests

- [ ] **C038**: Create `tests/test_runners_base.py`
  - Test RunContext dataclass creation
  - Test RunResult dataclass creation
  - Test get_runner() factory for "local" runner
  - Test get_runner() raises ValueError for unknown runner

- [ ] **C039**: Add test for RunContext validation
  - Create RunContext with all required fields
  - Assert all fields set correctly

- [ ] **C040**: Add test for RunResult validation
  - Create RunResult with success=True
  - Create RunResult with success=False and error message
  - Assert fields set correctly

- [ ] **C041**: Create `tests/test_local_runner.py`
  - Import pytest, mock, LocalRunner, RunContext, GPUNotAvailableError
  - Add module docstring

- [ ] **C042**: Test LocalRunner with CPU
  - Create LocalRunner(accelerator="cpu")
  - Create mock RunContext
  - Call validate_preconditions() - should not raise
  - Assert self.gpu_info is None

- [ ] **C043**: Test LocalRunner with GPU available
  - Mock detect_local_gpu() to return GPUInfo(available=True, backend="cuda", ...)
  - Create LocalRunner(accelerator="gpu")
  - Call validate_preconditions() - should not raise
  - Assert self.gpu_info.backend == "cuda"

- [ ] **C044**: Test LocalRunner with GPU not available
  - Mock detect_local_gpu() to return GPUInfo(available=False, ...)
  - Create LocalRunner(accelerator="gpu")
  - Call validate_preconditions()
  - Assert raises GPUNotAvailableError

- [ ] **C045**: Test LocalRunner.run() returns valid RunResult
  - Mock TabularTrainingEngine
  - Mock engine.train() to return CV scores
  - Mock engine.predict() to return submission path
  - Call runner.run(ctx)
  - Assert RunResult.success == True
  - Assert RunResult.submission_path is not None

### Integration Test

- [ ] **C046**: Create `tests/integration/test_local_runner.py`
  - Create test data directory with sample CSV files
  - Create ModelingStrategy mock
  - Create RunContext with real paths
  - Run LocalRunner(accelerator="cpu").run(ctx)
  - Assert submission.csv created
  - Clean up test files

- [ ] **C047**: Run integration test
  - Execute: `uv run pytest tests/integration/test_local_runner.py -v`
  - Verify test passes

**Checkpoint**: Run `uv run pytest tests/test_runners_base.py tests/test_local_runner.py -v`
All tests should pass.

---

## Phase C3: Kernel Package Generation (Week 3)

**Goal**: Generate valid Kaggle kernel packages

### Kernel Package Setup

- [ ] **C048**: Create `src/kagglebot/kernel/` package
  - Create directory: `mkdir -p src/kagglebot/kernel`
  - Create `__init__.py` with package docstring

### Dependencies

- [ ] **C049**: Add Jinja2 dependency
  - Run: `uv add jinja2`
  - Verify: `uv.lock` updated

### Kernel Exceptions

- [ ] **C050**: Create `src/kagglebot/kernel/exceptions.py`
  - Define `KernelError(Exception)` base class
  - Define `KernelTimeoutError(KernelError)` with exit_code=11
  - Define `KernelFailedError(KernelError)` with exit_code=12
  - Define `MissingSubmissionError(KernelError)` with exit_code=7
  - Add docstrings

### Kernel Metadata

- [ ] **C051**: Create `src/kagglebot/kernel/metadata.py`
  - Import typing, json
  - Add module docstring

- [ ] **C052**: Implement `generate_kernel_metadata()` function
  - Accept: username, slug, run_id, accelerator, enable_internet
  - Create kernel slug: `f"{username}/{slug}-{run_id.replace('_', '-')}"`
  - Set id, title, code_file="main.py", language="python"
  - Set kernel_type="script", is_private=true
  - Set competition_sources=[slug] (NO "c/" prefix)
  - Set enable_gpu=(accelerator=="gpu"), enable_tpu=(accelerator=="tpu")
  - Validate: gpu and tpu never both true
  - Set enable_internet=enable_internet (lowercase JSON boolean)
  - Return dict
  - Add comprehensive docstring with examples

### Tests for Metadata

- [ ] **C053**: Create `tests/test_kernel_metadata.py`
  - Test generate_kernel_metadata() with accelerator="cpu"
  - Assert enable_gpu=false, enable_tpu=false

- [ ] **C054**: Test metadata with enable_internet
  - Call generate_kernel_metadata(..., enable_internet=True)
  - Assert enable_internet=true in result
  - Call with enable_internet=False
  - Assert enable_internet=false in result

### Kernel Templates

- [ ] **C055**: Create `src/kagglebot/kernel/templates/` directory
  - Create directory: `mkdir -p src/kagglebot/kernel/templates`
  - No __init__.py needed (not a package)

- [ ] **C056**: Create `kernel/templates/tabular_script.py.j2`
  - Add shebang: `#!/usr/bin/env python3`
  - Add docstring with competition_slug, run_id variables
  - Import: json, logging, pathlib, pandas, numpy, sklearn, catboost
  - Setup logging
  - Define Kaggle paths: INPUT_DIR, WORKING_DIR
  - Define load_data() function
  - Define load_plan() function (embeds plan_json variable)
  - Define preprocess() function
  - Define train_model() function (use CatBoost with GPU)
  - Define predict() function
  - Define main() function
  - Add `if __name__ == "__main__": main()`

- [ ] **C057**: Add error handling to template
  - Wrap main() in try/except
  - Log exceptions
  - Save error to error.json if failure

- [ ] **C058**: Add logging to template
  - Log at start of main()
  - Log after loading data
  - Log after training each fold
  - Log mean CV score
  - Log after saving submission

### Kernel Packager

- [ ] **C059**: Create `src/kagglebot/kernel/packager.py`
  - Import: Jinja2, json, pathlib, logging
  - Define KernelPackager class
  - Add class docstring

- [ ] **C060**: Implement `KernelPackager.__init__()`
  - Accept username parameter
  - Store self.username
  - Set self.template_dir = Path(__file__).parent / "templates"
  - Initialize Jinja2 environment with FileSystemLoader
  - Add docstring

- [ ] **C061**: Implement `KernelPackager.generate_package()`
  - Accept: slug, run_id, strategy, accelerator, enable_internet
  - Create package directory: artifacts/{slug}/{run_id}/kernel/
  - Render script template with context (slug, run_id, plan_json)
  - Write rendered script to main.py
  - Call generate_kernel_metadata() from metadata.py
  - Write metadata to kernel-metadata.json
  - Call validate_kernel_package() on package directory
  - Return package directory path
  - Add comprehensive docstring

- [ ] **C062**: Implement `validate_kernel_package()` function
  - Read main.py and kernel-metadata.json
  - Check for secret patterns (kaggle.json, api_key, token, password)
  - Raise ValueError if secrets detected
  - Log validation success
  - Add docstring

### Strategy Serialization

- [ ] **C063**: Add `to_dict()` method to `ModelingStrategy` in analyzer
  - Return dict with: target, features, models, preprocessing, cv_folds
  - Ensure all values are JSON-serializable
  - Add docstring

### Unit Tests

- [ ] **C064**: Create `tests/test_kernel_metadata.py` (if not already created)
  - Add test for all accelerator values (cpu, gpu, tpu)
  - Assert correct enable_gpu and enable_tpu values

- [ ] **C065**: Test metadata generation for GPU
  - Call generate_kernel_metadata(..., accelerator="gpu")
  - Assert enable_gpu=true, enable_tpu=false

- [ ] **C066**: Test metadata validation
  - Manually create metadata with both gpu=true and tpu=true
  - Call validation function
  - Assert raises ValueError (if validation implemented)

- [ ] **C067**: Test competition_sources format
  - Generate metadata for "titanic"
  - Assert competition_sources=["titanic"] (no "c/" prefix)

- [ ] **C068**: Create `tests/test_kernel_packager.py`
  - Mock ModelingStrategy with to_dict()
  - Create KernelPackager instance

- [ ] **C069**: Test package generation for Titanic
  - Create temp directory
  - Call packager.generate_package(slug="titanic", ...)
  - Assert main.py exists
  - Assert kernel-metadata.json exists
  - Assert main.py contains competition slug
  - Clean up temp directory

- [ ] **C070**: Test template rendering with sample strategy
  - Create simple ModelingStrategy
  - Render template
  - Assert plan_json embedded correctly
  - Assert no syntax errors in rendered Python

- [ ] **C071**: Test secret detection
  - Create package with "kaggle.json" in code
  - Call validate_kernel_package()
  - Assert raises ValueError

### Integration Test

- [ ] **C072**: Integration test with Kaggle CLI dry-run
  - Generate full kernel package for Titanic
  - Run: `kaggle kernels push -p <package_dir> --dry-run` (if supported)
  - Or: manually validate JSON structure
  - Clean up

**Checkpoint**: Run `uv run pytest tests/test_kernel_metadata.py tests/test_kernel_packager.py -v`
All tests should pass.

---

## Phase C4: Kernel Lifecycle Management (Week 4)

**Goal**: Push, poll, and download Kaggle kernels

### Kernel Manager

- [ ] **C073**: Create `src/kagglebot/kernel/manager.py`
  - Import: subprocess, time, pathlib, logging, dataclasses
  - Add module docstring

- [ ] **C074**: Define `KernelStatus` dataclass
  - Fields: kernel_id, state, metadata
  - Add docstring

- [ ] **C075**: Implement `KernelManager.__init__()`
  - Accept: username, config
  - Store self.username, self.config
  - Add docstring

### Push Kernel

- [ ] **C076**: Implement `KernelManager.push_kernel()`
  - Accept: kernel_dir (Path)
  - Build command: ["kaggle", "kernels", "push", "-p", str(kernel_dir)]
  - Run subprocess.run() with capture_output=True, check=True
  - Parse stdout for kernel_id (format: "username/slug")
  - Handle CalledProcessError (credentials, quota, invalid metadata)
  - Log "Kernel pushed: {kernel_id}"
  - Return kernel_id
  - Add docstring

### Get Status

- [ ] **C077**: Implement `KernelManager.get_status()`
  - Accept: kernel_id
  - Build command: ["kaggle", "kernels", "status", kernel_id]
  - Run subprocess.run() with capture_output=True, check=True
  - Parse stdout for status (e.g., "queued", "running", "complete", "error")
  - Create KernelStatus dataclass
  - Return KernelStatus
  - Add docstring

### Poll Until Complete

- [ ] **C078**: Implement `KernelManager.poll_until_complete()`
  - Accept: kernel_id, timeout_minutes (default 120)
  - Calculate end_time = time.time() + timeout_minutes * 60
  - Initialize backoff = 10 (seconds)
  - Loop:
    - Call get_status(kernel_id)
    - If state in ["complete", "error", "cancelled"]: break
    - If time.time() > end_time: raise KernelTimeoutError
    - Log "Polling kernel {kernel_id}, state={state}, backoff={backoff}s"
    - Sleep for backoff seconds
    - Increase backoff = min(backoff * 1.5, 60)
  - If state == "error": raise KernelFailedError
  - Return final KernelStatus
  - Add comprehensive docstring

- [ ] **C079**: Test exponential backoff logic separately
  - Unit test for backoff calculation
  - Assert starts at 10s, increases to max 60s

### Download Outputs

- [ ] **C080**: Implement `KernelManager.download_outputs()`
  - Accept: kernel_id, dest (Path)
  - Create dest directory if not exists
  - Build command: ["kaggle", "kernels", "output", kernel_id, "-p", str(dest)]
  - Run subprocess.run() with check=True
  - Verify output files exist in dest
  - Log "Outputs downloaded to {dest}"
  - Return dest
  - Add docstring

### Logging and Error Handling

- [ ] **C081**: Add structured logging
  - Log at INFO level for all operations
  - Include kernel_id in all log messages
  - Use structured format: "Operation: {op}, kernel_id: {id}, result: {result}"

- [ ] **C082**: Handle all CLI error cases
  - Credentials not found: raise with clear message
  - Network errors: retry with backoff (up to 3 retries)
  - Quota exceeded: raise with quota message
  - Invalid metadata: raise with validation message
  - Timeout: raise KernelTimeoutError with kernel URL

### Unit Tests

- [ ] **C083**: Create `tests/test_kernel_manager.py`
  - Import pytest, mock, KernelManager, exceptions

- [ ] **C084**: Test push_kernel() with mocked subprocess
  - Mock subprocess.run() to return success with kernel_id in stdout
  - Call manager.push_kernel(package_dir)
  - Assert returns correct kernel_id

- [ ] **C085**: Test get_status() with various states
  - Mock subprocess to return "queued", "running", "complete", "error"
  - Call get_status()
  - Assert KernelStatus.state matches

- [ ] **C086**: Test poll_until_complete() succeeds
  - Mock get_status() to return "queued", "running", "complete" in sequence
  - Call poll_until_complete()
  - Assert returns KernelStatus with state="complete"

- [ ] **C087**: Test poll_until_complete() times out
  - Mock get_status() to always return "running"
  - Call poll_until_complete(timeout_minutes=0.01)
  - Assert raises KernelTimeoutError

- [ ] **C088**: Test download_outputs() with mocked subprocess
  - Mock subprocess.run() to succeed
  - Mock Path.exists() to return True
  - Call download_outputs()
  - Assert returns dest path

- [ ] **C089**: Test error handling
  - Test credentials error: mock subprocess to raise with specific message
  - Test quota error: mock subprocess to raise with quota message
  - Assert correct exceptions raised

### Integration Test

- [ ] **C090**: Integration test with mocked CLI
  - Mock all subprocess calls
  - Run full flow: push → poll → download
  - Assert all methods called correctly
  - Assert no actual network calls made

**Checkpoint**: Run `uv run pytest tests/test_kernel_manager.py -v`
All tests should pass.

---

## Phase C5: KaggleNotebookRunner (Week 5)

**Goal**: Implement KaggleNotebookRunner with full integration

### KaggleNotebookRunner Class

- [ ] **C091**: Create `src/kagglebot/runners/kaggle_notebook.py`
  - Import: logging, pathlib, typing, base, packager, manager, exceptions
  - Define KaggleNotebookRunner class extending Runner
  - Add class docstring

- [ ] **C092**: Implement `KaggleNotebookRunner.__init__()`
  - Call super().__init__(accelerator)
  - Initialize self.packager = None
  - Initialize self.manager = None
  - Add docstring

### Validate Preconditions

- [ ] **C093**: Implement `validate_preconditions()`
  - Import KaggleApi
  - Try: api = KaggleApi(); api.authenticate()
  - Catch exception: raise ComputeError with message about credentials
  - Get username from api.get_config_value("username")
  - Check if username is None: raise ValueError
  - Import check_rules_accepted from kaggle_cli
  - Call check_rules_accepted(ctx.competition_slug)
  - If not accepted: raise RulesNotAcceptedError
  - Initialize self.packager = KernelPackager(username)
  - Initialize self.manager = KernelManager(username, ctx.config)
  - Log "Credentials validated for {username}"
  - Add comprehensive docstring

### Run Method

- [ ] **C094**: Implement `run()` method
  - Log "Starting Kaggle kernel execution on {accelerator}"
  - Try: generate kernel package using self.packager.generate_package()
  - Catch exception: return RunResult with success=False, error=str(e)
  - Try: push kernel using self.manager.push_kernel()
  - Catch exception: return RunResult with success=False, error=str(e)
  - Try: poll until complete using self.manager.poll_until_complete()
  - Catch KernelTimeoutError: return RunResult with success=False, error, kernel_id
  - Catch KernelFailedError: return RunResult with success=False, error, kernel_id
  - Try: download outputs using self.manager.download_outputs()
  - Catch exception: return RunResult with success=False, error, kernel_id
  - Try: locate submission using _locate_submission()
  - Catch MissingSubmissionError: return RunResult with success=False, error, kernel_id
  - Parse metrics.json (if exists)
  - Build summary dict with kernel_id, kernel_url, accelerator, metrics
  - Return RunResult with success=True, submission_path, summary, kernel_id
  - Add comprehensive docstring

### Helper Methods

- [ ] **C095**: Implement `_locate_submission()` helper
  - Accept: outputs_dir (Path)
  - Check if outputs_dir / "submission.csv" exists
  - If not: raise MissingSubmissionError with message
  - Return submission_path
  - Add docstring

- [ ] **C096**: Implement `cleanup()` method
  - Log "Kernel metadata saved to artifacts"
  - For now: no actual cleanup needed
  - Add docstring

### Runner Registry Update

- [ ] **C097**: Add KaggleNotebookRunner to registry
  - In `runners/__init__.py`: import KaggleNotebookRunner
  - Add to _REGISTRY: "kaggle_notebook": KaggleNotebookRunner

### Unit Tests

- [ ] **C098**: Create `tests/test_kaggle_notebook_runner.py`
  - Import pytest, mock, KaggleNotebookRunner, exceptions

- [ ] **C099**: Test validate_preconditions() with valid credentials
  - Mock KaggleApi().authenticate() to succeed
  - Mock api.get_config_value("username") to return "testuser"
  - Mock check_rules_accepted() to return True
  - Create runner and call validate_preconditions()
  - Assert self.packager and self.manager initialized

- [ ] **C100**: Test validate_preconditions() with missing credentials
  - Mock KaggleApi().authenticate() to raise exception
  - Call validate_preconditions()
  - Assert raises ComputeError

- [ ] **C101**: Test validate_preconditions() with rules not accepted
  - Mock authentication to succeed
  - Mock check_rules_accepted() to return False
  - Call validate_preconditions()
  - Assert raises RulesNotAcceptedError

- [ ] **C102**: Test run() full flow (mocked)
  - Mock packager.generate_package() to return temp dir
  - Mock manager.push_kernel() to return "user/kernel"
  - Mock manager.poll_until_complete() to return KernelStatus(state="complete")
  - Mock manager.download_outputs() to return outputs_dir
  - Create fake submission.csv in outputs_dir
  - Call runner.run(ctx)
  - Assert RunResult.success == True
  - Assert RunResult.kernel_id == "user/kernel"

- [ ] **C103**: Test run() with kernel timeout
  - Mock poll_until_complete() to raise KernelTimeoutError
  - Call runner.run(ctx)
  - Assert RunResult.success == False
  - Assert RunResult.error contains "timeout"

- [ ] **C104**: Test run() with missing submission
  - Mock all steps to succeed
  - Don't create submission.csv in outputs
  - Call runner.run(ctx)
  - Assert RunResult.success == False
  - Assert RunResult.error contains "submission.csv"

### Integration Test

- [ ] **C105**: Integration test with mocked CLI
  - Mock all Kaggle API and CLI calls
  - Create full RunContext with real paths
  - Run KaggleNotebookRunner.run()
  - Assert kernel package generated
  - Assert all lifecycle methods called
  - Assert RunResult valid

**Checkpoint**: Run `uv run pytest tests/test_kaggle_notebook_runner.py -v`
All tests should pass.

---

## Phase C6: CLI Integration (Week 6)

**Goal**: Wire compute switching into CLI and orchestrator

### CLI Updates

- [ ] **C106**: Update `src/kagglebot/cli.py` run command
  - Add parameter: `compute: Literal["local_cpu", "local_gpu", "kaggle_gpu", "kaggle_tpu"] = "local_cpu"`
  - Add parameter: `strict: bool = False`
  - Add parameter: `kaggle_username: str | None = None`
  - Add parameter: `enable_internet: bool = False`
  - Add parameter: `max_kernel_runtime: int = 120`
  - Keep old parameters: `runner: str | None = None`, `accelerator: str | None = None` (for deprecation)
  - Update docstring

- [ ] **C107**: Add deprecation handling
  - Check if runner or accelerator flags used
  - If used: log deprecation warning
  - Translate to compute value: (runner="local", accelerator="cpu") → "local_cpu", etc.
  - Use warnings.warn() with DeprecationWarning

- [ ] **C108**: Create ComputePlan in CLI
  - Import create_compute_plan from compute.planner
  - Build config dict with kaggle_username, enable_internet, max_kernel_runtime
  - Try: compute_plan = create_compute_plan(compute, strict, config)
  - Catch ComputeError: print error message, exit with e.exit_code
  - Catch Exception: print error, exit with code 1

- [ ] **C109**: Update Pipeline to accept ComputePlan
  - Update Pipeline.__init__() signature to accept compute_plan: ComputePlan
  - Store self.compute_plan
  - Remove old runner/accelerator parameters

- [ ] **C110**: Update Pipeline.execute() to use runner factory
  - Import get_runner from runners
  - Call: runner = get_runner(self.compute_plan.runner, self.compute_plan.accelerator)
  - Create RunContext with all required fields
  - Call runner.validate_preconditions(ctx)
  - Call runner.run(ctx)
  - Call runner.cleanup(ctx)
  - Handle exceptions with proper logging

- [ ] **C111**: Pass runner config from CLI to Pipeline
  - Pass enable_internet to RunContext.config
  - Pass max_kernel_runtime to RunContext.config
  - Pass kaggle_username to config (if provided)

- [ ] **C112**: Add flag validation in CLI
  - If compute in ["kaggle_gpu", "kaggle_tpu"] and kaggle_username is None:
    - Try to auto-detect from Kaggle API
    - If can't detect: print error, exit with code 1
  - If enable_internet is True: log warning
  - If max_kernel_runtime > 540 (Kaggle limit): log warning

- [ ] **C113**: Update help text
  - Update --compute flag help with all 4 values
  - Add examples to --help output
  - Document all new flags

- [ ] **C114**: Add usage examples to help
  - Local CPU example
  - Local GPU example with strict mode
  - Kaggle GPU example
  - Kaggle TPU example

### Error Messages

- [ ] **C115**: Update error messages with remediation hints
  - For GPUNotAvailableError: suggest --compute local_cpu or --compute kaggle_gpu
  - For KernelTimeoutError: suggest --max-kernel-runtime increase
  - For KernelFailedError: include kernel URL for logs
  - Use ERROR_MESSAGES dict from ARCHITECTURE_COMPUTE.md

### Unit Tests

- [ ] **C116**: Create `tests/test_cli_compute.py`
  - Import CliRunner, cli module, mock

- [ ] **C117**: Test CLI parsing for all compute values
  - Invoke CLI with --compute local_cpu
  - Invoke CLI with --compute local_gpu
  - Invoke CLI with --compute kaggle_gpu
  - Invoke CLI with --compute kaggle_tpu
  - Assert correct compute plan created

- [ ] **C118**: Test --strict flag
  - Invoke with --compute local_gpu --strict
  - Mock GPU not available
  - Assert exit code 10

- [ ] **C119**: Test deprecation warnings
  - Invoke with --runner local --accelerator cpu
  - Assert deprecation warning logged
  - Assert translates to --compute local_cpu

- [ ] **C120**: Test flag validation
  - Invoke with --compute kaggle_gpu without --kaggle-username
  - Assert error if username can't be detected via `KAGGLE_USERNAME` or `~/.kaggle/kaggle.json`

### Integration Tests

- [ ] **C121**: Integration test: local_cpu end-to-end
  - Create test competition data
  - Run CLI: `kagglebot run test-comp --compute local_cpu`
  - Assert submission.csv created
  - Assert no errors

- [ ] **C122**: Integration test: local_gpu with mocked GPU
  - Mock detect_local_gpu() to return GPU available
  - Run CLI: `kagglebot run test-comp --compute local_gpu --strict`
  - Assert uses GPU parameters

- [ ] **C123**: Integration test: kaggle_gpu dry-run
  - Run CLI: `kagglebot run test-comp --compute kaggle_gpu --dry-run`
  - Assert kernel package generated
  - Assert no actual push (dry-run)

**Checkpoint**: Run `uv run pytest tests/test_cli_compute.py -v`
All tests should pass. Run integration tests separately.

---

## Phase C7: Documentation and Polish (Week 7)

**Goal**: Production-ready documentation and final testing

### Documentation

- [ ] **C124**: Update README.md
  - Add "Compute Switching" section after "Features"
  - Show example for each compute mode (local_cpu, local_gpu, kaggle_gpu, kaggle_tpu)
  - Document --compute flag and related flags (--strict, --kaggle-username optional, etc.)
  - Update "Features" to mention GPU support

- [ ] **C125**: Update CLAUDE.md (already done in this session)
  - Verify "Compute Switching Architecture" section present
  - Verify uv usage emphasized
  - Verify non-interactive emphasized
  - Verify rules acceptance manual only

- [ ] **C126**: Create TUTORIAL_COMPUTE.md
  - Introduction: What is compute switching?
  - Section 1: Local CPU (default, works everywhere)
  - Section 2: Local GPU (CUDA/MPS detection, fallback, strict mode)
  - Section 3: Kaggle GPU (kernel generation, push, poll, download)
  - Section 4: Troubleshooting (GPU not detected, kernel timeout, etc.)
  - Include screenshots/examples

- [ ] **C127**: Update SECURITY.md
  - Add "Kernel Security" section
  - Document secret detection validation
  - Document enable_internet risks
  - Document subprocess safety (no shell=True)
  - Add kernel package validation checklist

- [ ] **C128**: Create config examples
  - Create `config/examples/` directory
  - Create `local_cpu.toml` with CPU-specific settings
  - Create `local_gpu.toml` with GPU-specific settings
  - Create `kaggle_gpu.toml` with kernel settings
  - Create `kaggle_tpu.toml` with TPU settings
  - Add comments explaining each setting

- [ ] **C129**: Review error messages
  - Check all error messages have actionable next steps
  - Include URLs where relevant (kernel logs, competition rules)
  - Test all error paths manually
  - Update ERROR_MESSAGES dict if needed

- [ ] **C130**: Add structured logging
  - Log compute plan decisions at INFO level
  - Log GPU detection results at INFO level
  - Log kernel lifecycle events at INFO level
  - Log fallback decisions at WARNING level
  - Use consistent format: "{action}: {details}"

### Manual Testing

- [ ] **C131**: Manual test: Local CPU on Titanic
  - Download Titanic data
  - Run: `uv run kagglebot run titanic --compute local_cpu`
  - Verify submission.csv created
  - Verify validation passes
  - Check artifacts directory structure

- [ ] **C132**: Manual test: Local GPU on Titanic (if GPU available)
  - Run: `uv run kagglebot run titanic --compute local_gpu`
  - Verify GPU detected (check logs)
  - Verify models use GPU params
  - Verify faster than CPU (optional)

- [ ] **C133**: Manual test: Kaggle GPU on Titanic
  - Run: `uv run kagglebot run titanic --compute kaggle_gpu --dry-run`
  - Verify kernel package generated
  - Manually push kernel: `kaggle kernels push -p <package-dir>`
  - Wait for completion
  - Download outputs manually
  - Verify submission.csv present

### Backward Compatibility

- [ ] **C134**: Verify backward compatibility
  - Run old command: `kagglebot run titanic` (no --compute)
  - Assert defaults to local_cpu
  - Run with old flags: `--runner local --accelerator cpu`
  - Assert deprecation warning shown
  - Assert still works correctly

### Test Coverage

- [ ] **C135**: Run test coverage
  - Execute: `uv run pytest --cov=kagglebot.compute --cov=kagglebot.runners --cov=kagglebot.kernel --cov-report=term-missing`
  - Review coverage report

- [ ] **C136**: Ensure >80% coverage for new code
  - Identify uncovered lines
  - Add tests for uncovered code
  - Re-run coverage
  - Verify >80% coverage

### Security Audit

- [ ] **C137**: Review all subprocess calls
  - Grep for `subprocess` in codebase
  - Verify no `shell=True` used
  - Verify all args are list (not string)
  - Test with malicious inputs (e.g., filename with semicolon)

- [ ] **C138**: Verify secret detection
  - Create test kernel with "kaggle.json" in code
  - Run validate_kernel_package()
  - Assert raises error
  - Test with various secret patterns

- [ ] **C139**: Check enable_internet default
  - Review all code paths that set enable_internet
  - Verify defaults to False
  - Verify requires explicit flag

### Final Steps

- [ ] **C140**: Final lint and format
  - Run: `uv run ruff check .`
  - Fix any issues
  - Run: `uv run ruff format .`
  - Commit formatting changes

**Checkpoint**: All tests pass, documentation complete, ready for release.

---

## Success Criteria

After completing all 140 tasks:

### Functionality ✅
- `kagglebot run titanic --compute local_cpu` works end-to-end
- `kagglebot run titanic --compute local_gpu` detects GPU and trains
- `kagglebot run titanic --compute kaggle_gpu --submit` works (mocked in tests, manual test succeeds)
- GPU fallback works (local_gpu → local_cpu when no GPU and not strict)
- All safety guardrails work (dry-run, dedup, validation)

### Quality ✅
- Test coverage >80% for new code
- All tests pass in CI
- No regressions in existing functionality
- Documentation complete and accurate
- Clear error messages for all failures

### Security ✅
- No secrets in kernel code
- No secrets in kernel packages
- `enable_internet` defaults to false
- Rules acceptance required (no automation)
- All subprocess calls secure (no shell injection)

---

## Notes for Implementers

1. **Test frequently**: Run `uv run pytest -q` after every few tasks
2. **Read the specs**: Refer to SPEC_COMPUTE.md and ARCHITECTURE_COMPUTE.md for details
3. **Ask questions**: If a task is unclear, check the design docs or ask for clarification
4. **Document as you go**: Add docstrings to every function/class
5. **Commit often**: Small commits make debugging easier

**Good luck!** 🚀
