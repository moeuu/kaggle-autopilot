# Compute Switching Implementation Plan

**Version**: 1.0
**Status**: Planning
**Last Updated**: 2026-01-01

---

## Overview

This document outlines the phased implementation plan for compute switching functionality in kagglebot, enabling execution across:
- **local_cpu**: Train on local CPU
- **local_gpu**: Train on local GPU (CUDA/MPS auto-detection)
- **kaggle_gpu**: Train on Kaggle GPU kernel
- **kaggle_tpu**: Train on Kaggle TPU kernel

**Timeline**: 6 weeks (assumes 1 developer, full-time)

**Prerequisites**: Phase 3 (Training Engine) from main PLAN.md must be complete

---

## Phase Breakdown

### Phase C1: Foundation (Week 1)

**Goal**: Set up compute planning infrastructure and GPU detection

**Tasks**:
1. Create `src/kagglebot/compute/` package
2. Implement GPU detection (`compute/gpu_detector.py`)
3. Implement compute planning (`compute/planner.py`)
4. Define compute-specific exceptions (`compute/exceptions.py`)
5. Add unit tests for GPU detection (with mocks)
6. Add unit tests for compute planning
7. Update exit codes in main `exceptions.py`

**Deliverables**:
- ✅ `compute/gpu_detector.py` with CUDA/MPS detection
- ✅ `compute/planner.py` with ComputePlan generation
- ✅ `compute/exceptions.py` with GPUNotAvailableError, InvalidComputeError
- ✅ Unit tests with >80% coverage
- ✅ Documentation in module docstrings

**Success Criteria**:
- `detect_local_gpu()` correctly identifies CUDA/MPS
- `create_compute_plan()` maps compute flags to runner/accelerator
- Fallback logic works (local_gpu → local_cpu when strict=False)
- All unit tests pass

**Dependencies**:
- PyTorch (for GPU detection)
- None (can be developed independently)

---

### Phase C2: Runner Interface (Week 2)

**Goal**: Implement Runner abstraction and LocalRunner

**Tasks**:
1. Create `src/kagglebot/runners/` package
2. Define Runner ABC (`runners/base.py`)
3. Define RunContext and RunResult dataclasses
4. Implement LocalRunner (`runners/local.py`)
5. Add runner registry and factory (`runners/__init__.py`)
6. Update `training/tabular_engine.py` with GPU support
7. Add LightGBM GPU parameters
8. Add CatBoost GPU parameters
9. Add XGBoost GPU parameters
10. Unit tests for LocalRunner
11. Integration test: LocalRunner on Titanic (CPU mode)

**Deliverables**:
- ✅ `runners/base.py` with Runner, RunContext, RunResult
- ✅ `runners/local.py` with CPU and GPU support
- ✅ `runners/__init__.py` with factory function
- ✅ `training/tabular_engine.py` with GPU parameters
- ✅ Unit tests for all components
- ✅ Integration test proving LocalRunner works

**Success Criteria**:
- LocalRunner(accelerator="cpu") trains models successfully
- LocalRunner(accelerator="gpu") detects GPU and uses correct parameters
- GPU parameters correctly set for LightGBM, CatBoost, XGBoost
- All tests pass

**Dependencies**:
- Phase C1 (GPU detection)
- Existing training engine from Phase 3

---

### Phase C3: Kernel Package Generation (Week 3)

**Goal**: Generate valid Kaggle kernel packages

**Tasks**:
1. Create `src/kagglebot/kernel/` package
2. Add Jinja2 dependency: `uv add jinja2`
3. Implement kernel metadata generation (`kernel/metadata.py`)
4. Create kernel templates directory (`kernel/templates/`)
5. Create `tabular_script.py.j2` template
6. Implement KernelPackager (`kernel/packager.py`)
7. Add secret detection validation
8. Add ModelingStrategy.to_dict() method
9. Unit tests for metadata generation
10. Unit tests for template rendering
11. Integration test: generate package for Titanic

**Deliverables**:
- ✅ `kernel/metadata.py` with metadata generation
- ✅ `kernel/packager.py` with package assembly
- ✅ `kernel/templates/tabular_script.py.j2`
- ✅ Secret detection in validation
- ✅ Unit tests for all components
- ✅ Sample kernel package (not committed)

**Success Criteria**:
- `generate_kernel_metadata()` produces valid kernel-metadata.json
- Template renders correctly with plan.json embedded
- Generated package passes Kaggle CLI validation (dry-run)
- Secret detection catches test secrets
- All tests pass

**Dependencies**:
- Phase C2 (need ModelingStrategy from training)
- Jinja2 library

---

### Phase C4: Kernel Lifecycle Management (Week 4)

**Goal**: Push, poll, and download Kaggle kernels

**Tasks**:
1. Implement KernelManager (`kernel/manager.py`)
2. Implement push_kernel() method
3. Implement get_status() method
4. Implement poll_until_complete() with exponential backoff
5. Implement download_outputs() method
6. Add KernelStatus dataclass
7. Define kernel-specific exceptions (`kernel/exceptions.py`)
8. Add structured logging for all operations
9. Unit tests with mocked Kaggle API calls
10. Integration test with mock subprocess

**Deliverables**:
- ✅ `kernel/manager.py` with full lifecycle methods
- ✅ `kernel/exceptions.py` with KernelTimeoutError, KernelFailedError
- ✅ Exponential backoff in polling
- ✅ Unit tests with mocks (no actual push)
- ✅ Integration test with mocked CLI

**Success Criteria**:
- KernelManager can push packages (mocked)
- Polling works with timeout enforcement
- Download outputs correctly retrieves files (mocked)
- Error handling covers all CLI error states
- All tests pass

**Dependencies**:
- Phase C3 (kernel package generation)
- Kaggle API library

---

### Phase C5: KaggleNotebookRunner (Week 5)

**Goal**: Implement KaggleNotebookRunner with full integration

**Tasks**:
1. Implement KaggleNotebookRunner (`runners/kaggle_notebook.py`)
2. Implement validate_preconditions() with rules check
3. Implement run() method (full flow)
4. Implement _locate_submission() helper
5. Implement cleanup() method
6. Add runner to registry
7. Unit tests for KaggleNotebookRunner (mocked KernelManager)
8. Integration test with mock CLI
9. Test error handling (timeout, no submission, etc.)
10. Test dry-run mode (no actual push)

**Deliverables**:
- ✅ `runners/kaggle_notebook.py` fully implemented
- ✅ Rules acceptance check integrated
- ✅ Submission extraction from kernel outputs
- ✅ Unit tests with mocked dependencies
- ✅ Integration test proving full flow

**Success Criteria**:
- KaggleNotebookRunner.run() executes full flow (mocked)
- Rules not accepted → raises RulesNotAcceptedError
- Kernel timeout → raises KernelTimeoutError
- Missing submission → raises MissingSubmissionError
- All tests pass

**Dependencies**:
- Phase C4 (KernelManager)
- Phase C2 (Runner interface)

---

### Phase C6: CLI Integration (Week 6)

**Goal**: Wire compute switching into CLI and orchestrator

**Tasks**:
1. Add `--compute` flag to `kagglebot run` command
2. Add `--strict` flag
3. Add `--kaggle-username` flag
4. Add `--enable-internet` flag
5. Add `--max-kernel-runtime` flag
6. Update `cli.py` to create ComputePlan
7. Update `orchestrator.py` to use runner factory
8. Pass ComputePlan to Pipeline
9. Update Pipeline to use get_runner()
10. Add deprecation warnings for old `--runner`/`--accelerator` flags
11. Update help text and examples
12. Add validation for flag combinations
13. Integration test: full pipeline with local_cpu
14. Integration test: full pipeline with kaggle_gpu (mocked)
15. Update error messages with remediation hints

**Deliverables**:
- ✅ CLI accepts all compute flags
- ✅ Orchestrator uses runner factory
- ✅ Full pipeline works with all compute modes
- ✅ Deprecation warnings for old flags
- ✅ Comprehensive help text
- ✅ Integration tests for all modes

**Success Criteria**:
- `kagglebot run titanic --compute local_cpu` works end-to-end
- `kagglebot run titanic --compute local_gpu` detects GPU and trains
- `kagglebot run titanic --compute kaggle_gpu --dry-run` generates kernel package
- `--strict` flag enforces no fallback
- All validation works correctly
- All tests pass

**Dependencies**:
- Phase C5 (both runners complete)
- Phase C1 (compute planning)
- Existing orchestrator from main PLAN.md

---

### Phase C7: Documentation and Polish (Week 7)

**Goal**: Production-ready documentation and final testing

**Tasks**:
1. Update README.md with compute switching examples
2. Update CLAUDE.md with compute emphasis
3. Create TUTORIAL_COMPUTE.md with step-by-step guide
4. Document all CLI flags in help text
5. Add troubleshooting section to docs
6. Update SECURITY.md for kernel security
7. Review all error messages for clarity
8. Add logging for compute decisions
9. Create example configs for each compute mode
10. Manual test: local_cpu on Titanic
11. Manual test: local_gpu on Titanic (if GPU available)
12. Manual test: kaggle_gpu on Titanic (actual push)
13. Verify backward compatibility (old commands still work)
14. Achieve >80% test coverage for new code
15. Final security audit (no secrets, safe subprocess calls)

**Deliverables**:
- ✅ Updated README.md with usage examples
- ✅ Updated CLAUDE.md with emphasis on compute, non-interactive, uv
- ✅ TUTORIAL_COMPUTE.md with walkthrough
- ✅ Complete documentation for all features
- ✅ Manual testing on real competition
- ✅ Security audit complete

**Success Criteria**:
- Documentation is clear and complete
- All examples work as documented
- Manual tests pass on real competition
- Test coverage >80% for new code
- No security vulnerabilities found
- Backward compatibility maintained

**Dependencies**:
- Phase C6 (full integration)

---

## Implementation Checklist

### C1: Foundation

- [ ] **C001**: Create `src/kagglebot/compute/` package with `__init__.py`
- [ ] **C002**: Create `compute/exceptions.py` with:
  - ComputeError base class
  - GPUNotAvailableError (exit_code=10)
  - InvalidComputeError (exit_code=1)
- [ ] **C003**: Create `compute/gpu_detector.py` with:
  - GPUInfo dataclass
  - detect_local_gpu() function
  - get_torch_device() function
- [ ] **C004**: Implement CUDA detection in detect_local_gpu()
- [ ] **C005**: Implement MPS detection in detect_local_gpu()
- [ ] **C006**: Add logging for GPU detection results
- [ ] **C007**: Create `compute/planner.py` with:
  - ComputePlan dataclass
  - _COMPUTE_MAPPING dict
  - create_compute_plan() function
- [ ] **C008**: Implement compute validation in create_compute_plan()
- [ ] **C009**: Implement GPU fallback logic (local_gpu → local_cpu)
- [ ] **C010**: Add logging for compute plan decisions
- [ ] **C011**: Create `tests/test_gpu_detector.py`
- [ ] **C012**: Test CUDA detection (mocked torch.cuda)
- [ ] **C013**: Test MPS detection (mocked torch.backends.mps)
- [ ] **C014**: Test no GPU available
- [ ] **C015**: Create `tests/test_compute_planner.py`
- [ ] **C016**: Test all compute values (local_cpu, local_gpu, kaggle_gpu, kaggle_tpu)
- [ ] **C017**: Test strict mode (raises GPUNotAvailableError)
- [ ] **C018**: Test fallback mode (local_gpu → local_cpu)
- [ ] **C019**: Test invalid compute value (raises InvalidComputeError)
- [ ] **C020**: Update main `src/kagglebot/exceptions.py` with exit codes 10-12

**Checkpoint**: GPU detection and compute planning work, all tests pass

---

### C2: Runner Interface

- [ ] **C021**: Create `src/kagglebot/runners/` package with `__init__.py`
- [ ] **C022**: Create `runners/base.py` with:
  - RunContext dataclass
  - RunResult dataclass
  - Runner ABC class
- [ ] **C023**: Define Runner.validate_preconditions() abstract method
- [ ] **C024**: Define Runner.run() abstract method
- [ ] **C025**: Define Runner.cleanup() abstract method
- [ ] **C026**: Create runner registry dict in `runners/__init__.py`
- [ ] **C027**: Implement get_runner() factory function
- [ ] **C028**: Create `runners/local.py` with LocalRunner class
- [ ] **C029**: Implement LocalRunner.__init__() (store accelerator)
- [ ] **C030**: Implement LocalRunner.validate_preconditions():
  - Check GPU availability if accelerator="gpu"
  - Raise GPUNotAvailableError if not available
- [ ] **C031**: Implement LocalRunner.run():
  - Initialize TabularTrainingEngine
  - Call engine.train()
  - Call engine.predict()
  - Return RunResult
- [ ] **C032**: Implement LocalRunner.cleanup() (no-op for now)
- [ ] **C033**: Update `training/tabular_engine.py` to accept:
  - accelerator parameter
  - gpu_backend parameter
- [ ] **C034**: Add _get_lightgbm_params() method with GPU support:
  - If accelerator="gpu" and gpu_backend="cuda": device="gpu"
  - If accelerator="gpu" and gpu_backend="mps": log warning, use CPU
- [ ] **C035**: Add _get_catboost_params() method with GPU support:
  - If accelerator="gpu": task_type="GPU"
  - Log backend (CUDA or MPS)
- [ ] **C036**: Add _get_xgboost_params() method with GPU support:
  - If accelerator="gpu" and gpu_backend="cuda": device="cuda"
  - If accelerator="gpu" and gpu_backend="mps": log warning, use CPU
- [ ] **C037**: Update model training to use accelerator-specific params
- [ ] **C038**: Create `tests/test_runners_base.py`
- [ ] **C039**: Test RunContext and RunResult dataclasses
- [ ] **C040**: Test get_runner() factory for all runner types
- [ ] **C041**: Create `tests/test_local_runner.py`
- [ ] **C042**: Test LocalRunner with accelerator="cpu"
- [ ] **C043**: Test LocalRunner with accelerator="gpu" (mocked GPU available)
- [ ] **C044**: Test LocalRunner with accelerator="gpu" (GPU not available, raises error)
- [ ] **C045**: Test LocalRunner.run() returns valid RunResult
- [ ] **C046**: Create `tests/integration/test_local_runner.py`
- [ ] **C047**: Integration test: LocalRunner on Titanic (CPU mode)

**Checkpoint**: LocalRunner works for CPU training, GPU params configured

---

### C3: Kernel Package Generation

- [ ] **C048**: Create `src/kagglebot/kernel/` package with `__init__.py`
- [ ] **C049**: Add Jinja2 dependency: `uv add jinja2`
- [ ] **C050**: Create `kernel/exceptions.py` with:
  - KernelError base class
  - KernelTimeoutError (exit_code=11)
  - KernelFailedError (exit_code=12)
  - MissingSubmissionError (exit_code=7)
- [ ] **C051**: Create `kernel/metadata.py`
- [ ] **C052**: Implement generate_kernel_metadata() function:
  - Accept: username, slug, run_id, accelerator, enable_internet
  - Return: dict with kernel-metadata.json structure
  - Validate: gpu/tpu never both true
  - Use lowercase true/false (JSON booleans)
  - competition_sources: [slug] (no "c/" prefix)
- [ ] **C053**: Test metadata generation with all accelerator values
- [ ] **C054**: Test metadata with enable_internet=true/false
- [ ] **C055**: Create `kernel/templates/` directory
- [ ] **C056**: Create `kernel/templates/tabular_script.py.j2` template:
  - Load data from /kaggle/input/{slug}/
  - Read embedded plan.json
  - Implement preprocessing
  - Train models with CV
  - Generate predictions
  - Save to /kaggle/working/submission.csv
  - Save metrics.json
- [ ] **C057**: Add error handling to template
- [ ] **C058**: Add logging to template
- [ ] **C059**: Create `kernel/packager.py` with KernelPackager class
- [ ] **C060**: Implement KernelPackager.__init__() (setup Jinja2 env)
- [ ] **C061**: Implement KernelPackager.generate_package():
  - Create package directory
  - Render script from template
  - Write main.py
  - Generate metadata
  - Write kernel-metadata.json
  - Return package path
- [ ] **C062**: Implement secret detection in validate_kernel_package()
- [ ] **C063**: Add ModelingStrategy.to_dict() method in analyzer
- [ ] **C064**: Create `tests/test_kernel_metadata.py`
- [ ] **C065**: Test metadata generation for all accelerators
- [ ] **C066**: Test metadata validation (gpu/tpu conflict)
- [ ] **C067**: Test competition_sources format (no "c/" prefix)
- [ ] **C068**: Create `tests/test_kernel_packager.py`
- [ ] **C069**: Test package generation for Titanic
- [ ] **C070**: Test template rendering with sample strategy
- [ ] **C071**: Test secret detection catches test secrets
- [ ] **C072**: Integration test: generate package and validate with Kaggle CLI (dry-run)

**Checkpoint**: Can generate valid kernel packages locally

---

### C4: Kernel Lifecycle Management

- [ ] **C073**: Create `kernel/manager.py` with KernelManager class
- [ ] **C074**: Create KernelStatus dataclass
- [ ] **C075**: Implement KernelManager.__init__() (store username, config)
- [ ] **C076**: Implement KernelManager.push_kernel():
  - Run `kaggle kernels push -p <dir>`
  - Parse kernel_id from stdout
  - Handle errors (credentials, quota, invalid metadata)
  - Return kernel_id
- [ ] **C077**: Implement KernelManager.get_status():
  - Run `kaggle kernels status <kernel_id>`
  - Parse status from stdout
  - Return KernelStatus
- [ ] **C078**: Implement KernelManager.poll_until_complete():
  - Loop with exponential backoff
  - Call get_status() each iteration
  - Check for complete/error/cancelled states
  - Enforce timeout (raise KernelTimeoutError)
  - Log progress
  - Return final KernelStatus
- [ ] **C079**: Implement exponential backoff logic:
  - Start: 10s, Max: 60s
  - Increase by 1.5x each iteration
- [ ] **C080**: Implement KernelManager.download_outputs():
  - Run `kaggle kernels output <kernel_id> -p <dest>`
  - Verify download success
  - Return output directory path
- [ ] **C081**: Add structured logging for all operations
- [ ] **C082**: Handle all CLI error cases:
  - Credentials not found
  - Network errors (retry with backoff)
  - Quota exceeded
  - Invalid kernel metadata
  - Timeout
- [ ] **C083**: Create `tests/test_kernel_manager.py`
- [ ] **C084**: Test push_kernel() with mocked subprocess
- [ ] **C085**: Test get_status() with various status values
- [ ] **C086**: Test poll_until_complete() succeeds
- [ ] **C087**: Test poll_until_complete() times out (raises KernelTimeoutError)
- [ ] **C088**: Test download_outputs() with mocked subprocess
- [ ] **C089**: Test error handling for all CLI errors
- [ ] **C090**: Integration test with mocked Kaggle CLI

**Checkpoint**: KernelManager can push, poll, download (with mocks)

---

### C5: KaggleNotebookRunner

- [ ] **C091**: Create `runners/kaggle_notebook.py` with KaggleNotebookRunner class
- [ ] **C092**: Implement KaggleNotebookRunner.__init__() (call super, init manager)
- [ ] **C093**: Implement KaggleNotebookRunner.validate_preconditions():
  - Check Kaggle credentials exist
  - Detect username (from config or auto-detect)
  - Check competition rules accepted
  - Initialize KernelPackager and KernelManager
- [ ] **C094**: Implement KaggleNotebookRunner.run():
  - Generate kernel package
  - Push kernel
  - Poll until complete
  - Download outputs
  - Locate submission.csv
  - Parse metrics.json (if exists)
  - Return RunResult with kernel_id
- [ ] **C095**: Implement _locate_submission() helper:
  - Search outputs_dir for submission.csv
  - Raise MissingSubmissionError if not found
- [ ] **C096**: Implement KaggleNotebookRunner.cleanup():
  - Save kernel metadata to artifacts
- [ ] **C097**: Add KaggleNotebookRunner to runner registry
- [ ] **C098**: Create `tests/test_kaggle_notebook_runner.py`
- [ ] **C099**: Test validate_preconditions() with valid credentials
- [ ] **C100**: Test validate_preconditions() with missing credentials (raises)
- [ ] **C101**: Test validate_preconditions() with rules not accepted (raises)
- [ ] **C102**: Test run() full flow (mocked KernelManager)
- [ ] **C103**: Test run() with kernel timeout (raises KernelTimeoutError)
- [ ] **C104**: Test run() with missing submission (raises MissingSubmissionError)
- [ ] **C105**: Integration test with mocked CLI

**Checkpoint**: KaggleNotebookRunner complete (unit/integration tests pass)

---

### C6: CLI Integration

- [ ] **C106**: Update `cli.py` run command signature:
  - Add compute: Literal["local_cpu", "local_gpu", "kaggle_gpu", "kaggle_tpu"]
  - Add strict: bool = False
  - Add kaggle_username: str | None
  - Add enable_internet: bool = False
  - Add max_kernel_runtime: int = 120
  - Keep old runner/accelerator flags (deprecated)
- [ ] **C107**: Add deprecation handling for --runner and --accelerator:
  - If used, print warning
  - Translate to --compute value
- [ ] **C108**: Call create_compute_plan() in run command:
  - Pass compute, strict, config
  - Handle exceptions (print error, exit with code)
- [ ] **C109**: Update Pipeline.__init__() to accept ComputePlan
- [ ] **C110**: Update Pipeline.execute() to use get_runner():
  - Get runner from factory
  - Call validate_preconditions()
  - Call run()
  - Call cleanup()
- [ ] **C111**: Update Orchestrator to pass runner config:
  - enable_internet from CLI
  - max_kernel_runtime from CLI
  - kaggle_username from CLI or config
- [ ] **C112**: Add flag validation:
  - If Kaggle username not found in `KAGGLE_USERNAME`, `~/.kaggle/kaggle.json`, or `--kaggle-username`: error
  - If --enable-internet: log warning
  - If --max-kernel-runtime > Kaggle limits: warn
- [ ] **C113**: Update help text for all flags
- [ ] **C114**: Add usage examples to --help:
  - Local CPU example
  - Local GPU example
  - Kaggle GPU example
  - Kaggle TPU example
- [ ] **C115**: Update error messages with remediation hints:
  - GPUNotAvailableError: suggest alternatives
  - KernelTimeoutError: suggest --max-kernel-runtime increase
  - KernelFailedError: link to kernel logs
- [ ] **C116**: Create `tests/test_cli_compute.py`
- [ ] **C117**: Test CLI parsing for all --compute values
- [ ] **C118**: Test --strict flag behavior
- [ ] **C119**: Test deprecation warnings for old flags
- [ ] **C120**: Test flag validation
- [ ] **C121**: Integration test: `kagglebot run titanic --compute local_cpu`
- [ ] **C122**: Integration test: `kagglebot run titanic --compute local_gpu --strict` (mocked GPU)
- [ ] **C123**: Integration test: `kagglebot run titanic --compute kaggle_gpu --dry-run` (mocked)

**Checkpoint**: Full CLI integration, all flags work

---

### C7: Documentation and Polish

- [ ] **C124**: Update README.md:
  - Add "Compute Switching" section
  - Show examples for all 4 compute modes
  - Document --compute flag and related flags
  - Update "Features" section
- [ ] **C125**: Update CLAUDE.md:
  - Emphasize compute switching in project overview
  - Add non-interactive requirement
  - Emphasize uv usage
  - Add rules acceptance requirement (manual only)
  - Add compute-specific security guidelines
- [ ] **C126**: Create TUTORIAL_COMPUTE.md:
  - Step-by-step walkthrough for each compute mode
  - Local CPU: Full example (Titanic)
  - Local GPU: GPU detection and fallback
  - Kaggle GPU: Kernel push and polling
  - Troubleshooting guide
- [ ] **C127**: Update SECURITY.md:
  - Add kernel security section
  - Document secret detection
  - Add enable_internet risks
  - Add subprocess safety for Kaggle CLI
- [ ] **C128**: Create config examples:
  - `config/examples/local_cpu.toml`
  - `config/examples/local_gpu.toml`
  - `config/examples/kaggle_gpu.toml`
  - `config/examples/kaggle_tpu.toml`
- [ ] **C129**: Review all error messages:
  - Ensure actionable next steps
  - Include URLs where relevant
  - Test all error paths
- [ ] **C130**: Add structured logging:
  - Log compute plan decisions
  - Log GPU detection results
  - Log kernel lifecycle events
  - Log fallback decisions
- [ ] **C131**: Manual test: Local CPU on Titanic
- [ ] **C132**: Manual test: Local GPU on Titanic (if GPU available)
- [ ] **C133**: Manual test: Kaggle GPU on Titanic (actual push to private kernel)
- [ ] **C134**: Verify backward compatibility:
  - Old commands still work
  - Default behavior unchanged (local_cpu)
- [ ] **C135**: Run test coverage: `uv run pytest --cov=kagglebot --cov-report=term-missing`
- [ ] **C136**: Ensure >80% coverage for new code
- [ ] **C137**: Security audit:
  - No shell=True in subprocess calls
  - Secret detection works
  - No secrets in kernel packages
  - enable_internet defaults to false
- [ ] **C138**: Final lint and format:
  - `uv run ruff check .`
  - `uv run ruff format .`
- [ ] **C139**: Update CHANGELOG.md with compute switching features
- [ ] **C140**: Create migration guide for users (old flags → new flags)

**Checkpoint**: Production-ready, documented, tested

---

## Dependencies Graph

```
C1 (Foundation)
  ↓
C2 (Runner Interface)
  ↓
C3 (Kernel Package) + C4 (Kernel Manager)  [parallel]
  ↓
C5 (KaggleNotebookRunner)
  ↓
C6 (CLI Integration)
  ↓
C7 (Documentation)
```

**Critical Path**:
- C1 must complete before C2 (GPU detection needed for LocalRunner)
- C2 must complete before C3/C4 (need RunContext/RunResult)
- C3 and C4 can be done in parallel
- C5 requires both C3 and C4
- C6 requires C5 (both runners complete)
- C7 is final polish after C6

---

## Testing Strategy

### Unit Tests

**Target**: >80% coverage for new code

**Mocking Strategy**:
- Mock `torch.cuda.is_available()` for GPU detection
- Mock `torch.backends.mps.is_available()` for MPS detection
- Mock `subprocess.run()` for Kaggle CLI calls
- Mock `KaggleApi()` for API calls

**Key Test Cases**:
- GPU detection (CUDA, MPS, none)
- Compute planning (all values, fallback, strict mode)
- LocalRunner (CPU, GPU available, GPU not available)
- KernelPackager (template rendering, secret detection)
- KernelManager (push, poll, download, timeout)
- KaggleNotebookRunner (full flow, errors)
- CLI parsing (all flags, validation, deprecation)

### Integration Tests

**Target**: Full pipeline for each compute mode

**Test Cases**:
1. Local CPU: Download → analyze → train → predict → validate
2. Local GPU: Same as CPU, but with GPU detection mocked
3. Kaggle GPU: Generate kernel package (mocked push)
4. Dry-run: All modes should work without network actions

**Mocking Strategy**:
- Use real files for data/artifacts
- Mock Kaggle API calls (don't actually push kernels)
- Mock GPU detection as needed

### Manual Tests

**Required for sign-off**:
1. ✅ Local CPU on Titanic (real run)
2. ✅ Local GPU on Titanic (if GPU available)
3. ✅ Kaggle GPU on Titanic (actual kernel push to private kernel)
4. ✅ Verify no secrets in pushed kernel
5. ✅ Verify kernel runs successfully on Kaggle
6. ✅ Verify submission downloaded and validated
7. ✅ Verify ledger records kernel_id

---

## Risk Mitigation

### High-Risk Areas

1. **GPU detection fails on some platforms**
   - Mitigation: Extensive testing on different platforms
   - Fallback: Always allow --compute local_cpu

2. **Kaggle CLI changes or breaks**
   - Mitigation: Wrap all CLI calls, add version checks
   - Fallback: Clear error messages with upgrade instructions

3. **Kernel quota limits hit during testing**
   - Mitigation: Use mocks for most tests
   - Manual tests: Only 1-2 actual pushes needed

4. **Template rendering bugs**
   - Mitigation: Comprehensive unit tests for templates
   - Validation: Test with various strategies

5. **GPU parameters incorrect for some models**
   - Mitigation: Test all 3 GBDT libraries (LightGBM, CatBoost, XGBoost)
   - Documentation: Clear warnings for MPS limitations

### Mitigation Strategies

- **Extensive mocking**: Don't rely on Kaggle availability for tests
- **Clear error messages**: Include remediation steps for all errors
- **Timeouts**: On all network operations and kernel polling
- **Validation**: At every stage (metadata, package, outputs)
- **Incremental rollout**: Start with tabular only, expand later
- **Backward compatibility**: Keep old flags working with deprecation warnings

---

## Rollout Plan

### Week 7-8: Internal Testing

- Test with maintainer accounts only
- Private kernels only
- Limited to Titanic competition
- Focus: Catch critical bugs

### Week 9-10: Beta Testing

- Open to volunteers (GitHub issue for beta signup)
- Still private kernels
- Test on various competitions
- Collect feedback and bug reports

### Week 11+: General Availability

- Document in README
- Announce in release notes
- Monitor for issues
- Prepare hotfixes if needed

---

## Success Metrics

### Functionality

- [ ] `kagglebot run titanic --compute local_cpu` works end-to-end
- [ ] `kagglebot run titanic --compute local_gpu` detects GPU and trains
- [ ] `kagglebot run titanic --compute kaggle_gpu --submit` works (mocked)
- [ ] GPU fallback works (local_gpu → local_cpu when strict=False)
- [ ] All safety guardrails work (dry-run, dedup, validation)

### Quality

- [ ] Test coverage >80% for new code
- [ ] All tests pass in CI
- [ ] No regressions in existing functionality
- [ ] Documentation complete and accurate
- [ ] Clear error messages for all failures

### Security

- [ ] No secrets in kernel code
- [ ] No secrets in kernel packages
- [ ] enable_internet defaults to false
- [ ] Rules acceptance required (no automation)
- [ ] All subprocess calls secure (no shell injection)

### Performance

- [ ] GPU training 2-5x faster than CPU (typical)
- [ ] Kernel polling efficient (exponential backoff)
- [ ] Template rendering <1s
- [ ] Package generation <5s

---

## Future Enhancements

After stable release:

1. **Auto-accelerator selection**: Analyze competition type and auto-select compute
2. **Kernel reuse**: Cache and reuse unchanged kernels
3. **Multi-kernel ensembles**: Run multiple kernels and blend predictions
4. **Streaming logs**: Show real-time kernel output
5. **Cost tracking**: Track compute hours used on Kaggle
6. **Notebook format**: Support .ipynb in addition to script
7. **Remote GPUs**: Support other cloud providers (Colab, AWS, etc.)

---

## Reference

**Related Documents**:
- SPEC_COMPUTE.md: CLI flags, exit codes, artifact layout
- ARCHITECTURE_COMPUTE.md: Module boundaries, runner interface
- DESIGN_NOTEBOOK_RUNNER.md: Detailed notebook runner design
- TASKS_NOTEBOOK_RUNNER.md: Alternative task breakdown (more granular)

**Key Decisions**:
- Single `--compute` flag (not separate --runner/--accelerator)
- Fallback by default (strict mode opt-in)
- GPU detection via PyTorch (CUDA/MPS)
- Submission always local (never from kernel)
- Rules acceptance always manual

---

## Appendix: Task Dependency Order

For implementers, tasks should be done in this order:

**Week 1**: C001-C020 (Foundation)
**Week 2**: C021-C047 (Runner Interface)
**Week 3**: C048-C072 (Kernel Package)
**Week 4**: C073-C090 (Kernel Manager)
**Week 5**: C091-C105 (KaggleNotebookRunner)
**Week 6**: C106-C123 (CLI Integration)
**Week 7**: C124-C140 (Documentation)

**Total**: 140 tasks over 7 weeks
