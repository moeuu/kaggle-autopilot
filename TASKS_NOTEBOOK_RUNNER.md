# Implementation Tasks: Kaggle Notebook Runner

**Priority**: Implement after Phase 3 (Training Engine) is complete

**Dependencies**: Requires working local training engine and validation

---

## Phase N1: Runner Abstraction (Week 1)

### N1.1 Runner Interface
- [ ] **N001**: Create `src/kagglebot/runners/` package
- [ ] **N002**: Create `runners/base.py` with interfaces:
  - `RunContext` dataclass
  - `RunResult` dataclass
  - `Runner` ABC (validate_preconditions, run, cleanup)
- [ ] **N003**: Create `runners/__init__.py` with registry and factory
- [ ] **N004**: Unit tests for data classes

### N1.2 LocalRunner Refactoring
- [ ] **N005**: Create `runners/local.py` with `LocalRunner` class
- [ ] **N006**: Wrap existing `TrainingEngine` in LocalRunner.run()
- [ ] **N007**: Implement `validate_preconditions()` (check disk space, memory)
- [ ] **N008**: Return `RunResult` with proper metadata
- [ ] **N009**: Unit tests for LocalRunner
- [ ] **N010**: Integration test: LocalRunner on Titanic

### N1.3 Orchestrator Integration
- [ ] **N011**: Update `orchestrator.py` to use Runner interface
  - Add `runner_name` parameter to `Pipeline.__init__()`
  - Call `runner.validate_preconditions(ctx)` before execution
  - Call `runner.run(ctx)` instead of direct TrainingEngine
- [ ] **N012**: Update CLI to accept `--runner` flag
- [ ] **N013**: Test LocalRunner via orchestrator
- [ ] **N014**: Verify backward compatibility (existing tests pass)

**Checkpoint**: LocalRunner works via new abstraction, all existing tests pass

---

## Phase N2: Kernel Package Generation (Week 2)

### N2.1 Metadata Generation
- [ ] **N015**: Create `src/kagglebot/kernel_metadata.py`
- [ ] **N016**: Implement `generate_kernel_metadata()` function:
  - Accept: username, slug, accelerator, enable_internet, run_id
  - Return: dict (kernel-metadata.json structure)
  - Validate: gpu/tpu never both true
  - Validate: competition_sources has no "c/" prefix
  - Use lowercase true/false (JSON booleans)
- [ ] **N017**: Unit tests for all accelerator combinations
- [ ] **N018**: Unit tests for edge cases (invalid accelerator, missing username)

### N2.2 Template System
- [ ] **N019**: Add Jinja2 to dependencies: `uv add jinja2`
- [ ] **N020**: Create `src/kagglebot/notebook_templates/` directory
- [ ] **N021**: Create `tabular_script.py.j2` template:
  - Read from `/kaggle/input/{slug}/`
  - Read `plan.json` for strategy
  - Implement preprocessing (from strategy)
  - Train models with CV
  - Generate predictions
  - Save to `/kaggle/working/submission.csv`
  - Save metrics.json
- [ ] **N022**: Create template renderer function
  - `render_kernel_script(template_name, context) -> str`
- [ ] **N023**: Unit tests for template rendering

### N2.3 Plan Serialization
- [ ] **N024**: Add `to_dict()` method to `ModelingStrategy` dataclass
- [ ] **N025**: Create `generate_plan_json(strategy: ModelingStrategy) -> dict`
  - Include: features, target, models, preprocessing, cv_folds
  - Must be JSON-serializable
- [ ] **N026**: Unit tests for plan serialization

### N2.4 Package Assembly
- [ ] **N027**: Create `kernel_packager.py`
- [ ] **N028**: Implement `generate_kernel_package()`:
  - Create kernel directory
  - Write kernel-metadata.json
  - Render and write main.py from template
  - Write plan.json
  - Return package directory path
- [ ] **N029**: Implement `validate_kernel_package()` (secret detection)
- [ ] **N030**: Unit tests for package generation
- [ ] **N031**: Integration test: generate package for Titanic

**Checkpoint**: Can generate valid kernel packages locally

---

## Phase N3: Kernel Lifecycle Management (Week 3)

### N3.1 KernelManager Class
- [ ] **N032**: Create `src/kagglebot/kernel_manager.py`
- [ ] **N033**: Create `KernelStatus` dataclass
- [ ] **N034**: Implement `KernelManager` class:
  - `__init__(username, config)`
  - `push_kernel(kernel_dir) -> kernel_id`
  - `get_status(kernel_id) -> KernelStatus`
  - `poll_until_complete(kernel_id, timeout) -> KernelStatus`
  - `download_outputs(kernel_id, dest) -> Path`
  - `delete_kernel(kernel_id)` (log warning for manual deletion)

### N3.2 Kaggle CLI Integration
- [ ] **N035**: Implement `push_kernel()`:
  - Run `kaggle kernels push -p <dir>`
  - Parse kernel_id from output
  - Handle errors (credentials, quota, invalid metadata)
- [ ] **N036**: Implement `get_status()`:
  - Run `kaggle kernels status <kernel_id>`
  - Parse status from output
  - Map to KernelStatus dataclass
- [ ] **N037**: Implement `poll_until_complete()`:
  - Loop with exponential backoff
  - Check for complete/error/cancelled states
  - Enforce timeout
  - Log progress
- [ ] **N038**: Implement `download_outputs()`:
  - Run `kaggle kernels output <kernel_id> -p <dest>`
  - Verify download success
  - Return output directory path

### N3.3 Error Handling
- [ ] **N039**: Handle all CLI error cases:
  - Credentials not found
  - Network errors (retry with backoff)
  - Quota exceeded
  - Invalid kernel metadata
  - Timeout
- [ ] **N040**: Add structured logging for all operations
- [ ] **N041**: Unit tests with mocked subprocess calls
- [ ] **N042**: Integration test with mock Kaggle CLI

**Checkpoint**: KernelManager can push, poll, download (with mocks)

---

## Phase N4: KaggleNotebookRunner Implementation (Week 4)

### N4.1 Core Runner Logic
- [ ] **N043**: Create `runners/kaggle_notebook.py`
- [ ] **N044**: Create `KaggleNotebookRunner` class (extends `Runner`)
- [ ] **N045**: Implement `validate_preconditions()`:
  - Check Kaggle CLI installed
  - Check credentials exist (~/.kaggle/)
  - Detect username (from config or auto-detect)
  - Check competition rules accepted
  - Validate accelerator choice
- [ ] **N046**: Implement `run()` method (main logic):
  1. Generate kernel package
  2. Push kernel
  3. Poll until complete
  4. Download outputs
  5. Locate submission.csv
  6. Return RunResult

### N4.2 Submission Handling
- [ ] **N047**: Implement `_locate_submission()`:
  - Search downloaded outputs for submission.csv
  - Validate file exists and is non-empty
  - Copy to run artifacts directory
- [ ] **N048**: Implement `_parse_metrics()`:
  - Load metrics.json from kernel outputs (if exists)
  - Extract CV scores
  - Add to RunResult.summary

### N4.3 Cleanup and Recovery
- [ ] **N049**: Implement `cleanup()`:
  - Save kernel metadata to artifacts
  - Optionally delete kernel (configurable)
- [ ] **N050**: Add resume support:
  - Check if kernel already exists for run_id
  - Skip push if kernel already complete
  - Download and continue from there

### N4.4 Testing
- [ ] **N051**: Unit tests for KaggleNotebookRunner (mocked KernelManager)
- [ ] **N052**: Integration test with mock CLI
- [ ] **N053**: Test error handling (timeout, no submission, etc.)

**Checkpoint**: KaggleNotebookRunner complete (unit/integration tests pass)

---

## Phase N5: Accelerator Selection (Week 5)

### N5.1 Auto-Detection Logic
- [ ] **N054**: Create `runners/accelerator.py`
- [ ] **N055**: Implement `select_accelerator()` function:
  - Accept: metadata, strategy, requested
  - Return: "gpu" | "tpu" | "none"
  - Implement heuristics from design doc
- [ ] **N056**: Add decision logging (structured logs)
- [ ] **N057**: Unit tests for all heuristic branches

### N5.2 Configuration
- [ ] **N058**: Add accelerator config to `config/default.toml`:
  - `auto_accelerator_for_tabular`
  - `auto_accelerator_for_image`
  - `auto_accelerator_for_text`
  - `auto_accelerator_for_timeseries`
- [ ] **N059**: Wire accelerator selection into KaggleNotebookRunner
- [ ] **N060**: Test auto-selection for various competition types

**Checkpoint**: Accelerator auto-selection works correctly

---

## Phase N6: CLI Integration (Week 6)

### N6.1 CLI Flags
- [ ] **N061**: Add flags to `kagglebot run` command:
  - `--runner {local,kaggle_notebook}` (default: local)
  - `--accelerator {none,gpu,tpu,auto}` (default: auto for notebook, none for local)
  - `--enable-internet` (flag, default: false)
  - `--kaggle-username TEXT`
  - `--kernel-slug TEXT`
  - `--max-kernel-runtime MINUTES` (default: 120)
- [ ] **N062**: Add validation:
  - Require --kaggle-username if not auto-detectable
  - Warn if --enable-internet is set
  - Validate --max-kernel-runtime within Kaggle limits

### N6.2 Orchestrator Updates
- [ ] **N063**: Pass runner_name from CLI to Pipeline
- [ ] **N064**: Pass accelerator and other notebook-specific config
- [ ] **N065**: Update dry-run mode to show kernel metadata preview
- [ ] **N066**: Add --runner to help text and examples

### N6.3 Error Messages
- [ ] **N067**: Add exit code 3 for missing username
- [ ] **N068**: Add exit code 5 for kernel timeout
- [ ] **N069**: Add exit code 7 for missing submission in outputs
- [ ] **N070**: Improve all error messages per design doc
- [ ] **N071**: Add remediation hints to each error

**Checkpoint**: Full CLI integration, all flags work

---

## Phase N7: Documentation and Polish (Week 7)

### N7.1 Documentation
- [ ] **N072**: Update README.md:
  - Add Kaggle Notebook runner section
  - Show usage examples
  - Document --runner and --accelerator flags
- [ ] **N073**: Create TUTORIAL_NOTEBOOK_RUNNER.md:
  - Step-by-step walkthrough
  - Titanic example with GPU
  - Troubleshooting guide
- [ ] **N074**: Update CLAUDE.md:
  - Add runner architecture guidance
  - Add security rules for kernels
- [ ] **N075**: Update AGENTS.md:
  - Add notebook runner implementation notes
- [ ] **N076**: Update SPEC.md:
  - Document new CLI flags
  - Add kernel artifacts layout
  - Update exit codes

### N7.2 Configuration Defaults
- [ ] **N077**: Create sensible config defaults:
  - Poll interval: 30 seconds
  - Max runtime: 120 min (CPU), 540 min (GPU), 180 min (TPU)
  - enable_internet: false
  - kernel_is_private: true
- [ ] **N078**: Document all config options in CONFIG.md

### N7.3 Testing and Validation
- [ ] **N079**: End-to-end test: Titanic with --runner kaggle_notebook --accelerator gpu
- [ ] **N080**: Verify dry-run doesn't push kernel
- [ ] **N081**: Verify rules acceptance check works
- [ ] **N082**: Verify no secrets in kernel package
- [ ] **N083**: Test submission validation after kernel run
- [ ] **N084**: Test ledger recording with kernel_id
- [ ] **N085**: Achieve 80%+ test coverage for new code

### N7.4 Security Audit
- [ ] **N086**: Review all subprocess calls (no shell=True)
- [ ] **N087**: Verify secret detection works
- [ ] **N088**: Test with various credential scenarios
- [ ] **N089**: Verify enable_internet defaults to false
- [ ] **N090**: Check kernel package for any leaks

**Checkpoint**: Production-ready, documented, tested

---

## Phase N8: Advanced Features (Future)

### N8.1 Notebook Format Support
- [ ] **N091**: Create .ipynb template (Jupyter notebook)
- [ ] **N092**: Add --kernel-type {script,notebook} flag
- [ ] **N093**: Render notebook JSON with code cells
- [ ] **N094**: Test notebook execution on Kaggle

### N8.2 Kernel Logs Streaming
- [ ] **N095**: Implement `KernelManager.stream_logs(kernel_id)`
- [ ] **N096**: Show real-time progress in terminal
- [ ] **N097**: Add --follow flag to stream logs

### N8.3 Kernel Reuse
- [ ] **N098**: Check if kernel already exists (by slug + hash)
- [ ] **N099**: Skip push if code unchanged
- [ ] **N100**: Add --reuse-kernel flag

### N8.4 Multi-Kernel Orchestration
- [ ] **N101**: Support multiple kernels per run
- [ ] **N102**: Ensemble predictions from multiple kernels
- [ ] **N103**: Parallel kernel execution

---

## Testing Checklist

Before merging each phase:

### Unit Tests
- [ ] All new classes have >80% coverage
- [ ] All error paths tested
- [ ] All edge cases covered
- [ ] Mocks used for Kaggle CLI calls

### Integration Tests
- [ ] Full flow with mocked CLI
- [ ] Dry-run mode (no side effects)
- [ ] Error handling (timeout, failures)
- [ ] Backward compatibility (local runner)

### Manual Tests (Requires Kaggle Account)
- [ ] Push kernel to Kaggle (private)
- [ ] Verify kernel runs successfully
- [ ] Download outputs
- [ ] Validate submission locally
- [ ] Submit from local machine
- [ ] Check ledger entry

---

## Success Metrics

After completing Phase N7:

### Functionality
- [ ] `kagglebot run titanic --runner kaggle_notebook --accelerator gpu --submit` works end-to-end
- [ ] Kernel executes on Kaggle GPU
- [ ] Submission downloaded and validated locally
- [ ] Submission submitted from local machine
- [ ] Ledger records kernel_id and metadata
- [ ] All safety guardrails work (dry-run, dedup, rate limits)

### Quality
- [ ] Test coverage > 80% for new code
- [ ] All tests pass in CI
- [ ] No regressions in existing functionality
- [ ] Documentation complete and accurate
- [ ] Clear error messages for all failures

### Security
- [ ] No secrets in kernel code
- [ ] No secrets in kernel package
- [ ] enable_internet defaults to false
- [ ] Rules acceptance required (no automation)
- [ ] All subprocess calls secure (no shell injection)

---

## Implementation Order

**Critical Path**:
```
N1 (Runner Abstraction)
  ↓
N2 (Kernel Package) + N3 (Kernel Manager)  [parallel]
  ↓
N4 (KaggleNotebookRunner)
  ↓
N5 (Accelerator) + N6 (CLI)  [parallel]
  ↓
N7 (Documentation)
```

**Recommended Pace**:
- Week 1: Complete N1 (foundation)
- Week 2-3: Complete N2-N3 (kernel generation and management)
- Week 4: Complete N4 (runner implementation)
- Week 5: Complete N5 (accelerator selection)
- Week 6: Complete N6 (CLI integration)
- Week 7: Complete N7 (docs and polish)

**Total: 7 weeks**

---

## Risk Mitigation

### High-Risk Areas
1. **Kaggle CLI changes**: Wrap all CLI calls, add version checks
2. **Kernel quota limits**: Handle quota errors gracefully
3. **Network timeouts**: Robust retry logic
4. **Template bugs**: Comprehensive testing with various strategies

### Mitigation Strategies
- Extensive mocking in tests (don't rely on Kaggle availability)
- Clear error messages with remediation steps
- Timeouts on all network operations
- Validation at every stage (metadata, package, outputs)
- Incremental rollout (start with tabular only)

---

## Dependencies

### New Python Packages
```bash
# Add to pyproject.toml
uv add jinja2  # Template rendering
```

### External Dependencies
- Kaggle CLI (must be installed and on PATH)
- Kaggle credentials (~/.kaggle/)
- Active internet connection (for push/poll)

---

## Rollout Plan

### Phase 1: Internal Testing (Week 7-8)
- Test with maintainer accounts only
- Private kernels only
- Limited to Titanic competition

### Phase 2: Beta (Week 9-10)
- Open to volunteers (GitHub issue for beta signup)
- Still private kernels
- Collect feedback and bug reports

### Phase 3: General Availability (Week 11+)
- Document in README
- Announce in release notes
- Monitor for issues

---

## Future Enhancements

After MVP is stable:
- **Notebook format**: Support .ipynb for interactive development
- **Kernel logs streaming**: Real-time progress updates
- **Kernel reuse**: Cache and reuse unchanged kernels
- **GPU quota tracking**: Warn before hitting limits
- **Multi-kernel ensembles**: Run multiple kernels and blend predictions
- **Web UI**: Monitor kernel status via dashboard
- **Cost tracking**: Track compute hours used
