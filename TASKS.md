# Implementation Task Checklist

This is an ordered, dependency-aware checklist for implementing the production-grade Kaggle Autopilot. Each task is small, testable, and builds on previous tasks.

**Important**: Follow tasks in order. Don't skip ahead - later tasks depend on earlier ones.

---

## Phase 0: Foundation (Weeks 1)

### 0.1 Testing Infrastructure
- [ ] **T001**: Add `pytest-cov` to dev dependencies
- [ ] **T002**: Create `.github/workflows/ci.yml` for GitHub Actions
  - Run tests on Python 3.11, 3.12, 3.13
  - Run `ruff check` and `ruff format --check`
  - Run `pyright` (if using)
  - Generate coverage report
- [ ] **T003**: Add pre-commit hooks config (`.pre-commit-config.yaml`)
  - ruff (lint and format)
  - trailing whitespace
  - end of file fixer
- [ ] **T004**: Achieve 80%+ test coverage on existing code

### 0.2 Configuration System
- [ ] **T005**: Create `src/kagglebot/config.py` with `Config` dataclass
  - Use attrs or pydantic for validation
  - Include all fields from SPEC.md config schema
  - Add `from_file(path)` classmethod (TOML support)
  - Add `save(path)` method
  - Add `merge(other: Config)` for inheritance
- [ ] **T006**: Create `config/default.toml` with sensible defaults
- [ ] **T007**: Write unit tests for config loading, merging, validation
- [ ] **T008**: Update `cli.py` to load config from file

### 0.3 Foundation Modules
- [ ] **T009**: Create `src/kagglebot/logger.py`
  - Structured logger class (JSON + text formats)
  - Log levels (DEBUG, INFO, WARNING, ERROR)
  - Log to file and/or console (configurable)
  - Sanitize sensitive fields (see SECURITY.md)
- [ ] **T010**: Enhance `src/kagglebot/kaggle_cli.py` (rename to `kaggle_api.py`)
  - Add retry logic with exponential backoff
  - Add progress bars for downloads (Rich)
  - Add `check_rules_acceptance(slug)` method
  - Improve error messages
  - Unit tests with mocked API
- [ ] **T011**: Create `src/kagglebot/hashing.py`
  - `compute_file_hash(path: Path, algorithm="sha256") -> str`
  - Unit tests
- [ ] **T012**: Update `src/kagglebot/paths.py`
  - Implement full artifact layout from SPEC.md
  - Add `safe_path()` function (see SECURITY.md)
  - Unit tests for path construction and safety

### 0.4 Existing MVP Enhancements
- [ ] **T013**: Refactor `bootstrap.py` to use new `Config` and `paths`
- [ ] **T014**: Update `validation.py`
  - Add strict schema validation (column names, row count, ID alignment)
  - Add value range checks
  - Better error messages with diffs
  - Unit tests with various invalid submissions
- [ ] **T015**: Enhance `history.py`
  - Use JSONL format for `ledger.jsonl`
  - Add rate limit checking methods
  - Add duplicate hash checking
  - Unit tests
- [ ] **T016**: Run full test suite and achieve >80% coverage

**Checkpoint**: All Phase 0 tests pass, CI is green

---

## Phase 1: Competition Analyzer (Week 2)

### 1.1 Type Detection
- [ ] **T017**: Create `src/kagglebot/analyzer/` package
- [ ] **T018**: Create `analyzer/types.py` with dataclasses:
  - `CompetitionMetadata`
  - `ModelingStrategy`
  - `SchemaInfo`
- [ ] **T019**: Create `analyzer/detector.py`
  - `detect_competition_type(data_dir) -> Literal["tabular", "text", "image", "timeseries", "code", "unknown"]`
  - Heuristics: check for CSV files, image dirs, text files, date columns
  - Unit tests with mock directory structures
- [ ] **T020**: Add `detect_task_type(train_df, sample_df) -> Literal["classification", "regression", "ranking", "other"]`
  - Check target dtype (int → classification, float → regression)
  - Check sample_submission values
  - Unit tests with various train/sample pairs

### 1.2 Schema Inference
- [ ] **T021**: Create `analyzer/schema.py`
  - `infer_schema(data_dir: Path) -> SchemaInfo`
  - Load train.csv, test.csv, sample_submission.csv
  - Identify ID column (heuristics: "id", first column, etc.)
  - Identify target columns (sample - ID columns)
  - Classify features (numeric, categorical, datetime, text)
  - Compute statistics (missing %, unique values, etc.)
- [ ] **T022**: Handle edge cases:
  - Multi-target competitions
  - No clear ID column
  - Complex submission formats
- [ ] **T023**: Unit tests with Titanic and synthetic competitions

### 1.3 Rules Parsing
- [ ] **T024**: Create `analyzer/rules_parser.py`
  - `parse_rules(slug: str) -> dict` (conservative parsing)
  - Fetch competition metadata via Kaggle API (if available)
  - Extract evaluation metric (heuristics + API)
  - Detect external data policy (default: not allowed)
  - Detect pretrained model policy (default: not allowed)
  - Log all assumptions made
- [ ] **T025**: Implement fallback logic for missing rules
- [ ] **T026**: Unit tests (mock API responses)

### 1.4 Strategy Generation
- [ ] **T027**: Create `analyzer/strategy.py`
  - `generate_strategy(metadata: CompetitionMetadata, config: Config) -> ModelingStrategy`
  - Select preprocessing steps based on schema
  - Select models from registry based on (type, task, dataset size)
  - Set CV strategy (K-fold, stratified, timeseries split)
  - Allocate time budget across models
- [ ] **T028**: Handle special cases:
  - Imbalanced datasets → stratified folds
  - Time-series → TimeSeriesSplit
  - Small datasets → more folds
  - Large datasets → fewer folds, fast models
- [ ] **T029**: Unit tests with various metadata combinations

### 1.5 Integration
- [ ] **T030**: Create `analyzer/__init__.py` with `analyze_competition()` function
- [ ] **T031**: Add `kagglebot analyze` CLI command
- [ ] **T032**: Integration test: `kagglebot analyze titanic` outputs valid JSON
- [ ] **T033**: Verify analysis correctness for Titanic (tabular, binary classification)

**Checkpoint**: Analyzer module complete, Titanic analyzed correctly

---

## Phase 2: Orchestrator (Week 3)

### 2.1 Pipeline State
- [ ] **T034**: Create `src/kagglebot/orchestrator.py`
- [ ] **T035**: Define `PipelineState` dataclass
  - run_id (UUID), slug, stage, timestamps, checkpoints
  - `save(path)` and `load(path)` methods (JSON)
- [ ] **T036**: Unit tests for state persistence

### 2.2 Pipeline Stages
- [ ] **T037**: Create `Pipeline` class in `orchestrator.py`
- [ ] **T038**: Implement `_check_rules_acceptance()` stage
  - Call `kaggle_api.check_rules_acceptance()`
  - If not accepted: print rules URL, exit code 2
  - Update state checkpoint
- [ ] **T039**: Implement `_fetch_data()` stage
  - Call `kaggle_api.competition_download_files()`
  - Extract archives
  - Verify downloads
  - Update state checkpoint
- [ ] **T040**: Implement `_analyze_competition()` stage
  - Call `analyzer.analyze_competition()`
  - Save `competition_analysis.json`
  - Update state checkpoint
- [ ] **T041**: Implement `_validate_submission()` stage
  - Call existing `validation.validate_submission()`
  - Compute hash
  - Check for duplicates in ledger
  - Update state checkpoint
- [ ] **T042**: Add stub stages (implement in Phase 3/4):
  - `_train_models()` → placeholder (return fake model path)
  - `_generate_predictions()` → placeholder (copy sample_submission)
  - `_submit_to_kaggle()` → placeholder (log only)

### 2.3 Safety Guardrails
- [ ] **T043**: Implement duplicate detection in `_validate_submission()`
  - Compute hash, check ledger
  - If duplicate: log warning, exit code 8
  - Allow `--force` to override
- [ ] **T044**: Implement rate limiting
  - Check timestamps in ledger
  - Enforce max submissions per day
  - Enforce min hours between submissions
  - Exit code 9 if exceeded
  - Allow `--force` to override (with warning)
- [ ] **T045**: Add `--force` flag handling
  - Log warnings when safety checks are bypassed

### 2.4 CLI Integration
- [ ] **T046**: Refactor `cli.py` to use new orchestrator
- [ ] **T047**: Implement `kagglebot run` command
  - Parse competition URL/slug (see SECURITY.md for validation)
  - Load config (global → competition-specific → CLI flags)
  - Create `Pipeline` instance
  - Call `pipeline.execute(submit=submit_flag)`
  - Handle exit codes properly
- [ ] **T048**: Add Rich progress bars and status display
- [ ] **T049**: Add `--resume RUN_ID` flag
  - Load state from previous run
  - Skip completed stages
  - Continue from checkpoint

### 2.5 Testing
- [ ] **T050**: Unit tests for each pipeline stage
- [ ] **T051**: Integration test: full pipeline on Titanic (without training)
  - Should download, analyze, validate (with placeholder submission)
- [ ] **T052**: Test resume functionality (interrupt and resume)
- [ ] **T053**: Test error handling and exit codes

**Checkpoint**: Pipeline orchestrates stages, Titanic runs end-to-end (except training)

---

## Phase 3: Training Engine (Weeks 4-5)

### 3.1 Model Interface
- [ ] **T054**: Create `src/kagglebot/training/` package
- [ ] **T055**: Create `training/models/base.py` with `BaseModel` ABC
  - Abstract methods: `fit()`, `predict()`, `save()`, `load()`
  - Optional: `predict_proba()`, `get_feature_importance()`
- [ ] **T056**: Unit tests for base model interface

### 3.2 Preprocessing
- [ ] **T057**: Create `training/preprocessor.py`
  - `TabularPreprocessor` class
  - Handle missing values (median for numeric, mode for categorical)
  - Categorical encoding (one-hot for low cardinality, target for high)
  - Feature scaling (optional, configurable)
  - Datetime feature extraction
  - Fit on train, transform train and test
  - Save/load fitted preprocessor
- [ ] **T058**: Make preprocessing data-driven (use schema from analyzer)
- [ ] **T059**: Unit tests with various feature types

### 3.3 Model Implementations
- [ ] **T060**: Create `training/models/linear.py`
  - `RidgeModel` (for regression)
  - `LogisticRegressionModel` (for classification)
  - Implement `BaseModel` interface
  - Unit tests
- [ ] **T061**: Create `training/models/gbdt.py`
  - `LightGBMModel`
  - `CatBoostModel`
  - Sensible hyperparameters (early stopping, etc.)
  - Implement `BaseModel` interface
  - Unit tests
- [ ] **T062**: Create `training/models/__init__.py` with MODEL_REGISTRY
  - `{"ridge": RidgeModel, "logreg": LogisticRegressionModel, "lgbm": LightGBMModel, "catboost": CatBoostModel}`
- [ ] **T063**: Create `training/models/stacking.py`
  - `StackedEnsemble` model
  - Takes base model predictions (OOF) and trains meta-learner
  - Implement `BaseModel` interface
  - Unit tests

### 3.4 Cross-Validation
- [ ] **T064**: Create `training/cv.py`
  - `create_cv_splits(X, y, strategy, n_folds)` → list of (train_idx, val_idx)
  - Support: K-Fold, StratifiedKFold, TimeSeriesSplit
  - `train_with_cv(model, X, y, splits, metric)` → CV scores + OOF predictions
  - Parallel fold execution (joblib)
  - Unit tests

### 3.5 Metrics
- [ ] **T065**: Create `training/metrics.py`
  - Classification: accuracy, AUC-ROC, F1, log-loss
  - Regression: RMSE, MAE, R²
  - `select_metric(task_type, metric_name)` → callable
  - Unit tests

### 3.6 Training Orchestrator
- [ ] **T066**: Create `training/engine.py` with `TrainingEngine` class
- [ ] **T067**: Implement `train()` method:
  1. Load data (train.csv)
  2. Preprocess features (fit preprocessor on train)
  3. Create CV splits
  4. For each model in strategy:
     - Train with CV
     - Collect OOF predictions
     - Log CV scores
  5. If stacking enabled:
     - Train meta-learner on OOF predictions
  6. Train final model(s) on full training data
  7. Save all models and artifacts
  8. Return `ModelArtifacts` dataclass
- [ ] **T068**: Add time budget management
  - Allocate time per model
  - Early stopping for GBDT
  - Skip models if time running out
  - Prioritize fast models first
- [ ] **T069**: Save artifacts:
  - Fitted preprocessor (`models/preprocessor.pkl`)
  - Models for each fold (`models/<model>/fold_<i>.pkl`)
  - Final model (`models/<model>/final.pkl`)
  - OOF predictions (`predictions/<model>_train.npy`)
  - CV results (`cv_results.json`)

### 3.7 Integration
- [ ] **T070**: Wire `TrainingEngine` into orchestrator `_train_models()` stage
- [ ] **T071**: Integration test: train on Titanic, verify models saved
- [ ] **T072**: Verify CV scores are reasonable (>0.75 for Titanic)

**Checkpoint**: Training engine works end-to-end, Titanic achieves CV > 0.80

---

## Phase 4: Prediction and Submission (Week 6)

### 4.1 Prediction Generation
- [ ] **T073**: Create `src/kagglebot/prediction.py`
- [ ] **T074**: Implement `generate_predictions(model_artifacts, test_data, metadata) -> Path`
  - Load test data
  - Apply fitted preprocessor
  - Load final model (stacked or best single)
  - Generate predictions
  - Format as submission.csv (match sample_submission)
  - Save to artifacts
- [ ] **T075**: Handle special cases (multi-target, probabilities, rankings)
- [ ] **T076**: Unit tests

### 4.2 Enhanced Validation
- [ ] **T077**: Enhance `validation.py`:
  - Strict column name matching
  - Row count validation
  - ID alignment check
  - Value range validation (e.g., probabilities in [0,1])
  - Data type validation
- [ ] **T078**: Add detailed error messages with diffs
- [ ] **T079**: Unit tests with various invalid submissions

### 4.3 Submission Manager
- [ ] **T080**: Update `submission.py` (or create if missing):
  - `submit_to_kaggle(submission_path, slug, message, ledger, config)`
  - Check duplicates (hash)
  - Check rate limits
  - Auto-generate message (include CV score)
  - Submit via Kaggle API
  - Parse response and get submission ID
  - Record in ledger
  - Handle errors gracefully
- [ ] **T081**: Unit tests (mock Kaggle API)

### 4.4 Integration
- [ ] **T082**: Wire prediction generation into orchestrator `_generate_predictions()` stage
- [ ] **T083**: Wire submission into orchestrator `_submit_to_kaggle()` stage
- [ ] **T084**: Add `--dry-run` flag (skip actual submission)
- [ ] **T085**: Update all exit codes per SPEC.md

### 4.5 End-to-End Testing
- [ ] **T086**: End-to-end test: `kagglebot run titanic` (no --submit)
  - Should complete all stages
  - Generate valid submission.csv
- [ ] **T087**: Test duplicate detection (run twice, second should be blocked)
- [ ] **T088**: Test rate limiting (submit multiple times rapidly)
- [ ] **T089**: Manual test: `kagglebot run titanic --submit` (verify on Kaggle)

**Checkpoint**: Full pipeline works, Titanic submits to Kaggle successfully

---

## Phase 5: Polish and Production Features (Week 7)

### 5.1 Resume and Checkpointing
- [ ] **T090**: Test resume from each checkpoint
- [ ] **T091**: Handle partial failures gracefully (cleanup on error)
- [ ] **T092**: Add `--clean-on-failure` flag (remove partial artifacts)

### 5.2 Monitoring and Logging
- [ ] **T093**: Add Rich progress bars for all stages
- [ ] **T094**: Improve terminal output:
  - Status dashboard during training
  - Table output for CV scores
  - Colorized error messages
- [ ] **T095**: Structured JSON logs for parsing
- [ ] **T096**: Add log rotation/cleanup logic

### 5.3 Artifact Management
- [ ] **T097**: Implement `kagglebot list-runs <slug>` command
  - Show table of runs with status, scores, timestamps
- [ ] **T098**: Implement `kagglebot show-run <slug> <run_id>` command
  - Display full run details (metadata, CV scores, models, etc.)
- [ ] **T099**: Implement `kagglebot list-submissions <slug>` command
  - Show ledger entries
- [ ] **T100**: Implement `kagglebot clean <slug>` command
  - Remove old runs (keep last N or last N days)
  - Dry-run mode to preview deletions

### 5.4 Error Messages and UX
- [ ] **T101**: Review all error messages
- [ ] **T102**: Add remediation hints to common errors
- [ ] **T103**: Pretty-print validation errors with diffs
- [ ] **T104**: Add examples to help text

### 5.5 Documentation
- [ ] **T105**: Update README.md with full examples
- [ ] **T106**: Create TUTORIAL.md for Titanic walkthrough
- [ ] **T107**: Document all config options (CONFIG.md or in README)
- [ ] **T108**: Create FAQ.md

### 5.6 Performance Optimization
- [ ] **T109**: Parallelize CV folds (use joblib or multiprocessing)
- [ ] **T110**: Cache preprocessed features to Parquet
- [ ] **T111**: Optimize data loading (chunking for large CSVs)
- [ ] **T112**: Profile slow paths and optimize

**Checkpoint**: Production-ready, polished UX, comprehensive docs

---

## Phase 6: Extensions (Weeks 8-10)

### 6.1 Plugin Architecture
- [ ] **T113**: Define `CompetitionHandler` interface
- [ ] **T114**: Implement handler registration and dispatch
- [ ] **T115**: Refactor tabular code into `TabularHandler`

### 6.2 Text Competitions
- [ ] **T116**: Create `TextHandler`
- [ ] **T117**: Implement TF-IDF vectorization
- [ ] **T118**: Add simple models (LogReg, Naive Bayes)
- [ ] **T119**: Test on example text competition

### 6.3 Image Competitions
- [ ] **T120**: Create `ImageHandler`
- [ ] **T121**: Implement image preprocessing
- [ ] **T122**: Add transfer learning (ResNet, EfficientNet)
- [ ] **T123**: Test on example image competition (requires GPU)

### 6.4 Time-Series Competitions
- [ ] **T124**: Create `TimeSeriesHandler`
- [ ] **T125**: Implement time-series CV splits
- [ ] **T126**: Add lag features and rolling statistics
- [ ] **T127**: Test on example time-series competition

**Checkpoint**: Basic support for text, image, time-series

---

## Continuous Tasks

These tasks span all phases:

- [ ] **T128**: Keep tests passing (run `uv run pytest` before each commit)
- [ ] **T129**: Keep linting clean (run `uv run ruff check .`)
- [ ] **T130**: Maintain >80% code coverage
- [ ] **T131**: Update documentation when behavior changes
- [ ] **T132**: Review SECURITY.md before any API or file handling changes
- [ ] **T133**: Add integration tests for new features
- [ ] **T134**: Benchmark performance on Titanic (time to completion)

---

## Success Metrics

After completing Phase 4 (Tabular MVP):
- [ ] `kagglebot run titanic --submit` works end-to-end
- [ ] No user prompts (except initial rules acceptance)
- [ ] Achieves CV score > 0.80 on Titanic
- [ ] Generates valid submission accepted by Kaggle
- [ ] Respects all safety guardrails (dedup, rate limits)
- [ ] Test coverage > 80%
- [ ] CI pipeline green
- [ ] Documentation complete

After completing Phase 6 (Multi-Type):
- [ ] Works on tabular, text, image, time-series competitions
- [ ] Each type has at least one successful end-to-end test
- [ ] Clear error messages for unsupported features

---

## Implementation Notes

1. **Test-Driven Development**: Write tests before implementation
2. **Small PRs**: Each task should be a reviewable unit of work
3. **Dependencies**: Don't skip tasks - later ones depend on earlier ones
4. **Documentation**: Update docs as you go, not at the end
5. **Review**: Self-review against ARCHITECTURE.md, SPEC.md, SECURITY.md before committing
6. **CI**: Keep CI green at all times

Good luck! 🚀
