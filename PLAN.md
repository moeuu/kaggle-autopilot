# Implementation Plan

## Overview

This plan breaks down the implementation into phases, each delivering working functionality. Each phase builds on the previous one, ensuring the system remains functional and testable throughout development.

## Phase 0: Foundation (Current State → Production Ready)

**Goal**: Solidify existing MVP, establish testing infrastructure, and prepare for automation.

### Tasks

#### 0.1 Code Quality & Testing
- [x] Set up pytest with comprehensive test coverage
- [x] Add type hints throughout codebase
- [x] Configure Ruff for linting and formatting
- [ ] Add pre-commit hooks (ruff, pyright, tests)
- [ ] Set up GitHub Actions CI pipeline
  - Run tests on Python 3.11, 3.12, 3.13
  - Lint and type-check
  - End-to-end test on Titanic
- [ ] Achieve 80%+ test coverage

#### 0.2 Configuration System
- [ ] Implement `Config` dataclass with full schema
- [ ] Add TOML config file support (load/save)
- [ ] Create `config/default.toml` with sensible defaults
- [ ] Support config inheritance (global → competition → CLI)
- [ ] Add config validation (pydantic or dataclasses)

#### 0.3 Foundation Modules
- [ ] Enhance `kaggle_api.py`:
  - Better error handling and retries
  - Progress bars for downloads
  - Test for rules acceptance
- [ ] Implement `paths.py` with full artifact layout
- [ ] Add `hashing.py` with submission hash computation
- [ ] Create `logger.py` for structured logging (JSON + text)

#### 0.4 Existing MVP Enhancements
- [ ] Refactor `bootstrap.py` to use new paths and config
- [ ] Update `validation.py` for strict schema checks
- [ ] Enhance `history.py` with JSONL ledger format
- [ ] Add rate limiting to submission ledger

**Deliverables**:
- ✅ All tests pass
- ✅ CI pipeline green
- ✅ Config system working
- ✅ Foundation modules complete

---

## Phase 1: Competition Analyzer (Non-Interactive Intelligence)

**Goal**: Implement automated competition analysis without user input.

### Tasks

#### 1.1 Competition Type Detection
- [ ] Create `analyzer/detector.py`:
  - Detect tabular (CSV files)
  - Detect image (image directories)
  - Detect text (text files, NLP indicators)
  - Detect time-series (datetime columns)
  - Detect code competitions (kernels-only)
- [ ] Heuristics for task type (classification vs regression)
  - Check target dtype in train.csv
  - Analyze sample_submission values
- [ ] Unit tests for each competition type

#### 1.2 Schema Inference
- [ ] Create `analyzer/schema.py`:
  - Parse train.csv, test.csv, sample_submission.csv
  - Identify ID column (common names + first column heuristic)
  - Identify target columns (sample_submission - ID columns)
  - Classify feature types (numeric, categorical, datetime, text)
  - Detect missing values and outliers
  - Compute basic statistics
- [ ] Handle edge cases:
  - Multi-target competitions
  - No clear ID column
  - Complex submission formats

#### 1.3 Rules and Constraints (Conservative Parsing)
- [ ] Create `analyzer/rules_parser.py`:
  - Fetch competition overview via API (if available)
  - Parse evaluation metric from description
  - Conservative defaults if metric unclear
  - Detect external data policy (default: not allowed)
  - Detect pretrained model policy (default: not allowed)
- [ ] Fallback logic when rules unavailable
- [ ] Log assumptions made

#### 1.4 Strategy Generator
- [ ] Create `analyzer/strategy.py`:
  - Map (competition_type, task_type, dataset_size) → ModelingStrategy
  - Select preprocessing steps
  - Select models from registry
  - Decide CV strategy (K-fold, stratified, time-series split)
  - Set time budget allocation per model
- [ ] Handle special cases:
  - Imbalanced datasets → stratified folds
  - Time-series → TimeSeriesSplit
  - Small datasets → more CV folds
  - Large datasets → fewer folds, fast models

#### 1.5 Integration
- [ ] Create `CompetitionMetadata` dataclass
- [ ] Implement `analyze_competition()` function
- [ ] Add `kagglebot analyze` CLI command
- [ ] Write integration tests with Titanic

**Deliverables**:
- ✅ `kagglebot analyze titanic` outputs full analysis JSON
- ✅ Correctly detects tabular binary classification
- ✅ Generates sensible modeling strategy
- ✅ 100% test coverage for analyzer module

---

## Phase 2: Orchestrator (Pipeline Management)

**Goal**: Implement the main pipeline that coordinates all stages.

### Tasks

#### 2.1 Pipeline State Management
- [ ] Create `PipelineState` dataclass
- [ ] Implement state persistence (JSON)
- [ ] Add checkpoint save/load logic
- [ ] Support for resume from checkpoint

#### 2.2 Pipeline Orchestrator
- [ ] Create `orchestrator.py` with `Pipeline` class
- [ ] Implement stage execution:
  - `_check_rules_acceptance()`
  - `_fetch_data()`
  - `_analyze_competition()`
  - `_train_models()` (stub for now)
  - `_generate_predictions()` (stub for now)
  - `_validate_submission()`
  - `_submit_to_kaggle()`
- [ ] Add error handling and rollback
- [ ] Implement stage checkpointing
- [ ] Add structured logging for each stage

#### 2.3 Safety Guardrails
- [ ] Implement duplicate detection:
  - Compute submission hash (SHA256 of file content)
  - Check against ledger before submitting
- [ ] Implement rate limiting:
  - Check submission timestamps in ledger
  - Enforce max submissions per day
  - Enforce min hours between submissions
- [ ] Add `--force` flag to override guardrails (with warning)

#### 2.4 CLI Integration
- [ ] Implement `kagglebot run` command
- [ ] Parse competition URL/slug
- [ ] Pass flags to orchestrator
- [ ] Handle exit codes properly
- [ ] Add Rich progress bars and status display

#### 2.5 Testing
- [ ] Unit tests for each pipeline stage
- [ ] Integration test: full pipeline (without training)
- [ ] Test resume functionality
- [ ] Test error handling and exit codes

**Deliverables**:
- ✅ `kagglebot run titanic` executes pipeline (except training)
- ✅ Downloads data
- ✅ Analyzes competition
- ✅ Validates existing submission
- ✅ Respects rate limits and duplicates
- ✅ Supports `--resume`

---

## Phase 3: Training Engine - Tabular (Production-Grade Models)

**Goal**: Implement serious tabular ML pipeline, not toy baselines.

### Tasks

#### 3.1 Model Interface
- [ ] Create `training/models/base.py` with `BaseModel` ABC
- [ ] Define interface:
  - `fit(X, y)`
  - `predict(X)`
  - `predict_proba(X)` (for classification)
  - `save(path)`
  - `load(path)` (classmethod)
  - `get_feature_importance()` (optional)

#### 3.2 Preprocessing Pipeline
- [ ] Create `training/preprocessor.py`:
  - Handle missing values (median for numeric, mode for categorical)
  - Categorical encoding (one-hot for low cardinality, target encoding for high)
  - Feature scaling (optional, configurable)
  - Datetime feature extraction
  - Text feature vectorization (TF-IDF for short text columns)
  - Feature selection (optional, based on importance)
- [ ] Make preprocessing strategy data-driven (based on schema analysis)
- [ ] Save fitted preprocessor for test-time application

#### 3.3 Model Implementations
- [ ] Linear models (`training/models/linear.py`):
  - `RidgeModel` (regression)
  - `LogisticRegressionModel` (classification)
- [ ] GBDT models (`training/models/gbdt.py`):
  - `LightGBMModel`
  - `CatBoostModel`
  - `XGBoostModel` (optional)
- [ ] Ensemble (`training/models/stacking.py`):
  - `StackedEnsemble` with meta-learner
  - Use out-of-fold predictions from base models

#### 3.4 Cross-Validation
- [ ] Create `training/cv.py`:
  - K-Fold CV with stratification for classification
  - TimeSeriesSplit for time-series data
  - Generate out-of-fold (OOF) predictions for stacking
  - Compute CV score with appropriate metric
  - Parallel fold execution (joblib)

#### 3.5 Metrics
- [ ] Create `training/metrics.py`:
  - Classification: accuracy, AUC, F1, log-loss
  - Regression: RMSE, MAE, R²
  - Auto-select metric based on task type
  - Support custom metrics from config

#### 3.6 Training Orchestrator
- [ ] Create `training/engine.py` with `TrainingEngine` class:
  - Load and preprocess data
  - Create CV splits
  - Train each model with CV
  - Collect OOF predictions
  - Train stacking ensemble (if enabled)
  - Train final model on full training data
  - Save all models and artifacts
  - Log CV scores and training times

#### 3.7 Time Budget Management
- [ ] Allocate time budget across models
- [ ] Early stopping for iterative models
- [ ] Skip slow models if time running out
- [ ] Prioritize fast models first

#### 3.8 Testing
- [ ] Unit tests for each model class
- [ ] Test preprocessing on various schemas
- [ ] Test CV with different splits
- [ ] Integration test: full training on Titanic
- [ ] Verify model artifacts can be loaded and used

**Deliverables**:
- ✅ Training engine produces multiple trained models
- ✅ CV scores logged for each model
- ✅ Stacking ensemble trained
- ✅ All models saved to artifacts
- ✅ Titanic end-to-end test achieves CV score > 0.80

---

## Phase 4: Prediction and Submission

**Goal**: Generate predictions and submit to Kaggle with full validation.

### Tasks

#### 4.1 Prediction Generation
- [ ] Create `prediction.py`:
  - Load test data
  - Apply fitted preprocessor
  - Load final model (stacked or best single model)
  - Generate predictions
  - Format as submission.csv (match sample_submission)
  - Handle special cases (multi-target, ranking, etc.)

#### 4.2 Enhanced Validation
- [ ] Enhance `validation.py`:
  - Check column names match exactly
  - Check row count matches
  - Check ID alignment (if applicable)
  - Check value ranges (e.g., probabilities in [0,1])
  - Check data types
  - Validate against competition-specific rules

#### 4.3 Submission Manager
- [ ] Enhance `submission.py`:
  - Integrate hash computation
  - Check duplicates in ledger
  - Check rate limits
  - Auto-generate submission message (include CV score)
  - Submit via Kaggle API
  - Record in ledger with Kaggle submission ID
  - Handle submission errors gracefully

#### 4.4 Integration
- [ ] Wire prediction generation into orchestrator
- [ ] Wire submission into orchestrator
- [ ] Add `--submit` and `--dry-run` flags
- [ ] Update exit codes

#### 4.5 Testing
- [ ] Test prediction generation on Titanic
- [ ] Validate submission format
- [ ] Test duplicate detection
- [ ] Test rate limiting
- [ ] End-to-end test with actual submission (manually verify once)

**Deliverables**:
- ✅ `kagglebot run titanic` generates valid submission.csv
- ✅ `kagglebot run titanic --submit` submits to Kaggle
- ✅ Duplicates are detected and prevented
- ✅ Rate limits are enforced
- ✅ All validations pass

---

## Phase 5: Polish and Production Features

**Goal**: Add production-quality features for reliability and usability.

### Tasks

#### 5.1 Resume and Checkpointing
- [ ] Implement full checkpoint/resume logic
- [ ] Handle partial failures gracefully
- [ ] Allow manual intervention and resume

#### 5.2 Monitoring and Logging
- [ ] Add Rich progress bars for all stages
- [ ] Improve terminal output (tables, colors, status)
- [ ] Structured JSON logs for parsing
- [ ] Log rotation and cleanup

#### 5.3 Artifact Management
- [ ] Implement `kagglebot list-runs`
- [ ] Implement `kagglebot show-run`
- [ ] Implement `kagglebot list-submissions`
- [ ] Implement `kagglebot clean` (remove old runs)

#### 5.4 Error Messages and UX
- [ ] Improve all error messages
- [ ] Add remediation hints
- [ ] Pretty-print validation errors
- [ ] Add help text and examples

#### 5.5 Documentation
- [ ] Complete README with examples
- [ ] Add tutorial for Titanic end-to-end
- [ ] Document all config options
- [ ] Add FAQ and troubleshooting guide

#### 5.6 Performance Optimization
- [ ] Parallelize CV folds
- [ ] Cache preprocessed features
- [ ] Optimize data loading (Parquet, chunking)
- [ ] Profile and optimize slow paths

**Deliverables**:
- ✅ Polished UX
- ✅ Comprehensive documentation
- ✅ Fast and efficient execution
- ✅ Production-ready error handling

---

## Phase 6: Extension - Other Competition Types

**Goal**: Support text, image, and time-series competitions.

### Tasks

#### 6.1 Plugin Architecture
- [ ] Define `CompetitionHandler` interface
- [ ] Implement handler registration system
- [ ] Create dispatcher based on competition type

#### 6.2 Text Competitions (Basic)
- [ ] Implement `TextHandler`
- [ ] TF-IDF vectorization
- [ ] Simple models (Logistic Regression, Naive Bayes)
- [ ] Placeholder for deep learning (future)

#### 6.3 Image Competitions (Basic)
- [ ] Implement `ImageHandler`
- [ ] Basic image preprocessing
- [ ] Transfer learning with pretrained models (ResNet, EfficientNet)
- [ ] Requires GPU support

#### 6.4 Time-Series Competitions
- [ ] Implement `TimeSeriesHandler`
- [ ] Time-series CV splits
- [ ] Lag features and rolling statistics
- [ ] Models: ARIMA, LightGBM, LSTM (future)

#### 6.5 Testing
- [ ] Find example competitions for each type
- [ ] End-to-end tests
- [ ] Verify submissions are valid

**Deliverables**:
- ✅ Basic support for text, image, time-series
- ✅ At least one end-to-end test per type
- ✅ Clear "not fully supported" warnings where needed

---

## Phase 7: Advanced Features (Future)

**Goal**: Advanced capabilities for serious usage.

### Features (Prioritize based on demand)

#### 7.1 Hyperparameter Tuning
- [ ] Optuna integration for HPO
- [ ] Auto-tuning within time budget
- [ ] Sensible default hyperparameters

#### 7.2 Feature Engineering
- [ ] Automated feature interactions
- [ ] Polynomial features
- [ ] Target encoding for high-cardinality categoricals
- [ ] Feature selection (importance-based, forward selection)

#### 7.3 Ensemble Strategies
- [ ] Weighted averaging
- [ ] Rank averaging
- [ ] Multi-layer stacking

#### 7.4 Distributed Training
- [ ] Multi-GPU support (for deep learning)
- [ ] Distributed CV (Dask, Ray)

#### 7.5 Experiment Tracking
- [ ] Integration with MLflow or Weights & Biases
- [ ] Track all experiments for comparison

#### 7.6 AutoML Integration
- [ ] H2O AutoML backend option
- [ ] FLAML integration
- [ ] Auto-sklearn for tabular

---

## Milestone Timeline (Rough Estimates)

| Phase | Description | Estimated Effort | Target Date |
|-------|-------------|------------------|-------------|
| 0 | Foundation | 1 week | Week 1 |
| 1 | Competition Analyzer | 1 week | Week 2 |
| 2 | Orchestrator | 1 week | Week 3 |
| 3 | Training Engine (Tabular) | 2 weeks | Week 5 |
| 4 | Prediction and Submission | 1 week | Week 6 |
| 5 | Polish and Production | 1 week | Week 7 |
| **MVP Complete** | **Fully working for tabular** | **7 weeks** | **End of Week 7** |
| 6 | Extension (Other Types) | 2-3 weeks | Week 10 |
| 7 | Advanced Features | Ongoing | As needed |

---

## Success Criteria

### Phase 3 Complete (Tabular MVP)
- [ ] `kagglebot run titanic --submit` works end-to-end
- [ ] No prompts or user interaction (except rules acceptance once)
- [ ] Achieves reasonable CV score (>0.80 for Titanic)
- [ ] Generates valid submission that Kaggle accepts
- [ ] Respects all safety guardrails
- [ ] Comprehensive test coverage (>80%)
- [ ] Clean CI pipeline (all tests pass)
- [ ] Documentation complete

### Phase 6 Complete (Multi-Type Support)
- [ ] Works on at least one competition of each type:
  - Tabular: Titanic
  - Text: Sentiment analysis competition
  - Image: Image classification competition
  - Time-series: Sales forecasting competition
- [ ] Each achieves reasonable baseline score
- [ ] Clear error messages for unsupported features

---

## Risk Management

### High-Risk Areas
1. **Kaggle API changes**: Wrap API calls, add version checks
2. **Competition rule parsing**: Use conservative defaults, log assumptions
3. **Model training failures**: Graceful degradation, fallback to simpler models
4. **Resource constraints**: Time budgets, memory limits, early stopping

### Mitigation Strategies
- Extensive testing on diverse competitions
- Clear logging and error messages
- Fallback strategies at each stage
- User-configurable overrides for edge cases

---

## Dependencies

### Critical Path
```
Phase 0 (Foundation)
    ↓
Phase 1 (Analyzer) + Phase 2 (Orchestrator)  [parallel]
    ↓
Phase 3 (Training Engine)
    ↓
Phase 4 (Prediction & Submission)
    ↓
Phase 5 (Polish)
    ↓
Phase 6 (Extensions)
```

### External Dependencies
- Kaggle API stability
- Python package ecosystem (pandas, sklearn, lightgbm, catboost)
- uv package manager
- GitHub Actions for CI

---

## Post-MVP Roadmap

### Short-term (Next 3 months)
- Support top 10 most popular competition types
- Add more models to registry
- Improve feature engineering
- Better hyperparameter defaults

### Medium-term (6 months)
- Deep learning support (PyTorch/TensorFlow)
- GPU acceleration
- Distributed training
- Web UI for monitoring

### Long-term (1 year+)
- Full AutoML capabilities
- Support for all Kaggle competition types
- Integration with cloud platforms (AWS SageMaker, GCP Vertex AI)
- Commercial features (team collaboration, advanced analytics)
