# Architecture

## Overview

Kagglebot is a production-grade, fully automated CLI tool for Kaggle competition workflows. It implements a pipeline from competition URL to submitted predictions with zero user interaction (except manual rules acceptance).

## Design Principles

1. **Non-interactive**: No prompts during execution. All decisions automated or configured.
2. **Extensible**: Plugin architecture for competition types (tabular, text, image, time-series).
3. **Safe by default**: Deduplication, validation, rate limiting, reproducibility.
4. **Production-grade models**: Not toy baselines - serious GBDT, stacking, CV.
5. **Clear failure modes**: Explicit exit codes, structured logging, artifact preservation.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         kagglebot CLI                            │
│                    (Typer + Rich output)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Module                           │
│  • Coordinates full pipeline                                    │
│  • Manages state transitions                                    │
│  • Enforces safety guardrails                                   │
│  • Logs all decisions                                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┬───────────────┐
         ▼               ▼               ▼               ▼
┌────────────────┐ ┌──────────┐ ┌──────────────┐ ┌─────────────┐
│  Competition   │ │   Data   │ │   Training   │ │ Submission  │
│    Analyzer    │ │ Fetcher  │ │   Engine     │ │  Manager    │
└────────────────┘ └──────────┘ └──────────────┘ └─────────────┘
         │               │               │               │
         ▼               ▼               ▼               ▼
┌────────────────────────────────────────────────────────────────┐
│                      Foundation Layer                           │
│  • Kaggle API wrapper      • Config management                 │
│  • Validation utilities    • Hashing & deduplication           │
│  • Logging & telemetry     • Path management                   │
└────────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. CLI Module (`cli.py`)

**Responsibilities:**
- Parse command-line arguments
- Validate inputs (URLs, paths, flags)
- Dispatch to orchestrator
- Format output for terminal

**Key Functions:**
```python
@app.command()
def run(
    competition: str,  # URL or slug
    submit: bool = False,
    time_budget_minutes: int = 60,
    config: Path | None = None,
) -> None:
    """Run full pipeline: analyze → train → predict → validate → submit."""
```

**Exit Codes:**
- `0`: Success
- `1`: General failure
- `2`: Rules not accepted (user action required)
- `3`: Invalid competition URL/slug
- `4`: Data download failed
- `5`: Training failed
- `6`: Submission validation failed
- `7`: Submission upload failed
- `8`: Duplicate submission (already recorded)

### 2. Orchestrator Module (`orchestrator.py`)

**Responsibilities:**
- Execute pipeline stages in sequence
- Handle failures and rollback
- Enforce guardrails (dedup, rate limits)
- Snapshot config and environment for reproducibility
- Write comprehensive logs

**Key Class:**
```python
class Pipeline:
    def __init__(self, slug: str, config: Config):
        self.slug = slug
        self.config = config
        self.state = PipelineState()
        self.logger = StructuredLogger(slug)

    def execute(self, submit: bool = False) -> PipelineResult:
        """Run full pipeline with checkpointing at each stage."""
        # Stage 1: Check rules acceptance
        self._check_rules_acceptance()

        # Stage 2: Download data
        self._fetch_data()

        # Stage 3: Analyze competition
        metadata = self._analyze_competition()

        # Stage 4: Train models
        model_artifacts = self._train_models(metadata)

        # Stage 5: Generate predictions
        submission_path = self._generate_predictions(model_artifacts)

        # Stage 6: Validate submission
        self._validate_submission(submission_path, metadata)

        # Stage 7: Submit (if requested)
        if submit:
            self._submit_to_kaggle(submission_path)

        return PipelineResult(success=True, submission_path=submission_path)
```

**State Management:**
```python
@dataclass
class PipelineState:
    slug: str
    run_id: str  # UUID for this run
    started_at: datetime
    stage: Literal["init", "fetching", "analyzing", "training", "predicting", "validating", "submitting", "done"]
    checkpoints: dict[str, Path]  # stage -> artifact path

    def save(self, path: Path) -> None:
        """Persist state to enable resume."""
```

### 3. Competition Analyzer Module (`analyzer/`)

**Responsibilities:**
- Detect competition type from files
- Infer schema (features, targets, IDs)
- Extract evaluation metric
- Parse rules for constraints (external data, ensembles, etc.)
- Generate modeling strategy

**Structure:**
```
analyzer/
├── __init__.py
├── detector.py        # Competition type detection
├── schema.py          # Schema inference from CSVs
├── rules_parser.py    # Extract rules (conservative fallback)
├── strategy.py        # Generate modeling plan
└── types.py           # CompetitionMetadata dataclass
```

**Key Types:**
```python
@dataclass
class CompetitionMetadata:
    slug: str
    type: Literal["tabular", "text", "image", "timeseries", "code", "unknown"]
    task: Literal["classification", "regression", "ranking", "other"]

    # Schema
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    id_column: str
    target_columns: list[str]
    feature_columns: list[str]

    # Evaluation
    metric: str  # "accuracy", "auc", "rmse", etc.
    metric_direction: Literal["maximize", "minimize"]

    # Constraints
    allows_external_data: bool
    allows_pretrained_models: bool

    # Recommended strategy
    strategy: ModelingStrategy

@dataclass
class ModelingStrategy:
    preprocessing: list[str]  # ["impute_median", "onehot_categorical"]
    models: list[str]  # ["ridge", "lgbm", "catboost"]
    cv_folds: int
    use_stacking: bool
    time_budget_minutes: int
```

**Detection Logic:**
```python
def detect_competition_type(data_dir: Path) -> str:
    """
    Heuristics:
    - Has train.csv + test.csv → tabular
    - Has train/ test/ dirs with images → image
    - Has .txt files or NLP in description → text
    - Has date/time column → timeseries
    """
```

### 4. Data Fetcher Module (`fetcher.py`)

**Responsibilities:**
- Download competition files via Kaggle API
- Extract ZIP archives
- Verify downloads (checksums if available)
- Detect rules acceptance status

**Key Functions:**
```python
def fetch_competition_data(slug: str, dest: Path) -> FetchResult:
    """
    Download all competition files to dest/<slug>/raw/.

    Returns:
        FetchResult(success=True, files=[...])

    Raises:
        RulesNotAcceptedError: If user hasn't joined competition
        DownloadFailedError: Network or permission issues
    """

def check_rules_acceptance(slug: str) -> bool:
    """
    Try to list competition files. If 403, rules not accepted.
    """
```

### 5. Training Engine (`training/`)

**Responsibilities:**
- Execute modeling strategy
- Train multiple models with CV
- Handle feature engineering
- Save model artifacts
- Log training metrics

**Structure:**
```
training/
├── __init__.py
├── engine.py          # Main training orchestrator
├── preprocessor.py    # Feature engineering pipeline
├── models/
│   ├── __init__.py
│   ├── base.py        # BaseModel interface
│   ├── linear.py      # Ridge, LogisticRegression
│   ├── gbdt.py        # LightGBM, CatBoost
│   └── stacking.py    # Ensemble meta-learner
├── cv.py              # Cross-validation utilities
└── metrics.py         # Metric implementations
```

**Base Model Interface:**
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train model."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Deserialize model."""
```

**Training Orchestrator:**
```python
class TrainingEngine:
    def train(self, metadata: CompetitionMetadata, time_budget: int) -> ModelArtifacts:
        """
        Execute full training pipeline:
        1. Load and preprocess data
        2. Create CV splits (respecting time-series if applicable)
        3. Train each model with CV
        4. Select best model or build ensemble
        5. Train final model on full data
        6. Save artifacts

        Returns model artifacts and CV scores.
        """
```

### 6. Submission Manager (`submission.py`)

**Responsibilities:**
- Generate predictions from trained models
- Validate submission format
- Compute submission hash
- Check for duplicates in ledger
- Enforce rate limits
- Submit via Kaggle API
- Record in ledger

**Key Functions:**
```python
def generate_submission(
    model: BaseModel,
    test_data: pd.DataFrame,
    metadata: CompetitionMetadata,
    output_path: Path,
) -> Path:
    """Generate submission CSV matching sample_submission format."""

def validate_submission_format(
    submission: Path,
    sample: Path,
    metadata: CompetitionMetadata,
) -> ValidationResult:
    """
    Strict validation:
    - Same columns as sample
    - Same row count
    - ID alignment
    - Value ranges (if known)
    """

class SubmissionLedger:
    """Track all submissions with hash-based deduplication."""

    def record(
        self,
        submission_hash: str,
        file_path: Path,
        run_id: str,
        message: str,
        kaggle_submission_id: str | None,
    ) -> None:
        """Append to JSONL ledger."""

    def is_duplicate(self, submission_hash: str) -> bool:
        """Check if this exact submission was already made."""

    def check_rate_limit(self, slug: str) -> tuple[bool, str]:
        """
        Check if we can submit now:
        - Max 5 submissions per day (configurable)
        - Min 1 hour between submissions (configurable)

        Returns (allowed, reason_if_not)
        """
```

### 7. Foundation Layer

#### Kaggle API Wrapper (`kaggle_api.py`)
```python
class KaggleAPI:
    """Thin wrapper around kaggle.api.KaggleApi with better error handling."""

    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()

    def competition_download_files(self, slug: str, dest: Path) -> list[Path]:
        """Download with retries and progress."""

    def competition_submit(
        self,
        slug: str,
        file: Path,
        message: str,
    ) -> SubmitResult:
        """Submit with validation and error wrapping."""

    def competition_list_files(self, slug: str) -> list[str]:
        """List files (test rules acceptance)."""
```

#### Config Management (`config.py`)
```python
@dataclass
class Config:
    # Paths
    data_root: Path = Path("data")
    artifacts_root: Path = Path("artifacts")

    # Training
    default_time_budget_minutes: int = 60
    cv_folds: int = 5
    use_stacking: bool = True
    random_seed: int = 42

    # Submission safety
    max_submissions_per_day: int = 5
    min_hours_between_submissions: float = 1.0

    # Model defaults
    tabular_models: list[str] = field(default_factory=lambda: ["ridge", "lgbm", "catboost"])

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load from TOML/YAML."""

    def save(self, path: Path) -> None:
        """Save current config."""
```

## Data Flow

### Directory Structure
```
kaggle-autopilot/
├── data/
│   └── <slug>/
│       ├── raw/                    # Downloaded files
│       │   ├── train.csv
│       │   ├── test.csv
│       │   └── sample_submission.csv
│       └── processed/              # Preprocessed features (cached)
│           ├── train_features.parquet
│           └── test_features.parquet
├── artifacts/
│   └── <slug>/
│       ├── runs/
│       │   └── <run_id>/
│       │       ├── config.json     # Snapshot of config used
│       │       ├── metadata.json   # CompetitionMetadata
│       │       ├── pipeline_state.json
│       │       ├── logs/
│       │       │   └── pipeline.log
│       │       ├── models/
│       │       │   ├── ridge_fold0.pkl
│       │       │   ├── lgbm_fold0.pkl
│       │       │   ├── catboost_fold0.pkl
│       │       │   └── final_model.pkl
│       │       ├── cv_results.json
│       │       └── submission.csv
│       └── submissions/
│           └── ledger.jsonl        # All submissions log
└── config/
    └── <slug>.toml                 # Competition-specific overrides
```

### Pipeline Execution Flow

```
1. URL/Slug Input
   ↓
2. Parse and validate
   ↓
3. Check if rules accepted (API test)
   ├─ Not accepted → Exit 2 with URL
   └─ Accepted → Continue
   ↓
4. Download data to data/<slug>/raw/
   ↓
5. Analyze competition
   ├─ Detect type (tabular/text/image/etc.)
   ├─ Infer schema
   ├─ Extract metric
   └─ Generate strategy
   ↓
6. Train models
   ├─ Preprocess features
   ├─ CV loop for each model
   ├─ Select best or ensemble
   └─ Train final model on full data
   ↓
7. Generate predictions
   ├─ Load test data
   ├─ Apply preprocessing
   ├─ Predict with final model
   └─ Format as submission.csv
   ↓
8. Validate submission
   ├─ Check format vs sample
   ├─ Compute hash
   └─ Check for duplicate
   ↓
9. If --submit:
   ├─ Check rate limit
   ├─ Submit via Kaggle API
   └─ Record in ledger
   └─ Done
```

## Extension Points

### Adding New Competition Types

Implement the `CompetitionHandler` interface:

```python
class CompetitionHandler(ABC):
    @abstractmethod
    def can_handle(self, metadata: CompetitionMetadata) -> bool:
        """Return True if this handler supports the competition type."""

    @abstractmethod
    def create_strategy(self, metadata: CompetitionMetadata, config: Config) -> ModelingStrategy:
        """Generate modeling strategy for this competition type."""

    @abstractmethod
    def train(self, metadata: CompetitionMetadata, strategy: ModelingStrategy) -> ModelArtifacts:
        """Execute training pipeline."""

# Register handler
HANDLERS = [
    TabularHandler(),
    TextHandler(),      # Future
    ImageHandler(),     # Future
    TimeSeriesHandler() # Future
]
```

### Adding New Models

Implement `BaseModel` interface and register:

```python
class XGBoostModel(BaseModel):
    # ... implementation ...

# Register in training/models/__init__.py
MODEL_REGISTRY = {
    "ridge": RidgeModel,
    "logreg": LogisticRegressionModel,
    "lgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "xgboost": XGBoostModel,  # New model
}
```

## Logging and Observability

### Structured Logging
```python
logger.info("stage_started", stage="training", run_id=run_id)
logger.info("cv_score", fold=0, model="lgbm", score=0.85, metric="auc")
logger.error("submission_failed", reason="rate_limit", retry_after=3600)
```

### Artifacts Preservation
Every run creates a complete snapshot:
- Config used
- Code version (git commit hash)
- Environment info (Python version, package versions)
- All models and predictions
- Full logs

This enables:
- Reproducibility
- Debugging failures
- Comparing runs
- Resuming interrupted runs

## Error Handling Strategy

1. **Fail fast**: Validate inputs early
2. **Clear errors**: Structured error messages with remediation steps
3. **Preserve state**: Save progress before each stage
4. **Enable resume**: Allow continuing from last checkpoint
5. **Log everything**: Full audit trail of decisions

## Security Considerations

See SECURITY.md for full details. Key points:
- Never commit credentials
- Use `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`
- Validate all inputs (URLs, file paths, competition slugs)
- Sandbox model training (resource limits)
- Rate limiting to prevent abuse

## Performance Considerations

- **Caching**: Cache preprocessed features to disk
- **Parallelism**: Train CV folds in parallel (configurable workers)
- **Memory management**: Stream large datasets, use chunking for big CSVs
- **Early stopping**: GBDT models with early stopping to save time
- **Time budgets**: Respect user-specified time limits

## Testing Strategy

- **Unit tests**: Each module has comprehensive tests
- **Integration tests**: Full pipeline on synthetic competitions
- **End-to-end tests**: Real competitions (Titanic) in CI
- **Regression tests**: Ensure submissions remain valid after changes
