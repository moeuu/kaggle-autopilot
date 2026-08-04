from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kagglebot.risk_coverage import risk_coverage_auc

Direction = Literal["maximize", "minimize"]
SplitName = Literal["kfold", "stratified_kfold", "group_kfold", "timeseries_split"]
ReadinessMethod = Literal["mean_std", "ci_bound"]
CiMethod = Literal["normal", "bootstrap"]


@dataclass(frozen=True)
class MetricDefinition:
    canonical_name: str
    direction: Direction


class MetricRegistry:
    _ALIASES = {
        "aurc": "aurc",
        "areaunderriskcoverage": "aurc",
        "areaunderriskcoveragecurve": "aurc",
        "riskcoverageauc": "aurc",
        "auc": "auc",
        "aucroc": "auc",
        "rocauc": "auc",
        "rocaucscore": "auc",
        "logloss": "logloss",
        "crossentropy": "logloss",
        "brier": "brier_score",
        "brierloss": "brier_score",
        "brierscore": "brier_score",
        "brierscoreloss": "brier_score",
        "accuracy": "accuracy",
        "acc": "accuracy",
        "f1": "f1",
        "f1score": "f1",
        "rmse": "rmse",
        "mae": "mae",
        "rmsle": "rmsle",
        "mape": "mape",
        "smape": "smape",
        "pearson": "pearson",
        "pearsonr": "pearson",
        "spearman": "spearman",
        "spearmanr": "spearman",
    }

    _DEFINITIONS = {
        "aurc": MetricDefinition(canonical_name="aurc", direction="minimize"),
        "auc": MetricDefinition(canonical_name="auc", direction="maximize"),
        "logloss": MetricDefinition(canonical_name="logloss", direction="minimize"),
        "brier_score": MetricDefinition(canonical_name="brier_score", direction="minimize"),
        "accuracy": MetricDefinition(canonical_name="accuracy", direction="maximize"),
        "f1": MetricDefinition(canonical_name="f1", direction="maximize"),
        "rmse": MetricDefinition(canonical_name="rmse", direction="minimize"),
        "mae": MetricDefinition(canonical_name="mae", direction="minimize"),
        "rmsle": MetricDefinition(canonical_name="rmsle", direction="minimize"),
        "mape": MetricDefinition(canonical_name="mape", direction="minimize"),
        "smape": MetricDefinition(canonical_name="smape", direction="minimize"),
        "pearson": MetricDefinition(canonical_name="pearson", direction="maximize"),
        "spearman": MetricDefinition(canonical_name="spearman", direction="maximize"),
    }

    @classmethod
    def canonical_metric(cls, metric_name: str) -> str:
        key = "".join(ch for ch in metric_name.lower().strip() if ch.isalnum())
        if not key:
            raise ValueError("Metric name is empty.")
        return cls._ALIASES.get(key, key)

    @classmethod
    def definition(cls, metric_name: str) -> MetricDefinition:
        canonical = cls.canonical_metric(metric_name)
        metric = cls._DEFINITIONS.get(canonical)
        if metric is None:
            supported = ", ".join(sorted(cls._DEFINITIONS))
            raise ValueError(f"Unsupported metric '{metric_name}'. Supported metrics: {supported}")
        return metric

    @classmethod
    def direction(cls, metric_name: str) -> Direction:
        return cls.definition(metric_name).direction

    @classmethod
    def score(cls, metric_name: str, y_true: Any, y_pred: Any) -> float:
        metric = cls.definition(metric_name).canonical_name
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if metric == "aurc":
            return risk_coverage_auc(y_true_arr, y_pred_arr)
        if metric == "auc":
            return cls._score_auc(y_true_arr, y_pred_arr)
        if metric == "logloss":
            return cls._score_logloss(y_true_arr, y_pred_arr)
        if metric == "brier_score":
            return cls._score_brier_score(y_true_arr, y_pred_arr)
        if metric == "accuracy":
            return float(accuracy_score(y_true_arr, cls._as_labels(y_pred_arr)))
        if metric == "f1":
            average = "binary" if np.unique(y_true_arr).size <= 2 else "macro"
            return float(f1_score(y_true_arr, cls._as_labels(y_pred_arr), average=average, zero_division=0))
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        if metric == "mae":
            return float(mean_absolute_error(y_true_arr, y_pred_arr))
        if metric == "rmsle":
            y_true_clip = np.clip(np.asarray(y_true_arr, dtype=float), 0.0, None)
            y_pred_clip = np.clip(np.asarray(y_pred_arr, dtype=float), 0.0, None)
            return float(np.sqrt(mean_squared_error(np.log1p(y_true_clip), np.log1p(y_pred_clip))))
        if metric == "mape":
            return float(mean_absolute_percentage_error(y_true_arr, y_pred_arr))
        if metric == "smape":
            return cls._score_smape(y_true_arr, y_pred_arr)
        if metric == "pearson":
            return cls._score_pearson(y_true_arr, y_pred_arr)
        if metric == "spearman":
            return cls._score_spearman(y_true_arr, y_pred_arr)
        raise ValueError(f"Unsupported metric '{metric_name}'.")

    @staticmethod
    def _as_labels(y_pred: np.ndarray) -> np.ndarray:
        if y_pred.ndim == 2:
            if y_pred.shape[1] == 1:
                return (y_pred[:, 0] >= 0.5).astype(int)
            return y_pred.argmax(axis=1)
        if np.issubdtype(y_pred.dtype, np.floating):
            if np.all((y_pred >= 0.0) & (y_pred <= 1.0)):
                return (y_pred >= 0.5).astype(int)
        return y_pred

    @staticmethod
    def _score_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim == 2:
            if y_pred.shape[1] > 2:
                return float(roc_auc_score(y_true, np.clip(y_pred, 1e-15, 1.0 - 1e-15), multi_class="ovr"))
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
            else:
                y_pred = y_pred[:, 0]
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return float(roc_auc_score(y_true, y_pred))

    @staticmethod
    def _score_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
            row_sum = y_pred.sum(axis=1, keepdims=True)
            row_sum[row_sum <= 0.0] = 1.0
            y_pred = y_pred / row_sum
            return float(log_loss(y_true, y_pred))
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-15, 1.0 - 1e-15)
        return float(log_loss(y_true, y_pred))

    @staticmethod
    def _score_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred_arr = np.asarray(y_pred, dtype=float)
        if y_pred_arr.ndim == 2:
            if y_pred_arr.shape[1] == 1:
                y_pred_arr = y_pred_arr[:, 0]
            else:
                y_pred_arr = y_pred_arr[:, -1]
        y_pred_arr = np.clip(y_pred_arr, 0.0, 1.0)
        return float(brier_score_loss(y_true, y_pred_arr))

    @staticmethod
    def _score_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        denom = np.abs(y_true_arr) + np.abs(y_pred_arr)
        smape = np.where(denom > 1e-15, 2.0 * np.abs(y_true_arr - y_pred_arr) / denom, 0.0)
        return float(np.mean(smape))

    @staticmethod
    def _score_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        corr, _ = pearsonr(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
        if np.isnan(corr):
            return 0.0
        return float(corr)

    @staticmethod
    def _score_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        corr, _ = spearmanr(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
        if np.isnan(corr):
            return 0.0
        return float(corr)


@dataclass(frozen=True)
class SplitStrategy:
    name: SplitName
    splitter: Any
    n_splits: int


class SplitStrategyFactory:
    _ALIASES = {
        "kfold": "kfold",
        "k": "kfold",
        "stratifiedkfold": "stratified_kfold",
        "stratified": "stratified_kfold",
        "groupkfold": "group_kfold",
        "group": "group_kfold",
        "timeseriessplit": "timeseries_split",
        "timeseries": "timeseries_split",
        "time": "timeseries_split",
    }

    @classmethod
    def create(
        cls,
        y: Any,
        *,
        strategy: str | None = None,
        n_splits: int = 5,
        seed: int = 42,
        groups: Any | None = None,
        time_values: Any | None = None,
        shuffle: bool = True,
    ) -> SplitStrategy:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = np.ravel(y_arr)
        if y_arr.size < 2:
            raise ValueError("At least 2 samples are required to build a split strategy.")

        requested = cls._normalize_strategy(strategy) if strategy else cls._infer_default(y_arr, groups, time_values)
        n_splits = max(2, min(int(n_splits), y_arr.size))

        if requested == "timeseries_split":
            effective = max(2, min(n_splits, y_arr.size - 1))
            return SplitStrategy(
                name="timeseries_split", splitter=TimeSeriesSplit(n_splits=effective), n_splits=effective
            )

        if requested == "group_kfold":
            if groups is None:
                return SplitStrategy(
                    name="kfold",
                    splitter=KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed),
                    n_splits=n_splits,
                )
            group_arr = np.asarray(groups)
            unique_groups = np.unique(group_arr)
            if unique_groups.size < 2:
                return SplitStrategy(
                    name="kfold",
                    splitter=KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed),
                    n_splits=n_splits,
                )
            effective = max(2, min(n_splits, int(unique_groups.size)))
            return SplitStrategy(name="group_kfold", splitter=GroupKFold(n_splits=effective), n_splits=effective)

        if requested == "stratified_kfold":
            min_count = cls._min_class_count(y_arr)
            if min_count >= 2:
                effective = max(2, min(n_splits, min_count))
                return SplitStrategy(
                    name="stratified_kfold",
                    splitter=StratifiedKFold(n_splits=effective, shuffle=shuffle, random_state=seed),
                    n_splits=effective,
                )
            return SplitStrategy(
                name="kfold", splitter=KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed), n_splits=n_splits
            )

        return SplitStrategy(
            name="kfold", splitter=KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed), n_splits=n_splits
        )

    @classmethod
    def _normalize_strategy(cls, strategy: str | None) -> SplitName:
        if not strategy:
            raise ValueError("Split strategy is empty.")
        key = "".join(ch for ch in strategy.lower().strip() if ch.isalnum())
        normalized = cls._ALIASES.get(key)
        if normalized is None:
            supported = ", ".join(sorted(set(cls._ALIASES.values())))
            raise ValueError(f"Unsupported split strategy '{strategy}'. Supported strategies: {supported}")
        return normalized  # type: ignore[return-value]

    @classmethod
    def _infer_default(cls, y: np.ndarray, groups: Any | None, time_values: Any | None) -> SplitName:
        if time_values is not None:
            return "timeseries_split"
        if groups is not None:
            group_arr = np.asarray(groups)
            if np.unique(group_arr).size >= 2:
                return "group_kfold"
        if cls._is_classification_target(y):
            return "stratified_kfold"
        return "kfold"

    @staticmethod
    def _is_classification_target(y: np.ndarray) -> bool:
        if y.dtype.kind in {"b", "O", "U", "S"}:
            return True
        unique = np.unique(y)
        if y.dtype.kind in {"i", "u"}:
            return unique.size <= max(20, int(y.size * 0.2))
        return False

    @staticmethod
    def _min_class_count(y: np.ndarray) -> int:
        _, counts = np.unique(y, return_counts=True)
        if counts.size == 0:
            return 0
        return int(counts.min())


@dataclass(frozen=True)
class FoldScore:
    repeat: int
    seed: int
    fold: int
    score: float
    train_size: int
    valid_size: int


@dataclass(frozen=True)
class CVRunResult:
    metric_name: str
    direction: Direction
    split_strategy: SplitName
    n_splits: int
    seeds: list[int]
    repeats: int
    per_fold_scores: list[float]
    fold_scores: list[FoldScore]


class RepeatedCVRunner:
    def __init__(
        self,
        *,
        metric_registry: type[MetricRegistry] = MetricRegistry,
        split_factory: type[SplitStrategyFactory] = SplitStrategyFactory,
    ) -> None:
        self.metric_registry = metric_registry
        self.split_factory = split_factory

    def run(
        self,
        *,
        y: Any,
        predict_fn: Any,
        metric_name: str,
        x: Any | None = None,
        strategy: str | None = None,
        n_splits: int = 5,
        seeds: list[int] | None = None,
        repeats: int = 1,
        groups: Any | None = None,
        time_values: Any | None = None,
    ) -> CVRunResult:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = np.ravel(y_arr)
        x_arr = self._as_split_input(x, y_arr.shape[0])
        seed_values = seeds or [42]

        all_fold_scores: list[FoldScore] = []
        split_name: SplitName | None = None
        effective_splits: int | None = None

        for repeat_idx in range(max(1, int(repeats))):
            for base_seed in seed_values:
                iter_seed = int(base_seed) + (repeat_idx * 997)
                split = self.split_factory.create(
                    y_arr,
                    strategy=strategy,
                    n_splits=n_splits,
                    seed=iter_seed,
                    groups=groups,
                    time_values=time_values,
                )
                split_name = split.name
                effective_splits = split.n_splits
                for fold_idx, (train_idx, valid_idx) in enumerate(self._iter_splits(split, x_arr, y_arr, groups)):
                    y_pred = predict_fn(train_idx, valid_idx, repeat_idx, iter_seed, fold_idx)
                    score = self.metric_registry.score(metric_name, y_arr[valid_idx], y_pred)
                    all_fold_scores.append(
                        FoldScore(
                            repeat=repeat_idx,
                            seed=iter_seed,
                            fold=fold_idx,
                            score=float(score),
                            train_size=int(len(train_idx)),
                            valid_size=int(len(valid_idx)),
                        )
                    )

        if split_name is None or effective_splits is None:
            raise RuntimeError("No CV folds were produced.")

        return CVRunResult(
            metric_name=self.metric_registry.definition(metric_name).canonical_name,
            direction=self.metric_registry.direction(metric_name),
            split_strategy=split_name,
            n_splits=effective_splits,
            seeds=[int(seed) for seed in seed_values],
            repeats=max(1, int(repeats)),
            per_fold_scores=[item.score for item in all_fold_scores],
            fold_scores=all_fold_scores,
        )

    @staticmethod
    def _as_split_input(x: Any | None, n_rows: int) -> np.ndarray:
        if x is None:
            return np.zeros((n_rows, 1), dtype=float)
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            return x_arr.reshape(-1, 1)
        return x_arr

    @staticmethod
    def _iter_splits(
        split: SplitStrategy,
        x: np.ndarray,
        y: np.ndarray,
        groups: Any | None,
    ):
        if split.name == "group_kfold":
            if groups is None:
                raise ValueError("GroupKFold selected but groups were not provided.")
            yield from split.splitter.split(x, y, groups=np.asarray(groups))
            return
        if split.name == "stratified_kfold":
            yield from split.splitter.split(x, y)
            return
        if split.name == "timeseries_split":
            yield from split.splitter.split(x)
            return
        yield from split.splitter.split(x, y)


@dataclass(frozen=True)
class UncertaintyStats:
    mean: float
    std: float
    ci_low: float
    ci_high: float


class UncertaintyEstimator:
    @staticmethod
    def estimate(
        scores: list[float] | np.ndarray,
        *,
        method: CiMethod = "normal",
        alpha: float = 0.05,
        bootstrap_iterations: int = 1000,
        random_state: int = 42,
    ) -> UncertaintyStats:
        values = np.asarray(scores, dtype=float)
        if values.size == 0:
            raise ValueError("At least one score is required to estimate uncertainty.")
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

        if method == "normal":
            if values.size < 2:
                return UncertaintyStats(mean=mean, std=std, ci_low=mean, ci_high=mean)
            z = float(norm.ppf(1.0 - alpha / 2.0))
            half = z * std / np.sqrt(values.size)
            return UncertaintyStats(mean=mean, std=std, ci_low=mean - half, ci_high=mean + half)

        if method == "bootstrap":
            rng = np.random.default_rng(random_state)
            sample_count = max(100, int(bootstrap_iterations))
            sampled = rng.choice(values, size=(sample_count, values.size), replace=True)
            sampled_means = sampled.mean(axis=1)
            low = float(np.quantile(sampled_means, alpha / 2.0))
            high = float(np.quantile(sampled_means, 1.0 - alpha / 2.0))
            return UncertaintyStats(mean=mean, std=std, ci_low=low, ci_high=high)

        raise ValueError(f"Unsupported CI method '{method}'.")


class DriftChecker:
    @staticmethod
    def adversarial_auc(
        train_x: pd.DataFrame | np.ndarray | None,
        test_x: pd.DataFrame | np.ndarray | None,
        *,
        enabled: bool = False,
        random_state: int = 42,
        n_splits: int = 5,
        max_rows_per_side: int = 20000,
    ) -> float | None:
        if not enabled:
            return None
        if train_x is None or test_x is None:
            return None

        train_df = DriftChecker._to_frame(train_x, prefix="f")
        test_df = DriftChecker._to_frame(test_x, prefix="f")
        if train_df.empty or test_df.empty:
            return None

        train_df = DriftChecker._sample_rows(train_df, max_rows_per_side, random_state)
        test_df = DriftChecker._sample_rows(test_df, max_rows_per_side, random_state + 1)

        x = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        y = np.concatenate([np.zeros(len(train_df), dtype=int), np.ones(len(test_df), dtype=int)])
        if np.unique(y).size < 2:
            return None

        min_count = int(np.bincount(y).min())
        effective_splits = max(2, min(int(n_splits), min_count))
        if effective_splits < 2:
            return None

        preprocessor = DriftChecker._build_preprocessor(x)
        model = LogisticRegression(max_iter=300, random_state=random_state)
        pipeline = Pipeline([("pre", preprocessor), ("model", model)])
        cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipeline, x, y, scoring="roc_auc", cv=cv)
        if scores.size == 0:
            return None
        return float(np.mean(scores))

    @staticmethod
    def _to_frame(data: pd.DataFrame | np.ndarray, *, prefix: str) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"{prefix}_{idx}" for idx in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    @staticmethod
    def _sample_rows(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
        if len(frame) <= max_rows:
            return frame
        return frame.sample(n=max_rows, random_state=random_state, replace=False)

    @staticmethod
    def _build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
        numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in frame.columns if col not in numeric_cols]

        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric_cols:
            numeric_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", numeric_pipe, numeric_cols))
        if categorical_cols:
            categorical_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("cat", categorical_pipe, categorical_cols))
        if not transformers:
            transformers.append(("num", "passthrough", frame.columns.tolist()))
        return ColumnTransformer(transformers=transformers)


class SubmissionReadinessScorer:
    @staticmethod
    def compute(
        *,
        direction: Direction,
        mean_score: float,
        std_score: float,
        ci_low: float | None = None,
        ci_high: float | None = None,
        method: ReadinessMethod = "ci_bound",
        k: float = 1.0,
        drift_auc: float | None = None,
        drift_enabled: bool = False,
        drift_weight: float = 1.0,
    ) -> float:
        if method == "ci_bound" and ci_low is not None and ci_high is not None:
            base = ci_low if direction == "maximize" else ci_high
        else:
            base = mean_score - (k * std_score) if direction == "maximize" else mean_score + (k * std_score)

        penalty = 0.0
        if drift_enabled and drift_auc is not None:
            penalty = max(0.0, float(drift_auc) - 0.5) * float(drift_weight)
        if direction == "maximize":
            return float(base - penalty)
        return float(base + penalty)


@dataclass(frozen=True)
class EvaluationReport:
    metric_name: str
    direction: Direction
    split_strategy: SplitName
    n_splits: int
    seeds: list[int]
    repeats: int
    per_fold_scores: list[float]
    mean: float
    std: float
    ci_low: float
    ci_high: float
    drift_auc: float | None
    readiness_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenericEvaluator:
    def __init__(
        self,
        *,
        metric_registry: type[MetricRegistry] = MetricRegistry,
        split_factory: type[SplitStrategyFactory] = SplitStrategyFactory,
        cv_runner: RepeatedCVRunner | None = None,
        uncertainty: UncertaintyEstimator | None = None,
        drift_checker: type[DriftChecker] = DriftChecker,
        readiness_scorer: type[SubmissionReadinessScorer] = SubmissionReadinessScorer,
    ) -> None:
        self.metric_registry = metric_registry
        self.split_factory = split_factory
        self.cv_runner = cv_runner or RepeatedCVRunner(metric_registry=metric_registry, split_factory=split_factory)
        self.uncertainty = uncertainty or UncertaintyEstimator()
        self.drift_checker = drift_checker
        self.readiness_scorer = readiness_scorer

    def evaluate(
        self,
        *,
        y: Any,
        predict_fn: Any,
        metric_name: str,
        x: Any | None = None,
        split_strategy: str | None = None,
        n_splits: int = 5,
        seeds: list[int] | None = None,
        repeats: int = 1,
        groups: Any | None = None,
        time_values: Any | None = None,
        ci_method: CiMethod = "normal",
        ci_alpha: float = 0.05,
        bootstrap_iterations: int = 1000,
        readiness_method: ReadinessMethod = "ci_bound",
        readiness_k: float = 1.0,
        drift_enabled: bool = False,
        drift_weight: float = 1.0,
        drift_train_x: pd.DataFrame | np.ndarray | None = None,
        drift_test_x: pd.DataFrame | np.ndarray | None = None,
    ) -> EvaluationReport:
        cv_result = self.cv_runner.run(
            y=y,
            predict_fn=predict_fn,
            metric_name=metric_name,
            x=x,
            strategy=split_strategy,
            n_splits=n_splits,
            seeds=seeds,
            repeats=repeats,
            groups=groups,
            time_values=time_values,
        )
        stats = self.uncertainty.estimate(
            cv_result.per_fold_scores,
            method=ci_method,
            alpha=ci_alpha,
            bootstrap_iterations=bootstrap_iterations,
            random_state=(seeds or [42])[0],
        )
        drift_auc = self.drift_checker.adversarial_auc(
            drift_train_x,
            drift_test_x,
            enabled=drift_enabled,
            random_state=(seeds or [42])[0],
        )
        readiness = self.readiness_scorer.compute(
            direction=cv_result.direction,
            mean_score=stats.mean,
            std_score=stats.std,
            ci_low=stats.ci_low,
            ci_high=stats.ci_high,
            method=readiness_method,
            k=readiness_k,
            drift_auc=drift_auc,
            drift_enabled=drift_enabled,
            drift_weight=drift_weight,
        )
        return EvaluationReport(
            metric_name=cv_result.metric_name,
            direction=cv_result.direction,
            split_strategy=cv_result.split_strategy,
            n_splits=cv_result.n_splits,
            seeds=cv_result.seeds,
            repeats=cv_result.repeats,
            per_fold_scores=cv_result.per_fold_scores,
            mean=stats.mean,
            std=stats.std,
            ci_low=stats.ci_low,
            ci_high=stats.ci_high,
            drift_auc=drift_auc,
            readiness_score=readiness,
        )
