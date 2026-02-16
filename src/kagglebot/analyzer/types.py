from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompetitionSchema:
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    id_column: str | None
    target_columns: list[str]
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["train_path"] = str(self.train_path)
        payload["test_path"] = str(self.test_path)
        payload["sample_submission_path"] = str(self.sample_submission_path)
        return payload


@dataclass(frozen=True)
class ModelingStrategy:
    preprocessing: list[str]
    models: list[str]
    cv_folds: int
    use_stacking: bool
    time_budget_minutes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CompetitionMetadata:
    slug: str
    competition_type: str
    task: str
    metric: str
    metric_direction: str
    prediction_kind: str
    schema: CompetitionSchema
    strategy: ModelingStrategy
    assumptions: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "slug": self.slug,
            "type": self.competition_type,
            "task": self.task,
            "metric": self.metric,
            "metric_direction": self.metric_direction,
            "prediction_kind": self.prediction_kind,
            "schema": self.schema.to_dict(),
            "strategy": self.strategy.to_dict(),
            "assumptions": self.assumptions,
        }
