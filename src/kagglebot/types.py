from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RulesInfo:
    url: str
    source: str
    file: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapMeta:
    slug: str
    created_at: str
    rules: RulesInfo

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rules"] = self.rules.to_dict()
        return payload


@dataclass
class PlanConfig:
    target_metric: str | None = None
    target_direction: str = "auto"
    target_score: float | None = None
    score_source: str = "auto"
    holdout_frac: float | None = None
    cv_folds: int | None = None
    seed: int | None = None
    time_budget_min: int | None = None
    kernel_name: str | None = None
    internet: str = "auto"
    max_iterations: int = 5
    max_total_min: int = 240
    patience: int = 2
    min_improvement: float = 0.0
    submit_policy: str = "on_target_only"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> PlanConfig:
        return cls(
            target_metric=payload.get("target_metric"),
            target_direction=payload.get("target_direction") or "auto",
            target_score=payload.get("target_score"),
            score_source=payload.get("score_source") or "auto",
            holdout_frac=payload.get("holdout_frac"),
            cv_folds=payload.get("cv_folds"),
            seed=payload.get("seed"),
            time_budget_min=payload.get("time_budget_min"),
            kernel_name=payload.get("kernel_name"),
            internet=payload.get("internet") or "auto",
            max_iterations=payload.get("max_iterations") or 5,
            max_total_min=payload.get("max_total_min") or 240,
            patience=payload.get("patience") or 2,
            min_improvement=payload.get("min_improvement") or 0.0,
            submit_policy=payload.get("submit_policy") or "on_target_only",
        )


@dataclass(frozen=True)
class AgentResult:
    transcript_path: str
    last_message_path: str
    returncode: int
    stdout: str
    stderr: str
