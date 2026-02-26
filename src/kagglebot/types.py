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
    score_source: str = "cv"
    holdout_frac: float | None = None
    cv_folds: int | None = None
    split_strategy: str | None = None
    seed: int | None = None
    eval_seeds: list[int] | None = None
    eval_repeats: int | None = None
    time_budget_min: int | None = None
    kernel_name: str | None = None
    internet: str = "on"
    max_iterations: int = 3
    max_total_min: int | None = None
    patience: int = 2
    min_improvement: float = 0.0
    submit_policy: str = "always"
    submission_gate: str | None = None
    readiness_target_score: float | None = None
    readiness_method: str | None = None
    readiness_k: float | None = None
    ci_method: str | None = None
    ci_alpha: float | None = None
    drift_check: bool | None = None
    drift_weight: float | None = None
    stop_min_delta: float | None = None
    stop_no_improve_patience: int | None = None
    stop_same_config_patience: int | None = None
    rank_force_major_max_percentile: float | None = None
    rank_force_major_min_teams: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> PlanConfig:
        return cls(
            target_metric=payload.get("target_metric"),
            target_direction=payload.get("target_direction") or "auto",
            target_score=payload.get("target_score"),
            score_source=payload.get("score_source") or "cv",
            holdout_frac=payload.get("holdout_frac"),
            cv_folds=payload.get("cv_folds"),
            split_strategy=payload.get("split_strategy"),
            seed=payload.get("seed"),
            eval_seeds=payload.get("eval_seeds"),  # type: ignore[arg-type]
            eval_repeats=payload.get("eval_repeats"),  # type: ignore[arg-type]
            time_budget_min=payload.get("time_budget_min"),
            kernel_name=payload.get("kernel_name"),
            internet=payload.get("internet") or "on",
            max_iterations=payload.get("max_iterations") or 3,
            max_total_min=payload.get("max_total_min"),
            patience=payload.get("patience") or 2,
            min_improvement=payload.get("min_improvement") or 0.0,
            submit_policy=payload.get("submit_policy") or "always",
            submission_gate=payload.get("submission_gate"),  # type: ignore[arg-type]
            readiness_target_score=payload.get("readiness_target_score"),
            readiness_method=payload.get("readiness_method"),  # type: ignore[arg-type]
            readiness_k=payload.get("readiness_k"),  # type: ignore[arg-type]
            ci_method=payload.get("ci_method"),  # type: ignore[arg-type]
            ci_alpha=payload.get("ci_alpha"),  # type: ignore[arg-type]
            drift_check=payload.get("drift_check"),  # type: ignore[arg-type]
            drift_weight=payload.get("drift_weight"),  # type: ignore[arg-type]
            stop_min_delta=payload.get("stop_min_delta"),  # type: ignore[arg-type]
            stop_no_improve_patience=payload.get("stop_no_improve_patience"),  # type: ignore[arg-type]
            stop_same_config_patience=payload.get("stop_same_config_patience"),  # type: ignore[arg-type]
            rank_force_major_max_percentile=payload.get("rank_force_major_max_percentile"),
            rank_force_major_min_teams=payload.get("rank_force_major_min_teams"),
        )


@dataclass(frozen=True)
class AgentResult:
    transcript_path: str
    last_message_path: str
    returncode: int
    stdout: str
    stderr: str
