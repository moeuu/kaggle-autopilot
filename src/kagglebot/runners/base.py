from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kagglebot.paths import CompetitionPaths


@dataclass(frozen=True)
class RunContext:
    competition: str
    slug: str
    run_id: str
    paths: CompetitionPaths
    workdir: Path
    dry_run: bool
    force: bool
    force_submit: bool
    message: str
    time_budget_minutes: int
    cv_folds: int
    model_names: list[str] | None
    use_stacking: bool
    compute: str
    accelerator: str
    enable_internet: bool
    kaggle_username: str | None
    strict_accelerator: bool
    candidate_budget_minutes: int | None = None
    max_candidates_per_iteration: int | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    runner: str
    submission_path: Path | None
    summary_path: Path | None
    analysis_path: Path | None
    kernel_slug: str | None


@dataclass(frozen=True)
class CandidateRunSpec:
    candidate_id: str
    node_id: str
    category: str
    method_id: str | None
    validation_profile_id: str | None
    expected_outputs: dict[str, str]
    node_type: str = ""
    adapter: str | None = None
    runtime_budget: dict[str, object] | None = None
    data_contract: dict[str, object] | None = None
    metric_contract: dict[str, object] | None = None
    dependency_check: dict[str, object] | None = None


@dataclass(frozen=True)
class CandidateRunResult:
    candidate_id: str
    node_id: str
    status: str
    metrics_path: Path | None = None
    oof_path: Path | None = None
    prediction_path: Path | None = None
    error: str | None = None


class Runner(Protocol):
    name: str

    def run(self, context: RunContext) -> RunResult: ...

    def run_one_candidate(self, context: RunContext, spec: CandidateRunSpec) -> CandidateRunResult: ...

    def run_candidate_batch(self, context: RunContext, specs: list[CandidateRunSpec]) -> list[CandidateRunResult]: ...
