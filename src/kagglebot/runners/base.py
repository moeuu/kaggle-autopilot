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


@dataclass(frozen=True)
class RunResult:
    run_id: str
    runner: str
    submission_path: Path | None
    summary_path: Path | None
    analysis_path: Path | None
    kernel_slug: str | None


class Runner(Protocol):
    name: str

    def run(self, context: RunContext) -> RunResult: ...
