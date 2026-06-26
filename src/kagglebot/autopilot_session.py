from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kagglebot import autopilot_submit as _autopilot_submit
from kagglebot.knowledge_phase import KnowledgePhase
from kagglebot.planning_phase import PlanningPhase


class AutopilotSessionConfig(Protocol):
    pass


@dataclass(frozen=True)
class SubmissionPhase:
    config: AutopilotSessionConfig
    run_id: str
    problem_types: list[str]
    submit_mode: str
    notebook_submit_artifact_mode: str = "wrapper"

    def attempt(self, *, submission_path: Path, best_score: float | None) -> dict[str, object] | None:
        return _autopilot_submit.attempt_submit_for_autopilot_run(
            config=self.config,
            run_id=self.run_id,
            submission_path=submission_path,
            best_score=best_score,
            problem_types=self.problem_types,
            submit_mode=self.submit_mode,
            notebook_submit_artifact_mode=self.notebook_submit_artifact_mode,
        )


@dataclass(frozen=True)
class AutopilotSession:
    config: AutopilotSessionConfig
    run_id: str
    resume_run: bool = False

    @property
    def planning(self) -> PlanningPhase:
        return PlanningPhase(config=self.config, run_id=self.run_id, resume_run=self.resume_run)

    @property
    def knowledge(self) -> KnowledgePhase:
        return KnowledgePhase(config=self.config)

    def run(self) -> None:
        from kagglebot.autopilot import run_autopilot_core

        run_autopilot_core(self.config, self.run_id, resume_run=self.resume_run)
