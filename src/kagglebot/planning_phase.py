from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rich import print

from kagglebot import plan_policy as _plan_policy
from kagglebot import planning_runner as _planning_runner
from kagglebot import watch_state as _watch_state
from kagglebot.types import PlanConfig


class PlanningPhaseConfig(Protocol):
    agent: str
    target_metric: str | None
    target_score: float | None
    target_direction: str | None
    paths: object


@dataclass(frozen=True)
class PlanningPhase:
    config: PlanningPhaseConfig
    run_id: str
    resume_run: bool

    def execute(self, plan: PlanConfig) -> PlanConfig:
        if _plan_policy.should_skip_planning_on_resume(
            resume_run=self.resume_run,
            plan_path=self.config.paths.plan_path,
            kernel_path=self.config.paths.kernel_source_dir / "kernel.py",
            completion_path=self.config.paths.run_dir(self.run_id) / "planning_complete.json",
            run_id=self.run_id,
            required_strategy_engine="oracle",
        ):
            print("[yellow]resume[/yellow]: skipping planning after restart; reusing existing plan")
            return plan
        if _plan_policy.needs_planning(
            agent=self.config.agent,
            config_target_metric=self.config.target_metric,
            config_target_score=self.config.target_score,
            config_target_direction=self.config.target_direction,
            plan_target_metric=plan.target_metric,
            plan_target_score=plan.target_score,
            plan_target_direction=plan.target_direction,
        ):
            print("[cyan]plan[/cyan]: generating initial plan")
            _watch_state.update_watch_phase(
                self.config,
                self.run_id,
                "gpt_planning",
                detail="GPT is drafting the initial competition plan.",
            )
            _planning_runner.run_plan_and_initial(self.config, self.run_id)
            return _plan_policy.load_plan_config(self.config.paths)
        return plan
