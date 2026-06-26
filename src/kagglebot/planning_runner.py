from __future__ import annotations

from typing import Protocol

from kagglebot import verify_artifacts as _verify_artifacts
from kagglebot.agents.identity import planning_flow_summary
from kagglebot.campaign import normalize_campaign_mode
from kagglebot.exec_utils import run_command
from kagglebot.method_scout import effective_method_scout_mode
from kagglebot.orchestrator.agent_pipeline import AgentPipelineConfig, run_agent_pipeline
from kagglebot.watch_state import update_watch_phase


class PlanningRunConfig(Protocol):
    slug: str
    competition_url: str
    compute: str
    accelerator: str
    internet: str | None
    dry_run: bool
    campaign_mode: str
    method_scout: str
    method_scout_max_sources: int | None
    hardware_profile: str | None
    time_budget_min: int | None
    verify_cmd: str | None
    paths: object


def run_plan_and_initial(config: PlanningRunConfig, run_id: str) -> None:
    print(f"[cyan]plan[/cyan]: {planning_flow_summary()}")
    update_watch_phase(
        config,
        run_id,
        "gpt_planning",
        detail=planning_flow_summary(),
    )
    planning_campaign_mode = normalize_campaign_mode(config.campaign_mode, deliverable_mode="leaderboard")
    pipeline_config = AgentPipelineConfig(
        slug=config.slug,
        competition_url=config.competition_url,
        compute=config.compute,
        accelerator=config.accelerator,
        internet=str(config.internet or "auto"),
        run_id=run_id,
        dry_run=config.dry_run,
        repo_root=config.paths.repo_root,
        method_scout=effective_method_scout_mode(
            requested_mode=config.method_scout,
            campaign_mode=planning_campaign_mode,
        ),
        method_scout_max_sources=int(config.method_scout_max_sources or 12),
        hardware_profile=config.hardware_profile,
        time_budget_min=config.time_budget_min,
    )
    run_agent_pipeline(paths=config.paths, config=pipeline_config)
    update_watch_phase(
        config,
        run_id,
        "verifying",
        detail="Verifying the generated plan and kernel scaffold.",
    )
    _verify_artifacts.run_repo_verify(
        config.verify_cmd,
        dry_run=config.dry_run,
        artifacts_dir=config.paths.artifacts_dir,
        run_command_fn=run_command,
    )
