from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from kagglebot import verify_artifacts as _verify_artifacts
from kagglebot.agents.identity import oracle_flow_token, planning_flow_summary
from kagglebot.agents.strategy_runner import resolve_strategy_engine
from kagglebot.campaign import normalize_campaign_mode
from kagglebot.exec_utils import run_command
from kagglebot.json_utils import write_json_object
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
    strategy_engine = _pipeline_strategy_engine_request()
    resolved_strategy_engine = resolve_strategy_engine(strategy_engine)
    strategy_token = oracle_flow_token() if resolved_strategy_engine == "oracle" else None
    flow_summary = planning_flow_summary(strategy_token=strategy_token)
    print(f"[cyan]plan[/cyan]: {flow_summary}")
    update_watch_phase(
        config,
        run_id,
        "gpt_planning",
        detail=flow_summary,
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
        strategy_engine=strategy_engine,
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
    if not config.dry_run:
        completion_path = config.paths.run_dir(run_id) / "planning_complete.json"
        write_json_object(
            completion_path,
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "run_id": run_id,
                "status": "complete",
                "strategy_engine": resolved_strategy_engine,
            },
            sort_keys=True,
        )


def _pipeline_strategy_engine_request() -> str:
    return "oracle"
