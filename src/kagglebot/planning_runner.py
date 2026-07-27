from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from kagglebot import verify_artifacts as _verify_artifacts
from kagglebot.agents.identity import oracle_flow_token, planning_flow_summary
from kagglebot.agents.strategy_runner import resolve_strategy_engine
from kagglebot.campaign import normalize_campaign_mode
from kagglebot.data_readiness import assess_local_training_data
from kagglebot.exceptions import MissingCompetitionDataError
from kagglebot.exec_utils import run_command
from kagglebot.json_utils import load_json_object_or_empty, write_json_object
from kagglebot.method_scout import effective_method_scout_mode
from kagglebot.orchestrator.agent_pipeline import AgentPipelineConfig, run_agent_pipeline
from kagglebot.watch_state import update_watch_phase
from kagglebot.writeup import normalize_deliverable_mode


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
        implementation = load_json_object_or_empty(config.paths.run_dir(run_id) / "implementation_verification.json")
        completion_payload: dict[str, object] = {
            "completed_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "status": "complete",
            "strategy_engine": resolved_strategy_engine,
        }
        implementation_status = implementation.get("status")
        if isinstance(implementation_status, str):
            completion_payload["implementation_status"] = implementation_status
        if implementation.get("blocked_reason") == "missing_competition_data":
            completion_payload["blocked_reason"] = "missing_competition_data"
        write_json_object(
            completion_path,
            completion_payload,
            sort_keys=True,
        )


def ensure_local_training_data_ready(config: PlanningRunConfig, run_id: str) -> None:
    plan = load_json_object_or_empty(config.paths.plan_path)
    if normalize_deliverable_mode(plan.get("deliverable_mode"), default="leaderboard") == "writeup":
        # Judged writeup/hackathon competitions may intentionally provide no
        # labeled competition dataset. Their project runtime and required
        # attachments enforce their own data and artifact contracts.
        return
    raw_runtime_budget = plan.get("runtime_budget")
    runtime_budget = raw_runtime_budget if isinstance(raw_runtime_budget, dict) else {}
    if runtime_budget.get("local_training_required") is not True:
        return

    readiness = assess_local_training_data(config.paths)
    if readiness.ready:
        return

    run_dir = config.paths.run_dir(run_id)
    implementation = load_json_object_or_empty(run_dir / "implementation_verification.json")
    run_payload = load_json_object_or_empty(run_dir / "run.json")
    run_payload.update(
        {
            "run_id": run_id,
            "slug": config.slug,
            "status": "blocked_on_data",
            "blocked_reason": "missing_competition_data",
            "data_readiness_reason": readiness.reason,
            "implementation_status": implementation.get("status", "unknown"),
            "training_performed": False,
            "score_reported": False,
            "submission_created": False,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )
    write_json_object(run_dir / "run.json", run_payload, sort_keys=True)
    update_watch_phase(
        config,
        run_id,
        "blocked_on_data",
        detail="Local training is required, but labeled competition data is unavailable.",
    )
    raise MissingCompetitionDataError(
        "Local training is required by the frozen plan, but labeled competition data is unavailable "
        f"({readiness.reason}). The implementation passed its data-free contract and can be resumed after the "
        "authorized training package is staged."
    )


def _pipeline_strategy_engine_request() -> str:
    return "oracle"
