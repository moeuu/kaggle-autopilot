from __future__ import annotations

from types import SimpleNamespace

from kagglebot import planning_runner


def _planning_config(tmp_path):
    paths = SimpleNamespace(repo_root=tmp_path, artifacts_dir=tmp_path / "artifacts")
    paths.artifacts_dir.mkdir()
    return SimpleNamespace(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        compute="local_gpu",
        accelerator="gpu",
        internet="auto",
        dry_run=False,
        campaign_mode="top1",
        method_scout="off",
        method_scout_max_sources=None,
        hardware_profile=None,
        time_budget_min=None,
        verify_cmd=None,
        paths=paths,
    )


def test_run_plan_and_initial_requires_oracle_even_when_environment_requests_auto(monkeypatch, tmp_path) -> None:
    config = _planning_config(tmp_path)
    phases: list[tuple[str, str]] = []
    pipeline_engines: list[str] = []

    monkeypatch.setenv("KAGGLEBOT_STRATEGY_ENGINE", "auto")
    requested_engines: list[str] = []
    monkeypatch.setattr(
        planning_runner,
        "resolve_strategy_engine",
        lambda requested: requested_engines.append(requested) or "oracle",
    )
    monkeypatch.setattr(
        planning_runner,
        "update_watch_phase",
        lambda _config, _run_id, phase, *, detail=None: phases.append((phase, detail or "")),
    )
    monkeypatch.setattr(
        planning_runner,
        "run_agent_pipeline",
        lambda *, paths, config: pipeline_engines.append(config.strategy_engine),  # noqa: ARG005
    )
    monkeypatch.setattr(planning_runner._verify_artifacts, "run_repo_verify", lambda *args, **kwargs: None)

    planning_runner.run_plan_and_initial(config, "run-1")

    assert requested_engines == ["oracle"]
    assert pipeline_engines == ["oracle"]
    assert phases[0][0] == "gpt_planning"
    assert "oracle(latest-pro)" in phases[0][1]


def test_run_plan_and_initial_auto_reports_oracle_when_available(monkeypatch, tmp_path) -> None:
    config = _planning_config(tmp_path)
    phases: list[tuple[str, str]] = []
    pipeline_engines: list[str] = []

    monkeypatch.setenv("KAGGLEBOT_STRATEGY_ENGINE", "auto")
    monkeypatch.setattr(planning_runner, "resolve_strategy_engine", lambda requested: "oracle")
    monkeypatch.setattr(
        planning_runner,
        "update_watch_phase",
        lambda _config, _run_id, phase, *, detail=None: phases.append((phase, detail or "")),
    )
    monkeypatch.setattr(
        planning_runner,
        "run_agent_pipeline",
        lambda *, paths, config: pipeline_engines.append(config.strategy_engine),  # noqa: ARG005
    )
    monkeypatch.setattr(planning_runner._verify_artifacts, "run_repo_verify", lambda *args, **kwargs: None)

    planning_runner.run_plan_and_initial(config, "run-1")

    assert pipeline_engines == ["oracle"]
    assert phases[0][0] == "gpt_planning"
    assert "oracle(latest-pro)" in phases[0][1]
