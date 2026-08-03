from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kagglebot import planning_runner
from kagglebot.exceptions import MissingCompetitionDataError
from kagglebot.paths import CompetitionPaths


def _planning_config(tmp_path):
    paths = SimpleNamespace(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        run_dir=lambda run_id: tmp_path / "artifacts" / "demo" / "runs" / run_id,
    )
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
    assert (config.paths.run_dir("run-1") / "planning_complete.json").exists()


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


def test_required_local_training_is_retryable_failure_when_labeled_data_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True)
    paths.data_dir.mkdir(parents=True)
    paths.run_dir("run-1").mkdir(parents=True)
    paths.plan_path.write_text(
        json.dumps({"runtime_budget": {"local_training_required": True}}),
        encoding="utf-8",
    )
    paths.dataset_profile_path.write_text(
        json.dumps({"status": "missing_required_files"}),
        encoding="utf-8",
    )
    (paths.run_dir("run-1") / "implementation_verification.json").write_text(
        json.dumps(
            {
                "status": "passed_contract_only",
                "blocked_reason": "missing_competition_data",
                "training_performed": False,
                "score_reported": False,
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(slug="demo", paths=paths)
    phases: list[str] = []
    monkeypatch.setattr(
        planning_runner,
        "update_watch_phase",
        lambda _config, _run_id, phase, **_kwargs: phases.append(phase),
    )

    with pytest.raises(MissingCompetitionDataError, match="can be resumed"):
        planning_runner.ensure_local_training_data_ready(config, "run-1")

    run_payload = json.loads((paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    assert run_payload["status"] == "failed"
    assert run_payload["blocked_reason"] == "missing_competition_data"
    assert run_payload["failure_kind"] == "blocked_on_data"
    assert run_payload["retryable"] is True
    assert run_payload["implementation_status"] == "passed_contract_only"
    assert run_payload["training_performed"] is False
    assert run_payload["score_reported"] is False
    assert run_payload["submission_created"] is False
    assert phases == ["blocked_on_data"]
    assert not list(paths.run_dir("run-1").rglob("submission.csv"))
    assert not list(paths.run_dir("run-1").rglob("oof_*.npy"))
    assert not list(paths.run_dir("run-1").rglob("*.pth"))

    (paths.data_dir / "HAR.zip").write_bytes(b"staged training archive")
    planning_runner.ensure_local_training_data_ready(config, "run-1")


def test_writeup_does_not_require_generic_labeled_competition_data(monkeypatch, tmp_path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True)
    paths.data_dir.mkdir(parents=True)
    paths.run_dir("run-1").mkdir(parents=True)
    paths.plan_path.write_text(
        json.dumps(
            {
                "deliverable_mode": "writeup",
                "runtime_budget": {"local_training_required": True},
            }
        ),
        encoding="utf-8",
    )
    paths.dataset_profile_path.write_text(
        json.dumps({"status": "missing_required_files"}),
        encoding="utf-8",
    )
    config = SimpleNamespace(slug="demo", paths=paths)
    phases: list[str] = []
    monkeypatch.setattr(
        planning_runner,
        "update_watch_phase",
        lambda _config, _run_id, phase, **_kwargs: phases.append(phase),
    )

    planning_runner.ensure_local_training_data_ready(config, "run-1")

    assert not (paths.run_dir("run-1") / "run.json").exists()
    assert phases == []
