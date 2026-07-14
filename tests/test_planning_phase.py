from __future__ import annotations

from types import SimpleNamespace

from kagglebot.planning_phase import PlanningPhase
from kagglebot.types import PlanConfig


def test_planning_phase_runs_initial_pipeline_when_plan_needs_agent_values(monkeypatch, tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    calls = {"planning": 0}

    config = SimpleNamespace(
        agent="gpt",
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        paths=SimpleNamespace(
            plan_path=plan_path,
            kernel_source_dir=kernel_dir,
            run_dir=lambda run_id: tmp_path / "runs" / run_id,
        ),
    )
    initial_plan = PlanConfig(target_metric=None, target_score=None, target_direction=None)
    planned = PlanConfig(target_metric="rmse", target_score=0.5, target_direction="minimize")

    monkeypatch.setattr("kagglebot.watch_state.update_watch_phase", lambda *args, **kwargs: None)

    def fake_run_plan_and_initial(config_arg, run_id_arg):  # noqa: ANN001
        assert config_arg is config
        assert run_id_arg == "run-1"
        calls["planning"] += 1

    monkeypatch.setattr("kagglebot.planning_runner.run_plan_and_initial", fake_run_plan_and_initial)
    monkeypatch.setattr("kagglebot.plan_policy.needs_planning", lambda **kwargs: True)
    monkeypatch.setattr("kagglebot.plan_policy.load_plan_config", lambda paths: planned)

    result = PlanningPhase(config=config, run_id="run-1", resume_run=False).execute(initial_plan)

    assert result == planned
    assert calls["planning"] == 1


def test_planning_phase_skips_restart_when_existing_plan_and_kernel_are_present(monkeypatch, tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "planning_complete.json").write_text(
        '{"run_id":"run-1","status":"complete","strategy_engine":"oracle"}\n',
        encoding="utf-8",
    )
    plan = PlanConfig(target_metric="rmse", target_score=0.5, target_direction="minimize")
    config = SimpleNamespace(
        agent="gpt",
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        paths=SimpleNamespace(
            plan_path=plan_path,
            kernel_source_dir=kernel_dir,
            run_dir=lambda run_id: tmp_path / "runs" / run_id,
        ),
    )

    monkeypatch.setattr(
        "kagglebot.planning_runner.run_plan_and_initial",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("planning should be skipped")),
    )

    result = PlanningPhase(config=config, run_id="run-1", resume_run=True).execute(plan)

    assert result is plan


def test_planning_phase_reruns_when_resume_marker_is_missing(monkeypatch, tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.py").write_text("print('stale')\n", encoding="utf-8")
    calls = {"planning": 0}
    config = SimpleNamespace(
        agent="gpt",
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        paths=SimpleNamespace(
            plan_path=plan_path,
            kernel_source_dir=kernel_dir,
            run_dir=lambda run_id: tmp_path / "runs" / run_id,
        ),
    )
    plan = PlanConfig(target_metric="rmse", target_score=0.5, target_direction="minimize")

    monkeypatch.setattr("kagglebot.plan_policy.needs_planning", lambda **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.planning_runner.run_plan_and_initial",
        lambda *args, **kwargs: calls.__setitem__("planning", calls["planning"] + 1),
    )
    monkeypatch.setattr("kagglebot.plan_policy.load_plan_config", lambda paths: plan)
    monkeypatch.setattr("kagglebot.watch_state.update_watch_phase", lambda *args, **kwargs: None)

    PlanningPhase(config=config, run_id="run-1", resume_run=True).execute(plan)

    assert calls["planning"] == 1
