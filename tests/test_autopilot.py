"""Tests for autopilot gating and iteration behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kagglebot.autopilot import AutopilotConfig, run_autopilot
from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_runner import KernelRunResult
from kagglebot.paths import CompetitionPaths, KnowledgePaths
from kagglebot.solver.evaluate import EvaluationResult
from kagglebot.solver.initial_model import TrainingOutcome
from kagglebot.types import PlanConfig


def _write_sample_submission(path: Path) -> None:
    df = pd.DataFrame({"id": [1, 2], "target": [0.5, 0.5]})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_plan(paths: CompetitionPaths, **overrides) -> None:
    plan = PlanConfig()
    payload = plan.to_dict()
    payload.update(overrides)
    paths.plan_path.parent.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_config(tmp_path: Path, **overrides) -> AutopilotConfig:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    _write_sample_submission(paths.sample_submission_path)
    paths.prompts_dir.mkdir(parents=True, exist_ok=True)
    paths.codex_improve_template.write_text("improve {slug} {iteration}\n", encoding="utf-8")
    paths.codex_kernel_fix_template.write_text("fix {slug} {iteration}\n", encoding="utf-8")
    paths.codex_plan_and_implement_prompt.write_text("plan+implement\n", encoding="utf-8")
    base = AutopilotConfig(
        run_id="run-1",
        slug="demo",
        competition_url=None,
        paths=paths,
        knowledge_paths=knowledge_paths,
        agent="codex",
        compute="local_cpu",
        accelerator="cpu",
        strict_accelerator=False,
        kaggle_username=None,
        kernel_name=None,
        internet=None,
        time_budget_min=None,
        seed=None,
        score_source=None,
        holdout_frac=None,
        cv_folds=None,
        target_metric=None,
        target_score=None,
        target_direction=None,
        max_iterations=1,
        max_total_min=60,
        patience=2,
        min_improvement=0.0,
        submit=False,
        force_submit=False,
        message=None,
        verify_cmd="uv run pytest -q",
        dry_run=False,
    )
    return base if not overrides else base.__class__(**{**base.__dict__, **overrides})


def test_autopilot_uses_plan_from_agent(monkeypatch, tmp_path: Path) -> None:
    def fake_plan(config: AutopilotConfig, run_id: str) -> None:  # noqa: ARG001
        _write_plan(
            config.paths,
            target_metric="rmse",
            target_score=0.5,
            target_direction="minimize",
        )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            value=0.4,
            std=None,
            train_score=None,
            val_score=None,
            fold_scores=None,
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="ridge",
            model_summary={},
            accelerator="cpu",
        )

    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", fake_plan)
    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path)
    run_autopilot(config)

    run_payload = json.loads((config.paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    assert run_payload["config"]["target_metric"] == "rmse"
    assert run_payload["config"]["target_score"] == 0.5


def test_autopilot_submit_when_top1_tier(monkeypatch, tmp_path: Path) -> None:
    submission_calls: list[Path] = []

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            value=0.4,
            std=None,
            train_score=None,
            val_score=None,
            fold_scores=None,
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="ridge",
            model_summary={},
            accelerator="cpu",
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.autopilot.submit_competition", lambda *args, **kwargs: submission_calls.append(args[1])
    )
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=3)
    run_autopilot(config)
    assert len(submission_calls) == 1


def test_autopilot_no_submit_when_top1_tier_without_submit(monkeypatch, tmp_path: Path) -> None:
    submission_calls: list[Path] = []

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            value=0.4,
            std=None,
            train_score=None,
            val_score=None,
            fold_scores=None,
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="ridge",
            model_summary={},
            accelerator="cpu",
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.autopilot.submit_competition", lambda *args, **kwargs: submission_calls.append(args[1])
    )
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=False, max_iterations=1)
    run_autopilot(config)
    assert submission_calls == []


def test_autopilot_submit_at_final_iteration(monkeypatch, tmp_path: Path) -> None:
    submission_calls: list[Path] = []

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            value=1.0,
            std=None,
            train_score=None,
            val_score=None,
            fold_scores=None,
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="ridge",
            model_summary={},
            accelerator="cpu",
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.1})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.autopilot.submit_competition", lambda *args, **kwargs: submission_calls.append(args[1])
    )
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_autopilot(config)
    assert len(submission_calls) == 1


def test_autopilot_stops_when_target_missing(monkeypatch, tmp_path: Path) -> None:
    calls = {"train": 0, "submit": 0}

    def fake_train(*args, **kwargs):  # noqa: ARG001
        calls["train"] += 1
        raise AssertionError("train should not be called when target missing")

    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})
    monkeypatch.setattr("kagglebot.autopilot.submit_competition", lambda *args, **kwargs: calls.update(submit=1))

    config = _make_config(tmp_path, submit=True)
    run_autopilot(config)
    run_payload = json.loads((config.paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    assert run_payload["status"] == "missing_target"
    assert calls["train"] == 0
    assert calls["submit"] == 0


def test_autopilot_creates_improve_prompt(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            value=1.0,
            std=None,
            train_score=None,
            val_score=None,
            fold_scores=None,
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="ridge",
            model_summary={},
            accelerator="cpu",
        )

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("improved features\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.1})
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, max_iterations=2)
    run_autopilot(config)
    iter_dir = config.paths.iter_dir(config.run_id or "run-1", 1)
    assert (iter_dir / "agent" / "prompt.md").exists()


def test_autopilot_runs_agent_pipeline(monkeypatch, tmp_path: Path) -> None:
    called = {"run": False}

    def fake_pipeline(*args, **kwargs):  # noqa: ARG001
        called["run"] = True

    monkeypatch.setattr("kagglebot.autopilot.run_agent_pipeline", fake_pipeline)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)

    config = _make_config(tmp_path)
    from kagglebot.autopilot import _run_plan_and_initial

    _run_plan_and_initial(config, config.run_id or "run-1")
    assert called["run"] is True


def test_autopilot_retries_kernel_failure(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        score_source="cv",
        cv_folds=3,
        seed=42,
    )

    calls = {"run_kernel": 0, "codex": 0, "kernel_fix": 0}

    def fake_run_kernel(**kwargs):
        calls["run_kernel"] += 1
        if calls["run_kernel"] == 1:
            raise KernelFailedError("kernel failed")
        output_dir = (
            kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{kwargs['iteration']}" / "output"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "score_source": "cv",
                    "metric": "rmse",
                    "direction": "minimize",
                    "offline_value": 0.4,
                    "offline_std": 0.01,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        submission_path = output_dir / "submission.csv"
        submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        return KernelRunResult(
            kernel_id="user/kernel",
            output_dir=output_dir,
            submission_path=submission_path,
            metrics_path=metrics_path,
        )

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool):  # noqa: ARG001
        calls["codex"] += 1
        if prompt_path.name == "kernel_fix_prompt.md":
            calls["kernel_fix"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("kernel fix applied\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.run_kernel", lambda **kwargs: fake_run_kernel(**kwargs))
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.kernel_runner.ensure_kernel_sources_valid", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, compute="kaggle_gpu", accelerator="gpu", max_iterations=1)
    run_autopilot(config)

    assert calls["run_kernel"] == 2
    assert calls["kernel_fix"] == 1


def test_autopilot_respects_max_iterations(monkeypatch, tmp_path: Path) -> None:
    calls = {"train": 0}

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        max_iterations=10,
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        calls["train"] += 1
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            value=1.0,
            std=None,
            train_score=None,
            val_score=None,
            fold_scores=None,
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="ridge",
            model_summary={},
            accelerator="cpu",
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, max_iterations=10)
    run_autopilot(config)
    assert calls["train"] == 10
