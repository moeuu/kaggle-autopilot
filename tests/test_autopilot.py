"""Tests for autopilot gating and iteration behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from kagglebot.autopilot import (
    AutopilotConfig,
    _build_submit_autofix_context,
    _resolve_plan,
    _resolve_submission_rank_payload,
    _resume_iteration_state,
    _run_autofix,
    _should_skip_planning,
    _write_iteration_state_marker,
    run_autopilot,
)
from kagglebot.exceptions import KaggleCliError, KernelFailedError, SubmissionCliError, SubmitAbortedError
from kagglebot.kernel_runner import KernelRunResult
from kagglebot.knowledge import resolve_problem_type_insights
from kagglebot.paths import CompetitionPaths, KnowledgePaths
from kagglebot.solver.evaluate import EvaluationResult
from kagglebot.submission.guard import compute_error_fingerprint
from kagglebot.types import PlanConfig


@dataclass(frozen=True)
class TrainingOutcome:
    submission_path: Path
    evaluation: EvaluationResult
    model_name: str
    model_summary: dict[str, object]
    accelerator: str


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


def _write_kernel_metrics(path: Path, *, metric: str = "rmse", value: float = 0.4) -> None:
    payload = {
        "score_source": "holdout",
        "metric": metric,
        "direction": "minimize",
        "offline_value": value,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_config(tmp_path: Path, **overrides) -> AutopilotConfig:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    _write_sample_submission(paths.sample_submission_path)
    paths.prompts_dir.mkdir(parents=True, exist_ok=True)
    paths.codex_improve_template.write_text("improve {slug} {iteration}\n", encoding="utf-8")
    paths.codex_kernel_fix_template.write_text("fix {slug} {iteration}\n", encoding="utf-8")
    paths.codex_plan_and_implement_prompt.write_text("plan+implement\n", encoding="utf-8")
    paths.kernel_source_dir.mkdir(parents=True, exist_ok=True)
    (paths.kernel_source_dir / "kernel.py").write_text("print('kernel stub')\n", encoding="utf-8")
    base = AutopilotConfig(
        run_id="run-1",
        slug="demo",
        competition_url=None,
        paths=paths,
        knowledge_paths=knowledge_paths,
        agent="codex",
        compute="local_gpu",
        accelerator="gpu",
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


def test_should_skip_planning_requires_kernel_py(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_plan(paths)
    (paths.context_dir / "agent").mkdir(parents=True, exist_ok=True)

    assert _should_skip_planning(resume_run=True, paths=paths) is False

    paths.kernel_source_dir.mkdir(parents=True, exist_ok=True)
    (paths.kernel_source_dir / "kernel.py").write_text("# generated kernel\n", encoding="utf-8")

    assert _should_skip_planning(resume_run=True, paths=paths) is True


@pytest.fixture(autouse=True)
def _stub_local_kernel_runner(monkeypatch) -> None:
    def fake_run_kernel_local(**kwargs):  # noqa: ANN003
        from kagglebot import autopilot as autopilot_module

        slug = kwargs["slug"]
        run_id = kwargs["run_id"]
        iteration = kwargs["iteration"]
        base_dir = Path(kwargs["base_dir"])
        metric = kwargs["metric"]
        direction = kwargs["direction"]
        score_source = kwargs["score_source"]
        seed = kwargs["seed"]
        holdout_frac = kwargs["holdout_frac"]
        cv_folds = kwargs["cv_folds"]
        timeout_minutes = kwargs["timeout_minutes"]

        output_dir = base_dir / slug / "runs" / run_id / f"iter-{iteration}" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"

        trainer = getattr(autopilot_module, "train_evaluate_and_predict", None)
        evaluation = None
        if callable(trainer):
            try:
                outcome = trainer(
                    data_dir=base_dir / slug / "data",
                    output_path=submission_path,
                    compute=None,
                    strict_accelerator=False,
                    seed=seed,
                    score_source=score_source,
                    metric=metric,
                    direction=direction,
                    holdout_frac=holdout_frac,
                    cv_folds=cv_folds,
                    plan_score_source=None,
                    target_override=None,
                    time_budget_min=timeout_minutes,
                )
                evaluation = outcome.evaluation
            except RuntimeError as exc:
                if "Legacy src local trainer has been removed" not in str(exc):
                    raise

        if not submission_path.exists():
            submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        if evaluation is None:
            evaluation = EvaluationResult(
                score_source="holdout",
                metric=str(metric),
                direction=str(direction),
                value=0.4,
                std=None,
                train_score=None,
                val_score=None,
                fold_scores=None,
            )

        metrics_payload: dict[str, object] = {
            "score_source": evaluation.score_source,
            "metric": evaluation.metric,
            "direction": evaluation.direction,
            "offline_value": evaluation.value,
        }
        if evaluation.std is not None:
            metrics_payload["offline_std"] = evaluation.std
        if evaluation.fold_scores is not None:
            metrics_payload["fold_scores"] = list(evaluation.fold_scores)
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        return KernelRunResult(
            kernel_id=f"local/{slug}",
            output_dir=output_dir,
            submission_path=submission_path,
            metrics_path=metrics_path,
        )

    monkeypatch.setattr("kagglebot.autopilot.run_kernel_local", fake_run_kernel_local)


def test_autopilot_local_requires_kernel_when_legacy_disabled(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=False, max_iterations=1)
    (config.paths.kernel_source_dir / "kernel.py").unlink(missing_ok=True)
    with pytest.raises(RuntimeError, match="requires kernel.py"):
        run_autopilot(config)


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
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.list_competition_submissions",
        lambda *args, **kwargs: [{"description": "", "status": "complete", "publicScore": "0.40"}],
    )

    config = _make_config(tmp_path)
    run_autopilot(config)

    run_payload = json.loads((config.paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    assert run_payload["config"]["target_metric"] == "rmse"
    assert run_payload["config"]["target_score"] == 0.5


def test_resolve_plan_preserves_requested_score_source(tmp_path: Path) -> None:
    config = _make_config(tmp_path, score_source="cv")
    plan = PlanConfig(score_source="cv")

    resolved = _resolve_plan(plan, config)

    assert resolved["score_source"] == "cv"


def test_resolve_plan_applies_evaluation_spec_when_plan_uses_defaults(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    spec = {
        "metric_name": "rmse",
        "direction": "minimize",
        "split_strategy": "kfold",
        "n_splits": 4,
        "seeds": [42, 2024, 777],
        "repeats": 2,
        "ci_method": "bootstrap",
        "ci_alpha": 0.1,
        "readiness_rule": {
            "method": "mean_std",
            "k": 0.7,
            "target_score": 0.33,
            "submission_gate": "final_only",
        },
        "drift_check": {"enabled": True, "drift_weight": 0.5},
        "stop_policy": {"min_delta": 0.01, "no_improve_patience": 3, "same_config_patience": 2},
    }
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.context_dir / "evaluation_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    plan = PlanConfig(target_metric="rmse", target_direction="minimize", target_score=0.4)
    resolved = _resolve_plan(plan, config)

    assert resolved["submission_gate"] == "final_only"
    assert resolved["readiness_method"] == "mean_std"
    assert resolved["readiness_k"] == pytest.approx(0.7)
    assert resolved["ci_method"] == "bootstrap"
    assert resolved["ci_alpha"] == pytest.approx(0.1)
    assert resolved["drift_check"] is True
    assert resolved["drift_weight"] == pytest.approx(0.5)
    assert resolved["stop_min_delta"] == pytest.approx(0.01)
    assert resolved["stop_no_improve_patience"] == 3
    assert resolved["stop_same_config_patience"] == 2


def test_autopilot_submit_when_top1_tier_single_iteration(monkeypatch, tmp_path: Path) -> None:
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
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: (
            submission_calls.append(kwargs["submission_file"]),
            type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        )[1],
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.list_competition_submissions",
        lambda *args, **kwargs: [{"description": "", "status": "complete", "publicScore": "0.49"}],
    )
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_autopilot(config)
    assert len(submission_calls) == 1


def test_autopilot_writes_evaluation_report_and_uses_readiness_loop_decision(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="auc",
        target_score=0.90,
        target_direction="maximize",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.9\n2,0.8\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="auc",
            direction="maximize",
            value=0.86,
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
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=False, max_iterations=2)
    run_autopilot(config)

    iter1_report_path = config.paths.iter_dir("run-1", 1) / "evaluation_report.json"
    iter2_report_path = config.paths.iter_dir("run-1", 2) / "evaluation_report.json"
    assert iter1_report_path.exists()
    assert iter2_report_path.exists()

    iter1_report = json.loads(iter1_report_path.read_text(encoding="utf-8"))
    iter1_metrics = json.loads((config.paths.iter_dir("run-1", 1) / "metrics.json").read_text(encoding="utf-8"))
    assert iter1_metrics["loop_decision"]["source"] == "readiness"
    assert iter1_metrics["loop_decision"]["value"] == pytest.approx(iter1_report["readiness_score"])

    run_report = json.loads((config.paths.run_dir("run-1") / "evaluation_report.json").read_text(encoding="utf-8"))
    assert run_report["latest_iteration"] == 2
    assert len(run_report["history"]) == 2


@pytest.mark.parametrize(
    ("metric", "direction", "value", "target", "expected_submit_calls"),
    [
        ("auc", "maximize", 0.70, 0.80, 0),
        ("rmse", "minimize", 0.40, 0.50, 1),
    ],
)
def test_autopilot_submission_gate_uses_readiness_direction(
    monkeypatch,
    tmp_path: Path,
    metric: str,
    direction: str,
    value: float,
    target: float,
    expected_submit_calls: int,
) -> None:
    submit_calls = {"count": 0}

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric=metric,
        target_score=target,
        target_direction=direction,
        submission_gate="readiness_only",
        readiness_target_score=target,
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="holdout",
            metric=metric,
            direction=direction,
            value=value,
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
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: (
            submit_calls.update(count=submit_calls["count"] + 1),
            type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        )[1],
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.list_competition_submissions",
        lambda *args, **kwargs: [{"description": "", "status": "complete", "publicScore": "0.40"}],
    )
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_autopilot(config)

    assert submit_calls["count"] == expected_submit_calls


def test_autopilot_aborts_on_repeated_submit_fingerprint(monkeypatch, tmp_path: Path) -> None:
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

    fail_calls = {"count": 0}

    def always_transient_fail(*args, **kwargs):  # noqa: ARG001
        fail_calls["count"] += 1
        raise SubmissionCliError(
            "submit failed",
            command=["kaggle", "competitions", "submit"],
            exit_code=1,
            output="503 temporary failure",
            stdout="503 temporary failure",
            stderr="ConnectionError: temporarily unavailable",
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", always_transient_fail)
    monkeypatch.setattr("kagglebot.autopilot._is_submit_abort_autofixable", lambda *args, **kwargs: False)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    with pytest.raises(SubmitAbortedError):
        run_autopilot(config)

    run_dir = config.paths.run_dir(config.run_id or "run-1")
    attempts_path = run_dir / "submit_attempts.jsonl"
    assert attempts_path.exists()
    rows = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 2
    assert fail_calls["count"] == 2
    required_fields = {
        "ts",
        "run_id",
        "sub_path",
        "sub_sha256",
        "ok",
        "exit_code",
        "fingerprint",
        "stdout_tail",
        "stderr_tail",
        "action_taken",
    }
    assert required_fields.issubset(rows[-1].keys())
    assert rows[-1]["action_taken"] == "abort"
    assert rows[-1]["reason"] == "same_error_fingerprint_recurred"
    run_state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert run_state["submit_attempted"] is True
    assert run_state["submit_ok"] is False
    assert isinstance(run_state.get("last_submit_fingerprint"), str)


def test_autopilot_transient_retry_stops_after_max_attempts(monkeypatch, tmp_path: Path) -> None:
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

    fail_calls = {"count": 0}

    def transient_fail_unique_fingerprint(*args, **kwargs):  # noqa: ARG001
        fail_calls["count"] += 1
        n = fail_calls["count"]
        raise SubmissionCliError(
            "submit failed",
            command=["kaggle", "competitions", "submit"],
            exit_code=1,
            output=f"503 Service Unavailable #{n}",
            stdout=f"503 Service Unavailable #{n}",
            stderr=f"ConnectionError: temporarily unavailable attempt={n}",
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", transient_fail_unique_fingerprint)
    monkeypatch.setattr("kagglebot.autopilot._is_submit_abort_autofixable", lambda *args, **kwargs: False)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.time.sleep", lambda *_args, **_kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    with pytest.raises(SubmitAbortedError):
        run_autopilot(config)

    run_dir = config.paths.run_dir(config.run_id or "run-1")
    rows = [json.loads(line) for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row]
    assert fail_calls["count"] == 3
    assert [row["action_taken"] for row in rows] == ["retry", "retry", "abort"]
    assert rows[-1]["reason"] == "network_or_timeout"


def test_autopilot_validation_failure_aborts_before_kaggle_submit(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,not_a_number\n", encoding="utf-8")
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

    rules_calls = {"count": 0}
    submit_calls = {"count": 0}

    def fake_check_rules(*args, **kwargs):  # noqa: ARG001
        rules_calls["count"] += 1
        return True

    def fake_submit(*args, **kwargs):  # noqa: ARG001
        submit_calls["count"] += 1
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", fake_check_rules)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fake_submit)
    monkeypatch.setattr("kagglebot.autopilot._is_submit_abort_autofixable", lambda *args, **kwargs: False)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    with pytest.raises(SubmitAbortedError):
        run_autopilot(config)

    run_dir = config.paths.run_dir(config.run_id or "run-1")
    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["action_taken"] == "abort"
    assert rows[-1]["reason"] == "local_submission_validation_failed"
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["submit_attempted"] is True
    assert state["submit_ok"] is False
    assert rules_calls["count"] == 0
    assert submit_calls["count"] == 0


def test_autopilot_resume_allows_submit_after_prior_attempt(monkeypatch, tmp_path: Path) -> None:
    submit_calls = {"count": 0}
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
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: (
            submit_calls.update(count=submit_calls["count"] + 1),
            type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        )[1],
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.list_competition_submissions",
        lambda *args, **kwargs: [{"description": "", "status": "complete", "publicScore": "0.49"}],
    )
    monkeypatch.setattr("kagglebot.autopilot._is_submit_abort_autofixable", lambda *args, **kwargs: False)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_dir = config.paths.run_dir(config.run_id or "run-1")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {"submit_attempted": True, "submit_ok": False, "last_submit_fingerprint": "abc"},
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "submit_attempts.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "fingerprint": "abc", "ok": False}) + "\n",
        encoding="utf-8",
    )

    run_autopilot(config)
    assert submit_calls["count"] == 1
    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["action_taken"] == "submit"


def test_autopilot_force_submit_aborts_on_state_fingerprint_repeat(monkeypatch, tmp_path: Path) -> None:
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

    repeated_stdout = "503 temporary failure"
    repeated_stderr = "ConnectionError: temporarily unavailable"
    repeated_fp = compute_error_fingerprint(repeated_stdout, repeated_stderr)

    def always_transient_fail(*args, **kwargs):  # noqa: ARG001
        raise SubmissionCliError(
            "submit failed",
            command=["kaggle", "competitions", "submit"],
            exit_code=1,
            output=f"{repeated_stdout}\n{repeated_stderr}",
            stdout=repeated_stdout,
            stderr=repeated_stderr,
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", always_transient_fail)
    monkeypatch.setattr("kagglebot.autopilot._is_submit_abort_autofixable", lambda *args, **kwargs: False)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1, force_submit=True)
    run_dir = config.paths.run_dir(config.run_id or "run-1")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_submit_fingerprint": repeated_fp,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(SubmitAbortedError):
        run_autopilot(config)

    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert rows[-1]["action_taken"] == "abort"
    assert rows[-1]["reason"] == "same_error_fingerprint_recurred"


def test_autopilot_submit_validation_error_autofixes_and_resubmits(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )

    train_calls = {"count": 0}

    def fake_train(*args, **kwargs):  # noqa: ARG001
        train_calls["count"] += 1
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,not_a_number\n", encoding="utf-8")
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

    submit_calls = {"count": 0}
    autofix_calls = {"count": 0}
    planning_calls = {"count": 0}

    def fake_submit(*args, **kwargs):  # noqa: ARG001
        submit_calls["count"] += 1
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    def fake_submit_autofix(*, config: AutopilotConfig, run_id: str, attempt: int, error: Exception):  # noqa: ARG001
        autofix_calls["count"] += 1
        iter_submission = config.paths.iter_dir(run_id, 1) / "submission.csv"
        iter_submission.parent.mkdir(parents=True, exist_ok=True)
        iter_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        output_submission = config.paths.iter_dir(run_id, 1) / "output" / "submission.csv"
        output_submission.parent.mkdir(parents=True, exist_ok=True)
        output_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fake_submit)
    monkeypatch.setattr(
        "kagglebot.autopilot._run_plan_and_initial",
        lambda *args, **kwargs: planning_calls.update(count=planning_calls["count"] + 1),
    )
    monkeypatch.setattr("kagglebot.autopilot._run_autofix", fake_submit_autofix)
    monkeypatch.setattr(
        "kagglebot.autopilot.list_competition_submissions",
        lambda *args, **kwargs: [{"description": "", "status": "complete", "publicScore": "0.40"}],
    )

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_autopilot(config)

    run_dir = config.paths.run_dir(config.run_id or "run-1")
    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert autofix_calls["count"] == 1
    assert submit_calls["count"] == 1
    assert train_calls["count"] == 1
    assert planning_calls["count"] == 1
    assert rows[0]["action_taken"] == "abort"
    assert rows[-1]["action_taken"] == "submit"


def test_build_submit_autofix_context_includes_latest_attempt(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    run_dir = config.paths.run_dir(config.run_id or "run-1")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_error_kind": "validation",
                "last_reason": "local_submission_validation_failed",
                "last_action": "abort",
                "last_submit_fingerprint": "abc123",
                "last_submission_path": "/tmp/submission.csv",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "submit_attempts.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-02-15T00:00:00+00:00",
                "ok": False,
                "exit_code": 6,
                "error_kind": "validation",
                "reason": "local_submission_validation_failed",
                "action_taken": "abort",
                "fingerprint": "abc123",
                "sub_path": "/tmp/submission.csv",
                "stdout_tail": "",
                "stderr_tail": "Submission validation failed: prediction column contains NaN",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    context = _build_submit_autofix_context(run_dir)

    assert "run_state:" in context
    assert "latest_submit_attempt:" in context
    assert "last_reason: local_submission_validation_failed" in context
    assert "error_kind: validation" in context
    assert "stderr_tail: Submission validation failed:" in context


def test_run_autofix_submit_error_always_runs_strategy_then_codex(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_error_kind": "validation",
                "last_reason": "local_submission_validation_failed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "submit_attempts.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-02-15T00:00:00+00:00",
                "ok": False,
                "exit_code": 6,
                "error_kind": "validation",
                "reason": "local_submission_validation_failed",
                "action_taken": "abort",
                "fingerprint": "abc123",
                "sub_path": "/tmp/submission.csv",
                "stderr_tail": "missing columns: ['target']",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls = {"strategy": 0, "codex": 0}

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    monkeypatch.setattr("kagglebot.autopilot._maybe_write_column_fill", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.autopilot._maybe_write_object_coerce", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.autopilot._maybe_write_device_coerce", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.autopilot._maybe_write_column_map", lambda *args, **kwargs: True)

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool):  # noqa: ARG001
        calls["strategy"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "strategy_last_message.txt"
        last_msg.write_text("1) fix submit format\n", encoding="utf-8")
        return DummyResult(last_msg)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        calls["codex"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("submit fix applied\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.run_strategy", fake_run_strategy)
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._backup_guarded_files", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._snapshot_tree", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._diff_snapshots", lambda *args, **kwargs: [])
    monkeypatch.setattr("kagglebot.autopilot._enforce_allowlist_changes", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._maybe_restart_for_src_changes", lambda *args, **kwargs: None)

    _run_autofix(
        config=config,
        run_id=run_id,
        attempt=1,
        error=SubmitAbortedError("Local submission validation failed; Kaggle CLI submit is skipped."),
    )

    assert calls["strategy"] == 1
    assert calls["codex"] == 1
    prompt_text = (run_dir / "autofix" / "attempt-1" / "prompt.md").read_text(encoding="utf-8")
    assert "## Submit Context" in prompt_text
    strategy_prompt = (run_dir / "autofix" / "attempt-1" / "gpt_strategy" / "gpt_strategy_prompt.md").read_text(
        encoding="utf-8"
    )
    assert "Stage: submit_autofix" in strategy_prompt


def test_extract_kernel_metric_from_oof_dict() -> None:
    from kagglebot.autopilot import _extract_kernel_metric

    payload = {
        "oof_rmse": {
            "lgb": 8.75,
            "catboost": 8.79,
            "xgboost": 8.84,
            "stacked": 8.76,
            "average": 8.77,
            "selected": 8.76,
        },
        "selection": "selected",
    }
    metric, value = _extract_kernel_metric(payload, "rmse")
    assert metric == "rmse"
    assert value == 8.76


def test_extract_kernel_metric_from_selected_combined_score_schema() -> None:
    from kagglebot.autopilot import _extract_kernel_metric

    payload = {
        "run_id": "run_20260217_043129",
        "primary_metric": "0.5*mAP@[0.5:0.95] + 0.5*F1",
        "selected": {
            "name": "yolo11m_kfold_wbf_geom_rp",
            "mean_map": 0.6698263357562932,
            "oof_f1": 0.6666666666666666,
            "combined_score": 0.66824650121148,
        },
    }
    metric, value = _extract_kernel_metric(payload, "0.5*mAP@[0.5:0.95] + 0.5*F1")
    assert metric == "0.5*mAP@[0.5:0.95] + 0.5*F1"
    assert value == 0.66824650121148


def test_extract_kernel_metric_prefers_map_when_primary_metric_is_map() -> None:
    from kagglebot.autopilot import _extract_kernel_metric

    payload = {
        "primary_metric": "mAP@[0.5:0.95]",
        "selected": {
            "mean_map": 0.669,
            "oof_f1": 0.123,
            "combined_score": 0.456,
        },
    }
    metric, value = _extract_kernel_metric(payload, "mAP@[0.5:0.95]")
    assert metric == "mAP@[0.5:0.95]"
    assert value == 0.669


def test_metric_mismatch_preserves_direction_when_metric_direction_is_unknown(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="0.5 * mAP@[0.5:0.95] + 0.5 * F1-score",
        target_score=0.9,
        target_direction="maximize",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="composite",
            direction="maximize",
            value=0.62,
            std=0.01,
            train_score=None,
            val_score=0.62,
            fold_scores=[0.61, 0.63],
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="vision",
            model_summary={},
            accelerator="cuda",
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path, submit=False, max_iterations=1)
    run_autopilot(config)

    persisted_plan = json.loads(config.paths.plan_path.read_text(encoding="utf-8"))
    assert persisted_plan["target_metric"] == "composite"
    assert persisted_plan["target_direction"] == "maximize"


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
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: (
            submission_calls.append(kwargs["submission_file"]),
            type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        )[1],
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.list_competition_submissions",
        lambda *args, **kwargs: [{"description": "", "status": "complete", "publicScore": "0.11"}],
    )
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_autopilot(config)
    assert len(submission_calls) == 1


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
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: (
            submission_calls.append(kwargs["submission_file"]),
            type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        )[1],
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.list_competition_submissions",
        lambda *args, **kwargs: [{"description": "", "status": "complete", "publicScore": "0.95"}],
    )
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_autopilot(config)
    assert len(submission_calls) == 1


def test_autopilot_skips_submit_when_kaggle_credentials_missing(monkeypatch, tmp_path: Path) -> None:
    submit_calls: list[Path] = []

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

    def fake_check_rules(*args, **kwargs):  # noqa: ARG001
        raise KaggleCliError(
            "Kaggle CLI failed with exit code 1.",
            command=["kaggle", "competitions", "list", "--search", "demo", "--csv"],
            exit_code=1,
            output=(
                "OSError: Could not find kaggle.json. Make sure it's located in "
                "/home/moeu/.config/kaggle. Or use the environment method."
            ),
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.1})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", fake_check_rules)
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: (
            submit_calls.append(kwargs["submission_file"]),
            type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        )[1],
    )
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    with pytest.raises(SubmitAbortedError):
        run_autopilot(config)

    run_payload = json.loads((config.paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    assert run_payload["status"] == "submit_failed"
    assert len(submit_calls) == 0


def test_autopilot_stops_when_target_missing(monkeypatch, tmp_path: Path) -> None:
    calls = {"train": 0, "submit": 0}

    def fake_train(*args, **kwargs):  # noqa: ARG001
        calls["train"] += 1
        raise AssertionError("train should not be called when target missing")

    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: (
            calls.update(submit=1),
            type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        )[1],
    )

    config = _make_config(tmp_path)
    run_autopilot(config)
    run_payload = json.loads((config.paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    assert run_payload["status"] == "missing_target"
    assert calls["train"] == 0
    assert calls["submit"] == 0


def test_autopilot_records_problem_knowledge_after_submission_result(monkeypatch, tmp_path: Path) -> None:
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
            value=0.9,
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
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.list_competition_submissions",
        lambda *args, **kwargs: [{"description": "", "status": "complete", "publicScore": "0.95"}],
    )
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_autopilot(config)

    insights = resolve_problem_type_insights(config.knowledge_paths, ["unknown"], limit=5)
    assert insights
    assert insights[0]["outcome_bucket"] in {"good", "low"}
    assert insights[0]["submission_score"] == 0.95


def test_autopilot_refreshes_knowledge_hints(monkeypatch, tmp_path: Path) -> None:
    search_calls = {"count": 0}

    def fake_resolve_similar_improvements(**kwargs):  # noqa: ARG001
        search_calls["count"] += 1
        return [
            {
                "slug": "house-prices-advanced-regression-techniques",
                "overlap": 2,
                "summary": "CatBoost + robust feature engineering improved offline RMSE.",
            }
        ]

    monkeypatch.setattr("kagglebot.autopilot.ensure_taxonomy", lambda *args, **kwargs: {"tags": [], "aliases": {}})
    monkeypatch.setattr("kagglebot.autopilot.resolve_similar_improvements", fake_resolve_similar_improvements)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path)
    config.paths.dataset_profile_path.parent.mkdir(parents=True, exist_ok=True)
    config.paths.dataset_profile_path.write_text(json.dumps({"tags": ["tabular", "binary"]}), encoding="utf-8")

    run_autopilot(config)

    hints = config.paths.knowledge_hints_path.read_text(encoding="utf-8")
    assert "house-prices-advanced-regression-techniques" in hints
    assert "CatBoost + robust feature engineering improved offline RMSE." in hints
    assert search_calls["count"] >= 1


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

    codex_kwargs_seen: list[dict[str, object]] = []

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        codex_kwargs_seen.append(kwargs)
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("improved features\n", encoding="utf-8")
        return DummyResult(last_msg)

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "strategy_last_message.txt"
        last_msg.write_text("1) tune model\n2) increase training budget\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.1})
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.autopilot.run_strategy", fake_run_strategy)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, max_iterations=2)
    run_autopilot(config)
    iter_dir = config.paths.iter_dir(config.run_id or "run-1", 1)
    assert (iter_dir / "agent" / "prompt.md").exists()
    assert any(kwargs.get("model") == "gpt-5.3-codex" for kwargs in codex_kwargs_seen)
    assert any(kwargs.get("reasoning_effort") == "extra_high" for kwargs in codex_kwargs_seen)


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

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        calls["codex"] += 1
        if prompt_path.name == "kernel_fix_prompt.md":
            calls["kernel_fix"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("kernel fix applied\n", encoding="utf-8")
        return DummyResult(last_msg)

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "strategy_last_message.txt"
        last_msg.write_text("1) Fix import path\n2) Update kernel fallback\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.run_kernel", lambda **kwargs: fake_run_kernel(**kwargs))
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.autopilot.run_strategy", fake_run_strategy)
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
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.list_competition_submissions",
        lambda *args, **kwargs: [{"description": "", "status": "complete", "publicScore": "1.00"}],
    )

    config = _make_config(tmp_path, max_iterations=10)
    run_autopilot(config)
    assert calls["train"] == 10


def test_autopilot_top1_stop_requires_submission_score(monkeypatch, tmp_path: Path) -> None:
    calls = {"submit": 0}
    submission_scores = [0.6, 0.4]

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        source = str(kwargs.get("score_source") or "holdout")
        value = 0.4 if source == "holdout" else 0.41
        evaluation = EvaluationResult(
            score_source=source,
            metric="rmse",
            direction="minimize",
            value=value,
            std=0.001 if source == "cv" else 0.0,
            train_score=None,
            val_score=value,
            fold_scores=[value - 0.001, value + 0.001] if source == "cv" else [value],
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="ridge",
            model_summary={},
            accelerator="cpu",
        )

    def fake_attempt_submit(*, config, run_id, submission_path, best_score, problem_types):  # noqa: ARG001
        calls["submit"] += 1
        score = submission_scores[min(calls["submit"] - 1, len(submission_scores) - 1)]
        rank = 10 if calls["submit"] == 1 else 1
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-02-16T00:00:00+00:00",
            "iteration": calls["submit"],
            "outcome": {
                "status": "complete",
                "score": score,
                "rank": rank,
                "total_teams": 2700,
                "rank_source": "submission_row",
            },
        }

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})

    config = _make_config(tmp_path, submit=True, max_iterations=3)
    run_autopilot(config)

    run_payload = json.loads((config.paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    assert calls["submit"] == 2
    assert run_payload.get("stop_reason") == "submission_rank_1"
    assert (config.paths.iter_dir("run-1", 3) / "metrics.json").exists() is False


def test_resolve_submission_rank_payload_keeps_estimate_separate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "kagglebot.autopilot.leaderboard_rank_for_score",
        lambda **kwargs: {
            "rank": 19,
            "total_teams": 151,
            "rank_percentile": 19 / 151,
            "source": "kaggle competitions leaderboard --download",
        },
    )
    payload = _resolve_submission_rank_payload(
        slug="demo",
        context_dir=tmp_path,
        direction="maximize",
        outcome={"status": "complete", "score": 0.307},
        dry_run=False,
    )
    assert "rank" not in payload
    assert "total_teams" not in payload
    assert payload["estimated_rank"] == 19
    assert payload["estimated_total_teams"] == 151
    assert payload["rank_estimate_source"] == "leaderboard_score_estimate"


def test_autopilot_does_not_stop_on_estimated_rank_one(monkeypatch, tmp_path: Path) -> None:
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
            std=0.001,
            train_score=None,
            val_score=0.4,
            fold_scores=[0.399, 0.401],
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="ridge",
            model_summary={},
            accelerator="cpu",
        )

    def fake_attempt_submit(*, config, run_id, submission_path, best_score, problem_types):  # noqa: ARG001
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-02-16T00:00:00+00:00",
            "iteration": 1,
            "outcome": {"status": "complete", "score": 0.307},
        }

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.2})
    monkeypatch.setattr(
        "kagglebot.autopilot.leaderboard_rank_for_score",
        lambda **kwargs: {
            "rank": 1,
            "total_teams": 100,
            "rank_percentile": 0.01,
            "source": "kaggle competitions leaderboard --download",
        },
    )

    config = _make_config(tmp_path, submit=True, max_iterations=2)
    run_autopilot(config)

    run_payload = json.loads((config.paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    assert run_payload.get("stop_reason") != "submission_rank_1"
    assert (config.paths.iter_dir("run-1", 2) / "metrics.json").exists()


def test_autopilot_submit_mode_ignores_no_improve_stop_policy(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        stop_no_improve_patience=1,
        stop_same_config_patience=1,
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            value=0.4,
            std=0.001,
            train_score=None,
            val_score=0.4,
            fold_scores=[0.399, 0.401],
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="ridge",
            model_summary={},
            accelerator="cpu",
        )

    def fake_attempt_submit(*, config, run_id, submission_path, best_score, problem_types):  # noqa: ARG001
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-02-16T00:00:00+00:00",
            "iteration": 1,
            "outcome": {"status": "complete", "score": 0.307},
        }

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path, submit=True, max_iterations=2)
    run_autopilot(config)

    run_payload = json.loads((config.paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    assert run_payload.get("stop_reason") != "submission_rank_1"
    assert (config.paths.iter_dir("run-1", 2) / "metrics.json").exists()


def test_autopilot_forces_major_overhaul_after_noise_limited_streak(monkeypatch, tmp_path: Path) -> None:
    forced_modes: list[tuple[int, str | None, str | None]] = []
    cv_values = {1: 0.95500, 2: 0.95505, 3: 0.95509, 4: 0.95512}

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="AUC-ROC",
        target_score=0.965,
        target_direction="maximize",
        submission_gate="always",
    )

    def _iter_from_output(path: Path) -> int:
        for parent in [path.parent, *path.parents]:
            name = parent.name
            if name.startswith("iter-"):
                return int(name.split("-", 1)[1])
        return 1

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.9\n2,0.8\n", encoding="utf-8")
        iteration = _iter_from_output(output_path)
        cv_value = cv_values.get(iteration, cv_values[max(cv_values)])
        fold_scores = [cv_value - 0.0012, cv_value - 0.0006, cv_value, cv_value + 0.0006, cv_value + 0.0012]
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="AUC-ROC",
            direction="maximize",
            value=cv_value,
            std=0.0012,
            train_score=None,
            val_score=cv_value,
            fold_scores=fold_scores,
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="catboost",
            model_summary={},
            accelerator="cuda",
        )

    def fake_improvement(**kwargs):
        forced_modes.append(
            (
                int(kwargs["iteration"]),
                kwargs.get("forced_improvement_mode"),
                kwargs.get("forced_improvement_reason"),
            )
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", fake_improvement)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path, submit=False, max_iterations=4)
    run_autopilot(config)

    assert len(forced_modes) == 3
    assert forced_modes[0][1] is None
    assert forced_modes[1][1] is None
    assert forced_modes[2][1] == "major_overhaul"
    assert forced_modes[2][2] and "noise-limited" in forced_modes[2][2]


def test_autopilot_forces_major_overhaul_when_submission_rank_is_poor(monkeypatch, tmp_path: Path) -> None:
    forced_modes: list[tuple[int, str | None, str | None]] = []

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="AUC-ROC",
        target_score=0.965,
        target_direction="maximize",
        submission_gate="always",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.9\n2,0.8\n", encoding="utf-8")
        source = str(kwargs.get("score_source") or "holdout")
        if source == "cv":
            evaluation = EvaluationResult(
                score_source="cv",
                metric="AUC-ROC",
                direction="maximize",
                value=0.9550,
                std=0.0010,
                train_score=None,
                val_score=0.9550,
                fold_scores=[0.9540, 0.9548, 0.9550, 0.9552, 0.9560],
            )
        else:
            evaluation = EvaluationResult(
                score_source=source,
                metric="AUC-ROC",
                direction="maximize",
                value=0.9551,
                std=0.0,
                train_score=None,
                val_score=0.9551,
                fold_scores=[0.9551],
            )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="catboost",
            model_summary={},
            accelerator="cuda",
        )

    def fake_attempt_submit(*, config, run_id, submission_path, best_score, problem_types):  # noqa: ARG001
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-02-16T00:00:00+00:00",
            "iteration": 1,
            "outcome": {
                "status": "complete",
                "score": 0.9532,
                "rank": 1300,
                "total_teams": 2700,
                "rank_source": "submission_row",
            },
        }

    def fake_improvement(**kwargs):
        forced_modes.append(
            (
                int(kwargs["iteration"]),
                kwargs.get("forced_improvement_mode"),
                kwargs.get("forced_improvement_reason"),
            )
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", fake_improvement)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path, submit=True, max_iterations=2)
    run_autopilot(config)

    assert len(forced_modes) == 1
    assert forced_modes[0][1] == "major_overhaul"
    assert forced_modes[0][2] and "1300/2700" in forced_modes[0][2]


def test_autopilot_evaluation_report_uses_multiseed_defaults(monkeypatch, tmp_path: Path) -> None:
    calls = {"train": 0}

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="AUC-ROC",
        target_score=0.965,
        target_direction="maximize",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        calls["train"] += 1
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.9\n2,0.8\n", encoding="utf-8")
        source = str(kwargs.get("score_source") or "holdout")
        if source == "cv":
            evaluation = EvaluationResult(
                score_source="cv",
                metric="AUC-ROC",
                direction="maximize",
                value=0.9550,
                std=0.0010,
                train_score=None,
                val_score=0.9550,
                fold_scores=[0.9540, 0.9548, 0.9550, 0.9552, 0.9560],
            )
        else:
            evaluation = EvaluationResult(
                score_source=source,
                metric="AUC-ROC",
                direction="maximize",
                value=0.9552,
                std=0.0,
                train_score=None,
                val_score=0.9552,
                fold_scores=[0.9552],
            )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="catboost",
            model_summary={},
            accelerator="cuda",
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path, submit=False, max_iterations=1)
    run_autopilot(config)

    report = json.loads((config.paths.iter_dir("run-1", 1) / "evaluation_report.json").read_text(encoding="utf-8"))
    assert report["seeds"] == [42, 2024, 777]
    assert report["repeats"] == 2
    assert isinstance(report["split_index_fingerprints"], list)
    assert calls["train"] == 1


def test_resume_iteration_state_does_not_complete_with_submission_only(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    iter_dir = paths.iter_dir("run-1", 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "submission.csv").write_text("id,target\n1,0.1\n", encoding="utf-8")

    start, best_score, best_submission = _resume_iteration_state(
        paths=paths,
        run_id="run-1",
        metric_direction="minimize",
        target_metric="rmse",
        max_iterations=3,
        require_submit_phase=False,
    )

    assert start == 1
    assert best_score is None
    assert best_submission is None


def test_autopilot_resume_submit_only_from_legacy_output_artifacts(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )

    train_calls = {"count": 0}
    submit_calls = {"count": 0}

    def fake_train(*args, **kwargs):  # noqa: ARG001
        train_calls["count"] += 1
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

    def fake_attempt_submit(*, config, run_id, submission_path, best_score, problem_types):  # noqa: ARG001
        submit_calls["count"] += 1
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-02-16T00:00:00+00:00",
            "iteration": 1,
            "outcome": {"status": "complete", "score": 0.49},
        }

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.autopilot._run_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    iter_output_dir = config.paths.iter_dir("run-1", 1) / "output"
    iter_output_dir.mkdir(parents=True, exist_ok=True)
    (iter_output_dir / "submission.csv").write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    _write_kernel_metrics(iter_output_dir / "metrics.json", value=0.4)

    run_autopilot(config)

    assert train_calls["count"] == 0
    assert submit_calls["count"] == 1
    assert (config.paths.iter_dir("run-1", 1) / "submission.csv").exists()
    assert (config.paths.iter_dir("run-1", 1) / "metrics.json").exists()


def test_resume_iteration_state_requires_submit_phase_for_legacy_runs(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    iter_dir = paths.iter_dir("run-1", 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    submission_path = iter_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    _write_kernel_metrics(iter_dir / "metrics.json", value=0.4321)

    start_before, _, _ = _resume_iteration_state(
        paths=paths,
        run_id="run-1",
        metric_direction="minimize",
        target_metric="rmse",
        max_iterations=3,
        require_submit_phase=True,
    )
    assert start_before == 1

    run_dir = paths.run_dir("run-1")
    (run_dir / "submit_attempts.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "sub_path": str(submission_path),
                "action_taken": "submit",
                "reason": "submitted",
                "ok": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    start_after, best_score, best_submission = _resume_iteration_state(
        paths=paths,
        run_id="run-1",
        metric_direction="minimize",
        target_metric="rmse",
        max_iterations=3,
        require_submit_phase=True,
    )

    assert start_after == 2
    assert best_score == pytest.approx(0.4321)
    assert best_submission == submission_path


def test_resume_iteration_state_uses_iteration_marker_for_submit_phase(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    iter_dir = paths.iter_dir("run-1", 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    submission_path = iter_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    _write_kernel_metrics(iter_dir / "metrics.json", value=0.401)
    (iter_dir / "iteration_state.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "iteration": 1,
                "iteration_complete": True,
                "submit_phase_finished": True,
                "submit_phase_state": "skipped_gate",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    start, best_score, best_submission = _resume_iteration_state(
        paths=paths,
        run_id="run-1",
        metric_direction="minimize",
        target_metric="rmse",
        max_iterations=3,
        require_submit_phase=True,
    )

    assert start == 2
    assert best_score == pytest.approx(0.401)
    assert best_submission == submission_path


def test_resume_iteration_state_requires_submit_when_gate_allows(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    iter_dir = paths.iter_dir("run-1", 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    submission_path = iter_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    _write_kernel_metrics(iter_dir / "metrics.json", value=0.351)
    (iter_dir / "iteration_state.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "iteration": 1,
                "iteration_complete": True,
                "submit_phase_finished": False,
                "submit_allowed_by_gate": True,
                "submitted": False,
                "submit_phase_state": "attempted_no_result",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    start, best_score, best_submission = _resume_iteration_state(
        paths=paths,
        run_id="run-1",
        metric_direction="minimize",
        target_metric="rmse",
        max_iterations=3,
        require_submit_phase=True,
    )

    assert start == 1
    assert best_score is None
    assert best_submission is None


def test_write_iteration_state_marker_derives_submit_phase_finished(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    iter_dir = paths.iter_dir("run-1", 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    submission_path = iter_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    metrics_path = iter_dir / "metrics.json"
    _write_kernel_metrics(metrics_path, value=0.401)
    evaluation_report_path = iter_dir / "evaluation_report.md"
    evaluation_report_path.write_text("# report\n", encoding="utf-8")

    _write_iteration_state_marker(
        iter_dir=iter_dir,
        run_id="run-1",
        iteration=1,
        submission_path=submission_path,
        metrics_path=metrics_path,
        evaluation_report_path=evaluation_report_path,
        submit_phase_required=True,
        submit_allowed_by_gate=True,
        submit_phase_state="attempted_no_result",
        submitted=False,
        readiness_score=0.123,
    )

    payload = json.loads((iter_dir / "iteration_state.json").read_text(encoding="utf-8"))
    assert payload["submit_phase_required"] is True
    assert payload["submit_phase_finished"] is False


def test_autofix_writes_column_fill(tmp_path: Path) -> None:
    from kagglebot.autopilot import _maybe_write_column_fill

    config = _make_config(tmp_path)
    error_text = "ValueError: test.csv missing columns: ['col_a', 'col_b']"
    assert _maybe_write_column_fill(config, error_text) is True
    fill_path = config.paths.context_dir / "column_fill.json"
    assert fill_path.exists()
    payload = json.loads(fill_path.read_text(encoding="utf-8"))
    assert payload["files"]["test.csv"] == ["col_a", "col_b"]


def test_autofix_allows_src_edits(tmp_path: Path) -> None:
    from kagglebot import autopilot as autopilot_mod
    from kagglebot.autopilot import _autofix_allowed_prefixes

    config = _make_config(tmp_path)
    allowed_prefixes = _autofix_allowed_prefixes(config)
    assert config.paths.repo_root / "src" in allowed_prefixes
    module_src_root = Path(autopilot_mod.__file__).resolve().parents[1]
    if module_src_root.name == "src":
        assert module_src_root in allowed_prefixes
    assert config.paths.base_dir not in allowed_prefixes
    assert config.paths.kernels_dir not in allowed_prefixes
    assert config.paths.data_dir not in allowed_prefixes


def test_autofix_writes_object_coerce(tmp_path: Path) -> None:
    from kagglebot.autopilot import _maybe_write_object_coerce

    config = _make_config(tmp_path)
    error_text = "TypeError: can't convert np.ndarray of type numpy.object_"
    assert _maybe_write_object_coerce(config, error_text) is True
    coerce_path = config.paths.context_dir / "object_coerce.json"
    assert coerce_path.exists()
    payload = json.loads(coerce_path.read_text(encoding="utf-8"))
    assert payload["enabled"] is True


def test_autofix_writes_device_coerce(tmp_path: Path) -> None:
    from kagglebot.autopilot import _maybe_write_device_coerce

    config = _make_config(tmp_path)
    error_text = (
        "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"
    )
    assert _maybe_write_device_coerce(config, error_text) is True
    coerce_path = config.paths.context_dir / "device_coerce.json"
    assert coerce_path.exists()
    payload = json.loads(coerce_path.read_text(encoding="utf-8"))
    assert payload["enabled"] is True
