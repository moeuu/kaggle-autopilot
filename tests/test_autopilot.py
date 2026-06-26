"""Tests for autopilot gating and iteration behavior."""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from kagglebot import autopilot as autopilot_mod
from kagglebot import autopilot_state as _autopilot_state_test
from kagglebot import kernel_metrics as _kernel_metrics
from kagglebot import plan_resolution as _plan_resolution_test
from kagglebot import submit_notebook as _submit_notebook_test
from kagglebot.agent_io import agent_failure_detail, is_agent_capacity_failure
from kagglebot.autopilot import (
    _DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
    _DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
    _DEFAULT_LIMITED_SUBMISSION_GATE,
    _DEFAULT_MAX_ITERATIONS,
    _DEFAULT_STRICT_COMPETITION_METRIC,
    _DEFAULT_TARGET_MEDAL,
    _HEAVY_LOCAL_GPU_MAX_CV_FOLDS,
    _LONG_LOCAL_GPU_ITERATION_BUDGET_MIN,
    _LONG_LOCAL_GPU_MAX_ITERATIONS,
    AutopilotConfig,
    SubmissionPhase,
    _attempt_submit,
    _run_autofix,
    _run_kernel_fix,
    run_autopilot,
)
from kagglebot.autopilot_state import (
    _load_submit_retry_artifacts,
    _resolve_iteration_submission_artifact,
    _resume_iteration_state,
    load_run_state,
    write_iteration_state_marker,
)
from kagglebot.competition_rules import load_competition_rule_constraints
from kagglebot.eval import EvaluationReport
from kagglebot.exceptions import (
    KaggleCliError,
    KernelCapacityError,
    KernelFailedError,
    SubmissionCliError,
    SubmissionValidationError,
    SubmitAbortedError,
)
from kagglebot.history import SubmissionLedger
from kagglebot.iteration_signals import detect_online_mismatch_signal as _detect_online_mismatch_signal
from kagglebot.iteration_signals import extract_missing_ensemble_signal as _extract_missing_ensemble_signal
from kagglebot.iteration_signals import extract_orig_proba_signal as _extract_orig_proba_signal
from kagglebot.iteration_signals import extract_original_data_unused_signal as _extract_original_data_unused_signal
from kagglebot.iteration_signals import extract_pseudo_label_failure_signal as _extract_pseudo_label_failure_signal
from kagglebot.iteration_signals import extract_same_family_plateau_signal as _extract_same_family_plateau_signal
from kagglebot.kernel_quality import (
    build_accuracy_potential,
    build_kernel_quality_guard,
    extract_competition_faithfulness,
)
from kagglebot.kernel_runner import KernelRunResult
from kagglebot.knowledge import resolve_problem_type_insights
from kagglebot.paths import CompetitionPaths, KnowledgePaths
from kagglebot.plan_policy import build_evaluation_contract
from kagglebot.solver.evaluate import EvaluationResult
from kagglebot.submission.guard import compute_error_fingerprint
from kagglebot.submission.outcome_service import SubmissionOutcomePollingError
from kagglebot.submission_history import (
    detect_online_regression_vs_submission_history,
    format_previous_submission_history_for_prompt,
)
from kagglebot.submission_policy import (
    count_submission_rows_in_recent_window,
    count_submission_rows_on_utc_day,
    has_spare_daily_submission_slot,
    quality_reasons_allow_initial_submit_probe,
    quality_reasons_allow_spare_submit,
    should_attempt_submit_for_readiness,
    should_force_initial_submit,
)
from kagglebot.submit_autofix import SubmitFileAutofixPreparation
from kagglebot.submit_failure_context import (
    SubmitAbortAutofixDecision,
    load_submit_failure_context,
    resolve_submit_abort_autofixability_for_run,
)
from kagglebot.submit_stage import (
    infer_iteration_from_submission_path,
    resolve_submission_message,
    resolve_submission_rank_payload,
)
from kagglebot.types import PlanConfig

pytestmark = pytest.mark.slow


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


def _resolve_plan(plan: PlanConfig, config: AutopilotConfig) -> dict[str, object]:
    return _plan_resolution_test.resolve_plan_for_autopilot_config(
        plan=plan,
        config=config,
        defaults=_plan_resolution_test.AutopilotPlanResolutionDefaults(
            strict_competition_metric=_DEFAULT_STRICT_COMPETITION_METRIC,
            target_medal=_DEFAULT_TARGET_MEDAL,
            limited_submission_gate=_DEFAULT_LIMITED_SUBMISSION_GATE,
            max_iterations=_DEFAULT_MAX_ITERATIONS,
            heavy_local_gpu_max_cv_folds=_HEAVY_LOCAL_GPU_MAX_CV_FOLDS,
            long_local_gpu_iteration_budget_min=_LONG_LOCAL_GPU_ITERATION_BUDGET_MIN,
            long_local_gpu_max_iterations=_LONG_LOCAL_GPU_MAX_ITERATIONS,
            force_major_rank_max_percentile=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
            force_major_rank_min_teams=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
        ),
        on_message=lambda _message: None,
    )


def _run_notebook_submission_for_config(
    *,
    config: AutopilotConfig,
    run_id: str,
    submission_path: Path,
    message: str,
    artifact_mode: str = "wrapper",
):
    return _submit_notebook_test.run_notebook_kernel_submission_for_run(
        slug=config.slug,
        run_id=run_id,
        paths=config.paths,
        kaggle_username=config.kaggle_username,
        kernel_name=config.kernel_name,
        accelerator=config.accelerator,
        strict_accelerator=config.strict_accelerator,
        submission_path=submission_path,
        message=message,
        artifact_mode=artifact_mode,
        dry_run=config.dry_run,
        timeout_minutes=config.time_budget_min,
        infer_iteration_from_submission_path=infer_iteration_from_submission_path,
        resolve_kaggle_username=autopilot_mod.resolve_kaggle_username,
        run_submit_kernel=autopilot_mod.run_submit_kernel,
        run_kaggle_submit_kernel=autopilot_mod.run_kaggle_submit_kernel,
        copy_submission_artifact_to_iteration_dir=_autopilot_state_test.copy_submission_artifact_to_iteration_dir,
        classify_submit_error=autopilot_mod.classify_submit_error,
        should_retry_ambiguous=autopilot_mod._submit_failure_policy.should_retry_ambiguous_notebook_submit_error,
        sleep=autopilot_mod.time.sleep,
        on_message=lambda message: None,
        is_capacity_error=lambda exc: isinstance(exc, KernelCapacityError),
        is_push_error=lambda exc: isinstance(exc, KaggleCliError)
        and _submit_notebook_test.is_submit_kernel_push_error(exc),
    )


def _resolve_submit_abort_autofixable_for_config(*, config: AutopilotConfig, run_id: str) -> bool:
    decision = resolve_submit_abort_autofixability_for_run(
        run_dir=config.paths.run_dir(run_id),
        load_run_state=load_run_state,
    )
    return decision.autofixable


def _write_plan(paths: CompetitionPaths, **overrides) -> None:
    plan = PlanConfig()
    payload = plan.to_dict()
    payload.update(overrides)
    paths.plan_path.parent.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_dataset_profile(paths: CompetitionPaths, **overrides) -> None:
    payload: dict[str, object] = {
        "task": "regression",
        "modality": "tabular",
        "dtype_by_column": {"id": "int64"},
    }
    payload.update(overrides)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_rules(paths: CompetitionPaths, text: str, *, html: str | None = None) -> None:
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_md_path.write_text(text, encoding="utf-8")
    if html is not None:
        paths.rules_html_path.write_text(html, encoding="utf-8")


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
    (paths.kernel_source_dir / "kernel.py").write_text(
        "print('kernel stub')\n# submission.csv\n# metrics.json\n",
        encoding="utf-8",
    )
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


def test_submission_message_default_is_compact(tmp_path: Path) -> None:
    config = _make_config(tmp_path, slug="deep-past-initiative-machine-translation", message=None)
    message = resolve_submission_message(
        context_dir=config.paths.context_dir,
        run_id="20260223T161151Z-596afe59",
        best_score=17.273744466930147,
        explicit_message=config.message,
        submission_path=Path("iter-1/submission.csv"),
        campaign_mode=config.campaign_mode,
        target_direction=config.target_direction,
    )
    assert message.startswith("kb 20260223T161151Z-596afe59 i=1 offline=17.2737")
    assert "deep-past-initiative-machine-translation" not in message


def test_autopilot_does_not_force_iter1_submit_when_quality_gate_blocks(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        max_iterations=1,
        submission_gate="always",
    )

    def fake_run_kernel_local(**kwargs):  # noqa: ANN003
        iteration = int(kwargs["iteration"])
        output_dir = kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{iteration}" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        metrics_path.write_text(
            json.dumps(
                {
                    "score_source": "cv",
                    "metric": "rmse",
                    "direction": "minimize",
                    "offline_value": 0.9,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return KernelRunResult(
            kernel_id=f"local/{kwargs['slug']}",
            output_dir=output_dir,
            submission_path=submission_path,
            metrics_path=metrics_path,
        )

    submit_calls = {"count": 0}

    def fake_attempt_submit(
        *, config, run_id, submission_path, best_score, problem_types, submit_mode, notebook_submit_artifact_mode
    ):  # noqa: ARG001
        submit_calls["count"] += 1
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-03-18T00:00:00+00:00",
            "iteration": 1,
            "outcome": {"status": "complete", "score": None},
        }

    monkeypatch.setattr("kagglebot.autopilot.run_kernel_local", fake_run_kernel_local)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.1})
    monkeypatch.setattr(
        "kagglebot.kernel_quality.build_kernel_quality_guard",
        lambda **kwargs: {"allow_submit": False, "block_submit": True, "reasons": ["below_code_reference_baseline"]},
    )
    monkeypatch.setattr("kagglebot.submission_policy.should_attempt_submit_for_readiness", lambda **kwargs: False)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_autopilot(config)

    assert submit_calls["count"] == 0
    iter_dir = config.paths.iter_dir(config.run_id or "run-1", 1)
    marker = json.loads((iter_dir / "iteration_state.json").read_text(encoding="utf-8"))
    assert marker["submit_allowed_by_gate"] is False
    assert marker["submit_phase_state"] == "blocked_quality_guard"
    assert marker["forced_submit_reason"] == ""
    metrics = json.loads((iter_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["forced_submit_reason"] == ""


def test_autopilot_forces_iter1_submit_through_soft_detected_baseline_guard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        max_iterations=1,
        submission_gate="always",
    )

    def fake_run_kernel_local(**kwargs):  # noqa: ANN003
        output_dir = kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / "iter-1" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        metrics_path.write_text(
            json.dumps(
                {
                    "score_source": "cv",
                    "metric": "rmse",
                    "direction": "minimize",
                    "offline_value": 0.9,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return KernelRunResult(
            kernel_id=f"local/{kwargs['slug']}",
            output_dir=output_dir,
            submission_path=submission_path,
            metrics_path=metrics_path,
        )

    submit_calls = {"count": 0}

    def fake_attempt_submit(
        *, config, run_id, submission_path, best_score, problem_types, submit_mode, notebook_submit_artifact_mode
    ):  # noqa: ARG001
        submit_calls["count"] += 1
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-03-18T00:00:00+00:00",
            "iteration": 1,
            "outcome": {"status": "complete", "score": None},
        }

    monkeypatch.setattr("kagglebot.autopilot.run_kernel_local", fake_run_kernel_local)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.1})
    monkeypatch.setattr(
        "kagglebot.kernel_quality.build_kernel_quality_guard",
        lambda **kwargs: {
            "allow_submit": False,
            "block_submit": True,
            "reasons": ["selected_worse_than_detected_baseline"],
        },
    )
    monkeypatch.setattr("kagglebot.submission_policy.should_attempt_submit_for_readiness", lambda **kwargs: False)

    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_autopilot(config)

    assert submit_calls["count"] == 1
    iter_dir = config.paths.iter_dir(config.run_id or "run-1", 1)
    marker = json.loads((iter_dir / "iteration_state.json").read_text(encoding="utf-8"))
    assert marker["submit_allowed_by_gate"] is True
    assert marker["submit_phase_state"] == "submitted"
    assert marker["forced_submit_reason"] == "initial_submit_contract_probe"


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
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)

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
    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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


def test_resolve_plan_overrides_auto_score_source_to_cv(tmp_path: Path) -> None:
    config = _make_config(tmp_path, score_source="auto")
    plan = PlanConfig(score_source="auto")

    resolved = _resolve_plan(plan, config)

    assert resolved["score_source"] == "cv"


def test_resolve_plan_uses_plan_max_iterations_when_cli_unspecified(tmp_path: Path) -> None:
    config = _make_config(tmp_path, max_iterations=None)
    plan = PlanConfig(max_iterations=9)

    resolved = _resolve_plan(plan, config)

    assert resolved["max_iterations"] == 9


def test_resolve_plan_caps_long_heavy_local_gpu_iterations(tmp_path: Path) -> None:
    config = _make_config(tmp_path, compute="local_gpu", max_iterations=5, time_budget_min=None)
    _write_dataset_profile(config.paths, task="ocr", modality="image")

    resolved = _resolve_plan(PlanConfig(max_iterations=5), config)

    assert resolved["max_iterations"] == 3
    assert resolved["time_budget_min"] is None


def test_resolve_plan_treats_rna_structure_as_heavy_local_gpu(tmp_path: Path) -> None:
    config = _make_config(tmp_path, compute="local_gpu", max_iterations=5, time_budget_min=999)
    _write_dataset_profile(config.paths, task="structure_prediction", modality="rna_structure")

    resolved = _resolve_plan(PlanConfig(max_iterations=5, cv_folds=5, eval_seeds=[1, 2, 3]), config)

    assert resolved["cv_folds"] == 3
    assert resolved["eval_seeds"] == [1]


def test_resolve_plan_applies_explicit_local_gpu_time_budget_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLEBOT_LOCAL_GPU_TIME_BUDGET_MIN", "120")
    config = _make_config(tmp_path, compute="local_gpu", max_iterations=5, time_budget_min=999)
    _write_dataset_profile(config.paths, task="ocr", modality="image")

    resolved = _resolve_plan(PlanConfig(max_iterations=5), config)

    assert resolved["max_iterations"] == 5
    assert resolved["time_budget_min"] == 120


def test_resolve_plan_falls_back_to_default_max_iterations_when_plan_invalid(tmp_path: Path) -> None:
    config = _make_config(tmp_path, max_iterations=None)
    plan = PlanConfig(max_iterations=0)

    resolved = _resolve_plan(plan, config)

    assert resolved["max_iterations"] == _DEFAULT_MAX_ITERATIONS


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
            "submission_gate": "always",
        },
        "drift_check": {"enabled": True, "drift_weight": 0.5},
        "stop_policy": {"min_delta": 0.01, "no_improve_patience": 3, "same_config_patience": 2},
    }
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.context_dir / "evaluation_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    plan = PlanConfig(target_metric="rmse", target_direction="minimize", target_score=0.4)
    resolved = _resolve_plan(plan, config)

    assert resolved["submission_gate"] == "always"
    assert resolved["readiness_method"] == "mean_std"
    assert resolved["readiness_k"] == pytest.approx(0.7)
    assert resolved["ci_method"] == "bootstrap"
    assert resolved["ci_alpha"] == pytest.approx(0.1)
    assert resolved["drift_check"] is True
    assert resolved["drift_weight"] == pytest.approx(0.5)
    assert resolved["stop_min_delta"] == pytest.approx(0.01)
    assert resolved["stop_no_improve_patience"] == 3
    assert resolved["stop_same_config_patience"] == 2
    contract = resolved.get("evaluation_contract")
    assert isinstance(contract, dict)
    assert contract["expected_metric"] == "rmse"
    assert contract["expected_split_strategy"] == "kfold"
    assert contract["accepted_score_sources"] == ["cv", "holdout"]


def test_resolve_plan_keeps_explicit_plan_metric_over_stale_evaluation_spec(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    spec = {
        "metric_name": "accuracy",
        "direction": "maximize",
        "split_strategy": "stratified_kfold",
    }
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.context_dir / "evaluation_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    plan = PlanConfig(target_metric="balanced_accuracy", target_direction="maximize", target_score=0.9)
    resolved = _resolve_plan(plan, config)

    assert resolved["target_metric"] == "balanced_accuracy"
    contract = resolved.get("evaluation_contract")
    assert isinstance(contract, dict)
    assert contract["expected_metric"] == "balanced_accuracy"


def test_resolve_plan_requires_full_dataset_for_urban_flood(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="urban-flood-modelling", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    _write_sample_submission(paths.sample_submission_path)
    config = AutopilotConfig(
        run_id="run-1",
        slug="urban-flood-modelling",
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
        seed=42,
        score_source="cv",
        holdout_frac=0.2,
        cv_folds=5,
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        max_iterations=1,
        max_total_min=5,
        patience=1,
        min_improvement=0.0,
        submit=False,
        force_submit=False,
        message=None,
        verify_cmd=":",
        dry_run=True,
    )

    resolved = _resolve_plan(PlanConfig(target_metric="rmse", target_direction="minimize", target_score=0.5), config)

    contract = resolved.get("evaluation_contract")
    assert isinstance(contract, dict)
    assert contract["require_full_dataset"] is True


def test_resolve_plan_upgrades_split_strategy_to_groupkfold_from_plan_hints(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_plan(
        config.paths,
        split_strategy="kfold",
        evaluation_protocol={"cv_type": "GroupKFold(job_id)"},
    )
    _write_dataset_profile(config.paths, task="regression", modality="tabular")

    resolved = _resolve_plan(PlanConfig(split_strategy="kfold"), config)

    assert resolved["split_strategy"] == "group_kfold"


def test_resolve_plan_upgrades_split_strategy_to_timeseries_from_plan_hints(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_plan(
        config.paths,
        split_strategy="kfold",
        evaluation_protocol={"cv_type": "TimeSeriesSplit"},
    )
    _write_dataset_profile(config.paths, task="regression", modality="tabular")

    resolved = _resolve_plan(PlanConfig(split_strategy="kfold"), config)

    assert resolved["split_strategy"] == "timeseries_split"


def test_resolve_plan_upgrades_split_strategy_to_stratified_for_classification(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_dataset_profile(config.paths, task="classification", modality="tabular")

    resolved = _resolve_plan(PlanConfig(split_strategy="kfold"), config)

    assert resolved["split_strategy"] == "stratified_kfold"


def test_resolve_plan_notebook_only_does_not_block_local_compute(tmp_path: Path) -> None:
    config = _make_config(tmp_path, compute="local_gpu")
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.rules_md_path.write_text(
        "Submissions to this competition must be made through Notebooks.\n",
        encoding="utf-8",
    )

    resolved = _resolve_plan(PlanConfig(), config)
    assert resolved["submission_gate"] in {"always", "readiness_or_final", "readiness_only", "final_only"}
    assert resolved["submit_mode"] == "notebook"


def test_resolve_plan_uses_inference_artifact_mode_for_code_competition(tmp_path: Path) -> None:
    config = _make_config(tmp_path, compute="local_gpu")
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.data_md_path.write_text(
        (
            "Please note that this is a Code Competition. "
            "The test.csv is dummy data and hidden/full test runs in Kaggle.\n"
        ),
        encoding="utf-8",
    )

    resolved = _resolve_plan(PlanConfig(), config)

    assert resolved["submit_mode"] == "notebook"
    assert resolved["code_competition"] is True
    assert resolved["notebook_submit_artifact_mode"] == "inference"


def test_resolve_plan_uses_inference_mode_for_notebook_tiny_public_test(tmp_path: Path) -> None:
    config = _make_config(tmp_path, compute="local_gpu")
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.context_dir / "evaluation_spec.json").write_text(
        json.dumps({"submit_mode": "notebook"}, indent=2),
        encoding="utf-8",
    )
    config.paths.data_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.data_dir / "test.csv").write_text("id,x\n1,a\n2,b\n3,c\n", encoding="utf-8")
    (config.paths.data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    resolved = _resolve_plan(PlanConfig(), config)

    assert resolved["submit_mode"] == "notebook"
    assert resolved["code_competition"] is True
    assert resolved["notebook_submit_artifact_mode"] == "inference"


def test_resolve_plan_defaults_to_winner_target_for_leaderboard(tmp_path: Path) -> None:
    config = _make_config(tmp_path, compute="local_gpu")
    _write_dataset_profile(config.paths, task="classification", modality="tabular")

    resolved = _resolve_plan(PlanConfig(), config)

    assert resolved["deliverable_mode"] == "leaderboard"
    assert resolved["target_medal"] == "winner"
    assert resolved["target_rank_percentile"] == pytest.approx(0.001)
    assert resolved["rank_force_major_max_percentile"] == pytest.approx(0.001)


def test_resolve_plan_infers_writeup_mode_from_rules(tmp_path: Path) -> None:
    config = _make_config(tmp_path, compute="local_gpu")
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.rules_md_path.write_text(
        "This judged hackathon is evaluated by rubric and requires a writeup.\n",
        encoding="utf-8",
    )
    config.paths.overview_md_path.write_text(
        "Documentation and writeup quality is part of the panel scoring.\n",
        encoding="utf-8",
    )

    resolved = _resolve_plan(PlanConfig(), config)

    assert resolved["deliverable_mode"] == "writeup"


def test_resolve_plan_keeps_explicit_csv_mode_when_context_mentions_writeup_negatively(tmp_path: Path) -> None:
    config = _make_config(tmp_path, compute="local_gpu")
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.rules_md_path.write_text("You may select up to two Final Submissions for judging.\n", encoding="utf-8")
    config.paths.context_dir.joinpath("eval_advisor").mkdir(parents=True, exist_ok=True)
    config.paths.context_dir.joinpath("eval_advisor", "sources_summary.md").write_text(
        "This supports deliverable_mode=csv rather than writeup.\n"
        "This is a normal leaderboard CSV competition, not a judged/writeup competition.\n",
        encoding="utf-8",
    )
    (config.paths.context_dir / "evaluation_spec.json").write_text(
        json.dumps({"deliverable_mode": "csv", "metric_name": "auc", "direction": "maximize"}, indent=2),
        encoding="utf-8",
    )

    resolved = _resolve_plan(PlanConfig(), config)

    assert resolved["deliverable_mode"] == "leaderboard"
    assert resolved["submit_mode"] == "file"


def test_resolve_plan_enforces_rules_internet_and_runtime_caps(tmp_path: Path) -> None:
    config = _make_config(tmp_path, compute="kaggle_gpu", internet="on", time_budget_min=999)
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.rules_md_path.write_text(
        "\n".join(
            [
                "Submissions to this competition must be made through Notebooks.",
                "GPU Notebook <= 5 hours run-time",
                "Internet access disabled",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = _resolve_plan(PlanConfig(), config)
    assert resolved["internet"] == "off"
    assert resolved["time_budget_min"] == 300


def test_resolve_plan_disables_submission_gate_when_no_limit_detected(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.rules_md_path.write_text("General rules text without submission quota.\n", encoding="utf-8")

    resolved = _resolve_plan(
        PlanConfig(
            submit_policy="readiness_only",
            submission_gate="readiness_only",
        ),
        config,
    )
    assert resolved["submit_policy"] == "always"
    assert resolved["submission_gate"] == "always"


def test_resolve_plan_enables_submission_gate_when_limit_detected(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.rules_md_path.write_text(
        "Submission limit: maximum number of submissions per day.\n",
        encoding="utf-8",
    )

    explicit = _resolve_plan(
        PlanConfig(
            submit_policy="readiness_only",
            submission_gate="readiness_only",
        ),
        config,
    )
    assert explicit["submit_policy"] == "readiness_only"
    assert explicit["submission_gate"] == "readiness_only"

    defaulted = _resolve_plan(PlanConfig(), config)
    assert defaulted["submit_policy"] == "readiness_or_final"
    assert defaulted["submission_gate"] == "readiness_or_final"


def test_resolve_plan_treats_unrestricted_attempt_rules_as_no_daily_limit(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.rules_md_path.write_text(
        "SUBMISSION LIMITS\n"
        "Participants may submit without restriction as to the number of attempts, "
        "subject to the technical limits of the Kaggle platform.\n",
        encoding="utf-8",
    )

    resolved = _resolve_plan(
        PlanConfig(submit_policy="improved", submission_gate="readiness_or_final"),
        config,
    )

    assert resolved["submission_limit_per_day"] is None
    assert resolved["submit_policy"] == "always"
    assert resolved["submission_gate"] == "always"


def test_resolve_plan_extracts_daily_submission_limit_count(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.rules_md_path.write_text(
        "You may submit a maximum of one (1) Submission per day.\n",
        encoding="utf-8",
    )

    resolved = _resolve_plan(PlanConfig(), config)
    assert resolved["submission_limit_per_day"] == 1


def test_resolve_plan_extracts_markdown_word_parenthetical_daily_submission_limit(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.rules_md_path.write_text(
        "a. You may submit a maximum of **five (5)** Submissions per day.\n",
        encoding="utf-8",
    )

    resolved = _resolve_plan(PlanConfig(), config)
    assert resolved["submission_limit_per_day"] == 5


def test_resolve_plan_extracts_rolling_24h_submission_limit_count(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.rules_md_path.write_text(
        "The submission limit is 2 submissions within 24 hours per team.\n"
        "Each team will have 2 submission per 24h interval.\n",
        encoding="utf-8",
    )

    resolved = _resolve_plan(PlanConfig(), config)
    assert resolved["submission_limit_per_day"] == 2


def test_count_submission_rows_on_utc_day_uses_kaggle_cli_dates() -> None:
    rows = [
        {"date": "2026-05-09 06:16:21.527000", "status": "COMPLETE"},
        {"date": "2026-05-09T23:59:59+00:00", "status": "ERROR"},
        {"date": "2026-05-08 22:44:27.263000", "status": "COMPLETE"},
        {"date": "not-a-date", "status": "COMPLETE"},
    ]

    assert count_submission_rows_on_utc_day(rows, now=datetime(2026, 5, 9, 16, tzinfo=UTC)) == 2


def test_count_submission_rows_in_recent_window_covers_rolling_daily_limits() -> None:
    rows = [
        {"date": "2026-05-09 06:16:21.527000", "status": "COMPLETE"},
        {"date": "2026-05-08 22:44:27.263000", "status": "COMPLETE"},
        {"date": "2026-05-08 12:48:07.633000", "status": "COMPLETE"},
    ]

    assert count_submission_rows_in_recent_window(rows, now=datetime(2026, 5, 9, 16, tzinfo=UTC)) == 2


def test_has_spare_daily_submission_slot_requires_slots_for_remaining_iterations() -> None:
    assert has_spare_daily_submission_slot(
        submission_limit_per_day=5,
        submissions_used_today=2,
        iteration=3,
        max_iterations=5,
    )
    assert not has_spare_daily_submission_slot(
        submission_limit_per_day=5,
        submissions_used_today=3,
        iteration=3,
        max_iterations=5,
    )


def test_quality_reasons_allow_spare_submit_only_for_soft_blocks() -> None:
    assert quality_reasons_allow_spare_submit(["selected_worse_than_detected_baseline"])
    assert quality_reasons_allow_spare_submit(
        ["selected_worse_than_detected_baseline", "below_code_reference_baseline"]
    )
    assert not quality_reasons_allow_spare_submit(["external_test_label_transfer_detected"])
    assert not quality_reasons_allow_spare_submit(["selected_worse_than_detected_baseline", "untrusted_score_source"])


def test_quality_reasons_allow_initial_submit_probe_only_for_detected_baseline_soft_block() -> None:
    assert quality_reasons_allow_initial_submit_probe(["selected_worse_than_detected_baseline"])
    assert not quality_reasons_allow_initial_submit_probe(["below_code_reference_baseline"])
    assert not quality_reasons_allow_initial_submit_probe(
        ["selected_worse_than_detected_baseline", "untrusted_score_source"]
    )


def test_should_force_initial_submit_ignores_improved_policy_and_single_daily_submission() -> None:
    assert should_force_initial_submit(
        deliverable_mode="leaderboard",
        iteration=1,
        submit_enabled=True,
        dry_run=False,
        submit_policy="improved",
        submission_limit_per_day=None,
    )
    assert should_force_initial_submit(
        deliverable_mode="leaderboard",
        iteration=1,
        submit_enabled=True,
        dry_run=False,
        submit_policy="improved",
        submission_limit_per_day=2,
    )
    assert should_force_initial_submit(
        deliverable_mode="leaderboard",
        iteration=1,
        submit_enabled=True,
        dry_run=False,
        submit_policy="auto",
        submission_limit_per_day=2,
    )
    assert should_force_initial_submit(
        deliverable_mode="leaderboard",
        iteration=1,
        submit_enabled=True,
        dry_run=False,
        submit_policy="improved",
        submission_limit_per_day=1,
    )


def test_should_attempt_submit_with_limit_uses_reserved_final_slot_policy() -> None:
    assert (
        should_attempt_submit_for_readiness(
            gate="readiness_or_final",
            readiness_score=0.10,
            readiness_target=0.90,
            direction="maximize",
            iteration=1,
            max_iterations=3,
            submission_limit_per_day=3,
            successful_submissions=0,
            top1_score=0.80,
        )
        is True
    )
    assert (
        should_attempt_submit_for_readiness(
            gate="readiness_or_final",
            readiness_score=0.20,
            readiness_target=0.90,
            direction="maximize",
            iteration=2,
            max_iterations=3,
            submission_limit_per_day=3,
            successful_submissions=2,
            top1_score=0.80,
        )
        is False
    )
    assert (
        should_attempt_submit_for_readiness(
            gate="readiness_or_final",
            readiness_score=0.85,
            readiness_target=0.90,
            direction="maximize",
            iteration=2,
            max_iterations=3,
            submission_limit_per_day=3,
            successful_submissions=2,
            top1_score=0.80,
        )
        is True
    )
    assert (
        should_attempt_submit_for_readiness(
            gate="readiness_or_final",
            readiness_score=0.95,
            readiness_target=0.90,
            direction="maximize",
            iteration=3,
            max_iterations=3,
            submission_limit_per_day=2,
            successful_submissions=2,
            top1_score=0.80,
        )
        is False
    )
    assert (
        should_attempt_submit_for_readiness(
            gate="readiness_or_final",
            readiness_score=0.20,
            readiness_target=0.90,
            direction="maximize",
            iteration=3,
            max_iterations=3,
            submission_limit_per_day=3,
            successful_submissions=2,
            top1_score=0.80,
        )
        is True
    )


def test_should_attempt_submit_with_limit_strictly_spaces_non_final_submissions() -> None:
    # Daily cap 5 with 10 iterations -> reserve 1 slot for final, spread 4 non-final checkpoints.
    assert (
        should_attempt_submit_for_readiness(
            gate="always",
            readiness_score=0.20,
            readiness_target=0.90,
            direction="maximize",
            iteration=1,
            max_iterations=10,
            submission_limit_per_day=5,
            successful_submissions=0,
            top1_score=0.80,
        )
        is False
    )
    assert (
        should_attempt_submit_for_readiness(
            gate="always",
            readiness_score=0.20,
            readiness_target=0.90,
            direction="maximize",
            iteration=2,
            max_iterations=10,
            submission_limit_per_day=5,
            successful_submissions=0,
            top1_score=0.80,
        )
        is True
    )
    assert (
        should_attempt_submit_for_readiness(
            gate="always",
            readiness_score=0.20,
            readiness_target=0.90,
            direction="maximize",
            iteration=9,
            max_iterations=10,
            submission_limit_per_day=5,
            successful_submissions=1,
            top1_score=0.80,
        )
        is True
    )
    assert (
        should_attempt_submit_for_readiness(
            gate="always",
            readiness_score=0.20,
            readiness_target=0.90,
            direction="maximize",
            iteration=3,
            max_iterations=10,
            submission_limit_per_day=5,
            successful_submissions=1,
            top1_score=0.80,
        )
        is False
    )
    assert (
        should_attempt_submit_for_readiness(
            gate="always",
            readiness_score=0.20,
            readiness_target=0.90,
            direction="maximize",
            iteration=4,
            max_iterations=10,
            submission_limit_per_day=5,
            successful_submissions=1,
            top1_score=0.80,
        )
        is True
    )
    assert (
        should_attempt_submit_for_readiness(
            gate="always",
            readiness_score=0.20,
            readiness_target=0.90,
            direction="maximize",
            iteration=10,
            max_iterations=10,
            submission_limit_per_day=5,
            successful_submissions=4,
            top1_score=0.80,
        )
        is True
    )


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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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


def test_autopilot_writes_evaluation_report_and_uses_offline_loop_decision(monkeypatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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

    iter1_metrics = json.loads((config.paths.iter_dir("run-1", 1) / "metrics.json").read_text(encoding="utf-8"))
    assert iter1_metrics["loop_decision"]["source"] == "holdout"
    assert iter1_metrics["loop_decision"]["value"] == pytest.approx(0.86)

    run_report = json.loads((config.paths.run_dir("run-1") / "evaluation_report.json").read_text(encoding="utf-8"))
    assert run_report["latest_iteration"] == 2
    assert len(run_report["history"]) == 2


@pytest.mark.parametrize(
    ("metric", "direction", "value", "target"),
    [
        ("auc", "maximize", 0.70, 0.80),
        ("rmse", "minimize", 0.40, 0.50),
    ],
)
def test_autopilot_submission_runs_every_iteration(
    monkeypatch,
    tmp_path: Path,
    metric: str,
    direction: str,
    value: float,
    target: float,
) -> None:
    submit_calls = {"count": 0}

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric=metric,
        target_score=target,
        target_direction=direction,
        submit_policy="always",
        submission_gate="always",
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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

    assert submit_calls["count"] == 1


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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", always_transient_fail)
    monkeypatch.setattr(
        "kagglebot.submit_failure_context.resolve_submit_abort_autofixability_for_run",
        lambda *args, **kwargs: SubmitAbortAutofixDecision(False, ""),
    )
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", transient_fail_unique_fingerprint)
    monkeypatch.setattr(
        "kagglebot.submit_failure_context.resolve_submit_abort_autofixability_for_run",
        lambda *args, **kwargs: SubmitAbortAutofixDecision(False, ""),
    )
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", fake_check_rules)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fake_submit)
    monkeypatch.setattr(
        "kagglebot.submit_failure_context.resolve_submit_abort_autofixability_for_run",
        lambda *args, **kwargs: SubmitAbortAutofixDecision(False, ""),
    )
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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
    monkeypatch.setattr(
        "kagglebot.submit_failure_context.resolve_submit_abort_autofixability_for_run",
        lambda *args, **kwargs: SubmitAbortAutofixDecision(False, ""),
    )
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", always_transient_fail)
    monkeypatch.setattr(
        "kagglebot.submit_failure_context.resolve_submit_abort_autofixability_for_run",
        lambda *args, **kwargs: SubmitAbortAutofixDecision(False, ""),
    )
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


def test_attempt_submit_allows_one_repeated_fingerprint_after_code_change(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1, force_submit=True)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    repeated_stdout = "503 temporary failure"
    repeated_stderr = "ConnectionError: temporarily unavailable"
    repeated_fp = compute_error_fingerprint(repeated_stdout, repeated_stderr)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_submit_fingerprint": repeated_fp,
                "last_submit_code_fingerprint": "old-code-fingerprint",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    calls = {"count": 0}

    def fail_then_succeed(*args, **kwargs):  # noqa: ARG001
        calls["count"] += 1
        if calls["count"] == 1:
            raise SubmissionCliError(
                "submit failed",
                command=["kaggle", "competitions", "submit"],
                exit_code=1,
                output=f"{repeated_stdout}\n{repeated_stderr}",
                stdout=repeated_stdout,
                stderr=repeated_stderr,
            )
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fail_then_succeed)
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.submit_retry_policy.compute_submit_code_fingerprint",
        lambda *args, **kwargs: "new-code",
    )
    monkeypatch.setattr("kagglebot.autopilot.time.sleep", lambda *_args, **_kwargs: None)

    result = _attempt_submit(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        best_score=0.4,
        problem_types=[],
    )

    assert result is not None
    assert calls["count"] == 2
    run_state = load_run_state(run_dir)
    assert run_state["same_fp_allowance_code_fingerprint"] == "new-code"
    assert run_state["same_fp_allowance_error_fingerprint"] == repeated_fp


def test_attempt_submit_consumes_repeated_fingerprint_allowance_once(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1, force_submit=True)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    repeated_stdout = "503 temporary failure"
    repeated_stderr = "ConnectionError: temporarily unavailable"
    repeated_fp = compute_error_fingerprint(repeated_stdout, repeated_stderr)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_submit_fingerprint": repeated_fp,
                "last_submit_code_fingerprint": "old-code-fingerprint",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    calls = {"count": 0}

    def always_fail(*args, **kwargs):  # noqa: ARG001
        calls["count"] += 1
        raise SubmissionCliError(
            "submit failed",
            command=["kaggle", "competitions", "submit"],
            exit_code=1,
            output=f"{repeated_stdout}\n{repeated_stderr}",
            stdout=repeated_stdout,
            stderr=repeated_stderr,
        )

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", always_fail)
    monkeypatch.setattr(
        "kagglebot.submit_retry_policy.compute_submit_code_fingerprint",
        lambda *args, **kwargs: "new-code",
    )
    monkeypatch.setattr("kagglebot.autopilot.time.sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(SubmitAbortedError):
        _attempt_submit(
            config=config,
            run_id=run_id,
            submission_path=submission_path,
            best_score=0.4,
            problem_types=[],
        )

    assert calls["count"] == 2
    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["reason"] == "same_error_fingerprint_recurred"


def test_attempt_submit_allows_one_repeated_fingerprint_for_legacy_state_without_code_fp(
    monkeypatch, tmp_path: Path
) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1, force_submit=True)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    repeated_stdout = "503 temporary failure"
    repeated_stderr = "ConnectionError: temporarily unavailable"
    repeated_fp = compute_error_fingerprint(repeated_stdout, repeated_stderr)
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

    calls = {"count": 0}

    def fail_then_succeed(*args, **kwargs):  # noqa: ARG001
        calls["count"] += 1
        if calls["count"] == 1:
            raise SubmissionCliError(
                "submit failed",
                command=["kaggle", "competitions", "submit"],
                exit_code=1,
                output=f"{repeated_stdout}\n{repeated_stderr}",
                stdout=repeated_stdout,
                stderr=repeated_stderr,
            )
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fail_then_succeed)
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.submit_retry_policy.compute_submit_code_fingerprint",
        lambda *args, **kwargs: "new-code",
    )
    monkeypatch.setattr("kagglebot.autopilot.time.sleep", lambda *_args, **_kwargs: None)

    result = _attempt_submit(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        best_score=0.4,
        problem_types=[],
    )

    assert result is not None
    assert calls["count"] == 2
    run_state = load_run_state(run_dir)
    assert run_state["same_fp_allowance_code_fingerprint"] == "new-code"
    assert run_state["same_fp_allowance_error_fingerprint"] == repeated_fp


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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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


def test_submit_autofix_does_not_force_resubmit_when_run_already_has_success(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "submit_attempts.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "fingerprint": "ok1", "ok": True}) + "\n",
        encoding="utf-8",
    )

    def always_submit_abort(self):  # noqa: ANN001
        raise SubmitAbortedError("submit failed")

    observed: dict[str, str | None] = {}

    def fake_run_autofix(*, config: AutopilotConfig, run_id: str, attempt: int, error: Exception) -> None:  # noqa: ARG001
        observed["force_resubmit"] = os.environ.get("KAGGLEBOT_FORCE_RESUBMIT")
        raise RuntimeError("stop_after_autofix_check")

    monkeypatch.delenv("KAGGLEBOT_FORCE_RESUBMIT", raising=False)
    monkeypatch.setattr("kagglebot.autopilot.AutopilotSession.run", always_submit_abort)
    monkeypatch.setattr(
        "kagglebot.submit_failure_context.resolve_submit_abort_autofixability_for_run",
        lambda *args, **kwargs: SubmitAbortAutofixDecision(True, ""),
    )
    monkeypatch.setattr("kagglebot.autopilot._run_autofix", fake_run_autofix)

    with pytest.raises(RuntimeError, match="stop_after_autofix_check"):
        run_autopilot(config)

    assert observed["force_resubmit"] is None


def test_submit_autofix_forces_resubmit_on_polling_abort_even_with_prior_success(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "submit_attempts.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "fingerprint": "ok1", "ok": True, "action_taken": "submit"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": True,
                "last_error_kind": "transient",
                "last_reason": "submission_polling_error",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def always_submit_abort(self):  # noqa: ANN001
        raise SubmitAbortedError("submit failed")

    observed: dict[str, str | None] = {}

    def fake_run_autofix(*, config: AutopilotConfig, run_id: str, attempt: int, error: Exception) -> None:  # noqa: ARG001
        observed["force_resubmit"] = os.environ.get("KAGGLEBOT_FORCE_RESUBMIT")
        raise RuntimeError("stop_after_autofix_check")

    monkeypatch.delenv("KAGGLEBOT_FORCE_RESUBMIT", raising=False)
    monkeypatch.setattr("kagglebot.autopilot.AutopilotSession.run", always_submit_abort)
    monkeypatch.setattr(
        "kagglebot.submit_failure_context.resolve_submit_abort_autofixability_for_run",
        lambda *args, **kwargs: SubmitAbortAutofixDecision(True, ""),
    )
    monkeypatch.setattr("kagglebot.autopilot._run_autofix", fake_run_autofix)

    with pytest.raises(RuntimeError, match="stop_after_autofix_check"):
        run_autopilot(config)

    assert observed["force_resubmit"] == "1"


def test_attempt_submit_does_not_skip_when_prepared_path_changes(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    prepared_path = config.paths.iter_dir(run_id, 1) / "submission.compact.csv"
    prepared_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_submission_path": str(submission_path),
                "last_submit_fingerprint": "abc123",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: prepared_path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)

    result = _attempt_submit(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        best_score=0.4,
        problem_types=[],
    )

    assert result is not None
    assert result["submission_path"] == str(prepared_path)
    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert rows[-1]["action_taken"] == "submit"
    assert rows[-1]["sub_path"] == str(prepared_path)


def test_attempt_submit_switches_to_notebook_submit_after_bad_request(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1, compute="local_gpu", accelerator="gpu")
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.competition_rules.load_competition_rule_constraints",
        lambda *args, **kwargs: type("C", (), {"notebook_submissions_only": False})(),
    )
    monkeypatch.setattr("kagglebot.autopilot.infer_code_competition_from_paths", lambda *args, **kwargs: True)

    submit_calls = {"file": 0, "notebook": 0}
    captured: dict[str, object] = {}

    def bad_request_submit(*args, **kwargs):  # noqa: ARG001
        submit_calls["file"] += 1
        raise SubmissionCliError(
            "Kaggle CLI submit failed.",
            command=["kaggle", "competitions", "submit"],
            exit_code=1,
            output=(
                "400 Client Error: Bad Request (submit-notebook)\n"
                "Code competition submissions require both the output file name and the version label"
            ),
            stdout="",
            stderr="",
        )

    def notebook_submit(*args, **kwargs):  # noqa: ARG001
        submit_calls["notebook"] += 1
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    def fake_run_submit_kernel(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        output_dir = (
            kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{kwargs['iteration']}" / "output"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_submission = output_dir / "submission.csv"
        out_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        return KernelRunResult(
            kernel_id="user/demo-kernel",
            output_dir=output_dir,
            submission_path=out_submission,
            metrics_path=None,
        )

    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", bad_request_submit)
    monkeypatch.setattr("kagglebot.autopilot.run_kaggle_submit_kernel", notebook_submit)
    monkeypatch.setattr(
        "kagglebot.autopilot.run_submit_kernel",
        lambda **kwargs: fake_run_submit_kernel(**kwargs),
    )

    result = _attempt_submit(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        best_score=0.4,
        problem_types=[],
    )

    assert result is not None
    assert result["submission_path"] == "kernel:user/demo-kernel"
    assert submit_calls["file"] == 1
    assert submit_calls["notebook"] == 1
    assert captured["mode"] == "inference"
    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert rows[-1]["action_taken"] == "submit"
    assert rows[-1]["sub_path"] == "kernel:user/demo-kernel"


def test_attempt_submit_does_not_switch_to_notebook_on_generic_bad_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1, compute="local_gpu", accelerator="gpu")
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.competition_rules.load_competition_rule_constraints",
        lambda *args, **kwargs: type("C", (), {"notebook_submissions_only": False})(),
    )
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SubmissionCliError(
                "Kaggle CLI submit failed.",
                command=["kaggle", "competitions", "submit"],
                exit_code=1,
                output="400 Client Error: Bad Request",
                stdout=(
                    "400 Client Error: Bad Request for url: "
                    "https://www.kaggle.com/api/v1/competitions/submissions/submit-notebook/demo"
                ),
                stderr="",
            )
        ),
    )

    notebook_calls = {"count": 0}

    def notebook_submit(*args, **kwargs):  # noqa: ARG001
        notebook_calls["count"] += 1
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("kagglebot.autopilot.run_kaggle_submit_kernel", notebook_submit)

    with pytest.raises(SubmitAbortedError):
        _attempt_submit(
            config=config,
            run_id=run_id,
            submission_path=submission_path,
            best_score=0.4,
            problem_types=[],
        )
    assert notebook_calls["count"] == 0
    context = load_submit_failure_context(run_dir)
    assert context["reason"] == "ambiguous_notebook_bad_request"
    assert context["repair_target"] == "manual_intervention"
    assert context["repairable"] is False
    assert _resolve_submit_abort_autofixable_for_config(config=config, run_id=run_id) is False


def test_load_submit_failure_context_normalizes_stale_ambiguous_notebook_context(tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1, compute="local_gpu", accelerator="gpu")
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "submit_failure_context.json").write_text(
        json.dumps(
            {
                "active": True,
                "reason": "ambiguous_notebook_bad_request",
                "error_kind": "unknown",
                "repair_target": "submit_mode_or_kernel",
                "repairable": True,
                "stdout_tail": (
                    "400 Client Error: Bad Request for url: "
                    "https://www.kaggle.com/api/v1/competitions/submissions/submit-notebook/demo"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    context = load_submit_failure_context(run_dir)

    assert context["repair_target"] == "manual_intervention"
    assert context["repairable"] is False
    assert "submit-notebook 400" in context["manual_next_step"]
    assert _resolve_submit_abort_autofixable_for_config(config=config, run_id=run_id) is False


def test_attempt_submit_treats_submission_limit_as_manual_blocker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1, compute="local_gpu", accelerator="gpu")
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr(
        "kagglebot.competition_rules.load_competition_rule_constraints",
        lambda *args, **kwargs: type("C", (), {"notebook_submissions_only": False})(),
    )
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SubmissionCliError(
                "Kaggle CLI submit failed.",
                command=["kaggle", "competitions", "submit"],
                exit_code=1,
                output="400 Client Error: Bad Request: submission limit reached",
                stdout="You have reached the maximum number of submissions for this competition.",
                stderr="",
            )
        ),
    )

    with pytest.raises(SubmitAbortedError):
        _attempt_submit(
            config=config,
            run_id=run_id,
            submission_path=submission_path,
            best_score=0.4,
            problem_types=[],
        )

    context = load_submit_failure_context(run_dir)
    assert context["repair_target"] == "manual_intervention"
    assert context["repairable"] is False
    assert "submission limit" in context["manual_next_step"].lower()
    assert _resolve_submit_abort_autofixable_for_config(config=config, run_id=run_id) is False


def test_attempt_submit_skips_duplicate_sha_before_notebook_submit(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1, compute="local_gpu", accelerator="gpu")
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    SubmissionLedger(config.paths.submission_ledger_path).record(
        slug=config.slug,
        message="previous message",
        submission_path=submission_path,
        run_id="prior-run",
    )

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.check_rules_accepted",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rules check should not run")),
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.run_submit_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("notebook kernel should not run")),
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.run_kaggle_submit_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("notebook submit should not run")),
    )

    result = _attempt_submit(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        best_score=0.4,
        problem_types=[],
        submit_mode="notebook",
    )

    assert result is not None
    assert result["skipped"] is True
    assert result["reason"] == "duplicate_submission_sha_seen"
    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["action_taken"] == "skip"
    assert rows[-1]["reason"] == "duplicate_submission_sha_seen"


def test_attempt_submit_retries_same_path_when_previous_bad_request(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1, compute="local_gpu", accelerator="gpu")
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_submission_path": str(submission_path),
                "last_reason": "bad_request",
                "last_submit_fingerprint": "prev",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.competition_rules.load_competition_rule_constraints",
        lambda *args, **kwargs: type("C", (), {"notebook_submissions_only": False})(),
    )
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SubmissionCliError(
                "Kaggle CLI submit failed.",
                command=["kaggle", "competitions", "submit"],
                exit_code=1,
                output="400 Client Error: Bad Request (submit-notebook)",
                stdout="Code competition submissions require both the output file name and the version label",
                stderr="",
            )
        ),
    )

    def fake_run_submit_kernel(**kwargs):  # noqa: ANN003
        output_dir = (
            kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{kwargs['iteration']}" / "output"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_submission = output_dir / "submission.csv"
        out_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        return KernelRunResult(
            kernel_id="user/demo-kernel",
            output_dir=output_dir,
            submission_path=out_submission,
            metrics_path=None,
        )

    monkeypatch.setattr(
        "kagglebot.autopilot.run_submit_kernel",
        lambda **kwargs: fake_run_submit_kernel(**kwargs),
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.run_kaggle_submit_kernel",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )

    result = _attempt_submit(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        best_score=0.4,
        problem_types=[],
    )
    assert result is not None
    assert result["submission_path"] == "kernel:user/demo-kernel"


def test_run_notebook_submission_for_config_forces_internet_off(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, compute="local_gpu", accelerator="gpu", internet="on")
    run_id = config.run_id or "run-1"
    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_submit_kernel(**kwargs):  # noqa: ANN003
        captured["run_submit_kernel"] = kwargs
        output_dir = (
            kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{kwargs['iteration']}" / "output"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_submission = output_dir / "submission.csv"
        out_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        return KernelRunResult(
            kernel_id="user/demo-submit-kernel",
            output_dir=output_dir,
            submission_path=out_submission,
            metrics_path=None,
        )

    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr("kagglebot.autopilot.run_submit_kernel", fake_run_submit_kernel)
    monkeypatch.setattr(
        "kagglebot.autopilot.run_kaggle_submit_kernel",
        lambda **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )

    _run_notebook_submission_for_config(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        message="test notebook submit",
    )

    assert captured["run_submit_kernel"]["enable_internet"] is False
    assert captured["run_submit_kernel"]["mode"] == "wrapper"


def test_run_notebook_submission_for_config_retries_cpu_after_gpu_capacity_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, submit=True, compute="local_gpu", accelerator="gpu", strict_accelerator=False)
    run_id = config.run_id or "run-1"
    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    accelerators: list[str] = []

    def fake_run_submit_kernel(**kwargs):  # noqa: ANN003
        accelerators.append(kwargs["accelerator"])
        if kwargs["accelerator"] == "gpu":
            raise KernelCapacityError(
                "Kaggle GPU session limit reached.",
                command=["kaggle", "kernels", "push"],
                exit_code=15,
                output="Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.",
            )
        output_dir = (
            kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{kwargs['iteration']}" / "output"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_submission = output_dir / "submission.csv"
        out_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        return KernelRunResult(
            kernel_id="user/demo-submit-kernel-cpu",
            output_dir=output_dir,
            submission_path=out_submission,
            metrics_path=None,
        )

    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr("kagglebot.autopilot.run_submit_kernel", fake_run_submit_kernel)
    monkeypatch.setattr(
        "kagglebot.autopilot.run_kaggle_submit_kernel",
        lambda **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )

    result, kernel_ref, artifact_path = _run_notebook_submission_for_config(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        message="test notebook cpu fallback",
    )

    assert result.returncode == 0
    assert kernel_ref == "kernel:user/demo-submit-kernel-cpu"
    assert artifact_path is not None
    assert accelerators == ["gpu", "cpu"]


def test_run_notebook_submission_for_config_retries_cpu_after_kernel_push_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, submit=True, compute="local_gpu", accelerator="gpu", strict_accelerator=False)
    run_id = config.run_id or "run-1"
    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    accelerators: list[str] = []

    def fake_run_submit_kernel(**kwargs):  # noqa: ANN003
        accelerators.append(kwargs["accelerator"])
        if kwargs["accelerator"] == "gpu":
            raise KaggleCliError(
                "Kaggle kernel push failed.",
                command=["kaggle", "kernels", "push", "-p", "kernel"],
                exit_code=4,
                output="Kernel push error: Notebook not found",
            )
        output_dir = (
            kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{kwargs['iteration']}" / "output"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_submission = output_dir / "submission.csv"
        out_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        return KernelRunResult(
            kernel_id="user/demo-submit-kernel-cpu",
            output_dir=output_dir,
            submission_path=out_submission,
            metrics_path=None,
        )

    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr("kagglebot.autopilot.run_submit_kernel", fake_run_submit_kernel)
    monkeypatch.setattr(
        "kagglebot.autopilot.run_kaggle_submit_kernel",
        lambda **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )

    result, kernel_ref, artifact_path = _run_notebook_submission_for_config(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        message="test notebook cpu fallback",
    )

    assert result.returncode == 0
    assert kernel_ref == "kernel:user/demo-submit-kernel-cpu"
    assert artifact_path is not None
    assert accelerators == ["gpu", "cpu"]


def test_run_notebook_submission_for_config_preserves_kaggle_cli_error_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, submit=True, compute="local_gpu", accelerator="gpu")
    run_id = config.run_id or "run-1"
    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    command = ["kaggle", "kernels", "push", "-p", "kernel"]
    output = "Kernel push error: Notebook not found"

    def fake_run_submit_kernel(**kwargs):  # noqa: ANN003, ARG001
        raise KaggleCliError(
            "Kaggle kernel push failed.",
            command=command,
            exit_code=4,
            output=output,
        )

    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr("kagglebot.autopilot.run_submit_kernel", fake_run_submit_kernel)

    with pytest.raises(SubmissionCliError) as exc_info:
        _run_notebook_submission_for_config(
            config=config,
            run_id=run_id,
            submission_path=submission_path,
            message="test notebook push failure",
        )

    exc = exc_info.value
    assert exc.command == command
    assert exc.exit_code == 4
    assert exc.output == output
    assert exc.stderr == output


def test_run_notebook_submission_for_config_does_not_retry_generic_submit_notebook_bad_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, submit=True, compute="local_gpu", accelerator="gpu")
    run_id = config.run_id or "run-1"
    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    submit_calls = {"count": 0}
    sleep_calls: list[float] = []

    def fake_run_submit_kernel(**kwargs):  # noqa: ANN003
        output_dir = (
            kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{kwargs['iteration']}" / "output"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_submission = output_dir / "submission.csv"
        out_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        return KernelRunResult(
            kernel_id="user/demo-submit-kernel",
            output_dir=output_dir,
            submission_path=out_submission,
            metrics_path=None,
        )

    def fake_run_kaggle_submit_kernel(**kwargs):  # noqa: ANN003, ARG001
        submit_calls["count"] += 1
        if submit_calls["count"] == 1:
            raise SubmissionCliError(
                "Kaggle CLI notebook submit failed.",
                command=["kaggle", "competitions", "submit"],
                exit_code=1,
                output=(
                    "400 Client Error: Bad Request for url: "
                    "https://www.kaggle.com/api/v1/competitions/submissions/submit-notebook/demo"
                ),
                stdout="",
                stderr="",
            )
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr("kagglebot.autopilot.run_submit_kernel", fake_run_submit_kernel)
    monkeypatch.setattr("kagglebot.autopilot.run_kaggle_submit_kernel", fake_run_kaggle_submit_kernel)
    monkeypatch.setattr("kagglebot.autopilot.time.sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(SubmissionCliError):
        _run_notebook_submission_for_config(
            config=config,
            run_id=run_id,
            submission_path=submission_path,
            message="test notebook retry",
        )

    assert submit_calls["count"] == 1
    assert sleep_calls == []


def test_run_notebook_submission_for_config_uses_inference_mode_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _make_config(tmp_path, submit=True, compute="local_gpu", accelerator="gpu")
    run_id = config.run_id or "run-1"
    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_submit_kernel(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        output_dir = (
            kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{kwargs['iteration']}" / "output"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_submission = output_dir / "submission.csv"
        out_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        return KernelRunResult(
            kernel_id="user/demo-submit-kernel",
            output_dir=output_dir,
            submission_path=out_submission,
            metrics_path=None,
        )

    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr("kagglebot.autopilot.run_submit_kernel", fake_run_submit_kernel)
    monkeypatch.setattr(
        "kagglebot.autopilot.run_kaggle_submit_kernel",
        lambda **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )

    _run_notebook_submission_for_config(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        message="test notebook inference submit",
        artifact_mode="inference",
    )

    assert captured["mode"] == "inference"


def test_attempt_submit_retries_same_path_after_code_change(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1, force_submit=False)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_submission_path": str(submission_path),
                "last_reason": "submission_poll_status_error",
                "last_submit_fingerprint": "prev",
                "last_submit_code_fingerprint": "old-code-fingerprint",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.competition_rules.load_competition_rule_constraints",
        lambda *args, **kwargs: type("C", (), {"notebook_submissions_only": False})(),
    )
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)

    result = _attempt_submit(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        best_score=0.4,
        problem_types=[],
    )

    assert result is not None
    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert rows[-1]["action_taken"] == "submit"
    assert rows[-1]["sub_path"] == str(submission_path)


def test_attempt_submit_allows_new_submission_after_prior_success(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1, force_submit=False)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    (run_dir / "submit_attempts.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "fingerprint": "abc", "ok": True, "action_taken": "submit"}) + "\n",
        encoding="utf-8",
    )

    validate_calls = {"count": 0}
    submit_calls = {"count": 0}

    def fake_validate(self, path):  # noqa: ARG001
        validate_calls["count"] += 1
        return path

    def fake_submit(*args, **kwargs):  # noqa: ARG001
        submit_calls["count"] += 1
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        fake_validate,
    )
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fake_submit)
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)

    result = _attempt_submit(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        best_score=0.4,
        problem_types=[],
    )
    assert result is not None
    assert validate_calls["count"] == 1
    assert submit_calls["count"] == 1

    run_state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert run_state["submit_ok"] is True

    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["action_taken"] == "submit"
    assert rows[-1]["reason"] == "submitted"


def test_attempt_submit_aborts_when_polling_reports_error_status(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )
    monkeypatch.setattr(
        "kagglebot.submit_stage.wait_for_submission_outcome",
        lambda **kwargs: {
            "status": "SubmissionStatus.ERROR",
            "score": None,
            "raw": {"status": "SubmissionStatus.ERROR", "errorDescription": "bad submission"},
        },
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.list_competition_submissions",
        lambda *args, **kwargs: [{"status": "error", "errorDescription": "bad submission from Kaggle"}],
    )

    with pytest.raises(SubmitAbortedError, match="error status 'error'"):
        _attempt_submit(
            config=config,
            run_id=run_id,
            submission_path=submission_path,
            best_score=0.4,
            problem_types=[],
        )

    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["action_taken"] for row in rows[-1:]] == ["abort"]
    assert rows[-1]["reason"] == "submission_poll_status_error"
    assert "Kaggle reported: bad submission from Kaggle" in rows[-1]["stderr_tail"]


def test_attempt_submit_aborts_when_complete_has_no_score(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )
    monkeypatch.setattr(
        "kagglebot.submit_stage.wait_for_submission_outcome",
        lambda **kwargs: {
            "status": "SubmissionStatus.COMPLETE",
            "score": None,
            "raw": {"status": "SubmissionStatus.COMPLETE", "publicScore": "", "privateScore": ""},
        },
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.list_competition_submissions",
        lambda *args, **kwargs: [{"status": "complete", "publicScore": "", "privateScore": "", "description": "demo"}],
    )

    with pytest.raises(SubmitAbortedError, match="no score"):
        _attempt_submit(
            config=config,
            run_id=run_id,
            submission_path=submission_path,
            best_score=0.4,
            problem_types=[],
        )

    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["action_taken"] for row in rows[-1:]] == ["abort"]
    assert rows[-1]["reason"] == "submission_poll_status_complete_no_score"
    assert "scoring error" in rows[-1]["stderr_tail"].lower()
    failure_context = json.loads((run_dir / "submit_failure_context.json").read_text(encoding="utf-8"))
    assert failure_context["active"] is True
    assert failure_context["repairable"] is True
    assert failure_context["repair_target"] == "submission_artifact"


def test_attempt_submit_prefers_repaired_submit_artifact_from_submit_autofix(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    original_submission = config.paths.iter_dir(run_id, 1) / "submission.csv"
    original_submission.parent.mkdir(parents=True, exist_ok=True)
    original_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    repaired_submission = config.paths.iter_dir(run_id, 1) / "output" / "submission-fixed.csv"
    repaired_submission.parent.mkdir(parents=True, exist_ok=True)
    repaired_submission.write_text("id,target\n1,0.3\n2,0.4\n", encoding="utf-8")

    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_error_kind": "validation",
                "last_reason": "submission_poll_status_error",
                "submit_autofix_submission_path": str(repaired_submission),
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
                "exit_code": None,
                "error_kind": "validation",
                "reason": "submission_poll_status_error",
                "action_taken": "abort",
                "fingerprint": "abc123",
                "sub_path": str(original_submission),
                "sub_sha256": "old-hash",
                "stderr_tail": "Kaggle reported: submission file columns mismatch",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    seen: dict[str, Path] = {}

    def fake_validate(self, path):  # noqa: ANN001
        seen["path"] = path
        return path

    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.SubmissionService.validate_and_prepare_submission", fake_validate)
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)

    result = _attempt_submit(
        config=config,
        run_id=run_id,
        submission_path=original_submission,
        best_score=0.4,
        problem_types=[],
    )

    assert result is not None
    assert seen["path"] == repaired_submission


def test_attempt_submit_does_not_reuse_stale_repaired_submit_artifact(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=2)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    original_submission = config.paths.iter_dir(run_id, 1) / "submission.csv"
    original_submission.parent.mkdir(parents=True, exist_ok=True)
    original_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    repaired_submission = config.paths.iter_dir(run_id, 1) / "output" / "submission-fixed.csv"
    repaired_submission.parent.mkdir(parents=True, exist_ok=True)
    repaired_submission.write_text("id,target\n1,0.3\n2,0.4\n", encoding="utf-8")
    new_submission = config.paths.iter_dir(run_id, 2) / "submission.csv"
    new_submission.parent.mkdir(parents=True, exist_ok=True)
    new_submission.write_text("id,target\n1,0.5\n2,0.6\n", encoding="utf-8")

    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_error_kind": "validation",
                "last_reason": "submission_poll_status_error",
                "submit_autofix_submission_path": str(repaired_submission),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "submit_failure_context.json").write_text(
        json.dumps(
            {
                "ts": "2026-02-15T00:00:01+00:00",
                "active": True,
                "repair_target": "submission_artifact",
                "repairable": True,
                "reason": "submission_poll_status_error",
                "error_kind": "validation",
                "submission_ref": str(original_submission),
                "submission_artifact_path": str(original_submission),
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
                "exit_code": None,
                "error_kind": "validation",
                "reason": "submission_poll_status_error",
                "action_taken": "abort",
                "fingerprint": "abc123",
                "sub_path": str(original_submission),
                "sub_sha256": "old-hash",
                "stderr_tail": "Kaggle reported: submission file columns mismatch",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    seen: dict[str, Path] = {}

    def fake_validate(self, path):  # noqa: ANN001
        seen["path"] = path
        return path

    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.SubmissionService.validate_and_prepare_submission", fake_validate)
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)

    result = _attempt_submit(
        config=config,
        run_id=run_id,
        submission_path=new_submission,
        best_score=0.4,
        problem_types=[],
    )

    assert result is not None
    assert seen["path"] == new_submission
    run_state = load_run_state(run_dir)
    assert run_state.get("submit_autofix_submission_path") == ""
    context = load_submit_failure_context(run_dir)
    assert context["superseded_by_submission_path"] == str(new_submission)


def test_resolve_iteration_submission_artifact_prefers_manifest_when_bundle_staging_exists(tmp_path: Path) -> None:
    iter_dir = tmp_path / "iter-1"
    bundle_dir = iter_dir / "output" / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "mask.tif").write_bytes(b"mask")
    manifest_path = iter_dir / "output" / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "staging_dir": "bundle",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert _resolve_iteration_submission_artifact(iter_dir) == manifest_path


def test_resolve_iteration_submission_artifact_uses_latest_fold_intermediate(tmp_path: Path) -> None:
    iter_dir = tmp_path / "iter-1"
    iter_dir.mkdir(parents=True, exist_ok=True)
    fold1 = iter_dir / "submission_model_fold1.csv"
    fold2 = iter_dir / "output" / "submission_model_fold2.csv"
    fold2.parent.mkdir()
    fold1.write_text("id,target\n1,0.1\n", encoding="utf-8")
    fold2.write_text("id,target\n1,0.2\n", encoding="utf-8")
    os.utime(fold1, (1000, 1000))
    os.utime(fold2, (2000, 2000))

    assert _resolve_iteration_submission_artifact(iter_dir) == fold2


def test_copy_submission_artifact_to_iteration_dir_skips_same_path(tmp_path: Path) -> None:
    iter_dir = tmp_path / "iter-1"
    iter_dir.mkdir()
    submission = iter_dir / "submission.csv"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")

    copied = _autopilot_state_test.copy_submission_artifact_to_iteration_dir(source=submission, iter_dir=iter_dir)

    assert copied == submission
    assert submission.read_text(encoding="utf-8") == "id,target\n1,0.1\n"


def test_attempt_submit_aborts_when_polling_raises_error(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )

    def raise_poll_error(**kwargs):  # noqa: ARG001
        raise SubmissionOutcomePollingError(
            "poll failed",
            attempt=3,
            consecutive_errors=3,
            detail="RuntimeError: kaggle api unreachable",
        )

    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", raise_poll_error)

    with pytest.raises(SubmitAbortedError, match="polling failed"):
        _attempt_submit(
            config=config,
            run_id=run_id,
            submission_path=submission_path,
            best_score=0.4,
            problem_types=[],
        )

    rows = [
        json.loads(line)
        for line in (run_dir / "submit_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["action_taken"] for row in rows[-1:]] == ["abort"]
    assert rows[-1]["reason"] == "submission_polling_error"
    assert rows[-1]["error_kind"] == "transient"


def test_autopilot_submits_every_iteration_without_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_MIN_HOURS_BETWEEN", "0")
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
        output_path.write_text(f"id,target\n1,{0.1 + train_calls['count'] * 0.001:.3f}\n2,0.2\n", encoding="utf-8")
        value = 0.4 - 0.01 * float(train_calls["count"] - 1)
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
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

    def fake_submit(*args, **kwargs):  # noqa: ARG001
        submit_calls["count"] += 1
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fake_submit)
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)
    monkeypatch.setattr("kagglebot.submission_policy.should_attempt_submit_for_readiness", lambda **kwargs: True)

    config = _make_config(tmp_path, submit=True, max_iterations=2, force_submit=False)
    run_autopilot(config)

    assert train_calls["count"] == 2
    assert submit_calls["count"] == 2

    iter2_state = json.loads(
        (config.paths.iter_dir(config.run_id or "run-1", 2) / "iteration_state.json").read_text(encoding="utf-8")
    )
    assert iter2_state["submit_phase_finished"] is True
    assert iter2_state["submit_allowed_by_gate"] is True
    assert iter2_state["submit_phase_state"] == "submitted"


def test_autopilot_allows_non_improving_submit_on_final_iteration(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_MIN_HOURS_BETWEEN", "0")
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
        output_path.write_text(f"id,target\n1,{0.1 + train_calls['count'] * 0.001:.3f}\n2,0.2\n", encoding="utf-8")
        value = 0.4 if train_calls["count"] == 1 else 0.45
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
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

    def fake_submit(*args, **kwargs):  # noqa: ARG001
        submit_calls["count"] += 1
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.35})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fake_submit)
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)
    monkeypatch.setattr("kagglebot.submission_policy.should_attempt_submit_for_readiness", lambda **kwargs: True)

    config = _make_config(tmp_path, submit=True, max_iterations=2, force_submit=False)
    run_autopilot(config)

    assert train_calls["count"] == 2
    assert submit_calls["count"] == 2

    iter2_state = json.loads(
        (config.paths.iter_dir(config.run_id or "run-1", 2) / "iteration_state.json").read_text(encoding="utf-8")
    )
    assert iter2_state["submit_phase_state"] == "submitted"


def test_autopilot_uses_spare_daily_slots_for_non_improving_soft_quality_guard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_MIN_HOURS_BETWEEN", "0")
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_plan(
        paths,
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )
    _write_rules(paths, "You may submit 5 submissions per day.\n")

    train_calls = {"count": 0}
    submit_calls = {"count": 0}

    def fake_train(*args, **kwargs):  # noqa: ARG001
        train_calls["count"] += 1
        output_path = kwargs["output_path"]
        output_path.write_text(f"id,target\n1,{0.1 + train_calls['count'] * 0.001:.3f}\n2,0.2\n", encoding="utf-8")
        values = {1: 0.30, 2: 0.31, 3: 0.32}
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            value=values[train_calls["count"]],
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

    def fake_submit(*args, **kwargs):  # noqa: ARG001
        submit_calls["count"] += 1
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    def fake_quality_guard(**kwargs):
        if kwargs["iteration"] == 1:
            return {
                "allow_submit": True,
                "reasons": [],
                "competition_faithfulness": {"faithful": True, "trusted": True},
            }
        return {
            "allow_submit": False,
            "reasons": ["selected_worse_than_detected_baseline"],
            "competition_faithfulness": {"faithful": True, "trusted": True},
        }

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.25})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.kernel_quality.build_kernel_quality_guard", fake_quality_guard)
    monkeypatch.setattr("kagglebot.submission_policy.submission_count_for_daily_limit", lambda **kwargs: 1)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fake_submit)
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: None)
    monkeypatch.setattr("kagglebot.submit_stage.resolve_submission_rank_payload", lambda **kwargs: {})

    config = _make_config(
        tmp_path, paths=paths, submit=True, max_iterations=3, force_submit=False, submit_policy="improved"
    )
    run_autopilot(config)

    assert train_calls["count"] == 3
    assert submit_calls["count"] == 3
    iter2_state = json.loads((config.paths.iter_dir(config.run_id or "run-1", 2) / "iteration_state.json").read_text())
    assert iter2_state["submit_phase_state"] == "submitted"
    assert iter2_state["forced_submit_reason"] == "spare_daily_submission_slot"


def test_autopilot_submit_improvement_prefers_online_submission_score(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_MIN_HOURS_BETWEEN", "0")
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
        output_path.write_text(f"id,target\n1,{0.1 + train_calls['count'] * 0.001:.3f}\n2,0.2\n", encoding="utf-8")
        values = {1: 0.30, 2: 0.31, 3: 0.305}
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            value=values[train_calls["count"]],
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

    def fake_submit(*args, **kwargs):  # noqa: ARG001
        submit_calls["count"] += 1
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    outcomes = [
        {"status": "complete", "score": 0.50},
        {"status": "complete", "score": 0.40},
        {"status": "complete", "score": 0.39},
    ]

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.35})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.submission_service.run_kaggle_submit", fake_submit)
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: outcomes.pop(0))
    monkeypatch.setattr("kagglebot.submit_stage.resolve_submission_rank_payload", lambda **kwargs: {})
    monkeypatch.setattr("kagglebot.submission_policy.should_attempt_submit_for_readiness", lambda **kwargs: True)

    config = _make_config(tmp_path, submit=True, max_iterations=3, force_submit=False)
    run_autopilot(config)

    assert train_calls["count"] == 3
    assert submit_calls["count"] == 3
    iter2_state = json.loads(
        (config.paths.iter_dir(config.run_id or "run-1", 2) / "iteration_state.json").read_text(encoding="utf-8")
    )
    assert iter2_state["submit_phase_state"] == "submitted"
    iter2_metrics = json.loads((config.paths.iter_dir(config.run_id or "run-1", 2) / "metrics.json").read_text())
    assert float(iter2_metrics["submission_score"]) == pytest.approx(0.4)


def test_load_run_state_infers_submit_ok_from_submit_attempts(tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_dir = config.paths.run_dir(config.run_id or "run-1")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_submit_fingerprint": "abc",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "submit_attempts.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "fingerprint": "abc", "ok": True}) + "\n",
        encoding="utf-8",
    )

    state = load_run_state(run_dir)
    assert state["submit_attempted"] is True
    assert state["submit_ok"] is True


def test_attempt_submit_persists_submit_failure_context_for_validation_abort(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    def fail_validation(self, path):  # noqa: ANN001
        raise SubmissionValidationError("prediction column contains NaN")

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission", fail_validation
    )

    with pytest.raises(SubmitAbortedError, match="Local submission validation failed"):
        _attempt_submit(
            config=config,
            run_id=run_id,
            submission_path=submission_path,
            best_score=0.4,
            problem_types=[],
        )

    context = load_submit_failure_context(run_dir)
    assert context["repair_target"] == "submission_artifact"
    assert context["repairable"] is True
    assert context["reason"] == "local_submission_validation_failed"
    assert context["submission_artifact_path"] == str(submission_path)
    assert "prediction column contains NaN" in str(context["summary"])


def test_attempt_submit_persists_manual_submit_failure_context_for_rules_block(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, submit=True, max_iterations=1)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = config.paths.iter_dir(run_id, 1) / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    monkeypatch.setattr(
        "kagglebot.submission_service.SubmissionService.validate_and_prepare_submission",
        lambda self, path: path,  # noqa: ARG005
    )
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: False)

    with pytest.raises(SubmitAbortedError, match="Competition rules are not accepted"):
        _attempt_submit(
            config=config,
            run_id=run_id,
            submission_path=submission_path,
            best_score=0.4,
            problem_types=[],
        )

    context = load_submit_failure_context(run_dir)
    assert context["repair_target"] == "manual_intervention"
    assert context["repairable"] is False
    assert context["reason"] == "rules_not_accepted"
    assert "Accept the competition rules" in str(context["manual_next_step"])
    assert _resolve_submit_abort_autofixable_for_config(config=config, run_id=run_id) is False


def test_run_autofix_submit_error_always_runs_strategy_then_codex(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    original_submission = config.paths.iter_dir(run_id, 1) / "submission.csv"
    original_submission.parent.mkdir(parents=True, exist_ok=True)
    original_submission.write_text("id,target\n1,0.1\n2,not_a_number\n", encoding="utf-8")
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_error_kind": "validation",
                "last_reason": "local_submission_validation_failed",
                "last_submission_path": str(original_submission),
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
                "sub_path": str(original_submission),
                "stderr_tail": "submission payload mismatch",
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

    monkeypatch.setattr("kagglebot.runtime_fixes.maybe_write_column_fill", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.runtime_fixes.maybe_write_object_coerce", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.runtime_fixes.maybe_write_device_coerce", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.runtime_fixes.maybe_write_column_map", lambda *args, **kwargs: True)

    def fake_prepare_submit_file_autofix_for_run(**kwargs: object) -> SubmitFileAutofixPreparation:
        fixed_submission = config.paths.iter_dir(run_id, 1) / "output" / "submission-fixed.csv"
        fixed_submission.parent.mkdir(parents=True, exist_ok=True)
        fixed_submission.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        save_repaired_path = kwargs["save_repaired_path"]
        assert callable(save_repaired_path)
        save_repaired_path(fixed_submission)
        return SubmitFileAutofixPreparation(
            path=fixed_submission,
            summary=f"fixed_submission_path: {fixed_submission}",
            file_fix_required=True,
        )

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

    def fail_if_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("autofix must not call write-guard")

    monkeypatch.setattr("kagglebot.autopilot.run_strategy", fake_run_strategy)
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr(
        "kagglebot.submit_autofix.prepare_submit_file_autofix_for_run",
        fake_prepare_submit_file_autofix_for_run,
    )
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._backup_guarded_files", fail_if_called)
    monkeypatch.setattr("kagglebot.autopilot._snapshot_tree", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._diff_snapshots", lambda *args, **kwargs: [])
    monkeypatch.setattr("kagglebot.autopilot._enforce_allowlist_changes", fail_if_called)
    monkeypatch.setattr(
        "kagglebot.autopilot._autofix_restart.maybe_restart_for_src_changes", lambda *args, **kwargs: None
    )

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
    assert "## Submission File Repair Contract" in prompt_text
    assert "Codex" not in prompt_text
    run_state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert run_state["submit_autofix_submission_path"].endswith("submission-fixed.csv")
    strategy_prompt = (run_dir / "autofix" / "attempt-1" / "gpt_strategy" / "gpt_strategy_prompt.md").read_text(
        encoding="utf-8"
    )
    assert "Stage: submit_autofix" in strategy_prompt


def test_run_autofix_submit_error_still_runs_strategy_for_internet_policy(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": False,
                "last_error_kind": "unknown",
                "last_reason": "bad_request",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    calls = {"strategy": 0, "codex": 0}

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool):  # noqa: ARG001
        calls["strategy"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "strategy_last_message.txt"
        last_msg.write_text("strategy\n", encoding="utf-8")
        return DummyResult(last_msg)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        calls["codex"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("submit fix applied\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.run_strategy", fake_run_strategy)
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._snapshot_tree", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._diff_snapshots", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "kagglebot.autopilot._autofix_restart.maybe_restart_for_src_changes", lambda *args, **kwargs: None
    )

    _run_autofix(
        config=config,
        run_id=run_id,
        attempt=1,
        error=SubmitAbortedError("Cannot submit: Your Notebook cannot use internet access in this competition."),
    )

    assert calls["strategy"] == 1
    assert calls["codex"] == 1


def test_run_autofix_submit_error_falls_back_to_direct_codex_when_strategy_empty(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    calls = {"codex": 0}

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        calls["codex"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("submit fix applied\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.agent_strategy.run_error_strategy_prompt", lambda **kwargs: "")
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._snapshot_tree", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._diff_snapshots", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "kagglebot.autopilot._autofix_restart.maybe_restart_for_src_changes", lambda *args, **kwargs: None
    )

    _run_autofix(
        config=config,
        run_id=run_id,
        attempt=1,
        error=SubmitAbortedError("status 'error' during polling; aborting submit stage for this run."),
    )

    assert calls["codex"] == 1
    prompt_text = (run_dir / "autofix" / "attempt-1" / "prompt.md").read_text(encoding="utf-8")
    assert "## GPT 5.4 Extra-High Error-Fix Strategy" not in prompt_text


def test_run_autofix_retries_same_attempt_when_verify_fails(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    run_id = config.run_id or "run-1"
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    calls = {"codex": 0, "verify": 0}

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        calls["codex"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / f"codex_last_message-{calls['codex']}.txt"
        last_msg.write_text(f"autofix pass {calls['codex']}\n", encoding="utf-8")
        return DummyResult(last_msg)

    def flaky_verify(*args, **kwargs):  # noqa: ANN002, ARG001
        calls["verify"] += 1
        if calls["verify"] == 1:
            raise RuntimeError("Verification failed: first pass")

    monkeypatch.setattr("kagglebot.agent_strategy.run_error_strategy_prompt", lambda **kwargs: "1) fix root cause\n")
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", flaky_verify)
    monkeypatch.setattr("kagglebot.autopilot._snapshot_tree", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._diff_snapshots", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "kagglebot.autopilot._autofix_restart.maybe_restart_for_src_changes", lambda *args, **kwargs: None
    )

    _run_autofix(
        config=config,
        run_id=run_id,
        attempt=1,
        error=RuntimeError("train crashed"),
    )

    assert calls["codex"] == 2
    assert calls["verify"] == 2
    retry_prompt = run_dir / "autofix" / "attempt-1" / "prompt-pass-02.md"
    assert retry_prompt.exists()
    assert "Retry Feedback (pass 1)" in retry_prompt.read_text(encoding="utf-8")


def test_run_kernel_fix_retries_same_attempt_when_verify_fails(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    run_id = config.run_id or "run-1"
    iter_dir = config.paths.iter_dir(run_id, 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    calls = {"codex": 0, "verify": 0}

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        calls["codex"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / f"kernel_fix_last_message-{calls['codex']}.txt"
        last_msg.write_text(f"kernel fix pass {calls['codex']}\n", encoding="utf-8")
        return DummyResult(last_msg)

    def flaky_verify(*args, **kwargs):  # noqa: ANN002, ARG001
        calls["verify"] += 1
        if calls["verify"] == 1:
            raise RuntimeError("Verification failed: first kernel-fix pass")

    monkeypatch.setattr("kagglebot.agent_strategy.run_error_strategy_prompt", lambda **kwargs: "")
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", flaky_verify)
    monkeypatch.setattr("kagglebot.autopilot._backup_guarded_files", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._snapshot_tree", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._diff_snapshots", lambda *args, **kwargs: ["src/kagglebot/autopilot.py"])
    monkeypatch.setattr("kagglebot.autopilot._enforce_allowlist_changes", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.autopilot._autofix_restart.maybe_restart_for_src_changes", lambda *args, **kwargs: None
    )

    _run_kernel_fix(
        config=config,
        run_id=run_id,
        iteration=1,
        iter_dir=iter_dir,
        error_message="RuntimeError: kernel failed",
        attempt=1,
        pending_error_fixes=[],
    )

    assert calls["codex"] == 2
    assert calls["verify"] == 2
    retry_prompt = iter_dir / "agent" / "kernel_fix_prompt-01-pass-02.md"
    assert retry_prompt.exists()
    assert "Retry Feedback (pass 1)" in retry_prompt.read_text(encoding="utf-8")


def test_run_kernel_fix_includes_subgroup_prompt_context(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    run_id = config.run_id or "run-1"
    iter_dir = config.paths.iter_dir(run_id, 1)
    (iter_dir / "output").mkdir(parents=True, exist_ok=True)
    (iter_dir / "output" / "metrics.json").write_text(
        json.dumps(
            {
                "cv_breakdown_by_model_node": {
                    "model_1_node_type_1": 0.020,
                    "model_1_node_type_2": 0.015,
                    "model_2_node_type_1": 0.240,
                    "model_2_node_type_2": 0.085,
                },
                "cv_step_buckets": {
                    "000-011": 0.05,
                    "144-155": 0.12,
                },
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        captured["prompt"] = prompt_path.read_text(encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("kernel fix applied\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.agent_strategy.run_error_strategy_prompt", lambda **kwargs: "")
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._backup_guarded_files", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._snapshot_tree", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._diff_snapshots", lambda *args, **kwargs: ["src/kagglebot/autopilot.py"])
    monkeypatch.setattr("kagglebot.autopilot._enforce_allowlist_changes", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.autopilot._autofix_restart.maybe_restart_for_src_changes", lambda *args, **kwargs: None
    )

    _run_kernel_fix(
        config=config,
        run_id=run_id,
        iteration=1,
        iter_dir=iter_dir,
        error_message="RuntimeError: kernel failed",
        attempt=1,
        pending_error_fixes=[],
    )

    prompt_text = captured["prompt"]
    assert "Subgroup repair target:" in prompt_text
    assert "model=2 node_type=1" in prompt_text
    assert "(model_id,node_type) granularity" in prompt_text


def test_build_evaluation_contract_prefers_competition_metric_override_for_deep_past(tmp_path: Path) -> None:
    paths = CompetitionPaths(
        slug="deep-past-initiative-machine-translation",
        artifacts_dir=tmp_path / "artifacts",
    )

    contract = build_evaluation_contract(
        slug=paths.slug,
        eval_spec={},
        target_metric="accuracy",
        target_direction="maximize",
        split_strategy="kfold",
    )

    assert contract["expected_metric"] == "geometric mean of the bleu and the chrf++ scores"
    assert contract["expected_direction"] == "maximize"
    assert contract["expected_split_strategy"] == "group_kfold"


def test_resolve_plan_overrides_stale_deep_past_accuracy_contract(tmp_path: Path) -> None:
    paths = CompetitionPaths(
        slug="deep-past-initiative-machine-translation",
        artifacts_dir=tmp_path / "artifacts",
    )
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    plan = PlanConfig(
        target_metric="accuracy",
        target_direction="maximize",
        target_score=39.5,
        split_strategy="kfold",
    )
    config = AutopilotConfig(
        run_id="run-1",
        slug="deep-past-initiative-machine-translation",
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

    resolved = _resolve_plan(plan, config)
    contract = resolved["evaluation_contract"]

    assert resolved["target_metric"] == "Geometric Mean of the BLEU and the chrF++ scores"
    assert resolved["split_strategy"] == "group_kfold"
    assert contract["expected_metric"] == "geometric mean of the bleu and the chrf++ scores"
    assert contract["expected_split_strategy"] == "group_kfold"


def test_resolve_plan_keeps_deep_past_override_even_with_stale_evaluation_spec(tmp_path: Path) -> None:
    paths = CompetitionPaths(
        slug="deep-past-initiative-machine-translation",
        artifacts_dir=tmp_path / "artifacts",
    )
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    _write_sample_submission(paths.sample_submission_path)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.joinpath("evaluation_spec.json").write_text(
        json.dumps(
            {
                "metric_name": "accuracy",
                "direction": "maximize",
                "split_strategy": "kfold",
                "n_splits": 7,
                "seeds": [42, 2024, 777],
                "repeats": 2,
                "ci_method": "normal",
                "ci_alpha": 0.05,
                "readiness_rule": {
                    "method": "ci_bound",
                    "k": 1.0,
                    "target_score": 39.5,
                    "submission_gate": "always",
                },
                "drift_check": {"enabled": False, "drift_weight": 1.0},
                "stop_policy": {
                    "min_delta": 0.0,
                    "no_improve_patience": 2,
                    "same_config_patience": 2,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config = AutopilotConfig(
        run_id="run-1",
        slug="deep-past-initiative-machine-translation",
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

    resolved = _resolve_plan(PlanConfig(target_metric="accuracy", split_strategy="kfold"), config)
    contract = resolved["evaluation_contract"]

    assert resolved["target_metric"] == "Geometric Mean of the BLEU and the chrF++ scores"
    assert resolved["split_strategy"] == "group_kfold"
    assert contract["expected_metric"] == "geometric mean of the bleu and the chrf++ scores"
    assert contract["expected_split_strategy"] == "group_kfold"


def test_load_competition_rule_constraints_detects_submit_page_internet_ban(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_rules(
        paths,
        "Your Notebook cannot use internet access in this competition. "
        "Please disable internet in the Notebook editor and save a new version.",
    )

    constraints = load_competition_rule_constraints(paths)

    assert constraints.internet_must_be_off is True


def test_resolve_plan_forces_internet_off_when_rules_ban_notebook_internet(tmp_path: Path) -> None:
    config = _make_config(tmp_path, compute="kaggle_gpu", internet="on")
    _write_rules(
        config.paths,
        "Code submissions to this competition must be made through Notebooks. "
        "Internet access is not allowed for submissions.",
    )

    resolved = _resolve_plan(PlanConfig(), config)

    assert resolved["submit_mode"] == "notebook"
    assert resolved["internet"] == "off"


def test_kernel_quality_guard_blocks_when_selected_worse_than_baseline() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="rmse",
        direction="minimize",
        value=0.5,
        std=0.01,
        train_score=None,
        val_score=None,
        fold_scores=[0.49, 0.51],
    )
    payload = {
        "selected_pipeline": "heavy_model",
        "pipelines": [
            {"name": "persistence_baseline", "offline_value": 0.1},
            {"name": "heavy_model", "offline_value": 0.5},
        ],
    }
    guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload=payload,
        evaluation_report=None,
        evaluation_contract=None,
        logs_dir=None,
        direction="minimize",
        iteration=1,
        max_iterations=3,
        force_submit=False,
    )
    assert guard["allow_submit"] is False
    reasons = guard.get("reasons")
    assert isinstance(reasons, list)
    assert "selected_worse_than_detected_baseline" in reasons


def test_kernel_quality_guard_ignores_baseline_fold_indices_in_logs(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "local_kernel_stdout.log").write_text(
        "\n".join(
            [
                "baseline_metadata fold=0 seed=3557 macro_f1=0.38366",
                "baseline_metadata fold=1 seed=3557 macro_f1=0.36519",
                "baseline_metadata fold=2 seed=3557 macro_f1=0.36136",
                "baseline_metadata fold=3 seed=3557 macro_f1=0.37635",
                "baseline_metadata cv_macro_f1=0.37159 mean_seed=0.37159",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation = EvaluationResult(
        score_source="cv",
        metric="macro_f1",
        direction="maximize",
        value=0.860307,
        std=0.015,
        train_score=None,
        val_score=None,
        fold_scores=[0.84, 0.85, 0.87, 0.88],
    )
    guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload={},
        evaluation_report=None,
        evaluation_contract=None,
        logs_dir=logs_dir,
        direction="maximize",
        iteration=2,
        max_iterations=5,
        force_submit=False,
    )
    reasons = guard.get("reasons")
    assert isinstance(reasons, list)
    assert "selected_worse_than_detected_baseline" not in reasons
    baseline = guard.get("baseline")
    assert isinstance(baseline, dict)
    assert baseline.get("best_score") == pytest.approx(0.37159)


def test_kernel_quality_guard_blocks_on_severe_validation_mismatch(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "local_kernel_stdout.log").write_text(
        "epoch=10 val_rmse=0.100000\nepoch=20 val_rmse=0.095000\n",
        encoding="utf-8",
    )
    evaluation = EvaluationResult(
        score_source="cv",
        metric="rmse",
        direction="minimize",
        value=1.2,
        std=0.05,
        train_score=None,
        val_score=None,
        fold_scores=[1.15, 1.25],
    )
    guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload={},
        evaluation_report=None,
        evaluation_contract=None,
        logs_dir=logs_dir,
        direction="minimize",
        iteration=1,
        max_iterations=3,
        force_submit=False,
    )
    assert guard["allow_submit"] is False
    reasons = guard.get("reasons")
    assert isinstance(reasons, list)
    assert "validation_metric_mismatch_vs_final_metric" in reasons

    final_guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload={},
        evaluation_report=None,
        evaluation_contract=None,
        logs_dir=logs_dir,
        direction="minimize",
        iteration=3,
        max_iterations=3,
        force_submit=False,
    )
    assert final_guard["allow_submit"] is True


def test_kernel_quality_guard_blocks_cv_selected_pipeline_with_worse_holdout_candidate() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="macro_f1",
        direction="maximize",
        value=0.0752,
        std=0.01,
        train_score=None,
        val_score=None,
        fold_scores=[0.07, 0.08],
    )
    payload = {
        "chosen_pipeline": "cv_only_sparse",
        "pipelines": [
            {
                "name": "public_like_reference",
                "cv_score": 0.0497,
                "holdout_score": 0.0384,
                "prediction_count_summary": {"test": {"mean": 10.0}},
            },
            {
                "name": "cv_only_sparse",
                "cv_score": 0.0752,
                "holdout_score": 0.0200,
                "prediction_count_summary": {"test": {"mean": 2.0}},
            },
        ],
    }
    guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload=payload,
        evaluation_report=None,
        evaluation_contract=None,
        logs_dir=None,
        direction="maximize",
        iteration=3,
        max_iterations=5,
        force_submit=False,
    )
    assert guard["allow_submit"] is False
    reasons = guard.get("reasons")
    assert isinstance(reasons, list)
    assert "selected_pipeline_validation_mismatch" in reasons
    assert "prediction_distribution_collapse_vs_candidates" in reasons
    mismatch = guard.get("candidate_selection_mismatch")
    assert isinstance(mismatch, dict)
    assert mismatch["best_secondary_candidate"] == "public_like_reference"


def test_kernel_quality_guard_surfaces_subgroup_collapse() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="rmse",
        direction="minimize",
        value=0.078,
        std=0.002,
        train_score=None,
        val_score=0.078,
        fold_scores=[0.077, 0.078, 0.079],
    )
    payload = {
        "cv_breakdown_by_model_node": {
            "model_1_node_type_1": 0.015,
            "model_1_node_type_2": 0.010,
            "model_2_node_type_1": 0.220,
            "model_2_node_type_2": 0.080,
        },
        "cv_step_buckets": {
            "000-011": 0.04,
            "012-023": 0.05,
            "144-155": 0.11,
        },
    }

    guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload=payload,
        evaluation_report=None,
        evaluation_contract=None,
        logs_dir=None,
        direction="minimize",
        iteration=1,
        max_iterations=3,
        force_submit=False,
    )

    assert "cv_subgroup_collapse_detected" in guard["warnings"]
    subgroup = guard["subgroup_collapse"]
    assert isinstance(subgroup, dict)
    assert subgroup["model_id"] == 2
    assert subgroup["node_type"] == 1
    assert subgroup["worst_key"] == "model_2_node_type_1"
    assert "Next iteration must use subgroup-aware selection" in str(subgroup["note"])


def test_kernel_quality_guard_blocks_oracle_or_untrusted_score_source() -> None:
    evaluation = EvaluationResult(
        score_source="oracle",
        metric="rmse",
        direction="minimize",
        value=0.1,
        std=0.01,
        train_score=None,
        val_score=None,
        fold_scores=[0.1, 0.1],
    )
    guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload={"oracle": {"mode_setting": "auto", "applied": True}},
        evaluation_report=None,
        evaluation_contract=None,
        logs_dir=None,
        direction="minimize",
        iteration=3,
        max_iterations=3,
        force_submit=False,
    )
    assert guard["allow_submit"] is False
    reasons = guard.get("reasons")
    assert isinstance(reasons, list)
    assert "untrusted_score_source" in reasons
    assert "oracle_override_detected" in reasons


def test_kernel_quality_guard_blocks_external_test_label_transfer_even_when_forced() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="macro_f1",
        direction="maximize",
        value=1.0,
        std=0.0,
        train_score=None,
        val_score=None,
        fold_scores=[1.0, 1.0, 1.0],
    )
    payload = {
        "final_selected_method": "official_multiview_overlap_mapping",
        "submission_rows": 6872,
        "submission_audit": {
            "external_overlap_trusted": True,
            "exact_coverage_pass": True,
            "external_root_path": "/kaggle/input/some-public-overlap",
            "test_selected_row_count": 6872,
            "uncovered_test_row_count": 0,
            "test_exact_sha1_matched_image_count": 926,
            "test_max_selected_image_distance": 0.0,
            "test_max_selected_bbox_distance": 0.0,
            "official_overlap_audit": {
                "match_type_counts": {
                    "test_selected_rows": {"exact_sha1": 6872},
                },
            },
        },
    }
    guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload=payload,
        evaluation_report=None,
        evaluation_contract=None,
        logs_dir=None,
        direction="maximize",
        iteration=3,
        max_iterations=3,
        force_submit=True,
    )

    assert guard["allow_submit"] is False
    reasons = guard.get("reasons")
    assert isinstance(reasons, list)
    assert "external_test_label_transfer_detected" in reasons
    transfer = guard.get("external_label_transfer")
    assert isinstance(transfer, dict)
    assert transfer["test_selected_row_count"] == 6872

    potential = build_accuracy_potential(
        score_source=evaluation.score_source,
        kernel_metrics_payload=payload,
        model_summary=None,
        quality_guard=guard,
        evaluation_contract=None,
    )
    assert potential["eligible"] is False
    assert potential["status"] == "blocked"
    assert potential["primary_reason"] == "external_test_label_transfer_detected"


def test_kernel_quality_guard_blocks_when_below_code_reference_baseline() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.62,
        std=0.01,
        train_score=None,
        val_score=None,
        fold_scores=[0.60, 0.64],
    )
    guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload={},
        evaluation_report=None,
        evaluation_contract=None,
        logs_dir=None,
        direction="maximize",
        iteration=3,
        max_iterations=3,
        force_submit=False,
        code_reference_score=0.741,
        code_reference_source="code_index:alice/ref-kernel",
    )
    assert guard["allow_submit"] is False
    reasons = guard.get("reasons")
    assert isinstance(reasons, list)
    assert "below_code_reference_baseline" in reasons
    code_ref = guard.get("code_reference")
    assert isinstance(code_ref, dict)
    assert code_ref.get("below_reference") is True


def test_kernel_quality_guard_normalizes_percent_code_reference_for_accuracy() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="accuracy",
        direction="maximize",
        value=0.995220,
        std=0.001,
        train_score=None,
        val_score=None,
        fold_scores=[0.995, 0.996],
    )
    guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload={},
        evaluation_report=None,
        evaluation_contract=None,
        logs_dir=None,
        direction="maximize",
        iteration=3,
        max_iterations=3,
        force_submit=False,
        code_reference_score=98.9,
        code_reference_source="code_index:abdulravoofshaik/denoising-autoencoder-lb-98-9",
    )
    reasons = guard.get("reasons")
    assert isinstance(reasons, list)
    assert "below_code_reference_baseline" not in reasons
    code_ref = guard.get("code_reference")
    assert isinstance(code_ref, dict)
    assert code_ref.get("comparison_score") == pytest.approx(0.989)
    assert code_ref.get("below_reference") is False


def test_kernel_quality_guard_blocks_competition_split_mismatch() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="rmse",
        direction="minimize",
        value=0.12,
        std=0.01,
        train_score=None,
        val_score=None,
        fold_scores=[0.11, 0.13],
    )
    report = EvaluationReport(
        metric_name="rmse",
        direction="minimize",
        split_strategy="kfold",
        n_splits=5,
        seeds=[42, 2024],
        repeats=2,
        per_fold_scores=[0.11, 0.12, 0.13],
        mean=0.12,
        std=0.01,
        ci_low=0.11,
        ci_high=0.13,
        drift_auc=None,
        readiness_score=0.12,
    )
    guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload={"readiness": {"split_strategy": "kfold"}},
        evaluation_report=report,
        evaluation_contract={
            "expected_metric": "rmse",
            "expected_split_strategy": "timeseries_split",
            "accepted_score_sources": ["cv", "holdout"],
            "require_metric_match": True,
            "require_split_match": True,
            "require_trusted_score_source": True,
            "require_competition_faithful": True,
            "require_full_dataset": False,
        },
        logs_dir=None,
        direction="minimize",
        iteration=1,
        max_iterations=3,
        force_submit=False,
    )
    assert guard["allow_submit"] is False
    reasons = guard.get("reasons")
    assert isinstance(reasons, list)
    assert "competition_split_mismatch" in reasons


def test_kernel_quality_guard_blocks_missing_competitive_data_from_sample_score_source() -> None:
    evaluation = EvaluationResult(
        score_source="sample_mode_smoke_cv",
        metric="rmse",
        direction="minimize",
        value=0.02,
        std=0.0,
        train_score=None,
        val_score=None,
        fold_scores=[0.02],
    )
    guard = build_kernel_quality_guard(
        evaluation=evaluation,
        kernel_metrics_payload={},
        evaluation_report=None,
        evaluation_contract={
            "expected_metric": "rmse",
            "expected_split_strategy": "timeseries_split",
            "accepted_score_sources": ["cv", "holdout"],
            "require_metric_match": True,
            "require_split_match": True,
            "require_trusted_score_source": True,
            "require_competition_faithful": True,
            "require_full_dataset": True,
        },
        logs_dir=None,
        direction="minimize",
        iteration=1,
        max_iterations=3,
        force_submit=False,
    )
    assert guard["allow_submit"] is False
    reasons = guard.get("reasons")
    assert isinstance(reasons, list)
    assert "competition_evaluation_unfaithful" in reasons
    assert "missing_competitive_data" in reasons


def test_build_accuracy_potential_prefers_high_capacity_candidate_when_not_yet_faithful() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="rmse",
        direction="minimize",
        value=0.12,
        std=0.01,
        train_score=None,
        val_score=None,
        fold_scores=[0.11, 0.13],
    )
    quality_guard = {
        "reasons": ["missing_competitive_data"],
        "competition_faithfulness": {
            "faithful": False,
            "metric_match": True,
            "split_match": True,
            "full_dataset_resolved": False,
        },
    }
    potential = build_accuracy_potential(
        score_source=evaluation.score_source,
        kernel_metrics_payload={"selected_pipeline": "graph_transformer_hybrid"},
        model_summary={"models": ["graph_transformer_hybrid"]},
        quality_guard=quality_guard,
        evaluation_contract={"require_full_dataset": True},
    )
    assert potential["eligible"] is True
    assert potential["status"] == "frontier"
    assert potential["capacity_tier"] in {"high", "extreme"}
    assert potential["data_tier"] == "minimum_submit_data"


def test_autopilot_missing_kernel_metric_triggers_kernel_fix(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import AutopilotSession

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.plan_policy.needs_planning", lambda **kwargs: False)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.MAX_KERNEL_FIX_ATTEMPTS", 1)

    calls: dict[str, int] = {"kernel_fix": 0}

    def fake_run_kernel_local(**kwargs):  # noqa: ANN003
        slug = kwargs["slug"]
        run_id = kwargs["run_id"]
        iteration = kwargs["iteration"]
        base_dir = Path(kwargs["base_dir"])
        output_dir = base_dir / slug / "runs" / run_id / f"iter-{iteration}" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        metrics_path.write_text(json.dumps({"metric": "rmse"}, indent=2), encoding="utf-8")
        return KernelRunResult(
            kernel_id=f"local/{slug}",
            output_dir=output_dir,
            submission_path=submission_path,
            metrics_path=metrics_path,
        )

    def fake_run_kernel_fix(*args, **kwargs):  # noqa: ANN002, ARG001
        calls["kernel_fix"] += 1

    monkeypatch.setattr("kagglebot.autopilot.run_kernel_local", fake_run_kernel_local)
    monkeypatch.setattr("kagglebot.autopilot._run_kernel_fix", fake_run_kernel_fix)

    config = _make_config(tmp_path, submit=False, max_iterations=1)
    session = AutopilotSession(config=config, run_id="run-1", resume_run=False)
    with pytest.raises(KernelFailedError, match="Local kernel metrics missing expected score"):
        session.run()
    assert calls["kernel_fix"] == 1


def test_metric_mismatch_keeps_competition_metric_in_strict_mode(monkeypatch, tmp_path: Path) -> None:
    target_metric = "0.5 * mAP@[0.5:0.95] + 0.5 * F1-score"
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric=target_metric,
        target_score=0.9,
        target_direction="maximize",
    )
    calls = {"metric_fix": 0, "metric_recheck": 0}

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

    def fake_metric_only_fix(**kwargs):  # noqa: ANN003
        calls["metric_fix"] += 1

    def fake_metric_recheck(**kwargs):  # noqa: ANN003
        calls["metric_recheck"] += 1
        submission_path = kwargs["submission_path"]
        evaluation = EvaluationResult(
            score_source="cv",
            metric=target_metric,
            direction="maximize",
            value=0.62,
            std=0.01,
            train_score=None,
            val_score=0.62,
            fold_scores=[0.61, 0.63],
        )
        payload = {
            "score_source": "cv",
            "metric": target_metric,
            "direction": "maximize",
            "offline_value": 0.62,
            "offline_std": 0.01,
            "val_score": 0.62,
            "fold_scores": [0.61, 0.63],
        }
        return evaluation, payload, submission_path

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.autopilot._run_metric_only_competition_metric_fix", fake_metric_only_fix)
    monkeypatch.setattr("kagglebot.autopilot._rerun_kernel_for_metric_recheck", fake_metric_recheck)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path, submit=False, max_iterations=1)
    run_autopilot(config)

    persisted_plan = json.loads(config.paths.plan_path.read_text(encoding="utf-8"))
    assert persisted_plan["target_metric"] == target_metric
    assert persisted_plan["target_direction"] == "maximize"
    assert calls["metric_recheck"] == 1
    assert calls["metric_fix"] == 0

    iter_metrics = json.loads((config.paths.iter_dir("run-1", 1) / "metrics.json").read_text(encoding="utf-8"))
    assert iter_metrics["metric"] == target_metric
    guard = iter_metrics.get("quality_guard", {})
    reasons = guard.get("reasons", []) if isinstance(guard, dict) else []
    assert "competition_metric_mismatch" not in reasons


def test_metric_alias_equivalence_does_not_trigger_mismatch(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="brier_score",
        target_score=0.2,
        target_direction="minimize",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="holdout",
            metric="brier",
            direction="minimize",
            value=0.19,
            std=0.01,
            train_score=None,
            val_score=0.19,
            fold_scores=[0.18, 0.2],
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="vision",
            model_summary={},
            accelerator="cuda",
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path, submit=False, max_iterations=1)
    run_autopilot(config)

    iter_metrics = json.loads((config.paths.iter_dir("run-1", 1) / "metrics.json").read_text(encoding="utf-8"))
    guard = iter_metrics.get("quality_guard", {})
    reasons = guard.get("reasons", []) if isinstance(guard, dict) else []
    assert "competition_metric_mismatch" not in reasons


def test_competition_faithfulness_prefers_metric_name_over_numeric_metric_payload() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="rmse",
        direction="minimize",
        value=0.123,
        std=None,
        train_score=None,
        val_score=0.123,
        fold_scores=[0.123],
    )
    faithfulness = extract_competition_faithfulness(
        evaluation_metric=evaluation.metric,
        evaluation_score_source=evaluation.score_source,
        kernel_metrics_payload={
            "metric": 0.123,
            "metric_name": "standardized_rmse",
            "score_source": "cv",
            "split_strategy": "timeseries_split",
            "full_dataset_resolved": True,
            "competition_faithful": True,
        },
        evaluation_report_split_strategy=None,
        evaluation_contract={
            "expected_metric": "standardized_rmse",
            "expected_split_strategy": "timeseries_split",
            "accepted_score_sources": ["cv", "holdout"],
            "require_metric_match": True,
            "require_split_match": True,
            "require_trusted_score_source": True,
            "require_competition_faithful": True,
            "require_full_dataset": True,
        },
    )

    assert faithfulness["actual_metric"] == "standardized_rmse"
    assert faithfulness["metric_match"] is True
    assert "competition_metric_mismatch" not in faithfulness["reasons"]


def test_metric_mismatch_can_follow_kernel_metric_when_strict_mode_disabled(monkeypatch, tmp_path: Path) -> None:
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

    monkeypatch.setenv("KAGGLEBOT_STRICT_COMPETITION_METRIC", "0")
    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path, submit=False, max_iterations=1)
    run_autopilot(config)

    persisted_plan = json.loads(config.paths.plan_path.read_text(encoding="utf-8"))
    assert persisted_plan["target_metric"] == "composite"
    assert persisted_plan["target_direction"] == "maximize"


def test_metric_only_fix_reruns_local_kernel_to_materialize_metric_outputs(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="auc",
        target_score=0.9,
        target_direction="maximize",
    )
    monkeypatch.setenv("KAGGLEBOT_STRICT_COMPETITION_METRIC", "1")
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.plan_policy.needs_planning", lambda **kwargs: False)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)

    calls = {"kernel_runs": 0, "metric_fix": 0}

    def fake_metric_only_fix(**kwargs):  # noqa: ANN003
        calls["metric_fix"] += 1

    def fake_run_kernel_local(**kwargs):  # noqa: ANN003
        calls["kernel_runs"] += 1
        slug = kwargs["slug"]
        run_id = kwargs["run_id"]
        iteration = kwargs["iteration"]
        base_dir = Path(kwargs["base_dir"])
        output_dir = base_dir / slug / "runs" / run_id / f"iter-{iteration}" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        metric = "accuracy" if calls["kernel_runs"] == 1 else "auc"
        metrics_path.write_text(
            json.dumps(
                {
                    "score_source": "cv",
                    "metric": metric,
                    "direction": "maximize",
                    "offline_value": 0.9 if metric == "accuracy" else 0.91,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return KernelRunResult(
            kernel_id=f"local/{slug}",
            output_dir=output_dir,
            submission_path=submission_path,
            metrics_path=metrics_path,
        )

    monkeypatch.setattr("kagglebot.autopilot._run_metric_only_competition_metric_fix", fake_metric_only_fix)
    monkeypatch.setattr("kagglebot.autopilot.run_kernel_local", fake_run_kernel_local)

    config = _make_config(tmp_path, submit=False, max_iterations=1)
    run_autopilot(config)

    assert calls["metric_fix"] == 1
    assert calls["kernel_runs"] == 2
    iter_metrics = json.loads((config.paths.iter_dir("run-1", 1) / "metrics.json").read_text(encoding="utf-8"))
    assert iter_metrics["metric"] == "auc"


def test_metric_recheck_uses_existing_artifacts_without_retraining(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _rerun_kernel_for_metric_recheck

    config = _make_config(tmp_path, compute="local_gpu", accelerator="gpu")
    iter_dir = config.paths.iter_dir("run-1", 1)
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "score_source": "cv",
                "metric": "rmse",
                "direction": "minimize",
                "offline_value": 0.321,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "kagglebot.autopilot.run_kernel",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_kernel must not be called")),
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.run_kernel_local",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_kernel_local must not be called")),
    )

    evaluation, payload, resolved_submission = _rerun_kernel_for_metric_recheck(
        config=config,
        run_id="run-1",
        iteration=1,
        submission_path=submission_path,
        iter_dir=iter_dir,
        metrics_artifact_path=metrics_path,
        kernel_name=None,
        enable_internet=False,
        score_source="cv",
        target_metric="rmse",
        metric_direction="minimize",
        holdout_frac=0.2,
        cv_folds=5,
        seed=42,
        time_budget_min=10,
    )

    assert evaluation.metric == "rmse"
    assert evaluation.value == pytest.approx(0.321)
    assert isinstance(payload, dict)
    assert payload.get("metric") == "rmse"
    assert resolved_submission.exists()


def test_metric_recheck_prefers_output_metrics_over_stale_iteration_metrics(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _rerun_kernel_for_metric_recheck

    config = _make_config(tmp_path, compute="local_gpu", accelerator="gpu")
    iter_dir = config.paths.iter_dir("run-1", 1)
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    stale_iter_metrics_path = iter_dir / "metrics.json"
    stale_iter_metrics_path.write_text(
        json.dumps(
            {
                "score_source": "cv",
                "metric": "accuracy",
                "direction": "maximize",
                "offline_value": 0.5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_metrics_path = output_dir / "metrics.json"
    output_metrics_path.write_text(
        json.dumps(
            {
                "score_source": "cv",
                "metric": "auc",
                "direction": "maximize",
                "offline_value": 0.92,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "kagglebot.autopilot.run_kernel",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_kernel must not be called")),
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.run_kernel_local",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_kernel_local must not be called")),
    )

    evaluation, payload, resolved_submission = _rerun_kernel_for_metric_recheck(
        config=config,
        run_id="run-1",
        iteration=1,
        submission_path=submission_path,
        iter_dir=iter_dir,
        metrics_artifact_path=stale_iter_metrics_path,
        kernel_name=None,
        enable_internet=False,
        score_source="cv",
        target_metric="auc",
        metric_direction="maximize",
        holdout_frac=0.2,
        cv_folds=5,
        seed=42,
        time_budget_min=10,
    )

    assert evaluation.metric == "auc"
    assert evaluation.value == pytest.approx(0.92)
    assert isinstance(payload, dict)
    assert payload.get("metric") == "auc"
    assert resolved_submission.exists()


def test_metric_recheck_ignores_submit_only_output_metrics(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _rerun_kernel_for_metric_recheck

    config = _make_config(tmp_path, compute="local_gpu", accelerator="gpu")
    iter_dir = config.paths.iter_dir("run-1", 1)
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")

    iter_metrics_path = iter_dir / "metrics.json"
    iter_metrics_path.write_text(
        json.dumps(
            {
                "score_source": "cv",
                "metric": "rmse",
                "direction": "minimize",
                "offline_value": 0.321,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_metrics_path = output_dir / "metrics.json"
    output_metrics_path.write_text(
        json.dumps({"schema_version": 1, "kind": "submit_only"}, indent=2),
        encoding="utf-8",
    )
    output_mtime = iter_metrics_path.stat().st_mtime + 10
    os.utime(output_metrics_path, (output_mtime, output_mtime))

    monkeypatch.setattr(
        "kagglebot.autopilot.run_kernel",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_kernel must not be called")),
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.run_kernel_local",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_kernel_local must not be called")),
    )

    evaluation, payload, resolved_submission = _rerun_kernel_for_metric_recheck(
        config=config,
        run_id="run-1",
        iteration=1,
        submission_path=submission_path,
        iter_dir=iter_dir,
        metrics_artifact_path=iter_metrics_path,
        kernel_name=None,
        enable_internet=False,
        score_source="cv",
        target_metric="rmse",
        metric_direction="minimize",
        holdout_frac=0.2,
        cv_folds=5,
        seed=42,
        time_budget_min=10,
    )

    assert evaluation.metric == "rmse"
    assert evaluation.value == pytest.approx(0.321)
    assert isinstance(payload, dict)
    assert payload.get("kind") != "submit_only"
    assert resolved_submission.exists()


def test_metric_recheck_recomputes_target_metric_from_oof_without_retraining(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _rerun_kernel_for_metric_recheck

    config = _make_config(tmp_path, compute="local_gpu", accelerator="gpu")
    iter_dir = config.paths.iter_dir("run-1", 1)
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "score_source": "cv",
                "metric": "accuracy",
                "direction": "maximize",
                "offline_value": 0.5,
                "offline_std": 0.01,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    oof_path = output_dir / "oof_predictions.csv"
    oof_path.write_text(
        "\n".join(
            [
                "row_id,y,oof_pred,oof_proba,fold",
                "0,0,0,0.01,1",
                "1,0,0,0.10,1",
                "2,1,1,0.90,2",
                "3,1,1,0.99,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "kagglebot.autopilot.run_kernel",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_kernel must not be called")),
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.run_kernel_local",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_kernel_local must not be called")),
    )

    evaluation, payload, resolved_submission = _rerun_kernel_for_metric_recheck(
        config=config,
        run_id="run-1",
        iteration=1,
        submission_path=submission_path,
        iter_dir=iter_dir,
        metrics_artifact_path=metrics_path,
        kernel_name=None,
        enable_internet=False,
        score_source="cv",
        target_metric="auc",
        metric_direction="maximize",
        holdout_frac=0.2,
        cv_folds=5,
        seed=42,
        time_budget_min=10,
    )

    assert evaluation.metric == "auc"
    assert evaluation.value == pytest.approx(1.0)
    assert isinstance(payload, dict)
    assert payload.get("metric") == "auc"
    assert payload.get("metric_recheck_without_retrain") is True
    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert persisted["metric"] == "auc"
    assert persisted["offline_value"] == pytest.approx(1.0)
    assert resolved_submission.exists()


def test_metric_recheck_resolves_oof_from_staged_local_kernel_outputs(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _rerun_kernel_for_metric_recheck

    config = _make_config(tmp_path, compute="local_gpu", accelerator="gpu")
    iter_dir = config.paths.iter_dir("run-1", 1)
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "score_source": "cv",
                "metric": "accuracy",
                "direction": "maximize",
                "offline_value": 0.5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    staged_oof_path = tmp_path / "artifacts" / "demo" / "kernels" / "run-1" / "local-iter-1" / "outputs"
    staged_oof_path.mkdir(parents=True, exist_ok=True)
    (staged_oof_path / "oof_predictions.csv").write_text(
        "\n".join(
            [
                "row_id,y,oof_pred,oof_proba,fold",
                "0,0,0,0.01,1",
                "1,0,0,0.10,1",
                "2,1,1,0.90,2",
                "3,1,1,0.99,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "kagglebot.autopilot.run_kernel",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_kernel must not be called")),
    )
    monkeypatch.setattr(
        "kagglebot.autopilot.run_kernel_local",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_kernel_local must not be called")),
    )

    evaluation, payload, resolved_submission = _rerun_kernel_for_metric_recheck(
        config=config,
        run_id="run-1",
        iteration=1,
        submission_path=submission_path,
        iter_dir=iter_dir,
        metrics_artifact_path=metrics_path,
        kernel_name=None,
        enable_internet=False,
        score_source="cv",
        target_metric="auc",
        metric_direction="maximize",
        holdout_frac=0.2,
        cv_folds=5,
        seed=42,
        time_budget_min=10,
    )

    assert evaluation.metric == "auc"
    assert evaluation.value == pytest.approx(1.0)
    assert isinstance(payload, dict)
    assert payload.get("metric") == "auc"
    assert payload.get("metric_recheck_without_retrain") is True
    assert resolved_submission.exists()


def test_local_kernel_oof_artifact_is_synced_for_metric_recheck(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="auc",
        target_score=0.6,
        target_direction="maximize",
    )
    calls = {"metric_fix": 0}

    def fake_run_kernel_local(**kwargs):  # noqa: ANN003
        slug = kwargs["slug"]
        run_id = kwargs["run_id"]
        iteration = kwargs["iteration"]
        base_dir = Path(kwargs["base_dir"])
        output_dir = base_dir / slug / "kernels" / run_id / f"local-iter-{iteration}" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / "submission.csv"
        submission_path.write_text("id,target\n1,0.1\n2,0.9\n", encoding="utf-8")
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "score_source": "cv",
                    "metric": "accuracy",
                    "direction": "maximize",
                    "offline_value": 0.5,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "oof_predictions.csv").write_text(
            "\n".join(
                [
                    "row_id,y,oof_pred,oof_proba,fold",
                    "0,0,0,0.01,1",
                    "1,0,0,0.10,1",
                    "2,1,1,0.90,2",
                    "3,1,1,0.99,2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return KernelRunResult(
            kernel_id=f"local/{slug}",
            output_dir=output_dir,
            submission_path=submission_path,
            metrics_path=metrics_path,
        )

    def fake_metric_only_fix(**kwargs):  # noqa: ANN003
        calls["metric_fix"] += 1

    monkeypatch.setattr("kagglebot.autopilot.run_kernel_local", fake_run_kernel_local)
    monkeypatch.setattr("kagglebot.autopilot._run_metric_only_competition_metric_fix", fake_metric_only_fix)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path, submit=False, max_iterations=1, compute="local_gpu", accelerator="gpu")
    run_autopilot(config)

    iter_output_dir = config.paths.iter_dir("run-1", 1) / "output"
    assert (iter_output_dir / "oof_predictions.csv").exists()
    iter_metrics = json.loads((config.paths.iter_dir("run-1", 1) / "metrics.json").read_text(encoding="utf-8"))
    assert iter_metrics["metric"] == "auc"
    assert calls["metric_fix"] == 0


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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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
    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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

    monkeypatch.setattr(
        "kagglebot.knowledge_context.ensure_taxonomy", lambda *args, **kwargs: {"tags": [], "aliases": {}}
    )
    monkeypatch.setattr("kagglebot.knowledge_context.resolve_similar_improvements", fake_resolve_similar_improvements)
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.1})
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.autopilot.run_strategy", fake_run_strategy)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, max_iterations=2)
    run_autopilot(config)
    iter_dir = config.paths.iter_dir(config.run_id or "run-1", 1)
    assert (iter_dir / "agent" / "prompt.md").exists()
    assert any(kwargs.get("model") == "gpt-5.5" for kwargs in codex_kwargs_seen)
    assert any(kwargs.get("reasoning_effort") == "xhigh" for kwargs in codex_kwargs_seen)


def test_run_improvement_allows_context_and_run_artifacts(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _run_improvement

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    config = _make_config(tmp_path, max_iterations=2)
    run_id = config.run_id or "run-1"
    iteration = 1
    iter_dir = config.paths.iter_dir(run_id, iteration)
    pending_problem_insights: list[dict[str, object]] = []

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        config.paths.context_dir.mkdir(parents=True, exist_ok=True)
        (config.paths.context_dir / "knowledge_hints.txt").write_text("updated hints\n", encoding="utf-8")

        run_dir = config.paths.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text('{"status":"updated"}\n', encoding="utf-8")
        (run_dir / "evaluation_report.json").write_text('{"ok":true}\n', encoding="utf-8")

        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("improved features\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.agent_strategy.run_improvement_strategy_prompt", lambda **kwargs: "")
    monkeypatch.setattr("kagglebot.knowledge_context.load_problem_type_knowledge_text", lambda *args, **kwargs: "")
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.record_improvement", lambda *args, **kwargs: None)

    evaluation = EvaluationResult(
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        value=0.45,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )
    _run_improvement(
        config=config,
        run_id=run_id,
        iteration=iteration,
        iter_dir=iter_dir,
        evaluation=evaluation,
        top1_info={"score": 0.12, "source": "leaderboard"},
        target_score=0.40,
        delta_offline=0.05,
        pending_problem_insights=pending_problem_insights,
    )

    assert (config.paths.context_dir / "knowledge_hints.txt").exists()
    assert (config.paths.run_dir(run_id) / "run.json").exists()
    assert (config.paths.run_dir(run_id) / "evaluation_report.json").exists()
    assert len(pending_problem_insights) == 1


def test_run_improvement_retries_transient_agent_capacity(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _run_improvement

    class DummyResult:
        def __init__(
            self,
            path: Path,
            *,
            returncode: int,
            stdout: str = "",
            stderr: str = "",
        ) -> None:
            self.returncode = returncode
            self.last_message_path = path
            self.stdout = stdout
            self.stderr = stderr

    config = _make_config(tmp_path, max_iterations=2)
    run_id = config.run_id or "run-1"
    iteration = 1
    iter_dir = config.paths.iter_dir(run_id, iteration)
    calls = {"codex": 0}
    output_dirs: list[Path] = []

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        calls["codex"] += 1
        output_dirs.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "codex_last_message.txt"
        if calls["codex"] == 1:
            last_msg.write_text("stale-looking success text\n", encoding="utf-8")
            return DummyResult(
                last_msg,
                returncode=1,
                stdout='{"type":"error","message":"Selected model is at capacity. Please try a different model."}\n',
            )
        last_msg.write_text("improved after retry\n", encoding="utf-8")
        return DummyResult(last_msg, returncode=0)

    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.agent_strategy.run_improvement_strategy_prompt", lambda **kwargs: "")
    monkeypatch.setattr("kagglebot.knowledge_context.load_problem_type_knowledge_text", lambda *args, **kwargs: "")
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.record_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.AGENT_CAPACITY_RETRY_SLEEP", 0.0)

    pending_problem_insights: list[dict[str, object]] = []
    evaluation = EvaluationResult(
        score_source="cv",
        metric="accuracy",
        direction="maximize",
        value=0.70,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )
    _run_improvement(
        config=config,
        run_id=run_id,
        iteration=iteration,
        iter_dir=iter_dir,
        evaluation=evaluation,
        top1_info={"score": 0.78, "source": "leaderboard"},
        target_score=0.78,
        delta_offline=-0.01,
        pending_problem_insights=pending_problem_insights,
    )

    assert calls["codex"] == 2
    assert output_dirs[0] == iter_dir / "agent"
    assert output_dirs[1].parent == iter_dir / "agent"
    assert output_dirs[1].name.startswith("improve_capacity_retry")


def test_agent_capacity_failure_detection_and_detail() -> None:
    class DummyResult:
        returncode = 1
        stdout = '{"type":"error","message":"Selected model is at capacity. Please try a different model."}'
        stderr = "stderr text"

    assert is_agent_capacity_failure(DummyResult(), "stale success text")
    detail = agent_failure_detail(DummyResult(), "stale success text")
    assert "returncode=1" in detail
    assert "stderr=stderr text" in detail
    assert "response=stale success text" in detail
    assert "transcript_tail=" in detail


def test_extract_iteration_policy_signals_detect_orig_proba_and_pseudo_label_failure() -> None:
    orig_signal = _extract_orig_proba_signal(
        {
            "original_data_found": False,
            "orig_proba_feature_status": "constant_fallback",
            "orig_proba_constant_cols": ["ORIG_proba_a", "ORIG_proba_b"],
        }
    )
    assert orig_signal is not None
    assert len(orig_signal["constant_cols"]) == 2
    assert "context/reference_inputs_manifest.json" in str(orig_signal["note"])

    pseudo_signal = _extract_pseudo_label_failure_signal(
        kernel_metrics_payload={"pseudo_label": {"accepted_folds": 0, "total_folds": 5}},
        diagnostics_text="Pseudo-label result: 0/5 accepted folds.",
    )
    assert pseudo_signal is not None
    assert pseudo_signal["accepted"] == 0
    assert pseudo_signal["total"] == 5

    mismatch = _detect_online_mismatch_signal(
        previous_best_offline=0.91,
        current_offline=0.92,
        previous_best_online=0.905,
        current_online=0.901,
        direction="maximize",
    )
    assert mismatch is not None
    assert "public leaderboard regressed" in str(mismatch["note"]).lower()

    missing_ensemble = _extract_missing_ensemble_signal(
        {
            "model_families": ["xgboost", "catboost"],
            "blend_method": "single",
            "component_models": ["xgb_only"],
        }
    )
    assert missing_ensemble is not None
    assert "weighted or rank OOF blend" in str(missing_ensemble["note"])

    original_data_unused = _extract_original_data_unused_signal(
        kernel_metrics_payload={"original_data_found": False, "external_data_used": False},
        reference_inputs_manifest_payload={
            "required_datasets": ["alice/original-data"],
            "reference_notebooks": [{"staged_sources": [{"kind": "dataset", "ref": "alice/original-data"}]}],
        },
    )
    assert original_data_unused is not None
    assert "staged but the kernel did not use them" in str(original_data_unused["note"])

    same_family = _extract_same_family_plateau_signal(
        {
            "model_families": ["xgboost"],
            "pipelines": [{"name": "xgb_a"}, {"name": "xgb_b"}],
            "selected_pipeline": "xgb_b",
        }
    )
    assert same_family is not None
    assert "same-family plateau" in str(same_family["note"])


def test_online_regression_signal_uses_historical_submission_baseline() -> None:
    history = {
        "direction": "minimize",
        "best_score": 9.600,
        "best": {"description": "kb previous i=5 offline=10.4211", "score": 9.600},
        "recent": [{"submitted_at": "2026-05-22T09:24:24+00:00", "score": 10.308}],
    }

    signal = detect_online_regression_vs_submission_history(
        previous_best_online=10.271,
        current_online=10.308,
        direction="minimize",
        history=history,
    )

    assert signal is not None
    assert signal["previous_best_online"] == pytest.approx(9.600)
    assert "historical_best=9.600000" in str(signal["note"])
    assert "materially different" in str(signal["note"])

    prompt = format_previous_submission_history_for_prompt(history)
    assert "Best historical public score: 9.600000" in prompt
    assert "Do not call a new iteration improved" in prompt


def test_run_improvement_appends_code_reference_gate_when_underperforming(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _run_improvement

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    config = _make_config(tmp_path, max_iterations=2)
    run_id = config.run_id or "run-1"
    iteration = 1
    iter_dir = config.paths.iter_dir(run_id, iteration)
    kernel_path = config.paths.kernel_source_dir / "kernel.py"

    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.code_md_path.write_text(
        (
            "# Code Notebook Snapshot\n\n"
            "## Required Reference Notebook (Execution baseline)\n"
            "- title: [Stock Pledge 2026] 0.741 FE + TabICL KFold\n"
        ),
        encoding="utf-8",
    )
    config.paths.code_notebooks_index_path.write_text(
        json.dumps(
            {
                "required_reference_kernel_id": "alice/ref-kernel",
                "notebooks": [
                    {"kernel_id": "alice/ref-kernel", "score": 0.741, "title": "ref"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        kernel_path.write_text(
            "\n".join(
                [
                    "# KAGGLEBOT_CODE_REFERENCE_IMPLEMENTED: alice/ref-kernel",
                    "import tabicl",
                    "print('ok')",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("improved features\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.agent_strategy.run_improvement_strategy_prompt", lambda **kwargs: "")
    monkeypatch.setattr("kagglebot.knowledge_context.load_problem_type_knowledge_text", lambda *args, **kwargs: "")
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.record_improvement", lambda *args, **kwargs: None)

    pending_problem_insights: list[dict[str, object]] = []
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.62,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )
    _run_improvement(
        config=config,
        run_id=run_id,
        iteration=iteration,
        iter_dir=iter_dir,
        evaluation=evaluation,
        top1_info={"score": 0.78, "source": "leaderboard"},
        target_score=0.78,
        delta_offline=-0.02,
        pending_problem_insights=pending_problem_insights,
    )

    prompt_text = (iter_dir / "agent" / "prompt.md").read_text(encoding="utf-8")
    assert "## Code Reference Gate" in prompt_text
    assert str(config.paths.code_md_path) in prompt_text
    assert str(config.paths.code_notebooks_index_path) in prompt_text
    assert "underperforming_code_reference" in prompt_text
    assert "Required Reference Notebook (Execution baseline)" in prompt_text


def test_run_improvement_appends_additional_policy_notes(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _run_improvement

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    config = _make_config(tmp_path, max_iterations=2)
    run_id = config.run_id or "run-1"
    iteration = 1
    iter_dir = config.paths.iter_dir(run_id, iteration)
    kernel_path = config.paths.kernel_source_dir / "kernel.py"
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        kernel_path.write_text("print('ok')\n", encoding="utf-8")
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("improved\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.agent_strategy.run_improvement_strategy_prompt", lambda **kwargs: "")
    monkeypatch.setattr("kagglebot.knowledge_context.load_problem_type_knowledge_text", lambda *args, **kwargs: "")
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.record_improvement", lambda *args, **kwargs: None)

    pending_problem_insights: list[dict[str, object]] = []
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.62,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )
    _run_improvement(
        config=config,
        run_id=run_id,
        iteration=iteration,
        iter_dir=iter_dir,
        evaluation=evaluation,
        top1_info={"score": 0.78, "source": "leaderboard"},
        target_score=0.78,
        delta_offline=-0.02,
        pending_problem_insights=pending_problem_insights,
        extra_policy_notes=[
            "Pseudo-labeling yielded 0/5 accepted folds. Disable pseudo-labeling next iteration.",
            "Recover ORIG_proba inputs from context/reference_inputs_manifest.json.",
        ],
    )

    prompt_text = (iter_dir / "agent" / "prompt.md").read_text(encoding="utf-8")
    assert "Additional repair targets:" in prompt_text
    assert "Disable pseudo-labeling next iteration" in prompt_text
    assert "context/reference_inputs_manifest.json" in prompt_text


def test_run_improvement_appends_competition_policy_override(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _run_improvement

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    config = _make_config(tmp_path, max_iterations=2)
    run_id = config.run_id or "run-1"
    iteration = 1
    iter_dir = config.paths.iter_dir(run_id, iteration)
    kernel_path = config.paths.kernel_source_dir / "kernel.py"
    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.competition_policy_path.write_text(
        json.dumps(
            {
                "required_capabilities": [
                    "recoverable_original_dataset",
                    "heterogeneous_tabular_ensemble",
                    "requires_oof_blend",
                    "text_translation_seq2seq",
                    "requires_grouped_text_cv",
                    "requires_candidate_rerank",
                    "supports_metadata_supervision",
                    "supports_soft_constraint_rewrite",
                ],
                "execution_hints": {
                    "tabular_original_data_usage": "prefer_recovered_public_original_dataset_when_staged",
                },
                "prompt": {
                    "ablation_groups": ["comp_only", "comp_plus_orig"],
                    "min_model_families_before_stop": 3,
                    "require_oof_blend_before_stop": True,
                    "prefer_ensemble_reference": True,
                },
                "evaluation": {"search_stop_rank_percentile": 0.08},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config.paths.code_notebooks_index_path.write_text(
        json.dumps(
            {
                "required_reference_kernel_id": "alice/ref-kernel",
                "ensemble_reference_kernel_id": "alice/blend-kernel",
                "notebooks": [
                    {"kernel_id": "alice/ref-kernel", "title": "Required ref"},
                    {"kernel_id": "alice/blend-kernel", "title": "Blend ref"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        kernel_path.write_text("print('ok')\n", encoding="utf-8")
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("improved\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.agent_strategy.run_improvement_strategy_prompt", lambda **kwargs: "")
    monkeypatch.setattr("kagglebot.knowledge_context.load_problem_type_knowledge_text", lambda *args, **kwargs: "")
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.record_improvement", lambda *args, **kwargs: None)

    pending_problem_insights: list[dict[str, object]] = []
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.62,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )
    _run_improvement(
        config=config,
        run_id=run_id,
        iteration=iteration,
        iter_dir=iter_dir,
        evaluation=evaluation,
        top1_info={"score": 0.78, "source": "leaderboard"},
        target_score=0.78,
        delta_offline=-0.02,
        pending_problem_insights=pending_problem_insights,
    )

    prompt_text = (iter_dir / "agent" / "prompt.md").read_text(encoding="utf-8")
    assert "Competition policy override is active." in prompt_text
    assert (
        "Required capabilities: recoverable_original_dataset, heterogeneous_tabular_ensemble, requires_oof_blend"
        in prompt_text
    )
    assert "Minimum model families before stop: 3" in prompt_text
    assert "wire them into training or feature generation" in prompt_text
    assert "weighted or rank blend artifact" in prompt_text
    assert "src/kagglebot/kernel_runtime/text_translation.py" in prompt_text
    assert "grouped text CV keyed by the plan/runtime group columns" in prompt_text
    assert "retrieval as a candidate source or fallback only" in prompt_text
    assert "text_runtime.required_aux_inputs" in prompt_text
    assert "soft constraint rewrites and rerank bonuses" in prompt_text
    assert "execution_hints:" in prompt_text
    assert "ensemble_kernel_id: alice/blend-kernel" in prompt_text


def test_run_improvement_retries_when_code_reference_impl_is_missing(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _run_improvement

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    config = _make_config(tmp_path, max_iterations=2)
    run_id = config.run_id or "run-1"
    iteration = 1
    iter_dir = config.paths.iter_dir(run_id, iteration)
    kernel_path = config.paths.kernel_source_dir / "kernel.py"

    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.code_notebooks_index_path.write_text(
        json.dumps(
            {
                "required_reference_kernel_id": "alice/ref-kernel",
                "notebooks": [
                    {
                        "kernel_id": "alice/ref-kernel",
                        "title": "TabICL reference",
                        "summary": "TabICL baseline notebook",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    codex_calls = {"count": 0}

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        codex_calls["count"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        if codex_calls["count"] == 2:
            kernel_path.write_text(
                "\n".join(
                    [
                        "# KAGGLEBOT_CODE_REFERENCE_IMPLEMENTED: alice/ref-kernel",
                        "import tabicl",
                        "print('reference path implemented')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        last_msg = output_dir / f"codex_last_message_{codex_calls['count']}.txt"
        last_msg.write_text("improved\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.agent_strategy.run_improvement_strategy_prompt", lambda **kwargs: "")
    monkeypatch.setattr("kagglebot.knowledge_context.load_problem_type_knowledge_text", lambda *args, **kwargs: "")
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.record_improvement", lambda *args, **kwargs: None)

    pending_problem_insights: list[dict[str, object]] = []
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.70,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )
    _run_improvement(
        config=config,
        run_id=run_id,
        iteration=iteration,
        iter_dir=iter_dir,
        evaluation=evaluation,
        top1_info={"score": 0.78, "source": "leaderboard"},
        target_score=0.78,
        delta_offline=-0.01,
        pending_problem_insights=pending_problem_insights,
        enforce_code_reference_implementation=True,
        code_reference_enforcement_reason="initial run under /code baseline",
    )

    assert codex_calls["count"] == 2
    kernel_text = kernel_path.read_text(encoding="utf-8")
    assert "KAGGLEBOT_CODE_REFERENCE_IMPLEMENTED: alice/ref-kernel" in kernel_text
    assert "tabicl" in kernel_text.lower()


def test_run_improvement_code_reference_repair_allows_src_edits(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _run_improvement

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    config = _make_config(tmp_path, max_iterations=2)
    run_id = config.run_id or "run-1"
    iteration = 1
    iter_dir = config.paths.iter_dir(run_id, iteration)
    kernel_path = config.paths.kernel_source_dir / "kernel.py"
    support_path = config.paths.repo_root / "src" / "support_fix.py"

    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.code_notebooks_index_path.write_text(
        json.dumps(
            {
                "required_reference_kernel_id": "alice/ref-kernel",
                "notebooks": [
                    {
                        "kernel_id": "alice/ref-kernel",
                        "title": "Reference kernel",
                        "summary": "starter baseline",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    codex_calls = {"count": 0}

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        codex_calls["count"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        if codex_calls["count"] == 2:
            support_path.parent.mkdir(parents=True, exist_ok=True)
            support_path.write_text("REPAIRED = True\n", encoding="utf-8")
            kernel_path.write_text(
                "# KAGGLEBOT_CODE_REFERENCE_IMPLEMENTED: alice/ref-kernel\nprint('ok')\n",
                encoding="utf-8",
            )
        last_msg = output_dir / f"codex_last_message_{codex_calls['count']}.txt"
        last_msg.write_text("improved\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.agent_strategy.run_improvement_strategy_prompt", lambda **kwargs: "")
    monkeypatch.setattr("kagglebot.knowledge_context.load_problem_type_knowledge_text", lambda *args, **kwargs: "")
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.record_improvement", lambda *args, **kwargs: None)

    pending_problem_insights: list[dict[str, object]] = []
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.70,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )
    _run_improvement(
        config=config,
        run_id=run_id,
        iteration=iteration,
        iter_dir=iter_dir,
        evaluation=evaluation,
        top1_info={"score": 0.78, "source": "leaderboard"},
        target_score=0.78,
        delta_offline=-0.01,
        pending_problem_insights=pending_problem_insights,
        enforce_code_reference_implementation=True,
        code_reference_enforcement_reason="repair needs shared helper edits",
    )

    assert codex_calls["count"] == 2
    assert support_path.exists()
    assert "KAGGLEBOT_CODE_REFERENCE_IMPLEMENTED: alice/ref-kernel" in kernel_path.read_text(encoding="utf-8")


def test_autopilot_runs_agent_pipeline(monkeypatch, tmp_path: Path) -> None:
    called = {"run": False}

    def fake_pipeline(*args, **kwargs):  # noqa: ARG001
        called["run"] = True

    monkeypatch.setattr("kagglebot.autopilot.run_agent_pipeline", fake_pipeline)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)

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
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})
    monkeypatch.setattr("kagglebot.autopilot.resolve_kaggle_username", lambda *args, **kwargs: "user")
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.kernel_runner.ensure_kernel_sources_valid", lambda *args, **kwargs: None)

    config = _make_config(tmp_path, compute="kaggle_gpu", accelerator="gpu", max_iterations=1)
    run_autopilot(config)

    assert calls["run_kernel"] == 2
    assert calls["kernel_fix"] == 1


def test_autopilot_preflight_fixes_kernel_sources_before_local_run(monkeypatch, tmp_path: Path) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        score_source="cv",
        cv_folds=3,
        seed=42,
    )
    config = _make_config(tmp_path, compute="local_gpu", accelerator="gpu", max_iterations=1, submit=False)
    kernel_path = config.paths.kernel_source_dir / "kernel.py"
    kernel_path.write_text("print('missing contract')\n", encoding="utf-8")

    calls = {"kernel_fix": 0, "run_kernel_local": 0}

    def fake_run_kernel_fix(**kwargs):  # noqa: ANN003
        calls["kernel_fix"] += 1
        kernel_path.write_text(
            "print('fixed')\n# submission.csv\n# metrics.json\n",
            encoding="utf-8",
        )

    def fake_run_kernel_local(**kwargs):  # noqa: ANN003
        calls["run_kernel_local"] += 1
        output_dir = (
            kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{kwargs['iteration']}" / "output"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        metrics_path.write_text(
            json.dumps(
                {
                    "score_source": "cv",
                    "metric": "rmse",
                    "direction": "minimize",
                    "offline_value": 0.4,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return KernelRunResult(
            kernel_id=f"local/{kwargs['slug']}",
            output_dir=output_dir,
            submission_path=submission_path,
            metrics_path=metrics_path,
        )

    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})
    monkeypatch.setattr("kagglebot.autopilot._run_kernel_fix", fake_run_kernel_fix)
    monkeypatch.setattr("kagglebot.autopilot.run_kernel_local", fake_run_kernel_local)

    run_autopilot(config)

    assert calls["kernel_fix"] == 1
    assert calls["run_kernel_local"] == 1


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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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


def test_autopilot_skips_fallback_submit_after_untrusted_final_iteration(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        max_iterations=2,
    )

    def fake_run_kernel_local(**kwargs):  # noqa: ANN003
        iteration = int(kwargs["iteration"])
        output_dir = kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{iteration}" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        score_source = "cv" if iteration == 1 else "sample_smoke"
        metrics_path.write_text(
            json.dumps(
                {
                    "score_source": score_source,
                    "metric": "rmse",
                    "direction": "minimize",
                    "offline_value": 0.4 if iteration == 1 else 0.2,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return KernelRunResult(
            kernel_id=f"local/{kwargs['slug']}",
            output_dir=output_dir,
            submission_path=submission_path,
            metrics_path=metrics_path,
        )

    submit_calls = {"count": 0}

    def fake_attempt_submit(
        *, config, run_id, submission_path, best_score, problem_types, submit_mode, notebook_submit_artifact_mode
    ):  # noqa: ARG001
        submit_calls["count"] += 1
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-03-07T00:00:00+00:00",
            "iteration": 1,
            "outcome": {"status": "complete", "score": 0.4},
        }

    monkeypatch.setattr("kagglebot.autopilot.run_kernel_local", fake_run_kernel_local)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.submission_policy.should_force_initial_submit", lambda **kwargs: False)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})
    monkeypatch.setattr(
        "kagglebot.competition_rules.load_competition_rule_constraints",
        lambda *args, **kwargs: type(
            "Constraints",
            (),
            {
                "notebook_submissions_only": False,
                "internet_must_be_off": False,
                "submission_limit_detected": True,
                "submission_limit_per_day": 1,
                "cpu_runtime_limit_min": None,
                "gpu_runtime_limit_min": None,
            },
        )(),
    )

    config = _make_config(tmp_path, submit=True, max_iterations=2)
    run_autopilot(config)

    captured = capsys.readouterr().out
    run_payload = json.loads((config.paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    assert submit_calls["count"] == 0
    assert run_payload["status"] == "completed"
    assert "iter 1/2 not attempted yet" in captured
    assert "iter 2/2 not attempted yet" in captured
    assert "refusing fallback submit" in captured


def test_autopilot_skips_fallback_submit_when_higher_potential_candidate_exists(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
        max_iterations=2,
    )

    def fake_run_kernel_local(**kwargs):  # noqa: ANN003
        iteration = int(kwargs["iteration"])
        output_dir = kwargs["base_dir"] / kwargs["slug"] / "runs" / kwargs["run_id"] / f"iter-{iteration}" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        if iteration == 1:
            payload = {
                "score_source": "cv",
                "metric": "rmse",
                "direction": "minimize",
                "offline_value": 0.3,
                "selected_pipeline": "graph_transformer_hybrid",
                "competition_faithful": False,
                "full_dataset_resolved": False,
            }
        else:
            payload = {
                "score_source": "cv",
                "metric": "rmse",
                "direction": "minimize",
                "offline_value": 0.35,
                "selected_pipeline": "ridge_baseline",
            }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return KernelRunResult(
            kernel_id=f"local/{kwargs['slug']}",
            output_dir=output_dir,
            submission_path=submission_path,
            metrics_path=metrics_path,
        )

    submit_calls = {"count": 0}

    def fake_attempt_submit(
        *, config, run_id, submission_path, best_score, problem_types, submit_mode, notebook_submit_artifact_mode
    ):  # noqa: ARG001
        submit_calls["count"] += 1
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-03-07T00:00:00+00:00",
            "iteration": 2,
            "outcome": {"status": "complete", "score": 0.35},
        }

    monkeypatch.setattr("kagglebot.autopilot.run_kernel_local", fake_run_kernel_local)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.submission_policy.should_force_initial_submit", lambda **kwargs: False)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})
    monkeypatch.setattr(
        "kagglebot.competition_rules.load_competition_rule_constraints",
        lambda *args, **kwargs: type(
            "Constraints",
            (),
            {
                "notebook_submissions_only": False,
                "internet_must_be_off": False,
                "submission_limit_detected": True,
                "submission_limit_per_day": 1,
                "cpu_runtime_limit_min": None,
                "gpu_runtime_limit_min": None,
            },
        )(),
    )

    config = _make_config(tmp_path, submit=True, max_iterations=2)
    run_autopilot(config)

    captured = capsys.readouterr().out
    run_payload = json.loads((config.paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    summary = run_payload.get("summary")
    assert submit_calls["count"] == 0
    assert run_payload["status"] == "completed"
    assert isinstance(summary, dict)
    assert summary.get("fallback_submit_blocked_reason") == "higher_potential_unsubmitted_candidate_exists"
    assert "higher_potential_unsubmitted_candidate_exists" in captured


def test_autopilot_top1_stop_requires_submission_score(monkeypatch, tmp_path: Path) -> None:
    calls = {"submit": 0}
    submission_scores = [0.6, 0.4]
    train_calls = {"count": 0}

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="rmse",
        target_score=0.5,
        target_direction="minimize",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        train_calls["count"] += 1
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
        source = str(kwargs.get("score_source") or "holdout")
        if source == "holdout":
            value = 0.4 - 0.01 * float(train_calls["count"] - 1)
        else:
            value = 0.41 - 0.01 * float(train_calls["count"] - 1)
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

    def fake_attempt_submit(
        *, config, run_id, submission_path, best_score, problem_types, submit_mode, notebook_submit_artifact_mode
    ):  # noqa: ARG001
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.5})

    config = _make_config(tmp_path, submit=True, max_iterations=3)
    run_autopilot(config)

    run_payload = json.loads((config.paths.run_dir("run-1") / "run.json").read_text(encoding="utf-8"))
    assert calls["submit"] == 2
    assert run_payload.get("stop_reason") == "submission_rank_1"
    assert (config.paths.iter_dir("run-1", 3) / "metrics.json").exists() is False


def test_submission_phase_attempt_passes_current_submit_contract(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_attempt_submit(
        *,
        config,
        run_id,
        submission_path,
        best_score,
        problem_types,
        submit_mode,
        notebook_submit_artifact_mode,
    ):  # noqa: ARG001
        captured["run_id"] = run_id
        captured["submission_path"] = submission_path
        captured["best_score"] = best_score
        captured["problem_types"] = problem_types
        captured["submit_mode"] = submit_mode
        captured["notebook_submit_artifact_mode"] = notebook_submit_artifact_mode
        return {"status": "ok"}

    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)

    config = _make_config(tmp_path)
    phase = SubmissionPhase(
        config=config,
        run_id="run-1",
        problem_types=["regression"],
        submit_mode="notebook",
        notebook_submit_artifact_mode="inference",
    )
    submission_path = config.paths.submissions_dir / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")

    result = phase.attempt(submission_path=submission_path, best_score=0.4)

    assert result == {"status": "ok"}
    assert captured == {
        "run_id": "run-1",
        "submission_path": submission_path,
        "best_score": 0.4,
        "problem_types": ["regression"],
        "submit_mode": "notebook",
        "notebook_submit_artifact_mode": "inference",
    }


def test_resolve_submission_rank_payload_keeps_estimate_separate(tmp_path: Path) -> None:
    payload = resolve_submission_rank_payload(
        slug="demo",
        context_dir=tmp_path,
        direction="maximize",
        outcome={"status": "complete", "score": 0.307},
        dry_run=False,
        leaderboard_rank_for_score=lambda **kwargs: {
            "rank": 19,
            "total_teams": 151,
            "rank_percentile": 19 / 151,
            "source": "kaggle competitions leaderboard --download",
        },
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

    def fake_attempt_submit(
        *, config, run_id, submission_path, best_score, problem_types, submit_mode, notebook_submit_artifact_mode
    ):  # noqa: ARG001
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-02-16T00:00:00+00:00",
            "iteration": 1,
            "outcome": {"status": "complete", "score": 0.307},
        }

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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

    def fake_attempt_submit(
        *, config, run_id, submission_path, best_score, problem_types, submit_mode, notebook_submit_artifact_mode
    ):  # noqa: ARG001
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-02-16T00:00:00+00:00",
            "iteration": 1,
            "outcome": {"status": "complete", "score": 0.307},
        }

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", fake_improvement)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path, submit=False, max_iterations=4)
    run_autopilot(config)

    assert len(forced_modes) == 3
    assert forced_modes[0][1] is None
    assert forced_modes[1][1] is None
    assert forced_modes[2][1] == "major_overhaul"
    assert forced_modes[2][2] and "noise-limited" in forced_modes[2][2]


def test_autopilot_skips_no_improve_major_override_when_best_is_outlier(monkeypatch, tmp_path: Path) -> None:
    forced_modes: list[tuple[int, str | None, str | None]] = []

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="AUC-ROC",
        target_score=0.78,
        target_direction="maximize",
        score_source="cv",
    )

    def _iter_from_output(path: Path) -> int:
        for parent in [path.parent, *path.parents]:
            name = parent.name
            if name.startswith("iter-"):
                return int(name.split("-", 1)[1])
        return 1

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.90\n2,0.10\n", encoding="utf-8")
        iteration = _iter_from_output(output_path)
        value = 0.799651 if iteration == 1 else 0.799700
        evaluation = EvaluationResult(
            score_source="cv",
            metric="AUC-ROC",
            direction="maximize",
            value=value,
            std=0.0020,
            train_score=None,
            val_score=value,
            fold_scores=[value - 0.0010, value, value + 0.0010],
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", fake_improvement)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.78})
    monkeypatch.setattr(
        "kagglebot.autopilot_state._resume_iteration_state",
        lambda **kwargs: (1, 0.999511, None),  # stale outlier best
    )

    config = _make_config(tmp_path, submit=False, max_iterations=2)
    run_autopilot(config)

    assert len(forced_modes) == 1
    assert forced_modes[0][1] is None

    metrics = json.loads((config.paths.iter_dir("run-1", 1) / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["best_score_guard"]["applied"] is True
    assert metrics["best_score_guard"]["prev_best"] == pytest.approx(0.999511)


def test_autopilot_forces_major_overhaul_when_below_code_reference(monkeypatch, tmp_path: Path) -> None:
    forced_modes: list[tuple[int, str | None, str | None, bool]] = []

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="AUC-ROC",
        target_score=0.78,
        target_direction="maximize",
        score_source="cv",
    )

    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.code_notebooks_index_path.write_text(
        json.dumps(
            {
                "required_reference_kernel_id": "alice/ref-kernel",
                "notebooks": [{"kernel_id": "alice/ref-kernel", "score": 0.741}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.9\n2,0.1\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="cv",
            metric="AUC-ROC",
            direction="maximize",
            value=0.62,
            std=0.002,
            train_score=None,
            val_score=0.62,
            fold_scores=[0.61, 0.63],
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
                bool(kwargs.get("enforce_code_reference_implementation")),
            )
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", fake_improvement)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.78})

    config = _make_config(tmp_path, submit=False, max_iterations=2)
    run_autopilot(config)

    assert len(forced_modes) == 1
    assert forced_modes[0][1] == "major_overhaul"
    assert forced_modes[0][2] and "code reference baseline" in forced_modes[0][2]
    assert forced_modes[0][3] is True


def test_autopilot_forces_major_overhaul_on_online_mismatch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_MIN_HOURS_BETWEEN", "0")
    forced_modes: list[tuple[int, str | None, str | None, list[str]]] = []
    train_calls = {"count": 0}

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="AUC-ROC",
        target_score=0.90,
        target_direction="maximize",
        submission_gate="always",
        score_source="cv",
        cv_folds=3,
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        train_calls["count"] += 1
        output_path = kwargs["output_path"]
        output_path.write_text(f"id,target\n1,{0.9 + train_calls['count'] * 0.001:.3f}\n2,0.1\n", encoding="utf-8")
        values = {1: 0.9100, 2: 0.9150, 3: 0.9140}
        evaluation = EvaluationResult(
            score_source="cv",
            metric="AUC-ROC",
            direction="maximize",
            value=values[train_calls["count"]],
            std=0.0010,
            train_score=None,
            val_score=values[train_calls["count"]],
            fold_scores=[
                values[train_calls["count"]] - 0.001,
                values[train_calls["count"]],
                values[train_calls["count"]] + 0.001,
            ],
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="xgboost",
            model_summary={"family": "xgboost"},
            accelerator="cuda",
        )

    outcomes = [
        {"status": "complete", "score": 0.9050},
        {"status": "complete", "score": 0.9030},
        {"status": "complete", "score": 0.9040},
    ]

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.submission_policy.should_attempt_submit_for_readiness", lambda **kwargs: True)
    monkeypatch.setattr("kagglebot.submit_stage.resolve_submission_rank_payload", lambda **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.93})
    monkeypatch.setattr("kagglebot.autopilot.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "kagglebot.submission_service.run_kaggle_submit",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})(),
    )
    monkeypatch.setattr("kagglebot.submit_stage.wait_for_submission_outcome", lambda **kwargs: outcomes.pop(0))

    def fake_improvement(**kwargs):
        forced_modes.append(
            (
                int(kwargs["iteration"]),
                kwargs.get("forced_improvement_mode"),
                kwargs.get("forced_improvement_reason"),
                list(kwargs.get("extra_policy_notes") or []),
            )
        )

    monkeypatch.setattr("kagglebot.autopilot._run_improvement", fake_improvement)

    config = _make_config(tmp_path, submit=True, max_iterations=3, force_submit=False)
    run_autopilot(config)

    assert len(forced_modes) == 2
    assert forced_modes[0][1] is None
    assert forced_modes[1][1] == "major_overhaul"
    assert forced_modes[1][2] and "public leaderboard regressed" in forced_modes[1][2].lower()
    assert any("same-family-only tuning" in note for note in forced_modes[1][3])

    insights = resolve_problem_type_insights(config.knowledge_paths, ["unknown"], limit=10)
    assert any("public leaderboard regressed" in str(item.get("why_poor", "")).lower() for item in insights)


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

    def fake_attempt_submit(
        *, config, run_id, submission_path, best_score, problem_types, submit_mode, notebook_submit_artifact_mode
    ):  # noqa: ARG001
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", fake_improvement)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": None})

    config = _make_config(tmp_path, submit=True, max_iterations=2)
    run_autopilot(config)

    assert len(forced_modes) == 1
    assert forced_modes[0][1] == "major_overhaul"
    assert forced_modes[0][2] and "1300/2700" in forced_modes[0][2]


def test_autopilot_blocks_minor_tuning_until_medal_band_is_reached(monkeypatch, tmp_path: Path) -> None:
    forced_modes: list[tuple[int, str | None, str | None]] = []

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="auc",
        target_score=0.91,
        target_direction="maximize",
        target_medal="bronze",
        target_rank_percentile=0.10,
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.9\n2,0.1\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="cv",
            metric="auc",
            direction="maximize",
            value=0.9095,
            std=0.001,
            train_score=None,
            val_score=0.9095,
            fold_scores=[0.9090, 0.9095, 0.9100],
        )
        return TrainingOutcome(
            submission_path=output_path,
            evaluation=evaluation,
            model_name="xgboost",
            model_summary={},
            accelerator="cuda",
        )

    def fake_improvement(**kwargs):
        forced_modes.append(
            (
                int(kwargs["iteration"]),
                kwargs.get("minimum_improvement_mode"),
                kwargs.get("minimum_improvement_reason"),
            )
        )

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", fake_improvement)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.91})

    config = _make_config(tmp_path, submit=False, max_iterations=2)
    run_autopilot(config)

    assert len(forced_modes) == 1
    assert forced_modes[0][1] == "moderate_update"
    assert forced_modes[0][2]
    assert "bronze" in forced_modes[0][2]


def test_autopilot_skips_no_improve_major_override_when_best_anchor_is_outlier(monkeypatch, tmp_path: Path) -> None:
    forced_modes: list[tuple[int, str | None, str | None]] = []

    _write_plan(
        CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"),
        target_metric="auc",
        target_score=0.78,
        target_direction="maximize",
        score_source="cv",
    )

    def fake_train(*args, **kwargs):  # noqa: ARG001
        output_path = kwargs["output_path"]
        output_path.write_text("id,target\n1,0.9\n2,0.1\n", encoding="utf-8")
        evaluation = EvaluationResult(
            score_source="cv",
            metric="auc",
            direction="maximize",
            value=0.799651,
            std=0.0015,
            train_score=None,
            val_score=0.799651,
            fold_scores=[0.7981, 0.7991, 0.7997, 0.8002, 0.8011],
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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.autopilot._run_improvement", fake_improvement)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot.leaderboard_top1", lambda *args, **kwargs: {"score": 0.78})
    monkeypatch.setattr(
        "kagglebot.autopilot_state._resume_iteration_state",
        lambda **kwargs: (1, 0.999511, None),  # noqa: ARG005
    )

    config = _make_config(tmp_path, submit=False, max_iterations=2)
    run_autopilot(config)

    assert len(forced_modes) == 1
    assert forced_modes[0][1] is None
    metrics = json.loads((config.paths.iter_dir("run-1", 1) / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["best_score_guard"]["applied"] is True


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

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        infer_iteration_from_submission_path=infer_iteration_from_submission_path,
    )

    assert start == 1
    assert best_score is None
    assert best_submission is None


def test_resume_iteration_state_accepts_zip_submission_artifact(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    iter_dir = paths.iter_dir("run-1", 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    submission_path = iter_dir / "submission.zip"
    with zipfile.ZipFile(submission_path, "w") as archive:
        archive.writestr("1.tif", b"dummy")
    _write_kernel_metrics(iter_dir / "metrics.json", value=0.2222)

    start, best_score, best_submission = _resume_iteration_state(
        paths=paths,
        run_id="run-1",
        metric_direction="minimize",
        target_metric="rmse",
        max_iterations=3,
        require_submit_phase=False,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        infer_iteration_from_submission_path=infer_iteration_from_submission_path,
    )

    assert start == 2
    assert best_score == pytest.approx(0.2222)
    assert best_submission == submission_path


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

    def fake_attempt_submit(
        *, config, run_id, submission_path, best_score, problem_types, submit_mode, notebook_submit_artifact_mode
    ):  # noqa: ARG001
        submit_calls["count"] += 1
        return {
            "message": "demo",
            "submission_path": str(submission_path),
            "submitted_at": "2026-02-16T00:00:00+00:00",
            "iteration": 1,
            "outcome": {"status": "complete", "score": 0.49},
        }

    monkeypatch.setattr("kagglebot.autopilot.train_evaluate_and_predict", fake_train, raising=False)
    monkeypatch.setattr("kagglebot.autopilot._attempt_submit", fake_attempt_submit)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
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


def test_load_submit_retry_artifacts_prefers_training_metrics_over_submit_only_output(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_dir = paths.run_dir("run-1")
    iter_dir = paths.iter_dir("run-1", 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    submission_path = iter_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    metrics_path = iter_dir / "metrics.json"
    _write_kernel_metrics(metrics_path, value=0.321)
    output_metrics_path = iter_dir / "output" / "metrics.json"
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(
        json.dumps({"schema_version": 1, "kind": "submit_only"}, indent=2),
        encoding="utf-8",
    )
    os.utime(output_metrics_path, (metrics_path.stat().st_mtime + 10, metrics_path.stat().st_mtime + 10))
    (iter_dir / "iteration_state.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "iteration": 1,
                "iteration_complete": True,
                "trained": True,
                "submission_path": str(submission_path),
                "metrics_path": str(metrics_path),
                "submit_phase_required": True,
                "submit_phase_finished": False,
                "submit_allowed_by_gate": True,
                "submitted": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = _load_submit_retry_artifacts(
        run_dir=run_dir,
        iter_dir=iter_dir,
        iteration=1,
        max_iterations=3,
        metric_direction="minimize",
        target_metric="rmse",
        require_submit_phase=True,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
    )

    assert result is not None
    result_submission_path, result_metrics_path, evaluation = result
    assert result_submission_path == submission_path
    assert result_metrics_path == metrics_path
    assert evaluation.value == pytest.approx(0.321)


def test_load_submit_retry_artifacts_ignores_later_submit_only_metrics_for_legacy_latest(
    tmp_path: Path,
) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_dir = paths.run_dir("run-1")
    iter1_dir = paths.iter_dir("run-1", 1)
    iter2_dir = paths.iter_dir("run-1", 2)
    iter1_dir.mkdir(parents=True, exist_ok=True)
    iter2_output_dir = iter2_dir / "output"
    iter2_output_dir.mkdir(parents=True, exist_ok=True)

    iter1_submission = iter1_dir / "submission.csv"
    iter1_submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    iter1_metrics = iter1_dir / "metrics.json"
    _write_kernel_metrics(iter1_metrics, value=0.456)
    (iter2_output_dir / "submission.csv").write_text("id,target\n1,0.1\n", encoding="utf-8")
    submit_only_metrics = iter2_output_dir / "metrics.json"
    submit_only_metrics.write_text(
        json.dumps({"schema_version": 1, "kind": "submit_only"}, indent=2),
        encoding="utf-8",
    )
    os.utime(submit_only_metrics, (iter1_metrics.stat().st_mtime + 10, iter1_metrics.stat().st_mtime + 10))

    result = _load_submit_retry_artifacts(
        run_dir=run_dir,
        iter_dir=iter1_dir,
        iteration=1,
        max_iterations=3,
        metric_direction="minimize",
        target_metric="rmse",
        require_submit_phase=True,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
    )

    assert result is not None
    result_submission_path, result_metrics_path, evaluation = result
    assert result_submission_path == iter1_submission
    assert result_metrics_path == iter1_metrics
    assert evaluation.value == pytest.approx(0.456)


def test_load_submit_retry_artifacts_ignores_prior_success_for_latest_failed_submit(
    tmp_path: Path,
) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_dir = paths.run_dir("run-1")
    iter1_dir = paths.iter_dir("run-1", 1)
    iter2_dir = paths.iter_dir("run-1", 2)
    iter1_dir.mkdir(parents=True, exist_ok=True)
    iter2_dir.mkdir(parents=True, exist_ok=True)
    iter1_submission = iter1_dir / "submission.csv"
    iter1_submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    _write_kernel_metrics(iter1_dir / "metrics.json", value=0.5)
    iter2_submission = iter2_dir / "submission.csv"
    iter2_submission.write_text("id,target\n1,0.2\n", encoding="utf-8")
    iter2_metrics = iter2_dir / "metrics.json"
    _write_kernel_metrics(iter2_metrics, value=0.321)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "submit_attempted": True,
                "submit_ok": True,
                "last_action": "abort",
                "last_reason": "ambiguous_notebook_bad_request",
                "last_submission_path": str(iter2_submission),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "submit_attempts.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "run-1",
                        "sub_path": str(iter1_submission),
                        "action_taken": "submit",
                        "reason": "submitted",
                        "ok": True,
                    }
                ),
                json.dumps(
                    {
                        "run_id": "run-1",
                        "sub_path": str(iter2_submission),
                        "action_taken": "abort",
                        "reason": "ambiguous_notebook_bad_request",
                        "ok": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _load_submit_retry_artifacts(
        run_dir=run_dir,
        iter_dir=iter2_dir,
        iteration=2,
        max_iterations=3,
        metric_direction="minimize",
        target_metric="rmse",
        require_submit_phase=True,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
    )

    assert result is not None
    result_submission_path, result_metrics_path, evaluation = result
    assert result_submission_path == iter2_submission
    assert result_metrics_path == iter2_metrics
    assert evaluation.value == pytest.approx(0.321)


def test_load_submit_retry_artifacts_treats_duplicate_skip_as_terminal(
    tmp_path: Path,
) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_dir = paths.run_dir("run-1")
    iter_dir = paths.iter_dir("run-1", 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    submission_path = iter_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    metrics_path = iter_dir / "metrics.json"
    _write_kernel_metrics(metrics_path, value=0.321)
    (iter_dir / "iteration_state.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "iteration": 1,
                "iteration_complete": True,
                "trained": True,
                "submission_path": str(submission_path),
                "metrics_path": str(metrics_path),
                "submit_phase_required": True,
                "submit_phase_finished": False,
                "submit_allowed_by_gate": True,
                "submitted": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "submit_attempts.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "sub_path": str(submission_path),
                "action_taken": "skip",
                "reason": "duplicate_submission_sha_seen",
                "ok": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _load_submit_retry_artifacts(
        run_dir=run_dir,
        iter_dir=iter_dir,
        iteration=1,
        max_iterations=3,
        metric_direction="minimize",
        target_metric="rmse",
        require_submit_phase=True,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
    )

    assert result is None


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
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        infer_iteration_from_submission_path=infer_iteration_from_submission_path,
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
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        infer_iteration_from_submission_path=infer_iteration_from_submission_path,
    )

    assert start_after == 2
    assert best_score == pytest.approx(0.4321)
    assert best_submission == submission_path


def test_resume_iteration_state_accepts_legacy_duplicate_skip(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    iter_dir = paths.iter_dir("run-1", 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    submission_path = iter_dir / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    _write_kernel_metrics(iter_dir / "metrics.json", value=0.4321)

    run_dir = paths.run_dir("run-1")
    (run_dir / "submit_attempts.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "sub_path": str(submission_path),
                "action_taken": "skip",
                "reason": "duplicate_submission_sha_seen",
                "ok": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    start, best_score, best_submission = _resume_iteration_state(
        paths=paths,
        run_id="run-1",
        metric_direction="minimize",
        target_metric="rmse",
        max_iterations=3,
        require_submit_phase=True,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        infer_iteration_from_submission_path=infer_iteration_from_submission_path,
    )

    assert start == 2
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
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        infer_iteration_from_submission_path=infer_iteration_from_submission_path,
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
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        infer_iteration_from_submission_path=infer_iteration_from_submission_path,
    )

    assert start == 1
    assert best_score is None
    assert best_submission is None


def test_resume_iteration_state_accepts_duplicate_submission_skip_marker(tmp_path: Path) -> None:
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
                "submit_phase_finished": True,
                "submit_allowed_by_gate": True,
                "submitted": False,
                "submit_phase_state": "duplicate_submission_sha_seen",
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
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        infer_iteration_from_submission_path=infer_iteration_from_submission_path,
    )

    assert start == 2
    assert best_score == pytest.approx(0.351)
    assert best_submission == submission_path


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

    write_iteration_state_marker(
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


def test_kernel_fix_uses_lightweight_column_fill_for_keyerror(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _run_kernel_fix

    config = _make_config(tmp_path)
    run_id = config.run_id or "run-1"
    iter_dir = config.paths.iter_dir(run_id, 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    calls = {"strategy": 0, "codex": 0}

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool):  # noqa: ARG001
        calls["strategy"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "strategy_last_message.txt"
        last_msg.write_text("should not be used\n", encoding="utf-8")
        return DummyResult(last_msg)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        calls["codex"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("should not be used\n", encoding="utf-8")
        return DummyResult(last_msg)

    monkeypatch.setattr("kagglebot.autopilot.run_strategy", fake_run_strategy)
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)

    pending: list[dict[str, object]] = []
    _run_kernel_fix(
        config=config,
        run_id=run_id,
        iteration=1,
        iter_dir=iter_dir,
        error_message="KeyError: \"['oare_id'] not in index\"",
        attempt=1,
        pending_error_fixes=pending,
    )

    fill_path = config.paths.context_dir / "column_fill.json"
    assert fill_path.exists()
    payload = json.loads(fill_path.read_text(encoding="utf-8"))
    assert payload["missing_columns"] == ["oare_id"]
    note_path = iter_dir / "agent" / "kernel_fix_note-01.txt"
    assert note_path.exists()
    assert "column_fill.json" in note_path.read_text(encoding="utf-8")
    assert calls["strategy"] == 0
    assert calls["codex"] == 0
    assert pending
    assert pending[0]["resolved"] is True


def test_kernel_fix_regenerates_when_codex_makes_no_changes(monkeypatch, tmp_path: Path) -> None:
    from kagglebot.autopilot import _run_kernel_fix

    config = _make_config(tmp_path)
    run_id = config.run_id or "run-1"
    iter_dir = config.paths.iter_dir(run_id, 1)
    iter_dir.mkdir(parents=True, exist_ok=True)
    calls = {"codex": 0, "regen": 0}

    class DummyResult:
        def __init__(self, path: Path) -> None:
            self.returncode = 0
            self.last_message_path = path

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        calls["codex"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        last_msg = output_dir / "codex_last_message.txt"
        last_msg.write_text("no changes applied\n", encoding="utf-8")
        return DummyResult(last_msg)

    def fake_replan(config_arg, run_id_arg):  # noqa: ANN001, ARG001
        calls["regen"] += 1

    def fail_if_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("allowlist enforcement must not run when no file changes are detected")

    monkeypatch.setattr("kagglebot.agent_strategy.run_error_strategy_prompt", lambda **kwargs: "")
    monkeypatch.setattr("kagglebot.autopilot.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.autopilot._run_plan_and_initial", fake_replan)
    monkeypatch.setattr("kagglebot.verify_artifacts.run_repo_verify", lambda *args, **kwargs: None)
    monkeypatch.setattr("kagglebot.autopilot._backup_guarded_files", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._snapshot_tree", lambda *args, **kwargs: {})
    monkeypatch.setattr("kagglebot.autopilot._diff_snapshots", lambda *args, **kwargs: [])
    monkeypatch.setattr("kagglebot.autopilot._enforce_allowlist_changes", fail_if_called)
    monkeypatch.setattr(
        "kagglebot.autopilot._autofix_restart.maybe_restart_for_src_changes", lambda *args, **kwargs: None
    )

    _run_kernel_fix(
        config=config,
        run_id=run_id,
        iteration=1,
        iter_dir=iter_dir,
        error_message="RuntimeError: kernel failed for unknown reason",
        attempt=1,
        pending_error_fixes=[],
    )

    assert calls["codex"] == 1
    assert calls["regen"] == 1
    marker_path = iter_dir / "agent" / "kernel_regenerated_once.json"
    assert marker_path.exists()

    with pytest.raises(KernelFailedError, match="produced no file changes"):
        _run_kernel_fix(
            config=config,
            run_id=run_id,
            iteration=1,
            iter_dir=iter_dir,
            error_message="RuntimeError: kernel failed for unknown reason",
            attempt=2,
            pending_error_fixes=[],
        )
