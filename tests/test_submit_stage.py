from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kagglebot.campaign import CampaignCandidate, campaign_state_path, candidate_registry_path, upsert_candidate
from kagglebot.submit_stage import (
    build_default_submission_problem_insight,
    build_submission_outcome_error_detail,
    build_submit_stage_success_record,
    classify_submission_outcome,
    classify_submit_stage_error,
    decide_initial_submit_stage_mode,
    decide_notebook_fallback_after_file_submit_error,
    decide_submission_outcome_abort,
    decide_submit_stage_error_action,
    ensure_submission_problem_insights,
    find_campaign_candidate_for_submission,
    format_iteration_submit_status_message,
    format_rank_force_reason,
    format_submission_rank_message,
    infer_iteration_from_submission_path,
    normalize_submission_outcome_status,
    record_submission_knowledge_entries,
    resolve_iteration_submit_phase_state,
    resolve_submission_knowledge_context,
    resolve_submission_knowledge_iteration,
    resolve_submission_message,
    resolve_submission_rank_payload,
    run_submit_stage_attempt,
    submission_score_for_tracking,
    wait_for_submission_outcome,
)


class FileSubmitResult:
    def __init__(self, submission_path: Path) -> None:
        self.submission_path = submission_path


class SubmitResultStub:
    def __init__(
        self,
        *,
        stdout: object = "",
        stderr: object = "",
        exit_code: int | None = None,
        returncode: int | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        if exit_code is not None:
            self.exit_code = exit_code
        if returncode is not None:
            self.returncode = returncode


def test_decide_initial_submit_stage_mode_keeps_file_submit() -> None:
    decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=False,
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        resolved_notebook_artifact_mode="inference",
    )

    assert decision.notebook_submit_required is False
    assert decision.notebook_fallback_activated is False
    assert decision.submission_artifact_mode == "wrapper"
    assert decision.messages == ()


def test_decide_initial_submit_stage_mode_uses_requested_notebook_mode() -> None:
    decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=True,
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        resolved_notebook_artifact_mode="inference",
    )

    assert decision.notebook_submit_required is True
    assert decision.notebook_fallback_activated is True
    assert decision.submission_artifact_mode == "inference"
    assert decision.messages == ("[yellow]submit mode[/yellow]: using notebook submit",)


def test_decide_initial_submit_stage_mode_forces_notebook_only_competition() -> None:
    decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=False,
        notebook_submissions_only=True,
        notebook_submit_artifact_mode="wrapper",
        resolved_notebook_artifact_mode="inference",
    )

    assert decision.notebook_submit_required is True
    assert decision.notebook_fallback_activated is True
    assert decision.submission_artifact_mode == "inference"
    assert decision.messages == (
        "[yellow]submit mode[/yellow]: notebook-only competition detected; forcing notebook submit",
        "[yellow]submit mode[/yellow]: using notebook submit",
    )


def test_normalize_submission_outcome_status_strips_enum_prefix() -> None:
    assert normalize_submission_outcome_status("SubmissionStatus.COMPLETE") == "complete"
    assert normalize_submission_outcome_status("") == "unknown"


def test_wait_for_submission_outcome_uses_fetch_adapter() -> None:
    submitted_at = datetime(2026, 1, 1, tzinfo=UTC)

    outcome = wait_for_submission_outcome(
        slug="demo",
        message="submission message",
        submitted_at=submitted_at,
        fetch_submission_rows=lambda slug: [
            {
                "description": "submission message",
                "date": "2026-01-01T00:01:00Z",
                "status": "SubmissionStatus.COMPLETE",
                "publicScore": "0.123",
            }
        ],
        max_attempts=1,
        poll_interval_sec=0.0,
        max_fetch_errors=1,
    )

    assert outcome is not None
    assert outcome["status"] == "complete"
    assert outcome["score"] == 0.123
    assert outcome["raw"]["description"] == "submission message"


def test_infer_iteration_from_submission_path_reads_iter_parent() -> None:
    assert infer_iteration_from_submission_path(Path("runs/run-1/iter-3/submission.csv")) == 3
    assert infer_iteration_from_submission_path(Path("submission.csv")) is None


def test_submission_score_for_tracking_prefers_finite_online_score() -> None:
    assert submission_score_for_tracking(offline_score=0.9, online_score=0.8) == (0.8, "submission_public_score")
    assert submission_score_for_tracking(offline_score=0.9, online_score=float("nan")) == (0.9, "offline")
    assert submission_score_for_tracking(offline_score=0.9, online_score=None) == (0.9, "offline")


def test_classify_submission_outcome_uses_target_score() -> None:
    assert classify_submission_outcome(score=0.4, direction="minimize", target_score=0.5, top1_score=None) == "good"
    assert classify_submission_outcome(score=0.6, direction="minimize", target_score=0.5, top1_score=None) == "low"
    assert classify_submission_outcome(score=0.8, direction="maximize", target_score=0.7, top1_score=None) == "good"
    assert classify_submission_outcome(score=0.6, direction="maximize", target_score=0.7, top1_score=None) == "low"


def test_classify_submission_outcome_treats_top1_near_miss_as_good() -> None:
    assert classify_submission_outcome(score=1.09, direction="minimize", target_score=None, top1_score=1.0) == "good"
    assert classify_submission_outcome(score=1.2, direction="minimize", target_score=None, top1_score=1.0) == "low"
    assert classify_submission_outcome(score=0.91, direction="maximize", target_score=None, top1_score=1.0) == "good"
    assert classify_submission_outcome(score=0.8, direction="maximize", target_score=None, top1_score=1.0) == "low"


def test_resolve_submission_knowledge_context_extracts_score_bucket_and_iteration() -> None:
    context = resolve_submission_knowledge_context(
        submission_result={"outcome": {"score": "0.42"}, "iteration": 3},
        metric_direction="minimize",
        target_score=0.5,
        top1_score=None,
    )

    assert context is not None
    assert context.online_score == 0.42
    assert context.outcome_bucket == "good"
    assert context.iteration == 3


def test_resolve_submission_knowledge_context_rejects_scoreless_result() -> None:
    assert (
        resolve_submission_knowledge_context(
            submission_result={"outcome": {"status": "complete"}, "iteration": 3},
            metric_direction="minimize",
            target_score=0.5,
            top1_score=None,
        )
        is None
    )
    assert (
        resolve_submission_knowledge_context(
            submission_result={"outcome": {"score": "nan"}, "iteration": "3"},
            metric_direction="minimize",
            target_score=0.5,
            top1_score=None,
        )
        is None
    )


def test_build_default_submission_problem_insight_uses_iteration_and_diagnostics() -> None:
    insight = build_default_submission_problem_insight(
        iteration=3,
        diagnostics_text="diagnostic details",
    )

    assert insight == {
        "iteration": 3,
        "why_poor": "diagnostic details",
        "how_improved": "Submitted iteration 3 result after validation.",
        "delta_offline": None,
    }


def test_ensure_submission_problem_insights_adds_default_with_diagnostics() -> None:
    pending: list[dict[str, object]] = []
    context = resolve_submission_knowledge_context(
        submission_result={"outcome": {"score": "0.42"}, "iteration": 3},
        metric_direction="minimize",
        target_score=0.5,
        top1_score=None,
    )
    assert context is not None

    ensure_submission_problem_insights(
        pending_problem_insights=pending,
        knowledge_context=context,
        load_diagnostics_text=lambda iteration: f"diagnostics for {iteration}",
    )

    assert pending == [
        {
            "iteration": 3,
            "why_poor": "diagnostics for 3",
            "how_improved": "Submitted iteration 3 result after validation.",
            "delta_offline": None,
        }
    ]


def test_ensure_submission_problem_insights_keeps_existing_items() -> None:
    pending: list[dict[str, object]] = [{"iteration": 2, "why_poor": "existing"}]
    context = resolve_submission_knowledge_context(
        submission_result={"outcome": {"score": "0.42"}, "iteration": 3},
        metric_direction="minimize",
        target_score=0.5,
        top1_score=None,
    )
    assert context is not None

    ensure_submission_problem_insights(
        pending_problem_insights=pending,
        knowledge_context=context,
        load_diagnostics_text=lambda _iteration: (_ for _ in ()).throw(AssertionError("should not load")),
    )

    assert pending == [{"iteration": 2, "why_poor": "existing"}]


def test_resolve_submission_knowledge_iteration_uses_fallback_for_bad_values() -> None:
    assert resolve_submission_knowledge_iteration(value="4", fallback_iteration=2) == 4
    assert resolve_submission_knowledge_iteration(value="", fallback_iteration=2) == 2
    assert resolve_submission_knowledge_iteration(value="bad", fallback_iteration=2) == 2
    assert resolve_submission_knowledge_iteration(value=None, fallback_iteration=None) == 1


def test_record_submission_knowledge_entries_records_problem_and_error_items() -> None:
    problem_calls: list[dict[str, object]] = []
    error_calls: list[dict[str, object]] = []
    knowledge_paths = object()
    context = resolve_submission_knowledge_context(
        submission_result={"outcome": {"score": "0.42"}, "iteration": 3},
        metric_direction="minimize",
        target_score=0.5,
        top1_score=None,
    )
    assert context is not None

    record_submission_knowledge_entries(
        knowledge_paths=knowledge_paths,
        slug="demo",
        run_id="run-1",
        problem_types=["tabular"],
        pending_problem_insights=[
            {"iteration": "", "why_poor": "too high", "how_improved": "tuned", "delta_offline": 0.1}
        ],
        pending_error_fixes=[
            {"iteration": "4", "error_message": "bad csv", "fix_summary": "fixed columns", "resolved": False}
        ],
        knowledge_context=context,
        record_problem_type_insight=lambda **kwargs: problem_calls.append(kwargs),
        record_error_fix_insight=lambda **kwargs: error_calls.append(kwargs),
    )

    assert problem_calls == [
        {
            "knowledge_paths": knowledge_paths,
            "slug": "demo",
            "run_id": "run-1",
            "iteration": 3,
            "problem_types": ["tabular"],
            "why_poor": "too high",
            "how_improved": "tuned",
            "delta_offline": 0.1,
            "outcome_bucket": "good",
            "submission_score": 0.42,
        }
    ]
    assert error_calls == [
        {
            "knowledge_paths": knowledge_paths,
            "slug": "demo",
            "run_id": "run-1",
            "iteration": 4,
            "problem_types": ["tabular"],
            "error_message": "bad csv",
            "fix_summary": "fixed columns",
            "resolved": False,
            "outcome_bucket": "good",
            "submission_score": 0.42,
        }
    ]


def test_resolve_submission_rank_payload_keeps_reported_rank() -> None:
    payload = resolve_submission_rank_payload(
        slug="demo",
        context_dir=Path("context"),
        direction="minimize",
        outcome={"rank": "2", "total_teams": "10", "rank_source": "submission_row"},
        dry_run=False,
        leaderboard_rank_for_score=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not estimate")),
    )

    assert payload == {
        "rank": 2,
        "total_teams": 10,
        "rank_source": "submission_row",
        "rank_percentile": 0.2,
    }


def test_resolve_submission_rank_payload_estimates_when_rank_missing() -> None:
    calls: list[dict[str, object]] = []

    def estimate(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"rank": 4, "total_teams": 20, "rank_percentile": 0.2, "source": "leaderboard"}

    payload = resolve_submission_rank_payload(
        slug="demo",
        context_dir=Path("context"),
        direction="maximize",
        outcome={"score": "0.75"},
        dry_run=True,
        leaderboard_rank_for_score=estimate,
    )

    assert payload == {
        "estimated_rank": 4,
        "estimated_total_teams": 20,
        "estimated_rank_percentile": 0.2,
        "rank_estimate_source": "leaderboard_score_estimate",
    }
    assert calls == [
        {
            "slug": "demo",
            "output_dir": Path("context"),
            "score": 0.75,
            "direction": "maximize",
            "dry_run": True,
        }
    ]


def test_format_rank_force_reason_includes_percentile_threshold_and_source() -> None:
    reason = format_rank_force_reason(
        rank=120,
        total_teams=1000,
        rank_percentile=None,
        max_percentile=0.01,
        min_teams=200,
        source="submission_row",
    )

    assert reason == (
        "Leaderboard rank indicates large headroom for improvement: "
        "120/1000 (percentile=12.00%, threshold=1.00%, min_teams=200). source=submission_row"
    )


def test_format_submission_rank_message_formats_observed_and_estimated_rank() -> None:
    observed = format_submission_rank_message(
        rank=12,
        total_teams=100,
        rank_percentile=None,
        source="submission_row",
    )
    estimated = format_submission_rank_message(
        rank=20,
        total_teams=200,
        rank_percentile=0.1,
        source="leaderboard_score_estimate",
        estimated=True,
    )

    assert observed == "[cyan]submission rank[/cyan]: 12/100 (percentile=12.00%) source=submission_row"
    assert estimated == (
        "[yellow]submission rank estimate[/yellow]: 20/200 (percentile=10.00%) source=leaderboard_score_estimate"
    )


def test_format_iteration_submit_status_message_handles_disabled_allowed_and_blocked() -> None:
    assert (
        format_iteration_submit_status_message(
            iteration=1,
            max_iterations=3,
            submit_enabled=False,
            submit_allowed_by_gate=False,
            submit_phase_state="disabled",
            quality_reasons=[],
        )
        is None
    )
    assert (
        format_iteration_submit_status_message(
            iteration=1,
            max_iterations=3,
            submit_enabled=True,
            submit_allowed_by_gate=True,
            submit_phase_state="ready",
            quality_reasons=[],
        )
        == "[cyan]submit[/cyan]: iter 1/3 attempting submission now."
    )

    blocked = format_iteration_submit_status_message(
        iteration=2,
        max_iterations=3,
        submit_enabled=True,
        submit_allowed_by_gate=False,
        submit_phase_state="blocked_quality_guard",
        quality_reasons=["collapsed_predictions", "weak_cv"],
        competition_faithfulness={
            "expected_metric": "logloss",
            "actual_metric": "accuracy",
            "expected_split_strategy": "group_kfold",
            "actual_split_strategy": "kfold",
            "dataset_mode": "sample",
        },
    )

    assert blocked == (
        "[cyan]submit[/cyan]: iter 2/3 not attempted yet "
        "(state=blocked_quality_guard reasons=collapsed_predictions,weak_cv "
        "metric=accuracy/logloss split=kfold/group_kfold dataset_mode=sample)."
    )


def test_resolve_iteration_submit_phase_state_prioritizes_hard_limit() -> None:
    assert (
        resolve_iteration_submit_phase_state(
            submit_enabled=True,
            daily_submission_limit_reached=True,
            force_initial_submit=True,
            quality_allows_submit=False,
            force_submit=False,
            submit_non_improving=False,
            defer_submit_for_accuracy_frontier=False,
            submit_limited_holdback=False,
        )
        == "daily_submission_limit_reached"
    )


def test_resolve_iteration_submit_phase_state_handles_quality_and_force() -> None:
    blocked = resolve_iteration_submit_phase_state(
        submit_enabled=True,
        daily_submission_limit_reached=False,
        force_initial_submit=False,
        quality_allows_submit=False,
        force_submit=False,
        submit_non_improving=False,
        defer_submit_for_accuracy_frontier=False,
        submit_limited_holdback=False,
    )
    forced = resolve_iteration_submit_phase_state(
        submit_enabled=True,
        daily_submission_limit_reached=False,
        force_initial_submit=False,
        quality_allows_submit=False,
        force_submit=True,
        submit_non_improving=False,
        defer_submit_for_accuracy_frontier=False,
        submit_limited_holdback=False,
    )

    assert blocked == "blocked_quality_guard"
    assert forced == "pending_submit"


def test_resolve_iteration_submit_phase_state_defers_override_pending_states() -> None:
    assert (
        resolve_iteration_submit_phase_state(
            submit_enabled=True,
            daily_submission_limit_reached=False,
            force_initial_submit=True,
            quality_allows_submit=True,
            force_submit=False,
            submit_non_improving=True,
            defer_submit_for_accuracy_frontier=False,
            submit_limited_holdback=False,
        )
        == "deferred_non_improving"
    )
    assert (
        resolve_iteration_submit_phase_state(
            submit_enabled=True,
            daily_submission_limit_reached=False,
            force_initial_submit=False,
            quality_allows_submit=True,
            force_submit=False,
            submit_non_improving=True,
            defer_submit_for_accuracy_frontier=True,
            submit_limited_holdback=True,
        )
        == "deferred_for_final_slot"
    )


def test_resolve_submission_message_builds_compact_default(tmp_path: Path) -> None:
    message = resolve_submission_message(
        context_dir=tmp_path / "context",
        run_id="run-1",
        best_score=0.123456,
        explicit_message=None,
        submission_path=Path("runs/run-1/iter-2/submission.csv"),
        campaign_mode="baseline",
        target_direction="minimize",
    )

    assert message == "kb run-1 i=2 offline=0.1235"


def test_resolve_submission_message_includes_campaign_candidate(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    submission_path = tmp_path / "runs" / "run-1" / "iter-2" / "submission.csv"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    upsert_candidate(
        candidate_registry_path(context_dir),
        CampaignCandidate(
            candidate_id="run-1-i002-strong_single",
            category="strong_single",
            run_id="run-1",
            iteration=2,
            direction="minimize",
            submission_path=str(submission_path),
        ),
    )
    campaign_state_path(context_dir).write_text(
        '{"campaign_id":"campaign-1","direction":"minimize","historical_best_score":0.2}\n',
        encoding="utf-8",
    )

    message = resolve_submission_message(
        context_dir=context_dir,
        run_id="run-1",
        best_score=0.15,
        explicit_message=None,
        submission_path=submission_path,
        campaign_mode="top1",
        target_direction="minimize",
    )

    assert "campaign=campaign-1" in message
    assert "candidate=run-1-i002-strong_single" in message
    assert "baseline_delta=+0.050000" in message


def test_find_campaign_candidate_for_submission_prefers_submission_path(tmp_path: Path) -> None:
    selected_path = tmp_path / "iter-1" / "submission.csv"
    other = CampaignCandidate(
        candidate_id="other",
        category="strong_single",
        run_id="run-1",
        iteration=1,
        direction="minimize",
    )
    selected = CampaignCandidate(
        candidate_id="selected",
        category="strong_single",
        run_id="run-2",
        iteration=2,
        direction="minimize",
        submission_path=str(selected_path),
    )

    assert (
        find_campaign_candidate_for_submission(
            candidates=[other, selected],
            submission_path=selected_path,
            run_id="run-1",
            iteration=1,
        )
        == selected
    )


def test_decide_submission_outcome_abort_handles_error_status() -> None:
    decision = decide_submission_outcome_abort(
        outcome_status="error",
        outcome_score=None,
        deliverable_mode="leaderboard",
        raw_detail="bad submission",
    )

    assert decision.should_abort is True
    assert decision.error_kind == "validation"
    assert decision.reason == "submission_poll_status_error"
    assert "error status 'error'" in decision.message
    assert decision.detail == "bad submission"


def test_decide_submission_outcome_abort_handles_scoreless_leaderboard_complete() -> None:
    decision = decide_submission_outcome_abort(
        outcome_status="complete",
        outcome_score=None,
        deliverable_mode="leaderboard",
        raw_detail="Kaggle reported: complete without score",
    )

    assert decision.should_abort is True
    assert decision.reason == "submission_poll_status_complete_no_score"
    assert "no score" in decision.message
    assert "scoring error inferred" in decision.detail.lower()


def test_decide_submission_outcome_abort_allows_scoreless_writeup_complete() -> None:
    decision = decide_submission_outcome_abort(
        outcome_status="complete",
        outcome_score=None,
        deliverable_mode="writeup",
        raw_detail="",
    )

    assert decision.should_abort is False


def test_build_submission_outcome_error_detail_prefers_matched_submission_row() -> None:
    rows = [
        {
            "description": "target message",
            "status": "SubmissionStatus.COMPLETE",
            "publicScore": "",
            "errorDescription": "Submission Scoring Error: incorrect format",
            "date": "2026-06-25T00:41:17Z",
        }
    ]

    detail = build_submission_outcome_error_detail(
        slug="demo",
        message="target message",
        submitted_at=datetime(2026, 6, 25, tzinfo=UTC),
        outcome={"status": "complete", "score": None, "raw": {"status": "complete"}},
        fetch_submission_rows=lambda slug: rows,
        normalize_detail=lambda text: text,
    )

    assert "Submission Scoring Error" in detail
    assert "errorDescription" in detail


def test_build_submission_outcome_error_detail_falls_back_to_raw_payload() -> None:
    detail = build_submission_outcome_error_detail(
        slug="demo",
        message="target message",
        submitted_at=datetime(2026, 6, 25, tzinfo=UTC),
        outcome={"status": "complete", "score": None, "raw": "raw failure"},
        fetch_submission_rows=lambda slug: (_ for _ in ()).throw(RuntimeError("unavailable")),
        normalize_detail=lambda text: text,
    )

    assert 'Kaggle submission raw payload: "raw failure"' in detail


def test_run_submit_stage_attempt_uses_file_submit_result_path(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared.csv"
    submitted_path = tmp_path / "submitted.csv"

    result = run_submit_stage_attempt(
        notebook_submit_required=False,
        file_submission_path=prepared_path,
        run_notebook_submit=lambda: (_ for _ in ()).throw(AssertionError("notebook should not run")),
        run_file_submit=lambda: FileSubmitResult(submitted_path),
    )

    assert isinstance(result.submission_result, FileSubmitResult)
    assert result.submission_reference == str(submitted_path)
    assert result.submission_artifact_path == submitted_path


def test_run_submit_stage_attempt_uses_notebook_submit_tuple(tmp_path: Path) -> None:
    notebook_artifact = tmp_path / "notebook-submission.csv"

    result = run_submit_stage_attempt(
        notebook_submit_required=True,
        file_submission_path=tmp_path / "prepared.csv",
        run_notebook_submit=lambda: ("notebook-result", "kernel:user/demo", notebook_artifact),
        run_file_submit=lambda: (_ for _ in ()).throw(AssertionError("file should not run")),
    )

    assert result.submission_result == "notebook-result"
    assert result.submission_reference == "kernel:user/demo"
    assert result.submission_artifact_path == notebook_artifact


def test_build_submit_stage_success_record_prefers_exit_code() -> None:
    record = build_submit_stage_success_record(
        submission_result=SubmitResultStub(stdout="ok", stderr="warn", exit_code=7, returncode=0),
        compute_error_fingerprint=lambda stdout, stderr: f"{stdout}:{stderr}",
    )

    assert record.exit_code == 7
    assert record.fingerprint == "ok:warn"
    assert record.stdout == "ok"
    assert record.stderr == "warn"


def test_build_submit_stage_success_record_uses_returncode_fallback() -> None:
    record = build_submit_stage_success_record(
        submission_result=SubmitResultStub(stdout=None, stderr=None, returncode=0),
        compute_error_fingerprint=lambda stdout, stderr: f"{stdout}:{stderr}",
    )

    assert record.exit_code == 0
    assert record.fingerprint == ":"
    assert record.stdout == ""
    assert record.stderr == ""


def test_classify_submit_stage_error_uses_output_fallback() -> None:
    calls: list[str] = []

    def classify(stdout: str, stderr: str, exit_code: int | None) -> dict[str, object]:  # noqa: ARG001
        calls.append(stderr)
        if "kernel must be specified" in stderr:
            return {
                "kind": "permanent",
                "reason": "ambiguous_notebook_bad_request",
                "retry_after_seconds": 4,
            }
        return {"reason": "unclassified_submit_error"}

    classification = classify_submit_stage_error(
        stdout="",
        stderr="",
        output="400 Client Error\nkernel must be specified",
        exit_code=1,
        classify_submit_error=classify,
    )

    assert classification.stderr == "400 Client Error\nkernel must be specified"
    assert classification.kind == "permanent"
    assert classification.reason == "ambiguous_notebook_bad_request"
    assert classification.retry_after_seconds == 4.0
    assert calls == ["", "400 Client Error\nkernel must be specified"]


def test_classify_submit_stage_error_defaults_unknown_kind_and_reason() -> None:
    classification = classify_submit_stage_error(
        stdout="",
        stderr="generic",
        output="",
        exit_code=1,
        classify_submit_error=lambda stdout, stderr, exit_code: {},
    )

    assert classification.stderr == "generic"
    assert classification.kind == "unknown"
    assert classification.reason == "unclassified_submit_error"
    assert classification.retry_after_seconds == 0.0


def test_classify_submit_stage_error_normalizes_blank_fields_and_bool_retry_after() -> None:
    classification = classify_submit_stage_error(
        stdout="",
        stderr="generic",
        output="",
        exit_code=1,
        classify_submit_error=lambda stdout, stderr, exit_code: {
            "kind": " ",
            "reason": "",
            "retry_after_seconds": True,
        },
    )

    assert classification.kind == "unknown"
    assert classification.reason == "unclassified_submit_error"
    assert classification.retry_after_seconds == 0.0


def test_classify_submit_stage_error_clamps_negative_retry_after() -> None:
    classification = classify_submit_stage_error(
        stdout="",
        stderr="generic",
        output="",
        exit_code=1,
        classify_submit_error=lambda stdout, stderr, exit_code: {
            "kind": "transient",
            "reason": "network_or_timeout",
            "retry_after_seconds": -3,
        },
    )

    assert classification.kind == "transient"
    assert classification.reason == "network_or_timeout"
    assert classification.retry_after_seconds == 0.0


def test_decide_submit_stage_error_action_aborts_repeated_fingerprint() -> None:
    decision = decide_submit_stage_error_action(
        fingerprint_seen=True,
        same_fingerprint_retry_allowed=False,
        classification_kind="transient",
        classification_reason="network_or_timeout",
        attempt=1,
        max_attempts=3,
        retry_after_seconds=0.0,
        backoff_seconds=2.0,
    )

    assert decision.action == "abort"
    assert decision.error_kind == "transient"
    assert decision.reason == "same_error_fingerprint_recurred"
    assert "Same submit error fingerprint recurred" in decision.abort_message
    assert decision.messages == ()


def test_decide_submit_stage_error_action_retries_transient_with_allowance_message() -> None:
    decision = decide_submit_stage_error_action(
        fingerprint_seen=True,
        same_fingerprint_retry_allowed=True,
        classification_kind="transient",
        classification_reason="network_or_timeout",
        attempt=1,
        max_attempts=3,
        retry_after_seconds=5.0,
        backoff_seconds=2.0,
    )

    assert decision.action == "retry"
    assert decision.error_kind == "transient"
    assert decision.reason == "network_or_timeout"
    assert decision.wait_seconds == 5.0
    assert "same fingerprint matched previous failures" in decision.messages[0]
    assert "transient submit error" in decision.messages[1]


def test_decide_submit_stage_error_action_aborts_after_retry_budget() -> None:
    decision = decide_submit_stage_error_action(
        fingerprint_seen=False,
        same_fingerprint_retry_allowed=False,
        classification_kind="transient",
        classification_reason="network_or_timeout",
        attempt=3,
        max_attempts=3,
        retry_after_seconds=0.0,
        backoff_seconds=8.0,
    )

    assert decision.action == "abort"
    assert decision.reason == "network_or_timeout"
    assert decision.abort_message == "Transient submit error exceeded retry budget; aborting this run."
    assert "no further retries" in decision.messages[0]


def test_decide_notebook_fallback_after_file_submit_error_retries_as_notebook() -> None:
    decision = decide_notebook_fallback_after_file_submit_error(
        notebook_submit_required=False,
        notebook_fallback_activated=False,
        should_use_notebook_fallback=True,
        resolved_notebook_artifact_mode="inference",
        current_submission_artifact_mode="wrapper",
    )

    assert decision.retry_as_notebook is True
    assert decision.notebook_submit_required is True
    assert decision.notebook_fallback_activated is True
    assert decision.submission_artifact_mode == "inference"
    assert decision.messages == (
        "[yellow]submit mode[/yellow]: file submit indicates notebook submit is required; "
        "retrying via notebook submit automatically.",
    )


def test_decide_notebook_fallback_after_file_submit_error_rejects_already_activated() -> None:
    decision = decide_notebook_fallback_after_file_submit_error(
        notebook_submit_required=False,
        notebook_fallback_activated=True,
        should_use_notebook_fallback=True,
        resolved_notebook_artifact_mode="inference",
        current_submission_artifact_mode="wrapper",
    )

    assert decision.retry_as_notebook is False
    assert decision.notebook_submit_required is False
    assert decision.notebook_fallback_activated is True
    assert decision.submission_artifact_mode == "wrapper"
    assert decision.messages == ()


def test_decide_notebook_fallback_after_file_submit_error_rejects_non_notebook_error() -> None:
    decision = decide_notebook_fallback_after_file_submit_error(
        notebook_submit_required=False,
        notebook_fallback_activated=False,
        should_use_notebook_fallback=False,
        resolved_notebook_artifact_mode="inference",
        current_submission_artifact_mode="wrapper",
    )

    assert decision.retry_as_notebook is False
    assert decision.notebook_submit_required is False
    assert decision.notebook_fallback_activated is False
    assert decision.submission_artifact_mode == "wrapper"
