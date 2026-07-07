from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from kagglebot.campaign import CampaignCandidate, campaign_state_path, candidate_registry_path, upsert_candidate
from kagglebot.history import SubmissionLedger
from kagglebot.json_utils import load_jsonl_records
from kagglebot.submit_abort import (
    SubmitRunAborter,
    SubmitRunRetryRecorder,
    abort_submit_for_run,
    build_submit_run_aborter_for_run,
    record_submit_abort_for_run,
    record_submit_stage_retry_attempt,
)
from kagglebot.submit_abort_specs import (
    build_kaggle_credentials_missing_abort_spec,
    build_local_submission_guardrail_abort_spec,
    build_local_submission_validation_abort_spec,
    build_rules_not_accepted_abort_spec,
    build_submission_outcome_abort_spec,
    build_submission_polling_error_abort_spec,
    build_submit_abort_spec_kwargs,
    build_submit_stage_error_action_abort_spec,
    resolve_kaggle_cli_submit_abort_spec,
    resolve_local_submission_guardrail_abort_spec,
)
from kagglebot.submit_attempt_loop import (
    run_submit_stage_attempt,
    run_submit_stage_attempts_until_success_or_abort,
)
from kagglebot.submit_attempts import SubmitAttemptStatePayloads, append_submit_attempt
from kagglebot.submit_cli_error_resolution import (
    SubmitStageRuntimeState,
    apply_notebook_fallback_retry_state,
    build_notebook_fallback_retry_state,
    decide_notebook_fallback_after_file_submit_error,
    decide_submit_stage_error_action,
)
from kagglebot.submit_context import build_submit_run_context, build_submit_runtime_context
from kagglebot.submit_failure_context import load_submit_failure_context, save_submit_failure_context
from kagglebot.submit_gate import decide_fallback_submit_gate, decide_iteration_submit_improvement_gate
from kagglebot.submit_knowledge import (
    build_default_submission_problem_insight,
    classify_submission_outcome,
    ensure_submission_problem_insights,
    record_submission_knowledge,
    record_submission_knowledge_entries,
    resolve_submission_knowledge_context,
    resolve_submission_knowledge_iteration,
)
from kagglebot.submit_message import find_campaign_candidate_for_submission, resolve_submission_message
from kagglebot.submit_outcome import (
    finalize_submit_outcome_for_run_or_abort,
    resolve_submission_outcome_after_submit,
    wait_for_submission_outcome,
)
from kagglebot.submit_outcome_decisions import (
    build_submission_outcome_error_detail,
    decide_submission_outcome_abort,
    evaluate_submission_outcome_after_poll,
    normalize_submission_outcome_status,
)
from kagglebot.submit_preflight import (
    SubmitPreparedSubmissionResolution,
    prepare_submission_for_run_or_abort,
    require_prepared_submission_path,
    resolve_prepared_submission_for_submit,
    resolve_rules_acceptance_for_submit,
)
from kagglebot.submit_rank import (
    format_rank_force_reason,
    format_submission_rank_message,
    resolve_submission_rank_payload,
    resolve_submission_rank_state,
)
from kagglebot.submit_stage import (
    prepare_and_resolve_submit_preflight_for_run_or_abort,
    resolve_submit_preflight_for_run_or_abort,
)
from kagglebot.submit_stage_duplicate import (
    apply_duplicate_submission_decision,
    apply_same_submission_path_decision,
    infer_iteration_from_submission_path,
    resolve_duplicate_submission_for_run,
    resolve_duplicate_submission_for_submit,
    resolve_same_submission_path_for_run,
    resolve_same_submission_path_for_submit,
)
from kagglebot.submit_stage_modes import (
    apply_initial_submit_stage_artifact_mode,
    build_submit_stage_runtime_state,
    decide_initial_submit_stage_mode,
    resolve_initial_submit_stage_runtime_state,
    resolve_iteration_submit_phase_state,
    update_submit_stage_artifact_mode,
)
from kagglebot.submit_success import (
    build_submit_stage_success_record,
    record_successful_submit_for_run,
    record_successful_submit_stage_result,
)
from kagglebot.submit_tracking import decide_submitted_tracking_score_update, submission_score_for_tracking


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


class SamePathDecisionStub:
    def __init__(
        self,
        *,
        action: str,
        reason: str = "",
        message: str = "",
        fingerprint: str = "",
    ) -> None:
        self.action = action
        self.reason = reason
        self.message = message
        self.fingerprint = fingerprint


class DuplicateDecisionStub:
    def __init__(
        self,
        *,
        action: str,
        reason: str = "",
        message: str = "",
        fingerprint: str = "",
        duplicate_sources: list[str] | None = None,
    ) -> None:
        self.action = action
        self.reason = reason
        self.message = message
        self.fingerprint = fingerprint
        self.duplicate_sources = duplicate_sources or []


class ArtifactModeDecisionStub:
    def __init__(self, *, mode: str, message: str = "") -> None:
        self.mode = mode
        self.message = message


class SubmitAttemptRecorderStub:
    def __init__(self) -> None:
        self.payloads: list[object] = []

    def append(self, payload: dict[str, object]) -> None:
        self.payloads.append(payload)

    def record_payloads(self, payloads: object) -> None:
        self.payloads.append(payloads)


class SubmitValidationStubError(ValueError):
    pass


class SubmitCliStubError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stdout: str = "",
        stderr: str = "",
        output: str = "",
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.output = output
        self.exit_code = exit_code


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


def test_decide_initial_submit_stage_mode_preserves_explicit_inference_when_resolver_returns_wrapper() -> None:
    decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=True,
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="inference",
        resolved_notebook_artifact_mode="wrapper",
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


def test_submit_stage_runtime_state_tracks_initial_artifact_and_fallback_updates() -> None:
    initial = decide_initial_submit_stage_mode(
        requested_notebook_submit=False,
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        resolved_notebook_artifact_mode="inference",
    )
    state = build_submit_stage_runtime_state(initial)

    assert state.notebook_submit_required is False
    assert state.notebook_fallback_activated is False
    assert state.submission_artifact_mode == "wrapper"

    state = update_submit_stage_artifact_mode(state, submission_artifact_mode="inference")
    assert state.notebook_submit_required is False
    assert state.notebook_fallback_activated is False
    assert state.submission_artifact_mode == "inference"

    fallback_decision = decide_notebook_fallback_after_file_submit_error(
        notebook_submit_required=False,
        notebook_fallback_activated=False,
        should_use_notebook_fallback=True,
        resolved_notebook_artifact_mode="wrapper",
        current_submission_artifact_mode=state.submission_artifact_mode,
    )
    fallback_state = build_notebook_fallback_retry_state(
        fallback_decision=fallback_decision,
        artifact_mode="wrapper",
        artifact_message="artifact message",
    )
    state = apply_notebook_fallback_retry_state(fallback_state)

    assert state.notebook_submit_required is True
    assert state.notebook_fallback_activated is True
    assert state.submission_artifact_mode == "wrapper"


def test_apply_initial_submit_stage_artifact_mode_emits_messages_and_updates_state() -> None:
    decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=True,
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        resolved_notebook_artifact_mode="inference",
    )
    calls: list[tuple[str, bool]] = []
    messages: list[str] = []

    state = apply_initial_submit_stage_artifact_mode(
        mode_decision=decision,
        resolve_artifact_mode=lambda mode, required: (
            calls.append((mode, required))
            or ArtifactModeDecisionStub(
                mode="inference",
                message="[yellow]submit mode[/yellow]: using inference artifact",
            )
        ),
        on_message=messages.append,
    )

    assert state.notebook_submit_required is True
    assert state.notebook_fallback_activated is True
    assert state.submission_artifact_mode == "inference"
    assert calls == [("inference", True)]
    assert messages == [
        "[yellow]submit mode[/yellow]: using notebook submit",
        "[yellow]submit mode[/yellow]: using inference artifact",
    ]


def test_resolve_initial_submit_stage_runtime_state_keeps_file_submit(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    sample_path = tmp_path / "sample_submission.csv"
    fallback_sample_path = tmp_path / "data_sample_submission.csv"
    calls: list[tuple[str, bool, bool]] = []
    messages: list[str] = []

    def resolve_notebook_mode(**kwargs: object) -> str:
        raise AssertionError(f"notebook mode should not be resolved for file submit: {kwargs}")

    def decide_artifact_mode(**kwargs: object) -> ArtifactModeDecisionStub:
        calls.append(
            (
                str(kwargs["requested_mode"]),
                bool(kwargs["notebook_submit_required"]),
                bool(kwargs["code_competition"]),
            )
        )
        return ArtifactModeDecisionStub(mode="wrapper")

    state = resolve_initial_submit_stage_runtime_state(
        submit_mode="file",
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        code_competition=False,
        sample_submission_path=sample_path,
        fallback_sample_submission_path=fallback_sample_path,
        submission_path=submission_path,
        resolve_notebook_submit_artifact_mode=resolve_notebook_mode,
        decide_notebook_submit_artifact_mode_for_paths=decide_artifact_mode,
        count_tabular_data_rows=lambda path: 3,
        on_message=messages.append,
    )

    assert state.notebook_submit_required is False
    assert state.notebook_fallback_activated is False
    assert state.submission_artifact_mode == "wrapper"
    assert calls == [("wrapper", False, False)]
    assert messages == []


def test_resolve_initial_submit_stage_runtime_state_forces_notebook_only_inference(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    sample_path = tmp_path / "sample_submission.csv"
    fallback_sample_path = tmp_path / "data_sample_submission.csv"
    artifact_calls: list[tuple[str, bool, bool]] = []
    resolver_calls: list[dict[str, object]] = []
    messages: list[str] = []

    def resolve_notebook_mode(**kwargs: object) -> str:
        resolver_calls.append(kwargs)
        return "inference"

    def decide_artifact_mode(**kwargs: object) -> ArtifactModeDecisionStub:
        artifact_calls.append(
            (
                str(kwargs["requested_mode"]),
                bool(kwargs["notebook_submit_required"]),
                bool(kwargs["code_competition"]),
            )
        )
        return ArtifactModeDecisionStub(
            mode="inference",
            message="[yellow]submit mode[/yellow]: tiny notebook sample/submission detected",
        )

    state = resolve_initial_submit_stage_runtime_state(
        submit_mode="file",
        notebook_submissions_only=True,
        notebook_submit_artifact_mode="wrapper",
        code_competition=True,
        sample_submission_path=sample_path,
        fallback_sample_submission_path=fallback_sample_path,
        submission_path=submission_path,
        resolve_notebook_submit_artifact_mode=resolve_notebook_mode,
        decide_notebook_submit_artifact_mode_for_paths=decide_artifact_mode,
        count_tabular_data_rows=lambda path: 3,
        on_message=messages.append,
    )

    assert state.notebook_submit_required is True
    assert state.notebook_fallback_activated is True
    assert state.submission_artifact_mode == "inference"
    assert resolver_calls == [{"submit_mode": "notebook", "code_competition": True}]
    assert artifact_calls == [("inference", True, True)]
    assert messages == [
        "[yellow]submit mode[/yellow]: notebook-only competition detected; forcing notebook submit",
        "[yellow]submit mode[/yellow]: using notebook submit",
        "[yellow]submit mode[/yellow]: tiny notebook sample/submission detected",
    ]


def test_apply_same_submission_path_decision_records_skip_payload(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    recorded: list[dict[str, object]] = []
    messages: list[str] = []

    skipped = apply_same_submission_path_decision(
        decision=SamePathDecisionStub(
            action="skip",
            reason="same_submission_path_reused_in_run",
            message="[yellow]submit skipped[/yellow]: same submission file already attempted in this run",
            fingerprint="known-fp",
        ),
        run_id="run-1",
        submission_path=submission_path,
        compute_submission_sha256=lambda path: "sha" if path == submission_path else None,
        record_submit_attempt=recorded.append,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert skipped is True
    assert messages == ["[yellow]submit skipped[/yellow]: same submission file already attempted in this run"]
    assert recorded == [
        {
            "run_id": "run-1",
            "sub_path": str(submission_path),
            "sub_sha256": "sha",
            "exit_code": None,
            "ok": False,
            "fingerprint": "known-fp",
            "error_kind": "unknown",
            "action_taken": "skip",
            "reason": "same_submission_path_reused_in_run",
            "stdout_tail": "",
            "stderr_tail": "",
        }
    ]


def test_apply_same_submission_path_decision_reports_retry_without_recording(tmp_path: Path) -> None:
    recorded: list[dict[str, object]] = []
    messages: list[str] = []

    skipped = apply_same_submission_path_decision(
        decision=SamePathDecisionStub(action="retry", message="[yellow]submit retry[/yellow]: retrying"),
        run_id="run-1",
        submission_path=tmp_path / "submission.csv",
        compute_submission_sha256=lambda path: "sha",
        record_submit_attempt=recorded.append,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert skipped is False
    assert messages == ["[yellow]submit retry[/yellow]: retrying"]
    assert recorded == []


def test_resolve_same_submission_path_for_submit_records_skip(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    recorded: list[dict[str, object]] = []
    messages: list[str] = []
    policy_calls: list[dict[str, object]] = []

    skipped = resolve_same_submission_path_for_submit(
        run_state={"last_submission_path": str(submission_path), "last_submit_fingerprint": "known-fp"},
        latest_submit_attempt={"sub_sha256": "sha"},
        prepared_submission_path=submission_path,
        current_submission_sha="sha",
        submit_code_fingerprint="code-fp",
        allow_force=False,
        notebook_submit_required=False,
        decide_same_submission_path_action=lambda **kwargs: policy_calls.append(kwargs)
        or SamePathDecisionStub(
            action="skip",
            reason="same_submission_path_reused_in_run",
            message="[yellow]submit skipped[/yellow]: same submission file already attempted in this run",
            fingerprint="known-fp",
        ),
        run_id="run-1",
        compute_submission_sha256=lambda path: "sha" if path == submission_path else None,
        record_submit_attempt=recorded.append,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert skipped is True
    assert policy_calls
    assert policy_calls[0]["prepared_submission_path"] == submission_path
    assert policy_calls[0]["current_submission_sha"] == "sha"
    assert policy_calls[0]["notebook_submit_required"] is False
    assert len(recorded) == 1
    assert recorded[0]["action_taken"] == "skip"
    assert recorded[0]["reason"] == "same_submission_path_reused_in_run"
    assert messages == ["[yellow]submit skipped[/yellow]: same submission file already attempted in this run"]


def test_resolve_same_submission_path_for_submit_skips_policy_for_notebook_submit(tmp_path: Path) -> None:
    recorded: list[dict[str, object]] = []
    messages: list[str] = []

    skipped = resolve_same_submission_path_for_submit(
        run_state={},
        latest_submit_attempt={},
        prepared_submission_path=tmp_path / "submission.csv",
        current_submission_sha="sha",
        submit_code_fingerprint="code-fp",
        allow_force=False,
        notebook_submit_required=True,
        decide_same_submission_path_action=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"same-path policy should not be called: {kwargs}")
        ),
        run_id="run-1",
        compute_submission_sha256=lambda path: "sha",
        record_submit_attempt=recorded.append,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert skipped is False
    assert recorded == []
    assert messages == []


def test_resolve_same_submission_path_for_run_binds_recorder(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    recorder = SubmitAttemptRecorderStub()
    messages: list[str] = []
    policy_calls: list[dict[str, object]] = []

    skipped = resolve_same_submission_path_for_run(
        run_id="run-1",
        run_state={"last_submission_path": str(submission_path), "last_submit_fingerprint": "known-fp"},
        latest_submit_attempt={"sub_sha256": "sha"},
        prepared_submission_path=submission_path,
        current_submission_sha="sha",
        submit_code_fingerprint="code-fp",
        allow_force=False,
        notebook_submit_required=False,
        decide_same_submission_path_action=lambda **kwargs: policy_calls.append(kwargs)
        or SamePathDecisionStub(
            action="skip",
            reason="same_submission_path_reused_in_run",
            message="[yellow]submit skipped[/yellow]: same submission file already attempted in this run",
            fingerprint="known-fp",
        ),
        compute_submission_sha256=lambda path: "sha" if path == submission_path else None,
        submit_attempt_recorder=recorder,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert skipped is True
    assert policy_calls
    assert recorder.payloads[0]["action_taken"] == "skip"
    assert recorder.payloads[0]["reason"] == "same_submission_path_reused_in_run"
    assert messages == ["[yellow]submit skipped[/yellow]: same submission file already attempted in this run"]


def test_apply_duplicate_submission_decision_records_skip_and_returns_payload(tmp_path: Path) -> None:
    prepared_submission_path = tmp_path / "prepared.csv"
    prepared_submission_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    source_submission_path = tmp_path / "iter-6" / "submission.csv"
    recorded_payloads: list[object] = []
    marked: list[tuple[str, str]] = []
    messages: list[str] = []
    submitted_at = datetime(2026, 6, 25, tzinfo=UTC)

    result = apply_duplicate_submission_decision(
        decision=DuplicateDecisionStub(
            action="skip",
            reason="duplicate_submission_sha_seen",
            message="[yellow]submit skipped[/yellow]: duplicate",
            fingerprint="fp",
            duplicate_sources=["run_attempts", "submission_ledger"],
        ),
        run_id="run-1",
        message="submit message",
        submitted_at=submitted_at,
        submission_path=source_submission_path,
        prepared_submission_path=prepared_submission_path,
        prepared_submission_sha="sha",
        code_fingerprint="code-fp",
        prior_state={"submit_attempts_count": 2},
        record_submit_attempt_payloads=recorded_payloads.append,
        mark_duplicate_skipped=lambda submission_ref, reason: marked.append((submission_ref, reason)),
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert len(recorded_payloads) == 1
    payloads = recorded_payloads[0]
    assert payloads.attempt_payload["action_taken"] == "skip"
    assert payloads.attempt_payload["duplicate_sources"] == ["run_attempts", "submission_ledger"]
    assert payloads.run_state_update["submit_attempts_count"] == 3
    assert marked == [(str(prepared_submission_path), "duplicate_submission_sha_seen")]
    assert messages == ["[yellow]submit skipped[/yellow]: duplicate"]
    assert result == {
        "message": "submit message",
        "submission_path": str(prepared_submission_path),
        "submitted_at": submitted_at.isoformat(),
        "iteration": 6,
        "skipped": True,
        "reason": "duplicate_submission_sha_seen",
        "duplicate_sources": ["run_attempts", "submission_ledger"],
    }


def test_apply_duplicate_submission_decision_ignores_non_skip(tmp_path: Path) -> None:
    recorded_payloads: list[object] = []
    marked: list[tuple[str, str]] = []
    messages: list[str] = []

    result = apply_duplicate_submission_decision(
        decision=DuplicateDecisionStub(action="proceed"),
        run_id="run-1",
        message="submit message",
        submitted_at=datetime(2026, 6, 25, tzinfo=UTC),
        submission_path=tmp_path / "iter-1" / "submission.csv",
        prepared_submission_path=tmp_path / "prepared.csv",
        prepared_submission_sha="sha",
        code_fingerprint="code-fp",
        prior_state={},
        record_submit_attempt_payloads=recorded_payloads.append,
        mark_duplicate_skipped=lambda submission_ref, reason: marked.append((submission_ref, reason)),
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert result is None
    assert recorded_payloads == []
    assert marked == []
    assert messages == []


def test_resolve_duplicate_submission_for_submit_records_skip(tmp_path: Path) -> None:
    prepared_submission_path = tmp_path / "prepared.csv"
    prepared_submission_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    source_submission_path = tmp_path / "iter-6" / "submission.csv"
    submitted_at = datetime(2026, 6, 25, tzinfo=UTC)
    recorded_payloads: list[object] = []
    marked: list[tuple[str, str]] = []
    messages: list[str] = []
    collect_calls: list[dict[str, object]] = []
    decide_calls: list[dict[str, object]] = []

    result = resolve_duplicate_submission_for_submit(
        slug="demo",
        run_id="run-1",
        message="submit message",
        submitted_at=submitted_at,
        submission_path=source_submission_path,
        prepared_submission_path=prepared_submission_path,
        prepared_submission_sha="sha",
        code_fingerprint="code-fp",
        allow_force=False,
        prior_state={"submit_attempts_count": 2},
        collect_duplicate_submission_sources=lambda **kwargs: collect_calls.append(kwargs)
        or ["run_attempts", "submission_ledger"],
        decide_duplicate_submission_action=lambda **kwargs: decide_calls.append(kwargs)
        or DuplicateDecisionStub(
            action="skip",
            reason="duplicate_submission_sha_seen",
            message="[yellow]submit skipped[/yellow]: duplicate",
            fingerprint="fp",
            duplicate_sources=["run_attempts", "submission_ledger"],
        ),
        submission_attempt_sha_seen=lambda submission_sha: True,
        submission_ledger_duplicate=lambda: True,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        record_submit_attempt_payloads=recorded_payloads.append,
        mark_duplicate_skipped=lambda submission_ref, reason: marked.append((submission_ref, reason)),
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert collect_calls
    assert collect_calls[0]["prepared_submission_sha"] == "sha"
    assert collect_calls[0]["allow_force"] is False
    assert callable(collect_calls[0]["submission_attempt_sha_seen"])
    assert callable(collect_calls[0]["submission_ledger_duplicate"])
    assert decide_calls
    assert decide_calls[0]["slug"] == "demo"
    assert decide_calls[0]["duplicate_sources"] == ["run_attempts", "submission_ledger"]
    assert callable(decide_calls[0]["compute_fingerprint"])
    assert len(recorded_payloads) == 1
    assert marked == [(str(prepared_submission_path), "duplicate_submission_sha_seen")]
    assert messages == ["[yellow]submit skipped[/yellow]: duplicate"]
    assert result is not None
    assert result["skipped"] is True
    assert result["reason"] == "duplicate_submission_sha_seen"


def test_resolve_duplicate_submission_for_submit_ignores_non_duplicate(tmp_path: Path) -> None:
    recorded_payloads: list[object] = []
    marked: list[tuple[str, str]] = []
    messages: list[str] = []

    result = resolve_duplicate_submission_for_submit(
        slug="demo",
        run_id="run-1",
        message="submit message",
        submitted_at=datetime(2026, 6, 25, tzinfo=UTC),
        submission_path=tmp_path / "iter-1" / "submission.csv",
        prepared_submission_path=tmp_path / "prepared.csv",
        prepared_submission_sha="sha",
        code_fingerprint="code-fp",
        allow_force=False,
        prior_state={},
        collect_duplicate_submission_sources=lambda **kwargs: [],
        decide_duplicate_submission_action=lambda **kwargs: DuplicateDecisionStub(action="proceed"),
        submission_attempt_sha_seen=lambda submission_sha: False,
        submission_ledger_duplicate=lambda: False,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        record_submit_attempt_payloads=recorded_payloads.append,
        mark_duplicate_skipped=lambda submission_ref, reason: marked.append((submission_ref, reason)),
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert result is None
    assert recorded_payloads == []
    assert marked == []
    assert messages == []


def test_resolve_duplicate_submission_for_run_binds_attempt_ledger_and_failure_context(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prepared_submission_path = tmp_path / "prepared.csv"
    prepared_submission_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    source_submission_path = tmp_path / "iter-6" / "submission.csv"
    source_submission_path.parent.mkdir(parents=True)
    source_submission_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    submitted_at = datetime(2026, 6, 25, tzinfo=UTC)
    append_submit_attempt(run_dir=run_dir, payload={"sub_sha256": "sha", "action_taken": "submit"})
    ledger_path = tmp_path / "ledger.jsonl"
    SubmissionLedger(ledger_path).record(
        slug="demo",
        message="submit message",
        submission_path=prepared_submission_path,
        run_id="run-1",
    )
    save_submit_failure_context(
        run_dir,
        {
            "active": True,
            "reason": "previous_submission_error",
            "submission_ref": "old-submission.csv",
        },
    )
    recorded_payloads: list[object] = []
    messages: list[str] = []
    collect_calls: list[dict[str, object]] = []
    load_state_calls: list[Path] = []

    def collect_duplicate_submission_sources(**kwargs: object) -> list[str]:
        collect_calls.append(kwargs)
        sources: list[str] = []
        if kwargs["submission_attempt_sha_seen"]("sha"):
            sources.append("run_attempts")
        if kwargs["submission_ledger_duplicate"]():
            sources.append("submission_ledger")
        return sources

    result = resolve_duplicate_submission_for_run(
        run_dir=run_dir,
        submission_ledger_path=ledger_path,
        slug="demo",
        run_id="run-1",
        message="submit message",
        submitted_at=submitted_at,
        submission_path=source_submission_path,
        prepared_submission_path=prepared_submission_path,
        prepared_submission_sha="sha",
        code_fingerprint="code-fp",
        allow_force=False,
        load_run_state=lambda state_run_dir: load_state_calls.append(state_run_dir) or {"submit_attempts_count": 1},
        collect_duplicate_submission_sources=collect_duplicate_submission_sources,
        decide_duplicate_submission_action=lambda **kwargs: DuplicateDecisionStub(
            action="skip",
            reason="duplicate_submission_sha_seen",
            message="[yellow]submit skipped[/yellow]: duplicate",
            fingerprint="fp",
            duplicate_sources=kwargs["duplicate_sources"],
        ),
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        record_submit_attempt_payloads=recorded_payloads.append,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert collect_calls
    assert load_state_calls == [run_dir]
    assert collect_calls[0]["prepared_submission_sha"] == "sha"
    assert result is not None
    assert result["skipped"] is True
    assert result["duplicate_sources"] == ["run_attempts", "submission_ledger"]
    assert recorded_payloads[0].attempt_payload["duplicate_sources"] == ["run_attempts", "submission_ledger"]
    failure_context = load_submit_failure_context(run_dir)
    assert failure_context["active"] is False
    assert failure_context["resolution"] == "duplicate_submission_sha_seen"
    assert failure_context["resolved_submission_ref"] == str(prepared_submission_path)
    assert messages == ["[yellow]submit skipped[/yellow]: duplicate"]


def test_resolve_submit_preflight_for_run_returns_runtime_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ledger_path = tmp_path / "ledger.jsonl"
    sample_path = tmp_path / "sample_submission.csv"
    fallback_sample_path = tmp_path / "data" / "sample_submission.csv"
    fallback_sample_path.parent.mkdir()
    prepared_submission_path = tmp_path / "prepared.csv"
    source_submission_path = tmp_path / "iter-2" / "submission.csv"
    source_submission_path.parent.mkdir()
    for path in (sample_path, fallback_sample_path, prepared_submission_path, source_submission_path):
        path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    recorder = SubmitAttemptRecorderStub()

    preflight = resolve_submit_preflight_for_run_or_abort(
        run_dir=run_dir,
        submission_ledger_path=ledger_path,
        slug="demo",
        run_id="run-1",
        message="submit message",
        submitted_at=datetime(2026, 6, 25, tzinfo=UTC),
        source_submission_path=source_submission_path,
        prepared_submission_path=prepared_submission_path,
        prepared_submission_sha="sha",
        code_fingerprint="code-fp",
        allow_force=False,
        run_state={},
        latest_submit_attempt={},
        submit_mode="file",
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        code_competition=False,
        sample_submission_path=sample_path,
        fallback_sample_submission_path=fallback_sample_path,
        load_run_state=lambda _run_dir: {},
        collect_duplicate_submission_sources=lambda **_kwargs: [],
        decide_duplicate_submission_action=lambda **_kwargs: DuplicateDecisionStub(action="proceed"),
        check_rules_accepted=lambda: True,
        cli_error_types=(RuntimeError,),
        is_missing_credentials_error=lambda _exc: False,
        rules_not_accepted_exit_code=64,
        resolve_notebook_submit_artifact_mode=lambda **_kwargs: "wrapper",
        decide_notebook_submit_artifact_mode_for_paths=lambda **_kwargs: ArtifactModeDecisionStub(mode="wrapper"),
        count_tabular_data_rows=lambda _path: 1,
        decide_same_submission_path_action=lambda **_kwargs: SamePathDecisionStub(action="retry"),
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        compute_submission_sha256=lambda path: "sha" if path == prepared_submission_path else None,
        submit_aborter=object(),
        submit_attempt_recorder=recorder,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=lambda _message: None,
    )

    assert preflight.duplicate_skip_result is None
    assert not preflight.same_submission_path_skipped
    assert preflight.submit_stage_state is not None
    assert not preflight.submit_stage_state.notebook_submit_required
    assert preflight.submit_stage_state.submission_artifact_mode == "wrapper"
    assert not preflight.code_competition
    assert preflight.seen_fingerprints == set()
    assert recorder.payloads == []


def test_prepare_and_resolve_submit_preflight_returns_prepared_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ledger_path = tmp_path / "ledger.jsonl"
    sample_path = tmp_path / "sample_submission.csv"
    fallback_sample_path = tmp_path / "data" / "sample_submission.csv"
    fallback_sample_path.parent.mkdir()
    input_submission_path = tmp_path / "input.csv"
    prepared_submission_path = tmp_path / "prepared.csv"
    source_submission_path = tmp_path / "iter-2" / "submission.csv"
    source_submission_path.parent.mkdir()
    for path in (
        sample_path,
        fallback_sample_path,
        input_submission_path,
        prepared_submission_path,
        source_submission_path,
    ):
        path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    recorder = SubmitAttemptRecorderStub()

    context = prepare_and_resolve_submit_preflight_for_run_or_abort(
        run_dir=run_dir,
        submission_ledger_path=ledger_path,
        slug="demo",
        run_id="run-1",
        message="submit message",
        submitted_at=datetime(2026, 6, 25, tzinfo=UTC),
        source_submission_path=source_submission_path,
        input_submission_path=input_submission_path,
        validate_and_prepare=lambda path: prepared_submission_path if path == input_submission_path else path,
        validation_error_types=(SubmitValidationStubError,),
        validation_exit_code=65,
        code_fingerprint="code-fp",
        allow_force=False,
        run_state={},
        latest_submit_attempt={},
        submit_mode="file",
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        code_competition=False,
        sample_submission_path=sample_path,
        fallback_sample_submission_path=fallback_sample_path,
        load_run_state=lambda _run_dir: {},
        collect_duplicate_submission_sources=lambda **_kwargs: [],
        decide_duplicate_submission_action=lambda **_kwargs: DuplicateDecisionStub(action="proceed"),
        check_rules_accepted=lambda: True,
        cli_error_types=(RuntimeError,),
        is_missing_credentials_error=lambda _exc: False,
        rules_not_accepted_exit_code=64,
        resolve_notebook_submit_artifact_mode=lambda **_kwargs: "wrapper",
        decide_notebook_submit_artifact_mode_for_paths=lambda **_kwargs: ArtifactModeDecisionStub(mode="wrapper"),
        count_tabular_data_rows=lambda _path: 1,
        decide_same_submission_path_action=lambda **_kwargs: SamePathDecisionStub(action="retry"),
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        compute_submission_sha256=lambda path: "sha" if path == prepared_submission_path else None,
        submit_aborter=object(),
        submit_attempt_recorder=recorder,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        build_error=RuntimeError,
        on_message=lambda _message: None,
    )

    assert context.prepared_context.prepared_submission_path == prepared_submission_path
    assert context.prepared_context.prepared_submission_sha == "sha"
    assert context.preflight_context.duplicate_skip_result is None
    assert context.preflight_context.submit_stage_state is not None
    assert context.preflight_context.submit_stage_state.submission_artifact_mode == "wrapper"


def test_prepare_and_resolve_submit_preflight_skips_static_validation_for_notebook_inference(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ledger_path = tmp_path / "ledger.jsonl"
    sample_path = tmp_path / "sample_submission.csv"
    fallback_sample_path = tmp_path / "data" / "sample_submission.csv"
    fallback_sample_path.parent.mkdir()
    input_submission_path = tmp_path / "submission.csv"
    source_submission_path = tmp_path / "iter-5" / "submission.csv"
    source_submission_path.parent.mkdir()
    for path in (sample_path, fallback_sample_path, input_submission_path, source_submission_path):
        path.write_text("id,prediction\n", encoding="utf-8")
    validate_calls: list[Path] = []
    recorder = SubmitAttemptRecorderStub()

    def fail_validation(path: Path) -> Path:
        validate_calls.append(path)
        raise SubmitValidationStubError("submission has no data rows")

    def fail_static_duplicate_check(**_kwargs: object) -> list[str]:
        raise AssertionError("static duplicate check should be skipped for notebook inference")

    artifact_mode_calls: list[dict[str, object]] = []

    def decide_artifact_mode(**kwargs: object) -> ArtifactModeDecisionStub:
        artifact_mode_calls.append(kwargs)
        return ArtifactModeDecisionStub(mode=str(kwargs["requested_mode"]))

    context = prepare_and_resolve_submit_preflight_for_run_or_abort(
        run_dir=run_dir,
        submission_ledger_path=ledger_path,
        slug="arc-prize-2026-arc-agi-3",
        run_id="run-1",
        message="submit message",
        submitted_at=datetime(2026, 6, 25, tzinfo=UTC),
        source_submission_path=source_submission_path,
        input_submission_path=input_submission_path,
        validate_and_prepare=fail_validation,
        validation_error_types=(SubmitValidationStubError,),
        validation_exit_code=65,
        code_fingerprint="code-fp",
        allow_force=False,
        run_state={},
        latest_submit_attempt={},
        submit_mode="notebook",
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="inference",
        code_competition=True,
        sample_submission_path=sample_path,
        fallback_sample_submission_path=fallback_sample_path,
        load_run_state=lambda _run_dir: {},
        collect_duplicate_submission_sources=fail_static_duplicate_check,
        decide_duplicate_submission_action=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("static duplicate action should be skipped for notebook inference")
        ),
        check_rules_accepted=lambda: True,
        cli_error_types=(RuntimeError,),
        is_missing_credentials_error=lambda _exc: False,
        rules_not_accepted_exit_code=64,
        resolve_notebook_submit_artifact_mode=lambda **_kwargs: "wrapper",
        decide_notebook_submit_artifact_mode_for_paths=decide_artifact_mode,
        count_tabular_data_rows=lambda _path: 0,
        decide_same_submission_path_action=lambda **_kwargs: SamePathDecisionStub(action="retry"),
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        compute_submission_sha256=lambda path: "input-sha" if path == input_submission_path else None,
        submit_aborter=object(),
        submit_attempt_recorder=recorder,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        build_error=RuntimeError,
        on_message=lambda _message: None,
    )

    assert validate_calls == []
    assert context.prepared_context.prepared_submission_path == input_submission_path
    assert context.prepared_context.prepared_submission_sha == "input-sha"
    assert context.preflight_context.submit_stage_state is not None
    assert context.preflight_context.submit_stage_state.notebook_submit_required
    assert context.preflight_context.submit_stage_state.submission_artifact_mode == "inference"
    assert artifact_mode_calls
    assert {call["requested_mode"] for call in artifact_mode_calls} == {"inference"}


def test_resolve_submit_preflight_for_run_returns_duplicate_skip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ledger_path = tmp_path / "ledger.jsonl"
    prepared_submission_path = tmp_path / "prepared.csv"
    source_submission_path = tmp_path / "iter-2" / "submission.csv"
    source_submission_path.parent.mkdir()
    prepared_submission_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    source_submission_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    recorder = SubmitAttemptRecorderStub()

    preflight = resolve_submit_preflight_for_run_or_abort(
        run_dir=run_dir,
        submission_ledger_path=ledger_path,
        slug="demo",
        run_id="run-1",
        message="submit message",
        submitted_at=datetime(2026, 6, 25, tzinfo=UTC),
        source_submission_path=source_submission_path,
        prepared_submission_path=prepared_submission_path,
        prepared_submission_sha="sha",
        code_fingerprint="code-fp",
        allow_force=False,
        run_state={},
        latest_submit_attempt={},
        submit_mode="file",
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        code_competition=False,
        sample_submission_path=tmp_path / "sample_submission.csv",
        fallback_sample_submission_path=tmp_path / "sample_submission.csv",
        load_run_state=lambda _run_dir: {},
        collect_duplicate_submission_sources=lambda **_kwargs: ["run_attempts"],
        decide_duplicate_submission_action=lambda **_kwargs: DuplicateDecisionStub(
            action="skip",
            reason="duplicate_submission_sha_seen",
            message="duplicate",
            fingerprint="fp",
            duplicate_sources=["run_attempts"],
        ),
        check_rules_accepted=lambda: (_ for _ in ()).throw(AssertionError("rules should not run")),
        cli_error_types=(RuntimeError,),
        is_missing_credentials_error=lambda _exc: False,
        rules_not_accepted_exit_code=64,
        resolve_notebook_submit_artifact_mode=lambda **_kwargs: "wrapper",
        decide_notebook_submit_artifact_mode_for_paths=lambda **_kwargs: ArtifactModeDecisionStub(mode="wrapper"),
        count_tabular_data_rows=lambda _path: 1,
        decide_same_submission_path_action=lambda **_kwargs: SamePathDecisionStub(action="retry"),
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        compute_submission_sha256=lambda path: "sha" if path == prepared_submission_path else None,
        submit_aborter=object(),
        submit_attempt_recorder=recorder,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=lambda _message: None,
    )

    assert preflight.duplicate_skip_result is not None
    assert preflight.duplicate_skip_result["skipped"] is True
    assert preflight.submit_stage_state is None
    assert recorder.payloads


def test_build_kaggle_credentials_missing_abort_spec_preserves_error_details() -> None:
    spec = build_kaggle_credentials_missing_abort_spec(
        stdout="stdout text",
        stderr="",
        output="missing kaggle.json",
        exit_code=2,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert spec.fingerprint == "fp:stdout text:missing kaggle.json"
    assert spec.error_kind == "permanent"
    assert spec.reason == "kaggle_credentials_missing"
    assert "Kaggle credentials not configured" in spec.message
    assert spec.stdout_tail == "stdout text"
    assert spec.stderr_tail == "missing kaggle.json"
    assert spec.exit_code == 2


def test_build_rules_not_accepted_abort_spec_sets_manual_blocker_contract() -> None:
    spec = build_rules_not_accepted_abort_spec(
        exit_code=77,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert spec.fingerprint == "fp::rules_not_accepted"
    assert spec.error_kind == "permanent"
    assert spec.reason == "rules_not_accepted"
    assert "Competition rules are not accepted" in spec.message
    assert spec.stdout_tail == ""
    assert spec.stderr_tail == "rules_not_accepted"
    assert spec.exit_code == 77


def test_resolve_rules_acceptance_for_submit_returns_accepted() -> None:
    resolution = resolve_rules_acceptance_for_submit(
        check_rules_accepted=lambda: True,
        cli_error_types=(SubmitCliStubError,),
        is_missing_credentials_error=lambda exc: False,
        rules_not_accepted_exit_code=77,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert resolution.rules_accepted is True
    assert resolution.abort_spec is None


def test_resolve_rules_acceptance_for_submit_maps_not_accepted_to_abort() -> None:
    resolution = resolve_rules_acceptance_for_submit(
        check_rules_accepted=lambda: False,
        cli_error_types=(SubmitCliStubError,),
        is_missing_credentials_error=lambda exc: False,
        rules_not_accepted_exit_code=77,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert resolution.rules_accepted is False
    assert resolution.abort_spec is not None
    assert resolution.abort_spec.reason == "rules_not_accepted"
    assert resolution.abort_spec.exit_code == 77


def test_resolve_rules_acceptance_for_submit_maps_missing_credentials_to_abort() -> None:
    resolution = resolve_rules_acceptance_for_submit(
        check_rules_accepted=lambda: (_ for _ in ()).throw(
            SubmitCliStubError(
                "missing credentials",
                stdout="stdout",
                stderr="",
                output="missing kaggle.json",
                exit_code=2,
            )
        ),
        cli_error_types=(SubmitCliStubError,),
        is_missing_credentials_error=lambda exc: True,
        rules_not_accepted_exit_code=77,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert resolution.rules_accepted is False
    assert resolution.abort_spec is not None
    assert resolution.abort_spec.reason == "kaggle_credentials_missing"
    assert resolution.abort_spec.fingerprint == "fp:stdout:missing kaggle.json"
    assert resolution.abort_spec.exit_code == 2


def test_resolve_rules_acceptance_for_submit_reraises_unmatched_cli_error() -> None:
    try:
        resolve_rules_acceptance_for_submit(
            check_rules_accepted=lambda: (_ for _ in ()).throw(SubmitCliStubError("other cli error")),
            cli_error_types=(SubmitCliStubError,),
            is_missing_credentials_error=lambda exc: False,
            rules_not_accepted_exit_code=77,
            compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        )
    except SubmitCliStubError as exc:
        assert str(exc) == "other cli error"
    else:
        raise AssertionError("SubmitCliStubError was not raised")


def test_build_local_submission_guardrail_abort_spec_sets_local_blocker_contract() -> None:
    spec = build_local_submission_guardrail_abort_spec(
        error=RuntimeError("duplicate submission sha"),
        exit_code=9,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert spec.fingerprint == "fp::duplicate submission sha"
    assert spec.error_kind == "permanent"
    assert spec.reason == "local_submission_guardrail"
    assert spec.message == "Local submission guardrail blocked submit: duplicate submission sha"
    assert spec.stdout_tail == ""
    assert spec.stderr_tail == "duplicate submission sha"
    assert spec.exit_code == 9


def test_resolve_local_submission_guardrail_abort_spec_reads_error_exit_code() -> None:
    error = SubmitCliStubError("rate limited", exit_code=88)

    spec = resolve_local_submission_guardrail_abort_spec(
        error=error,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert spec.reason == "local_submission_guardrail"
    assert spec.stderr_tail == "rate limited"
    assert spec.exit_code == 88


def test_resolve_local_submission_guardrail_abort_spec_uses_default_exit_code() -> None:
    spec = resolve_local_submission_guardrail_abort_spec(
        error=RuntimeError("duplicate submission sha"),
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert spec.reason == "local_submission_guardrail"
    assert spec.exit_code == 1


def test_resolve_kaggle_cli_submit_abort_spec_maps_missing_credentials() -> None:
    error = SubmitCliStubError(
        "missing credentials",
        stdout="stdout",
        stderr="",
        output="missing kaggle.json",
        exit_code=2,
    )

    spec = resolve_kaggle_cli_submit_abort_spec(
        error=error,
        is_missing_credentials_error=lambda exc: True,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert spec is not None
    assert spec.reason == "kaggle_credentials_missing"
    assert spec.fingerprint == "fp:stdout:missing kaggle.json"
    assert spec.exit_code == 2


def test_resolve_kaggle_cli_submit_abort_spec_ignores_other_cli_errors() -> None:
    spec = resolve_kaggle_cli_submit_abort_spec(
        error=SubmitCliStubError("server unavailable"),
        is_missing_credentials_error=lambda exc: False,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert spec is None


def test_build_local_submission_validation_abort_spec_sets_validation_contract() -> None:
    spec = build_local_submission_validation_abort_spec(
        error=ValueError("row count mismatch"),
        exit_code=65,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert spec.fingerprint == "fp::row count mismatch"
    assert spec.error_kind == "validation"
    assert spec.reason == "local_submission_validation_failed"
    assert spec.message == "Local submission validation failed; Kaggle CLI submit is skipped."
    assert spec.stdout_tail == ""
    assert spec.stderr_tail == "row count mismatch"
    assert spec.exit_code == 65


def test_resolve_prepared_submission_for_submit_returns_prepared_path(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    prepared_path = tmp_path / "prepared.csv"

    resolution = resolve_prepared_submission_for_submit(
        input_submission_path=input_path,
        validate_and_prepare=lambda path: prepared_path if path == input_path else path,
        validation_error_types=(SubmitValidationStubError,),
        validation_exit_code=65,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert resolution.prepared_submission_path == prepared_path
    assert resolution.abort_spec is None


def test_prepare_submission_for_run_or_abort_returns_prepared_context(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    prepared_path = tmp_path / "prepared.csv"

    context = prepare_submission_for_run_or_abort(
        input_submission_path=input_path,
        validate_and_prepare=lambda path: prepared_path if path == input_path else path,
        validation_error_types=(SubmitValidationStubError,),
        validation_exit_code=65,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        submit_aborter=object(),
        submit_attempt_recorder=object(),
        code_fingerprint="code-fp",
        compute_submission_sha256=lambda path: "sha" if path == prepared_path else None,
        build_error=RuntimeError,
    )

    assert context.prepared_submission_path == prepared_path
    assert context.prepared_submission_sha == "sha"


def test_prepare_submission_for_run_or_abort_delegates_validation_abort(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    abort_calls: list[dict[str, object]] = []

    class Aborter:
        @staticmethod
        def abort(**kwargs: object) -> None:
            abort_calls.append(kwargs)
            raise RuntimeError("aborted")

    try:
        prepare_submission_for_run_or_abort(
            input_submission_path=input_path,
            validate_and_prepare=lambda path: (_ for _ in ()).throw(SubmitValidationStubError("bad rows")),
            validation_error_types=(SubmitValidationStubError,),
            validation_exit_code=65,
            compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
            submit_aborter=Aborter(),
            submit_attempt_recorder=object(),
            code_fingerprint="code-fp",
            compute_submission_sha256=lambda path: None,
            build_error=RuntimeError,
        )
    except RuntimeError as exc:
        assert str(exc) == "aborted"
    else:
        raise AssertionError("aborter did not raise")

    assert abort_calls
    assert abort_calls[0]["submission_ref"] == input_path
    assert abort_calls[0]["code_fingerprint"] == "code-fp"
    assert abort_calls[0]["fingerprint"] == "fp::bad rows"
    assert abort_calls[0]["reason"] == "local_submission_validation_failed"


def test_require_prepared_submission_path_returns_path(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared.csv"

    assert (
        require_prepared_submission_path(
            SubmitPreparedSubmissionResolution(prepared_submission_path=prepared_path),
            build_error=RuntimeError,
        )
        == prepared_path
    )


def test_require_prepared_submission_path_raises_factory_error() -> None:
    try:
        require_prepared_submission_path(
            SubmitPreparedSubmissionResolution(prepared_submission_path=None),
            build_error=RuntimeError,
        )
    except RuntimeError as exc:
        assert str(exc) == "Submit validation did not produce a prepared submission path."
    else:
        raise AssertionError("RuntimeError was not raised")


def test_resolve_prepared_submission_for_submit_maps_validation_error_to_abort(tmp_path: Path) -> None:
    resolution = resolve_prepared_submission_for_submit(
        input_submission_path=tmp_path / "input.csv",
        validate_and_prepare=lambda path: (_ for _ in ()).throw(SubmitValidationStubError("bad rows")),
        validation_error_types=(SubmitValidationStubError,),
        validation_exit_code=65,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert resolution.prepared_submission_path is None
    assert resolution.abort_spec is not None
    assert resolution.abort_spec.fingerprint == "fp::bad rows"
    assert resolution.abort_spec.error_kind == "validation"
    assert resolution.abort_spec.reason == "local_submission_validation_failed"
    assert resolution.abort_spec.exit_code == 65


def test_resolve_prepared_submission_for_submit_reraises_unmatched_error(tmp_path: Path) -> None:
    try:
        resolve_prepared_submission_for_submit(
            input_submission_path=tmp_path / "input.csv",
            validate_and_prepare=lambda path: (_ for _ in ()).throw(RuntimeError("unexpected")),
            validation_error_types=(SubmitValidationStubError,),
            validation_exit_code=65,
            compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        )
    except RuntimeError as exc:
        assert str(exc) == "unexpected"
    else:
        raise AssertionError("RuntimeError was not raised")


def test_build_submit_abort_spec_kwargs_maps_abort_spec_fields() -> None:
    spec = build_local_submission_validation_abort_spec(
        error=ValueError("bad rows"),
        exit_code=65,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert build_submit_abort_spec_kwargs(spec) == {
        "fingerprint": "fp::bad rows",
        "error_kind": "validation",
        "reason": "local_submission_validation_failed",
        "message": "Local submission validation failed; Kaggle CLI submit is skipped.",
        "stdout_tail": "",
        "stderr_tail": "bad rows",
        "exit_code": 65,
    }


def test_build_submission_polling_error_abort_spec_sets_transient_contract() -> None:
    spec = build_submission_polling_error_abort_spec(
        error=RuntimeError("poll failed"),
        detail="  timeout while polling  ",
        normalize_detail=lambda text: " ".join(text.split()),
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert spec.fingerprint == "fp::timeout while polling"
    assert spec.error_kind == "transient"
    assert spec.reason == "submission_polling_error"
    assert spec.message == "Submission outcome polling failed; aborting submit stage for this run."
    assert spec.stdout_tail == ""
    assert spec.stderr_tail == "timeout while polling"
    assert spec.exit_code is None


def test_build_submission_outcome_abort_spec_maps_decision_contract() -> None:
    decision = decide_submission_outcome_abort(
        outcome_status="error",
        outcome_score=None,
        deliverable_mode="leaderboard",
        raw_detail="bad submission row",
    )

    spec = build_submission_outcome_abort_spec(
        decision=decision,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert spec.fingerprint == "fp::bad submission row"
    assert spec.error_kind == "validation"
    assert spec.reason == "submission_poll_status_error"
    assert "Submission finished with error status" in spec.message
    assert spec.stdout_tail == ""
    assert spec.stderr_tail == "bad submission row"
    assert spec.exit_code is None


def test_build_submit_stage_error_action_abort_spec_maps_action_contract() -> None:
    action = decide_submit_stage_error_action(
        fingerprint_seen=False,
        same_fingerprint_retry_allowed=False,
        classification_kind="permanent",
        classification_reason="bad_request",
        attempt=1,
        max_attempts=2,
        retry_after_seconds=0.0,
        backoff_seconds=1.0,
    )

    spec = build_submit_stage_error_action_abort_spec(
        action=action,
        fingerprint="fp",
        stdout="stdout text",
        stderr="stderr text",
        exit_code=1,
    )

    assert spec.fingerprint == "fp"
    assert spec.error_kind == "permanent"
    assert spec.reason == "bad_request"
    assert spec.message == "Submit failed and is not retryable in this run."
    assert spec.stdout_tail == "stdout text"
    assert spec.stderr_tail == "stderr text"
    assert spec.exit_code == 1


def test_record_submit_stage_retry_attempt_records_attempt_and_knowledge(tmp_path: Path) -> None:
    recorder = SubmitAttemptRecorderStub()
    calls: list[dict[str, object]] = []
    artifact_path = tmp_path / "iter-4" / "submission.csv"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    action = decide_submit_stage_error_action(
        fingerprint_seen=False,
        same_fingerprint_retry_allowed=False,
        classification_kind="transient",
        classification_reason="network_or_timeout",
        attempt=1,
        max_attempts=2,
        retry_after_seconds=0.0,
        backoff_seconds=3.25,
    )

    recorded = record_submit_stage_retry_attempt(
        submit_attempt_recorder=recorder,
        run_id="run-1",
        slug="demo",
        problem_types=["tabular"],
        submission_ref="submission.csv",
        submission_artifact_path=artifact_path,
        fallback_submission_path=tmp_path / "fallback.csv",
        compute_submission_sha256=lambda path: "sha" if path == artifact_path else None,
        exit_code=1,
        fingerprint="fp",
        action=action,
        stdout="abcdef",
        stderr="uvwxyz",
        attempt=1,
        stdout_tail_chars=3,
        stderr_tail_chars=4,
        knowledge_paths=object(),
        normalize_detail=lambda text, max_chars: str(text)[:max_chars],
        record_error_fix_insight=lambda **kwargs: calls.append(kwargs),
    )

    assert recorded is True
    assert recorder.payloads == [
        {
            "run_id": "run-1",
            "sub_path": "submission.csv",
            "sub_sha256": "sha",
            "exit_code": 1,
            "ok": False,
            "fingerprint": "fp",
            "error_kind": "transient",
            "action_taken": "retry",
            "reason": "network_or_timeout",
            "stdout_tail": "def",
            "stderr_tail": "wxyz",
        }
    ]
    assert calls[0]["iteration"] == 4
    assert calls[0]["fix_summary"] == "submit_action=retry; detail=attempt=1; wait=3.2s"


def test_submit_run_retry_recorder_binds_run_callbacks(tmp_path: Path) -> None:
    recorder = SubmitAttemptRecorderStub()
    calls: list[dict[str, object]] = []
    artifact_path = tmp_path / "iter-4" / "submission.csv"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    action = decide_submit_stage_error_action(
        fingerprint_seen=False,
        same_fingerprint_retry_allowed=False,
        classification_kind="transient",
        classification_reason="network_or_timeout",
        attempt=1,
        max_attempts=2,
        retry_after_seconds=0.0,
        backoff_seconds=2.5,
    )
    run_recorder = SubmitRunRetryRecorder(
        submit_attempt_recorder=recorder,
        run_id="run-1",
        slug="demo",
        problem_types=["tabular"],
        knowledge_paths=object(),
        compute_submission_sha256=lambda path: "sha" if path == artifact_path else None,
        stdout_tail_chars=3,
        stderr_tail_chars=4,
        normalize_detail=lambda text, max_chars: str(text)[:max_chars],
        record_error_fix_insight=lambda **kwargs: calls.append(kwargs),
    )

    recorded = run_recorder.record(
        submission_ref="submission.csv",
        submission_artifact_path=artifact_path,
        fallback_submission_path=tmp_path / "fallback.csv",
        exit_code=1,
        fingerprint="fp",
        action=action,
        stdout="abcdef",
        stderr="uvwxyz",
        attempt=1,
    )

    assert recorded is True
    assert recorder.payloads[0]["sub_sha256"] == "sha"
    assert recorder.payloads[0]["reason"] == "network_or_timeout"
    assert recorder.payloads[0]["stdout_tail"] == "def"
    assert recorder.payloads[0]["stderr_tail"] == "wxyz"
    assert calls[0]["iteration"] == 4
    assert calls[0]["fix_summary"] == "submit_action=retry; detail=attempt=1; wait=2.5s"


def test_record_submit_abort_for_run_persists_context_and_records_knowledge(tmp_path: Path) -> None:
    recorder = SubmitAttemptRecorderStub()
    run_dir = tmp_path / "run"
    submission = tmp_path / "iter-5" / "submission.csv"
    submission.parent.mkdir(parents=True)
    submission.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    persisted: dict[str, object] = {}
    insights: list[dict[str, object]] = []
    messages: list[str] = []

    def persist_submit_abort_failure(**kwargs: object) -> None:
        persisted.update(kwargs)

    record_submit_abort_for_run(
        run_dir=run_dir,
        run_id="run-1",
        slug="demo",
        knowledge_paths=object(),
        problem_types=["tabular"],
        submission_ref=submission,
        submission_artifact_path=None,
        artifact_mode="wrapper",
        code_fingerprint="code-fp",
        fingerprint="fp",
        error_kind="validation",
        reason="local_submission_validation_failed",
        message="Local validation failed.",
        stdout_tail="stdout",
        stderr_tail="stderr",
        exit_code=6,
        submit_attempt_recorder=recorder,
        resolve_submit_abort_artifact_path=lambda **kwargs: kwargs["submission_ref"],
        persist_submit_abort_failure=persist_submit_abort_failure,
        load_run_state=lambda _run_dir: {"submit_ok": False},
        load_latest_submit_attempt=lambda _run_dir: {},
        has_successful_submit_attempt=lambda _run_dir: False,
        compute_submission_sha256=lambda path: "sha" if path == submission else None,
        stdout_tail_chars=10,
        stderr_tail_chars=11,
        now_iso="2026-06-25T00:00:00+00:00",
        normalize_detail=lambda text, max_chars: str(text)[:max_chars],
        record_error_fix_insight=lambda **kwargs: insights.append(kwargs),
        on_message=messages.append,
    )

    assert persisted["submission_ref"] == str(submission)
    assert persisted["submission_sha256"] == "sha"
    assert persisted["artifact_path"] == submission
    assert persisted["prior_submit_ok"] is False
    assert persisted["now_iso"] == "2026-06-25T00:00:00+00:00"
    assert insights[0]["iteration"] == 5
    assert insights[0]["fix_summary"] == "submit_action=abort; detail=Local validation failed."
    assert messages == ["[red]submit aborted[/red]: Local validation failed."]


def test_abort_submit_for_run_records_and_raises(tmp_path: Path) -> None:
    recorder = SubmitAttemptRecorderStub()
    run_dir = tmp_path / "run"
    submission = tmp_path / "iter-5" / "submission.csv"
    submission.parent.mkdir(parents=True)
    submission.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    persisted: dict[str, object] = {}
    messages: list[str] = []

    class SubmitAbortStubError(RuntimeError):
        pass

    try:
        abort_submit_for_run(
            run_dir=run_dir,
            run_id="run-1",
            slug="demo",
            knowledge_paths=object(),
            problem_types=["tabular"],
            submission_ref=submission,
            submission_artifact_path=None,
            artifact_mode="wrapper",
            code_fingerprint=None,
            fingerprint="fp",
            error_kind="validation",
            reason="local_submission_validation_failed",
            message="Local validation failed.",
            stdout_tail="stdout",
            stderr_tail="stderr",
            exit_code=6,
            submit_attempt_recorder=recorder,
            save_run_state=lambda _updates: None,
            resolve_submit_abort_artifact_path=lambda **kwargs: kwargs["submission_ref"],
            persist_submit_abort_failure=lambda **kwargs: persisted.update(kwargs),
            load_run_state=lambda _run_dir: {"submit_ok": False},
            load_latest_submit_attempt=lambda _run_dir: {},
            has_successful_submit_attempt=lambda _run_dir: False,
            compute_submission_sha256=lambda path: "sha" if path == submission else None,
            stdout_tail_chars=10,
            stderr_tail_chars=11,
            now_iso="2026-06-25T00:00:00+00:00",
            normalize_detail=lambda text, max_chars: str(text)[:max_chars],
            record_error_fix_insight=lambda **_kwargs: None,
            on_message=messages.append,
            build_error=SubmitAbortStubError,
        )
    except SubmitAbortStubError as exc:
        assert str(exc) == "Local validation failed."
    else:
        raise AssertionError("abort_submit_for_run did not raise")

    assert persisted["submission_ref"] == str(submission)
    assert persisted["submission_sha256"] == "sha"
    assert persisted["code_fingerprint"] == ""
    assert messages == ["[red]submit aborted[/red]: Local validation failed."]


def test_submit_run_aborter_binds_run_callbacks_and_raises(tmp_path: Path) -> None:
    recorder = SubmitAttemptRecorderStub()
    run_dir = tmp_path / "run"
    submission = tmp_path / "iter-5" / "submission.csv"
    submission.parent.mkdir(parents=True)
    submission.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    persisted: dict[str, object] = {}
    messages: list[str] = []

    class SubmitAbortStubError(RuntimeError):
        pass

    aborter = SubmitRunAborter(
        run_dir=run_dir,
        run_id="run-1",
        slug="demo",
        knowledge_paths=object(),
        problem_types=["tabular"],
        save_run_state_for_run=lambda _run_dir, _updates: None,
        resolve_submit_abort_artifact_path=lambda **kwargs: kwargs["submission_ref"],
        persist_submit_abort_failure=lambda **kwargs: persisted.update(kwargs),
        load_run_state=lambda _run_dir: {"submit_ok": False},
        load_latest_submit_attempt=lambda _run_dir: {},
        has_successful_submit_attempt=lambda _run_dir: False,
        compute_submission_sha256=lambda path: "sha" if path == submission else None,
        stdout_tail_chars=10,
        stderr_tail_chars=11,
        now_iso=lambda: "2026-06-25T00:00:00+00:00",
        normalize_detail=lambda text, max_chars: str(text)[:max_chars],
        record_error_fix_insight=lambda **_kwargs: None,
        on_message=messages.append,
        build_error=SubmitAbortStubError,
    )

    try:
        aborter.abort(
            submission_ref=submission,
            submission_artifact_path=None,
            artifact_mode="wrapper",
            code_fingerprint=None,
            fingerprint="fp",
            error_kind="validation",
            reason="local_submission_validation_failed",
            message="Local validation failed.",
            stdout_tail="stdout",
            stderr_tail="stderr",
            exit_code=6,
            submit_attempt_recorder=recorder,
        )
    except SubmitAbortStubError as exc:
        assert str(exc) == "Local validation failed."
    else:
        raise AssertionError("SubmitRunAborter.abort did not raise")

    assert persisted["submission_ref"] == str(submission)
    assert persisted["submission_sha256"] == "sha"
    assert persisted["code_fingerprint"] == ""
    assert persisted["now_iso"] == "2026-06-25T00:00:00+00:00"
    assert messages == ["[red]submit aborted[/red]: Local validation failed."]


def test_build_submit_run_aborter_for_run_wires_standard_failure_helpers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    submission = tmp_path / "iter-5" / "submission.csv"
    submission.parent.mkdir(parents=True)
    submission.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    saved_updates: list[dict[str, object]] = []
    messages: list[str] = []

    class SubmitAbortStubError(RuntimeError):
        pass

    aborter = build_submit_run_aborter_for_run(
        run_dir=run_dir,
        run_id="run-1",
        slug="demo",
        knowledge_paths=object(),
        problem_types=["tabular"],
        save_run_state_for_run=lambda _run_dir, updates: saved_updates.append(updates),
        load_run_state=lambda _run_dir: {"submit_ok": False},
        compute_submission_sha256=lambda path: "sha" if path == submission else None,
        stdout_tail_chars=10,
        stderr_tail_chars=11,
        now_iso=lambda: "2026-06-25T00:00:00+00:00",
        normalize_detail=lambda text, max_chars: str(text)[:max_chars],
        record_error_fix_insight=lambda **_kwargs: None,
        on_message=messages.append,
        build_error=SubmitAbortStubError,
    )

    try:
        aborter.abort(
            submission_ref=submission,
            submission_artifact_path=None,
            artifact_mode="wrapper",
            code_fingerprint=None,
            fingerprint="fp",
            error_kind="validation",
            reason="local_submission_validation_failed",
            message="Local validation failed.",
            stdout_tail="stdout",
            stderr_tail="stderr",
            exit_code=6,
            submit_attempt_recorder=None,
        )
    except SubmitAbortStubError as exc:
        assert str(exc) == "Local validation failed."
    else:
        raise AssertionError("aborter did not raise")

    context = load_submit_failure_context(run_dir)
    assert context["submission_ref"] == str(submission)
    assert context["submission_artifact_sha256"] == "sha"
    assert context["reason"] == "local_submission_validation_failed"
    assert context["artifact_mode"] == "wrapper"
    assert saved_updates
    assert saved_updates[0]["last_reason"] == "local_submission_validation_failed"
    assert messages == ["[red]submit aborted[/red]: Local validation failed."]


def test_build_submit_run_context_wires_attempt_autofix_and_retry_helpers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    submission = tmp_path / "iter-2" / "submission.csv"
    submission.parent.mkdir(parents=True)
    submission.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    src_root = tmp_path / "src"
    kernel_source_dir = tmp_path / "kernel"
    src_root.mkdir()
    kernel_source_dir.mkdir()
    saved_updates: list[dict[str, object]] = []
    fingerprint_calls: list[dict[str, object]] = []

    class SubmitAbortStubError(RuntimeError):
        pass

    context = build_submit_run_context(
        run_dir=run_dir,
        run_id="run-1",
        slug="demo",
        submission_path=submission,
        src_root=src_root,
        kernel_source_dir=kernel_source_dir,
        knowledge_paths=object(),
        problem_types=["tabular"],
        force_submit=False,
        force_resubmit=True,
        save_run_state_for_run=lambda _run_dir, updates: saved_updates.append(updates),
        load_run_state=lambda _run_dir: {"status": "running"},
        compute_submit_code_fingerprint=lambda **kwargs: fingerprint_calls.append(kwargs) or "code-fp",
        compute_submission_sha256=lambda path: "sha" if path == submission else None,
        stdout_tail_chars=10,
        stderr_tail_chars=11,
        now_iso=lambda: "2026-06-25T00:00:00+00:00",
        normalize_detail=lambda text, max_chars: str(text)[:max_chars],
        record_error_fix_insight=lambda **_kwargs: None,
        on_message=lambda _message: None,
        build_error=SubmitAbortStubError,
    )

    assert context.submit_attempt_recorder is not None
    assert context.input_submission_path == submission
    assert context.run_state == {"status": "running"}
    assert context.latest_submit_attempt == {}
    assert context.submit_code_fingerprint == "code-fp"
    assert context.allow_force is True
    assert context.submit_aborter.run_id == "run-1"
    assert context.submit_retry_recorder.run_id == "run-1"
    assert fingerprint_calls == [
        {
            "src_root": src_root,
            "kernel_source_dir": kernel_source_dir,
            "sha256_or_none": context.submit_retry_recorder.compute_submission_sha256,
        }
    ]
    assert saved_updates == []


def test_build_submit_runtime_context_wires_message_service_and_timestamp(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir()
    data_dir.mkdir()
    sample_path = data_dir / "sample_submission.csv"
    ledger_path = tmp_path / "ledger.jsonl"
    submission = tmp_path / "iter-3" / "submission.csv"
    submission.parent.mkdir(parents=True)
    sample_path.write_text("id,pred\n1,0\n", encoding="utf-8")
    submission.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    submitted_at = datetime(2026, 6, 26, tzinfo=UTC)
    messages: list[str] = []

    context = build_submit_runtime_context(
        slug="demo",
        context_dir=context_dir,
        run_id="run-1",
        best_score=0.42,
        explicit_message="",
        submission_path=submission,
        campaign_mode="off",
        target_direction="max",
        data_dir=data_dir,
        sample_submission_path=sample_path,
        submission_ledger_path=ledger_path,
        dry_run=False,
        force_submit=True,
        now=lambda: submitted_at,
        on_message=messages.append,
    )

    assert "run-1" in context.message
    assert "offline=0.4200" in context.message
    assert context.submitted_at == submitted_at
    assert context.submission_service._config.slug == "demo"  # noqa: SLF001
    assert context.submission_service._config.force_submit is True  # noqa: SLF001
    assert messages == ["[cyan]submit[/cyan]: demo"]


def test_submit_run_aborter_binds_run_state_save_for_created_recorder(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    submission = tmp_path / "iter-5" / "submission.csv"
    submission.parent.mkdir(parents=True)
    submission.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    saved_updates: list[tuple[Path, dict[str, object]]] = []

    class SubmitAbortStubError(RuntimeError):
        pass

    def persist_submit_abort_failure(**kwargs: object) -> None:
        recorder = kwargs["submit_attempt_recorder"]
        recorder.record_payloads(
            SubmitAttemptStatePayloads(
                attempt_payload={"run_id": "run-1", "ok": False},
                run_state_update={"submit_ok": False, "last_error_kind": "validation"},
            )
        )

    aborter = SubmitRunAborter(
        run_dir=run_dir,
        run_id="run-1",
        slug="demo",
        knowledge_paths=object(),
        problem_types=["tabular"],
        save_run_state_for_run=lambda run_path, updates: saved_updates.append((run_path, updates)),
        resolve_submit_abort_artifact_path=lambda **kwargs: kwargs["submission_ref"],
        persist_submit_abort_failure=persist_submit_abort_failure,
        load_run_state=lambda _run_dir: {"submit_ok": False},
        load_latest_submit_attempt=lambda _run_dir: {},
        has_successful_submit_attempt=lambda _run_dir: False,
        compute_submission_sha256=lambda path: "sha" if path == submission else None,
        stdout_tail_chars=10,
        stderr_tail_chars=11,
        now_iso=lambda: "2026-06-25T00:00:00+00:00",
        normalize_detail=lambda text, max_chars: str(text)[:max_chars],
        record_error_fix_insight=lambda **_kwargs: None,
        on_message=lambda _message: None,
        build_error=SubmitAbortStubError,
    )

    try:
        aborter.abort(
            submission_ref=submission,
            submission_artifact_path=None,
            artifact_mode="wrapper",
            code_fingerprint=None,
            fingerprint="fp",
            error_kind="validation",
            reason="local_submission_validation_failed",
            message="Local validation failed.",
            stdout_tail="stdout",
            stderr_tail="stderr",
            exit_code=6,
            submit_attempt_recorder=None,
        )
    except SubmitAbortStubError:
        pass
    else:
        raise AssertionError("SubmitRunAborter.abort did not raise")

    assert saved_updates == [(run_dir, {"submit_ok": False, "last_error_kind": "validation"})]
    rows = load_jsonl_records(run_dir / "submit_attempts.jsonl")
    assert rows[0]["run_id"] == "run-1"


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


def test_resolve_submission_outcome_after_submit_returns_normalized_outcome() -> None:
    submitted_at = datetime(2026, 1, 1, tzinfo=UTC)

    resolution = resolve_submission_outcome_after_submit(
        slug="demo",
        message="submission message",
        submitted_at=submitted_at,
        deliverable_mode="leaderboard",
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
        normalize_detail=lambda text: text,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert isinstance(resolution.outcome, dict)
    assert resolution.outcome["status"] == "complete"
    assert resolution.outcome["score"] == 0.123
    assert resolution.abort_spec is None


def test_resolve_submission_outcome_after_submit_maps_scoreless_leaderboard_to_abort() -> None:
    submitted_at = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {
            "description": "submission message",
            "date": "2026-01-01T00:01:00Z",
            "status": "SubmissionStatus.COMPLETE",
            "publicScore": "",
            "errorDescription": "Submission Scoring Error: incorrect format",
        }
    ]

    resolution = resolve_submission_outcome_after_submit(
        slug="demo",
        message="submission message",
        submitted_at=submitted_at,
        deliverable_mode="leaderboard",
        fetch_submission_rows=lambda slug: rows,
        max_attempts=1,
        poll_interval_sec=0.0,
        max_fetch_errors=1,
        normalize_detail=lambda text: text,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert isinstance(resolution.outcome, dict)
    assert resolution.outcome["status"] == "complete"
    assert resolution.abort_spec is not None
    assert resolution.abort_spec.error_kind == "validation"
    assert resolution.abort_spec.reason == "submission_poll_status_complete_no_score"
    assert "Submission Scoring Error" in resolution.abort_spec.stderr_tail


def test_resolve_submission_outcome_after_submit_maps_polling_error_to_abort() -> None:
    submitted_at = datetime(2026, 1, 1, tzinfo=UTC)

    resolution = resolve_submission_outcome_after_submit(
        slug="demo",
        message="submission message",
        submitted_at=submitted_at,
        deliverable_mode="leaderboard",
        fetch_submission_rows=lambda slug: (_ for _ in ()).throw(RuntimeError("network unavailable")),
        max_attempts=1,
        poll_interval_sec=0.0,
        max_fetch_errors=1,
        normalize_detail=lambda text: " ".join(text.split()),
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
    )

    assert resolution.outcome is None
    assert resolution.abort_spec is not None
    assert resolution.abort_spec.error_kind == "transient"
    assert resolution.abort_spec.reason == "submission_polling_error"
    assert "network unavailable" in resolution.abort_spec.stderr_tail


def test_finalize_submit_outcome_for_run_or_abort_records_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    source_submission = run_dir / "iter-4" / "submission.csv"
    source_submission.parent.mkdir(parents=True)
    source_submission.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    artifact_path = tmp_path / "kernel-output" / "submission.csv"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    recorded_payloads: list[object] = []
    messages: list[str] = []

    result = finalize_submit_outcome_for_run_or_abort(
        run_dir=run_dir,
        submission_ledger_path=tmp_path / "ledger.jsonl",
        slug="demo",
        run_id="run-1",
        message="submission message",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
        submission_ref="kernel:user/demo/submission.csv",
        submission_result=SubmitResultStub(stdout="ok", stderr="", exit_code=0),
        source_submission_path=source_submission,
        submission_artifact_path=artifact_path,
        submit_stage_state=SubmitStageRuntimeState(
            notebook_submit_required=True,
            notebook_fallback_activated=True,
            submission_artifact_mode="inference",
        ),
        code_fingerprint="code-fp",
        deliverable_mode="leaderboard",
        fetch_submission_rows=lambda _slug: [
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
        normalize_detail=lambda text: text,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        compute_submission_sha256=lambda path: "sha" if path == artifact_path else None,
        load_run_state=lambda _run_dir: {},
        record_submit_attempt_payloads=recorded_payloads.append,
        submit_aborter=object(),
        submit_attempt_recorder=object(),
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert result["submission_path"] == "kernel:user/demo/submission.csv"
    assert result["outcome"]["score"] == 0.123
    assert recorded_payloads
    assert messages == [
        "[green]submission recorded[/green]",
        "[cyan]submission result[/cyan]: status=complete score=0.123000",
    ]


def test_finalize_submit_outcome_for_run_or_abort_delegates_outcome_abort(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    artifact_path = tmp_path / "kernel-output" / "submission.csv"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("id,pred\n1,0.1\n", encoding="utf-8")
    abort_calls: list[dict[str, object]] = []

    class Aborter:
        @staticmethod
        def abort(**kwargs: object) -> None:
            abort_calls.append(kwargs)
            raise RuntimeError("aborted")

    try:
        finalize_submit_outcome_for_run_or_abort(
            run_dir=run_dir,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            slug="demo",
            run_id="run-1",
            message="submission message",
            submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
            submission_ref="kernel:user/demo/submission.csv",
            submission_result=SubmitResultStub(stdout="ok", stderr="", exit_code=0),
            source_submission_path=run_dir / "iter-4" / "submission.csv",
            submission_artifact_path=artifact_path,
            submit_stage_state=SubmitStageRuntimeState(
                notebook_submit_required=True,
                notebook_fallback_activated=True,
                submission_artifact_mode="inference",
            ),
            code_fingerprint="code-fp",
            deliverable_mode="leaderboard",
            fetch_submission_rows=lambda _slug: [
                {
                    "description": "submission message",
                    "date": "2026-01-01T00:01:00Z",
                    "status": "SubmissionStatus.COMPLETE",
                    "publicScore": "",
                    "errorDescription": "Submission Scoring Error",
                }
            ],
            max_attempts=1,
            poll_interval_sec=0.0,
            max_fetch_errors=1,
            normalize_detail=lambda text: text,
            compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
            compute_submission_sha256=lambda path: "sha" if path == artifact_path else None,
            load_run_state=lambda _run_dir: {},
            record_submit_attempt_payloads=lambda _payload: None,
            submit_aborter=Aborter(),
            submit_attempt_recorder=object(),
            stdout_tail_chars=20,
            stderr_tail_chars=20,
            on_message=lambda _message: None,
        )
    except RuntimeError as exc:
        assert str(exc) == "aborted"
    else:
        raise AssertionError("aborter did not raise")

    assert abort_calls
    assert abort_calls[0]["submission_ref"] == "kernel:user/demo/submission.csv"
    assert abort_calls[0]["submission_artifact_path"] == artifact_path
    assert abort_calls[0]["artifact_mode"] == "inference"
    assert abort_calls[0]["reason"] == "submission_poll_status_complete_no_score"


def test_infer_iteration_from_submission_path_reads_iter_parent() -> None:
    assert infer_iteration_from_submission_path(Path("runs/run-1/iter-3/submission.csv")) == 3
    assert infer_iteration_from_submission_path(Path("runs/run-1/iter-4/output/submission.csv")) == 4
    assert infer_iteration_from_submission_path(Path("runs/run-1/iter-5/output/predictions.npy")) == 5
    assert infer_iteration_from_submission_path(Path("submission.csv")) is None


def test_submission_score_for_tracking_prefers_finite_online_score() -> None:
    assert submission_score_for_tracking(offline_score=0.9, online_score=0.8) == (0.8, "submission_public_score")
    assert submission_score_for_tracking(offline_score=0.9, online_score=float("nan")) == (0.9, "offline")
    assert submission_score_for_tracking(offline_score=0.9, online_score=None) == (0.9, "offline")


def test_decide_submitted_tracking_score_update_prefers_online_score() -> None:
    decision = decide_submitted_tracking_score_update(
        submission_result={"outcome": {"score": "0.8"}},
        offline_score=0.9,
        previous_best_score=None,
        direction="minimize",
    )

    assert decision.online_score == 0.8
    assert decision.tracking_score == 0.8
    assert decision.tracking_source == "submission_public_score"
    assert decision.update_best_submitted_score is True
    assert decision.best_submitted_score == 0.8


def test_decide_submitted_tracking_score_update_falls_back_to_offline_score() -> None:
    decision = decide_submitted_tracking_score_update(
        submission_result={"outcome": {"score": ""}},
        offline_score=0.9,
        previous_best_score=1.0,
        direction="minimize",
    )

    assert decision.online_score is None
    assert decision.tracking_score == 0.9
    assert decision.tracking_source == "offline"
    assert decision.update_best_submitted_score is True
    assert decision.best_submitted_score == 0.9


def test_decide_submitted_tracking_score_update_keeps_better_previous_score() -> None:
    decision = decide_submitted_tracking_score_update(
        submission_result={"outcome": {"score": "0.95"}},
        offline_score=0.9,
        previous_best_score=0.8,
        direction="minimize",
    )

    assert decision.tracking_score == 0.95
    assert decision.update_best_submitted_score is False
    assert decision.best_submitted_score == 0.8


def test_decide_submitted_tracking_score_update_noops_without_offline_score() -> None:
    decision = decide_submitted_tracking_score_update(
        submission_result={"outcome": {"score": "0.8"}},
        offline_score=None,
        previous_best_score=0.7,
        direction="minimize",
    )

    assert decision.tracking_score is None
    assert decision.tracking_source == "unavailable"
    assert decision.update_best_submitted_score is False
    assert decision.best_submitted_score == 0.7


def test_decide_fallback_submit_gate_blocks_improved_policy_without_prior_submission() -> None:
    decision = decide_fallback_submit_gate(
        submit_improved_only=True,
        force_submit=False,
        require_submit_improvement=True,
        best_submittable_score=0.8,
        best_submitted_score=None,
        direction="minimize",
        min_improvement=0.001,
        final_iteration_reached=True,
    )

    assert decision.allow_submit is False
    assert decision.message == ""


def test_decide_fallback_submit_gate_requires_improvement_when_configured() -> None:
    decision = decide_fallback_submit_gate(
        submit_improved_only=False,
        force_submit=False,
        require_submit_improvement=True,
        best_submittable_score=0.9,
        best_submitted_score=0.8,
        direction="minimize",
        min_improvement=0.001,
        final_iteration_reached=False,
    )

    assert decision.allow_submit is False
    assert decision.message == ""


def test_decide_fallback_submit_gate_allows_final_iteration_override_for_non_improved_policy() -> None:
    decision = decide_fallback_submit_gate(
        submit_improved_only=False,
        force_submit=False,
        require_submit_improvement=True,
        best_submittable_score=0.9,
        best_submitted_score=0.8,
        direction="minimize",
        min_improvement=0.001,
        final_iteration_reached=True,
    )

    assert decision.allow_submit is True
    assert "final iteration reached" in decision.message


def test_decide_fallback_submit_gate_allows_force_submit() -> None:
    decision = decide_fallback_submit_gate(
        submit_improved_only=True,
        force_submit=True,
        require_submit_improvement=True,
        best_submittable_score=0.9,
        best_submitted_score=0.8,
        direction="minimize",
        min_improvement=0.001,
        final_iteration_reached=False,
    )

    assert decision.allow_submit is True
    assert decision.message == ""


def test_decide_iteration_submit_improvement_gate_defers_without_prior_submission() -> None:
    decision = decide_iteration_submit_improvement_gate(
        submit_improved_only=True,
        force_submit=False,
        require_submit_improvement=True,
        best_submitted_score=None,
        current_score=0.8,
        direction="minimize",
        min_improvement=0.001,
        final_iteration=False,
        submit_enabled=True,
        quality_allows_submit=False,
        spare_daily_submission_slot=False,
        submission_limit_per_day=3,
        forced_submit_reason=None,
        spare_submit_reason="spare_daily_submission_slot",
    )

    assert decision.submit_improvement_allowed is False
    assert decision.submit_non_improving is True
    assert "prior submitted checkpoint" in decision.message


def test_decide_iteration_submit_improvement_gate_allows_spare_slot_without_prior_submission() -> None:
    decision = decide_iteration_submit_improvement_gate(
        submit_improved_only=True,
        force_submit=False,
        require_submit_improvement=True,
        best_submitted_score=None,
        current_score=0.8,
        direction="minimize",
        min_improvement=0.001,
        final_iteration=False,
        submit_enabled=True,
        quality_allows_submit=True,
        spare_daily_submission_slot=True,
        submission_limit_per_day=3,
        forced_submit_reason=None,
        spare_submit_reason="spare_daily_submission_slot",
    )

    assert decision.submit_improvement_allowed is True
    assert decision.submit_non_improving is False
    assert decision.forced_submit_reason == "spare_daily_submission_slot"
    assert "allowing submit without a prior submitted checkpoint" in decision.message


def test_decide_iteration_submit_improvement_gate_defers_non_improving_score() -> None:
    decision = decide_iteration_submit_improvement_gate(
        submit_improved_only=False,
        force_submit=False,
        require_submit_improvement=True,
        best_submitted_score=0.8,
        current_score=0.9,
        direction="minimize",
        min_improvement=0.001,
        final_iteration=False,
        submit_enabled=True,
        quality_allows_submit=True,
        spare_daily_submission_slot=False,
        submission_limit_per_day=3,
        forced_submit_reason=None,
        spare_submit_reason="spare_daily_submission_slot",
    )

    assert decision.submit_improvement_allowed is False
    assert decision.submit_non_improving is True
    assert "score did not improve" in decision.message


def test_decide_iteration_submit_improvement_gate_allows_final_iteration_override() -> None:
    decision = decide_iteration_submit_improvement_gate(
        submit_improved_only=False,
        force_submit=False,
        require_submit_improvement=True,
        best_submitted_score=0.8,
        current_score=0.9,
        direction="minimize",
        min_improvement=0.001,
        final_iteration=True,
        submit_enabled=True,
        quality_allows_submit=True,
        spare_daily_submission_slot=False,
        submission_limit_per_day=3,
        forced_submit_reason=None,
        spare_submit_reason="spare_daily_submission_slot",
    )

    assert decision.submit_improvement_allowed is True
    assert decision.submit_non_improving is False
    assert "final iteration reached" in decision.message


def test_decide_iteration_submit_improvement_gate_allows_improved_score() -> None:
    decision = decide_iteration_submit_improvement_gate(
        submit_improved_only=False,
        force_submit=False,
        require_submit_improvement=True,
        best_submitted_score=0.8,
        current_score=0.7,
        direction="minimize",
        min_improvement=0.001,
        final_iteration=False,
        submit_enabled=True,
        quality_allows_submit=True,
        spare_daily_submission_slot=False,
        submission_limit_per_day=3,
        forced_submit_reason=None,
        spare_submit_reason="spare_daily_submission_slot",
    )

    assert decision.submit_improvement_allowed is True
    assert decision.submit_non_improving is False
    assert decision.message == ""


def test_classify_submission_outcome_uses_target_score() -> None:
    assert classify_submission_outcome(score=0.4, direction="minimize", target_score=0.5, top1_score=None) == "good"
    assert classify_submission_outcome(score=0.4, direction="MINIMIZE", target_score=0.5, top1_score=None) == "good"
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


def test_record_submission_knowledge_prepares_default_and_records_entries() -> None:
    problem_calls: list[dict[str, object]] = []
    error_calls: list[dict[str, object]] = []
    pending_problem_insights: list[dict[str, object]] = []
    pending_error_fixes = [{"iteration": "", "error_message": "bad csv", "fix_summary": "fixed", "resolved": True}]

    recorded = record_submission_knowledge(
        knowledge_paths=object(),
        slug="demo",
        run_id="run-1",
        problem_types=["tabular"],
        pending_problem_insights=pending_problem_insights,
        pending_error_fixes=pending_error_fixes,
        submission_result={"outcome": {"score": "0.42"}, "iteration": 3},
        metric_direction="minimize",
        target_score=0.5,
        top1_score=None,
        load_diagnostics_text=lambda iteration: f"diagnostics for {iteration}",
        record_problem_type_insight=lambda **kwargs: problem_calls.append(kwargs),
        record_error_fix_insight=lambda **kwargs: error_calls.append(kwargs),
    )

    assert recorded is True
    assert pending_problem_insights[0]["why_poor"] == "diagnostics for 3"
    assert problem_calls[0]["iteration"] == 3
    assert problem_calls[0]["submission_score"] == 0.42
    assert error_calls[0]["iteration"] == 3
    assert error_calls[0]["resolved"] is True


def test_record_submission_knowledge_skips_invalid_result_without_loading_diagnostics() -> None:
    recorded = record_submission_knowledge(
        knowledge_paths=object(),
        slug="demo",
        run_id="run-1",
        problem_types=["tabular"],
        pending_problem_insights=[],
        pending_error_fixes=[],
        submission_result={"outcome": {"status": "complete"}, "iteration": 3},
        metric_direction="minimize",
        target_score=0.5,
        top1_score=None,
        load_diagnostics_text=lambda _iteration: (_ for _ in ()).throw(AssertionError("should not load")),
        record_problem_type_insight=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not record")),
        record_error_fix_insight=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not record")),
    )

    assert recorded is False


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


def test_resolve_submission_rank_state_formats_observed_rank_and_guard() -> None:
    calls: list[dict[str, object]] = []

    def force_by_rank(**kwargs: object) -> bool:
        calls.append(kwargs)
        return True

    state = resolve_submission_rank_state(
        rank_payload={"rank": "120", "total_teams": "1000", "rank_source": "submission_row"},
        rank_force_major_max_percentile=0.01,
        rank_force_major_min_teams=200,
        should_force_major_overhaul_by_rank=force_by_rank,
    )

    assert state.rank == 120
    assert state.total_teams == 1000
    assert state.rank_percentile == 0.12
    assert state.rank_source == "submission_row"
    assert state.force_major_overhaul is True
    assert state.force_reason == (
        "Leaderboard rank indicates large headroom for improvement: "
        "120/1000 (percentile=12.00%, threshold=1.00%, min_teams=200). source=submission_row"
    )
    assert state.messages == (
        "[cyan]submission rank[/cyan]: 120/1000 (percentile=12.00%) source=submission_row",
        "[yellow]rank guard[/yellow]: "
        "Leaderboard rank indicates large headroom for improvement: "
        "120/1000 (percentile=12.00%, threshold=1.00%, min_teams=200). source=submission_row",
    )
    assert calls == [
        {
            "rank": 120,
            "total_teams": 1000,
            "max_percentile": 0.01,
            "min_teams": 200,
        }
    ]


def test_resolve_submission_rank_state_formats_estimated_rank_without_guard() -> None:
    def force_by_rank(**kwargs: object) -> bool:
        raise AssertionError("estimated rank should not trigger observed-rank guard")

    state = resolve_submission_rank_state(
        rank_payload={
            "estimated_rank": "20",
            "estimated_total_teams": "200",
            "rank_estimate_source": "leaderboard_score_estimate",
        },
        rank_force_major_max_percentile=0.01,
        rank_force_major_min_teams=200,
        should_force_major_overhaul_by_rank=force_by_rank,
    )

    assert state.rank is None
    assert state.estimated_rank == 20
    assert state.estimated_total_teams == 200
    assert state.estimated_rank_percentile == 0.1
    assert state.rank_estimate_source == "leaderboard_score_estimate"
    assert state.force_major_overhaul is False
    assert state.messages == (
        "[yellow]submission rank estimate[/yellow]: 20/200 (percentile=10.00%) source=leaderboard_score_estimate",
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


def test_evaluate_submission_outcome_after_poll_normalizes_and_aborts_scoreless_leaderboard() -> None:
    rows = [
        {
            "description": "target message",
            "status": "SubmissionStatus.COMPLETE",
            "publicScore": "",
            "errorDescription": "Submission Scoring Error: incorrect format",
            "date": "2026-06-25T00:41:17Z",
        }
    ]

    decision = evaluate_submission_outcome_after_poll(
        slug="demo",
        message="target message",
        submitted_at=datetime(2026, 6, 25, tzinfo=UTC),
        outcome={"status": "SubmissionStatus.COMPLETE", "score": None, "raw": {"status": "complete"}},
        deliverable_mode="leaderboard",
        fetch_submission_rows=lambda slug: rows,
        normalize_detail=lambda text: text,
    )

    assert isinstance(decision.outcome, dict)
    assert decision.outcome["status"] == "complete"
    assert decision.abort_decision.should_abort is True
    assert decision.abort_decision.reason == "submission_poll_status_complete_no_score"
    assert "Submission Scoring Error" in decision.abort_decision.detail


def test_evaluate_submission_outcome_after_poll_allows_writeup_and_non_dict_outcomes() -> None:
    writeup = evaluate_submission_outcome_after_poll(
        slug="demo",
        message="target message",
        submitted_at=datetime(2026, 6, 25, tzinfo=UTC),
        outcome={"status": "SubmissionStatus.COMPLETE", "score": None},
        deliverable_mode="writeup",
        fetch_submission_rows=lambda slug: [],
        normalize_detail=lambda text: text,
    )
    missing = evaluate_submission_outcome_after_poll(
        slug="demo",
        message="target message",
        submitted_at=datetime(2026, 6, 25, tzinfo=UTC),
        outcome=None,
        deliverable_mode="leaderboard",
        fetch_submission_rows=lambda slug: [],
        normalize_detail=lambda text: text,
    )

    assert writeup.outcome == {"status": "complete", "score": None}
    assert writeup.abort_decision.should_abort is False
    assert missing.outcome is None
    assert missing.abort_decision.should_abort is False


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


def test_run_submit_stage_attempts_until_success_returns_file_submit_result(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared.csv"
    submitted_path = tmp_path / "submitted.csv"

    result = run_submit_stage_attempts_until_success_or_abort(
        run_dir=tmp_path,
        run_id="run-1",
        state=SubmitStageRuntimeState(False, False, "wrapper"),
        prepared_submission_path=prepared_path,
        message="submit message",
        code_competition=False,
        max_attempts=2,
        backoff_base_seconds=0.0,
        sample_submission_path=tmp_path / "sample_submission.csv",
        fallback_sample_submission_path=tmp_path / "data" / "sample_submission.csv",
        submit_code_fingerprint="code-fp",
        run_state={},
        seen_fingerprints=set(),
        run_notebook_submit=lambda _state: (_ for _ in ()).throw(AssertionError("notebook should not run")),
        run_file_submit=lambda: FileSubmitResult(submitted_path),
        submit_aborter=object(),
        submit_attempt_recorder=SubmitAttemptRecorderStub(),
        submit_retry_recorder=SubmitRunRetryRecorder(
            submit_attempt_recorder=SubmitAttemptRecorderStub(),
            run_id="run-1",
            slug="demo",
            problem_types=[],
            knowledge_paths=object(),
            compute_submission_sha256=lambda _path: None,
            stdout_tail_chars=100,
            stderr_tail_chars=100,
            normalize_detail=lambda text, max_chars: str(text)[:max_chars],
            record_error_fix_insight=lambda **kwargs: None,
        ),
        submission_cli_error_types=(SubmitCliStubError,),
        local_guardrail_error_types=(ValueError,),
        kaggle_cli_error_types=(KeyError,),
        classify_submit_error=lambda stdout, stderr, exit_code: {"kind": "permanent", "reason": "bad_request"},
        should_use_notebook_fallback=lambda **kwargs: False,
        resolve_notebook_submit_artifact_mode=lambda **kwargs: "wrapper",
        decide_notebook_submit_artifact_mode_for_paths=lambda **kwargs: ArtifactModeDecisionStub(mode="wrapper"),
        count_tabular_data_rows=lambda _path: 0,
        compute_error_fingerprint=lambda stdout, stderr: f"{stdout}:{stderr}",
        decide_submit_fingerprint_reuse=lambda **kwargs: SimpleNamespace(
            fingerprint_seen=False,
            same_fingerprint_retry_allowed=False,
        ),
        compute_submit_backoff=lambda **kwargs: 0.0,
        save_run_state_for_run=lambda _run_dir, _updates: None,
        is_missing_credentials_error=lambda _error: False,
        build_submit_aborted_error=RuntimeError,
        sleep=lambda _seconds: None,
        on_message=lambda _message: None,
    )

    assert isinstance(result.submission_result, FileSubmitResult)
    assert result.submission_reference == str(submitted_path)
    assert result.submission_artifact_path == submitted_path
    assert result.submit_stage_state.submission_artifact_mode == "wrapper"


def test_run_submit_stage_attempts_until_success_records_transient_retry(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared.csv"
    submitted_path = tmp_path / "submitted.csv"
    recorder = SubmitAttemptRecorderStub()
    sleeps: list[float] = []
    calls = {"file_submit": 0}

    def run_file_submit():
        calls["file_submit"] += 1
        if calls["file_submit"] == 1:
            raise SubmitCliStubError(
                "temporary failure",
                stdout="temporary stdout",
                stderr="temporary stderr",
                exit_code=4,
            )
        return FileSubmitResult(submitted_path)

    result = run_submit_stage_attempts_until_success_or_abort(
        run_dir=tmp_path,
        run_id="run-1",
        state=SubmitStageRuntimeState(False, False, "wrapper"),
        prepared_submission_path=prepared_path,
        message="submit message",
        code_competition=False,
        max_attempts=2,
        backoff_base_seconds=3.5,
        sample_submission_path=tmp_path / "sample_submission.csv",
        fallback_sample_submission_path=tmp_path / "data" / "sample_submission.csv",
        submit_code_fingerprint="code-fp",
        run_state={},
        seen_fingerprints=set(),
        run_notebook_submit=lambda _state: (_ for _ in ()).throw(AssertionError("notebook should not run")),
        run_file_submit=run_file_submit,
        submit_aborter=object(),
        submit_attempt_recorder=recorder,
        submit_retry_recorder=SubmitRunRetryRecorder(
            submit_attempt_recorder=recorder,
            run_id="run-1",
            slug="demo",
            problem_types=["tabular"],
            knowledge_paths=object(),
            compute_submission_sha256=lambda path: "sha" if path == prepared_path else None,
            stdout_tail_chars=100,
            stderr_tail_chars=100,
            normalize_detail=lambda text, max_chars: str(text)[:max_chars],
            record_error_fix_insight=lambda **kwargs: None,
        ),
        submission_cli_error_types=(SubmitCliStubError,),
        local_guardrail_error_types=(ValueError,),
        kaggle_cli_error_types=(KeyError,),
        classify_submit_error=lambda stdout, stderr, exit_code: {
            "kind": "transient",
            "reason": "network_or_timeout",
        },
        should_use_notebook_fallback=lambda **kwargs: False,
        resolve_notebook_submit_artifact_mode=lambda **kwargs: "wrapper",
        decide_notebook_submit_artifact_mode_for_paths=lambda **kwargs: ArtifactModeDecisionStub(mode="wrapper"),
        count_tabular_data_rows=lambda _path: 0,
        compute_error_fingerprint=lambda stdout, stderr: "retry-fp",
        decide_submit_fingerprint_reuse=lambda **kwargs: SimpleNamespace(
            fingerprint_seen=False,
            same_fingerprint_retry_allowed=False,
        ),
        compute_submit_backoff=lambda **kwargs: 3.5,
        save_run_state_for_run=lambda _run_dir, _updates: None,
        is_missing_credentials_error=lambda _error: False,
        build_submit_aborted_error=RuntimeError,
        sleep=sleeps.append,
        on_message=lambda _message: None,
    )

    assert calls["file_submit"] == 2
    assert result.submission_reference == str(submitted_path)
    assert sleeps == [3.5]
    assert recorder.payloads[0]["action_taken"] == "retry"
    assert recorder.payloads[0]["fingerprint"] == "retry-fp"


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


def test_record_successful_submit_stage_result_records_attempt_outcome_and_payload(tmp_path: Path) -> None:
    submission_path = tmp_path / "iter-7" / "submission.csv"
    submission_path.parent.mkdir(parents=True)
    artifact_path = tmp_path / "kernel-output" / "submission.csv"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("id,prediction\n1,0.5\n", encoding="utf-8")
    outcome = {"status": "complete", "score": 0.12345}
    recorded_payloads: list[object] = []
    recorded_outcomes: list[tuple[Path, dict[str, object]]] = []
    marked_refs: list[str] = []
    messages: list[str] = []
    submitted_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    result = record_successful_submit_stage_result(
        run_id="run-1",
        message="submit message",
        submitted_at=submitted_at,
        submission_ref="kernel:user/demo/submission.csv",
        submission_result=SubmitResultStub(stdout="ok", stderr="warn", exit_code=0),
        submission_path=submission_path,
        submission_artifact_path=artifact_path,
        outcome=outcome,
        code_fingerprint="code-fp",
        prior_state={"submit_attempts_count": 2},
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        compute_submission_sha256=lambda path: "sha256" if path == artifact_path else None,
        record_submit_attempt_payloads=recorded_payloads.append,
        record_outcome=lambda path, ledger_outcome: recorded_outcomes.append((path, ledger_outcome)),
        mark_failure_context_submitted=marked_refs.append,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=messages.append,
    )

    assert len(recorded_payloads) == 1
    payloads = recorded_payloads[0]
    assert payloads.attempt_payload["ok"] is True
    assert payloads.attempt_payload["fingerprint"] == "fp:ok:warn"
    assert payloads.attempt_payload["sub_sha256"] == "sha256"
    assert payloads.run_state_update["submit_attempts_count"] == 3
    assert recorded_outcomes == [(artifact_path, outcome)]
    assert marked_refs == ["kernel:user/demo/submission.csv"]
    assert messages == [
        "[green]submission recorded[/green]",
        "[cyan]submission result[/cyan]: status=complete score=0.123450",
    ]
    assert result == {
        "message": "submit message",
        "submission_path": "kernel:user/demo/submission.csv",
        "submitted_at": submitted_at.isoformat(),
        "iteration": 7,
        "outcome": outcome,
    }


def test_record_successful_submit_for_run_records_ledger_and_resolves_failure_context(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    save_submit_failure_context(
        run_dir,
        {
            "active": True,
            "reason": "submission_poll_status_complete_no_score",
            "submission_ref": "old-submission.csv",
        },
    )
    submission_path = run_dir / "iter-2" / "submission.csv"
    submission_path.parent.mkdir(parents=True)
    artifact_path = tmp_path / "kernel-output" / "submission.csv"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("id,prediction\n1,0.5\n", encoding="utf-8")
    recorded_payloads: list[object] = []
    load_state_calls: list[Path] = []

    result = record_successful_submit_for_run(
        run_dir=run_dir,
        submission_ledger_path=tmp_path / "ledger.jsonl",
        slug="demo",
        run_id="run-1",
        message="submit message",
        submitted_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        submission_ref="kernel:user/demo/submission.csv",
        submission_result=SubmitResultStub(stdout="ok", stderr="", exit_code=0),
        submission_path=submission_path,
        submission_artifact_path=artifact_path,
        outcome={"status": "complete", "score": 0.25},
        code_fingerprint="code-fp",
        load_run_state=lambda state_run_dir: load_state_calls.append(state_run_dir) or {},
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        compute_submission_sha256=lambda path: "sha256" if path == artifact_path else None,
        record_submit_attempt_payloads=recorded_payloads.append,
        stdout_tail_chars=20,
        stderr_tail_chars=20,
        on_message=lambda _message: None,
    )

    ledger_records = load_jsonl_records(tmp_path / "ledger.jsonl")
    assert load_state_calls == [run_dir]
    assert ledger_records[0]["event"] == "outcome"
    assert ledger_records[0]["slug"] == "demo"
    assert ledger_records[0]["outcome"] == {"status": "complete", "score": 0.25}
    failure_context = load_submit_failure_context(run_dir)
    assert failure_context["active"] is False
    assert failure_context["resolution"] == "submitted"
    assert failure_context["resolved_submission_ref"] == "kernel:user/demo/submission.csv"
    assert recorded_payloads[0].attempt_payload["sub_sha256"] == "sha256"
    assert result["submission_path"] == "kernel:user/demo/submission.csv"
