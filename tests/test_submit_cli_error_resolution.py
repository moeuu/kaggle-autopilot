from __future__ import annotations

from pathlib import Path

from kagglebot.submit_cli_error_resolution import (
    SubmitStageRuntimeState,
    apply_notebook_fallback_decision,
    build_notebook_fallback_retry_state,
    classify_submit_stage_error,
    decide_notebook_fallback_after_file_submit_error,
    decide_submit_stage_error_action,
    decide_submit_stage_error_action_from_classification,
    resolve_notebook_fallback_after_file_submit_error,
    resolve_submit_cli_error,
    resolve_submit_cli_error_for_run,
)


class FingerprintReuseDecisionStub:
    def __init__(self, *, fingerprint_seen: bool, same_fingerprint_retry_allowed: bool) -> None:
        self.fingerprint_seen = fingerprint_seen
        self.same_fingerprint_retry_allowed = same_fingerprint_retry_allowed


class ArtifactModeDecisionStub:
    def __init__(self, *, mode: str, message: str = "") -> None:
        self.mode = mode
        self.message = message


def _runtime_state() -> SubmitStageRuntimeState:
    return SubmitStageRuntimeState(
        notebook_submit_required=False,
        notebook_fallback_activated=False,
        submission_artifact_mode="wrapper",
    )


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


def test_decide_submit_stage_error_action_from_classification_uses_normalized_values() -> None:
    classification = classify_submit_stage_error(
        stdout="",
        stderr="network down",
        output="",
        exit_code=1,
        classify_submit_error=lambda stdout, stderr, exit_code: {
            "kind": "transient",
            "reason": "network_or_timeout",
            "retry_after_seconds": 5.5,
        },
    )

    decision = decide_submit_stage_error_action_from_classification(
        fingerprint_seen=False,
        same_fingerprint_retry_allowed=False,
        classification=classification,
        attempt=1,
        max_attempts=3,
        backoff_seconds=2.0,
    )

    assert decision.action == "retry"
    assert decision.error_kind == "transient"
    assert decision.reason == "network_or_timeout"
    assert decision.wait_seconds == 5.5


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


def test_build_notebook_fallback_retry_state_combines_artifact_mode_and_messages() -> None:
    decision = decide_notebook_fallback_after_file_submit_error(
        notebook_submit_required=False,
        notebook_fallback_activated=False,
        should_use_notebook_fallback=True,
        resolved_notebook_artifact_mode="wrapper",
        current_submission_artifact_mode="wrapper",
    )

    state = build_notebook_fallback_retry_state(
        fallback_decision=decision,
        artifact_mode="inference",
        artifact_message="[yellow]submit mode[/yellow]: using inference artifact",
    )

    assert state.notebook_submit_required is True
    assert state.notebook_fallback_activated is True
    assert state.submission_artifact_mode == "inference"
    assert state.messages == (
        "[yellow]submit mode[/yellow]: file submit indicates notebook submit is required; "
        "retrying via notebook submit automatically.",
        "[yellow]submit mode[/yellow]: using inference artifact",
    )


def test_apply_notebook_fallback_decision_resolves_artifact_and_updates_state() -> None:
    decision = decide_notebook_fallback_after_file_submit_error(
        notebook_submit_required=False,
        notebook_fallback_activated=False,
        should_use_notebook_fallback=True,
        resolved_notebook_artifact_mode="wrapper",
        current_submission_artifact_mode="wrapper",
    )
    calls: list[tuple[str, bool]] = []
    messages: list[str] = []

    applied = apply_notebook_fallback_decision(
        state=_runtime_state(),
        fallback_decision=decision,
        resolve_artifact_mode=lambda mode, required: (
            calls.append((mode, required))
            or ArtifactModeDecisionStub(
                mode="inference",
                message="[yellow]submit mode[/yellow]: using inference artifact",
            )
        ),
        on_message=messages.append,
    )

    assert applied.retry_as_notebook is True
    assert applied.state.notebook_submit_required is True
    assert applied.state.notebook_fallback_activated is True
    assert applied.state.submission_artifact_mode == "inference"
    assert calls == [("wrapper", True)]
    assert messages == [
        "[yellow]submit mode[/yellow]: file submit indicates notebook submit is required; "
        "retrying via notebook submit automatically.",
        "[yellow]submit mode[/yellow]: using inference artifact",
    ]


def test_apply_notebook_fallback_decision_keeps_state_when_not_retrying() -> None:
    state = _runtime_state()
    decision = decide_notebook_fallback_after_file_submit_error(
        notebook_submit_required=False,
        notebook_fallback_activated=False,
        should_use_notebook_fallback=False,
        resolved_notebook_artifact_mode="wrapper",
        current_submission_artifact_mode=state.submission_artifact_mode,
    )
    messages: list[str] = []

    applied = apply_notebook_fallback_decision(
        state=state,
        fallback_decision=decision,
        resolve_artifact_mode=lambda mode, required: (_ for _ in ()).throw(AssertionError("not used")),
        on_message=messages.append,
    )

    assert applied.retry_as_notebook is False
    assert applied.state == state
    assert messages == []


def test_resolve_notebook_fallback_after_file_submit_error_updates_state(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    sample_path = tmp_path / "sample_submission.csv"
    fallback_sample_path = tmp_path / "data_sample_submission.csv"
    resolver_calls: list[dict[str, object]] = []
    artifact_calls: list[tuple[str, bool, bool]] = []
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
            message="[yellow]submit mode[/yellow]: using inference artifact",
        )

    applied = resolve_notebook_fallback_after_file_submit_error(
        state=_runtime_state(),
        should_use_notebook_fallback=True,
        code_competition=True,
        sample_submission_path=sample_path,
        fallback_sample_submission_path=fallback_sample_path,
        submission_path=submission_path,
        resolve_notebook_submit_artifact_mode=resolve_notebook_mode,
        decide_notebook_submit_artifact_mode_for_paths=decide_artifact_mode,
        count_tabular_data_rows=lambda path: 3,
        on_message=messages.append,
    )

    assert applied.retry_as_notebook is True
    assert applied.state.notebook_submit_required is True
    assert applied.state.notebook_fallback_activated is True
    assert applied.state.submission_artifact_mode == "inference"
    assert resolver_calls == [{"submit_mode": "notebook", "code_competition": True}]
    assert artifact_calls == [("inference", True, True)]
    assert messages == [
        "[yellow]submit mode[/yellow]: file submit indicates notebook submit is required; "
        "retrying via notebook submit automatically.",
        "[yellow]submit mode[/yellow]: using inference artifact",
    ]


def test_resolve_notebook_fallback_after_file_submit_error_keeps_state_when_disabled(tmp_path: Path) -> None:
    state = _runtime_state()
    messages: list[str] = []

    def resolve_notebook_mode(**kwargs: object) -> str:
        raise AssertionError(f"notebook resolver should not be used: {kwargs}")

    applied = resolve_notebook_fallback_after_file_submit_error(
        state=state,
        should_use_notebook_fallback=False,
        code_competition=False,
        sample_submission_path=tmp_path / "sample_submission.csv",
        fallback_sample_submission_path=tmp_path / "data_sample_submission.csv",
        submission_path=tmp_path / "submission.csv",
        resolve_notebook_submit_artifact_mode=resolve_notebook_mode,
        decide_notebook_submit_artifact_mode_for_paths=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"artifact resolver should not be used: {kwargs}")
        ),
        count_tabular_data_rows=lambda path: 3,
        on_message=messages.append,
    )

    assert applied.retry_as_notebook is False
    assert applied.state == state
    assert messages == []


def test_resolve_submit_cli_error_retries_via_notebook_fallback(tmp_path: Path) -> None:
    messages: list[str] = []

    resolution = resolve_submit_cli_error(
        state=_runtime_state(),
        stdout="",
        stderr="notebook submissions only",
        output="",
        exit_code=1,
        attempt=1,
        max_attempts=2,
        backoff_base_seconds=1.0,
        classify_submit_error=lambda stdout, stderr, exit_code: {
            "kind": "permanent",
            "reason": "notebook_submit_required",
        },
        should_use_notebook_fallback=lambda **kwargs: True,
        code_competition=True,
        sample_submission_path=tmp_path / "sample_submission.csv",
        fallback_sample_submission_path=tmp_path / "data_sample_submission.csv",
        submission_path=tmp_path / "submission.csv",
        resolve_notebook_submit_artifact_mode=lambda **kwargs: "inference",
        decide_notebook_submit_artifact_mode_for_paths=lambda **kwargs: ArtifactModeDecisionStub(mode="inference"),
        count_tabular_data_rows=lambda path: 3,
        compute_error_fingerprint=lambda stdout, stderr: "unused",
        decide_submit_fingerprint_reuse=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"fingerprint reuse should not be resolved: {kwargs}")
        ),
        compute_submit_backoff=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"backoff should not be resolved: {kwargs}")
        ),
        seen_fingerprints=set(),
        run_state={},
        code_fingerprint="code-fp",
        save_run_state=lambda updates: None,
        on_message=messages.append,
    )

    assert resolution.classification.reason == "notebook_submit_required"
    assert resolution.fallback_application.retry_as_notebook is True
    assert resolution.fallback_application.state.submission_artifact_mode == "inference"
    assert resolution.fingerprint == ""
    assert resolution.error_action is None
    assert messages == [
        "[yellow]submit mode[/yellow]: file submit indicates notebook submit is required; "
        "retrying via notebook submit automatically."
    ]


def test_resolve_submit_cli_error_builds_retry_action(tmp_path: Path) -> None:
    messages: list[str] = []
    fingerprint_calls: list[dict[str, object]] = []

    resolution = resolve_submit_cli_error(
        state=_runtime_state(),
        stdout="stdout",
        stderr="network down",
        output="",
        exit_code=1,
        attempt=1,
        max_attempts=2,
        backoff_base_seconds=3.0,
        classify_submit_error=lambda stdout, stderr, exit_code: {
            "kind": "transient",
            "reason": "network_or_timeout",
        },
        should_use_notebook_fallback=lambda **kwargs: False,
        code_competition=False,
        sample_submission_path=tmp_path / "sample_submission.csv",
        fallback_sample_submission_path=tmp_path / "data_sample_submission.csv",
        submission_path=tmp_path / "submission.csv",
        resolve_notebook_submit_artifact_mode=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"notebook resolver should not be used: {kwargs}")
        ),
        decide_notebook_submit_artifact_mode_for_paths=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"artifact resolver should not be used: {kwargs}")
        ),
        count_tabular_data_rows=lambda path: 3,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        decide_submit_fingerprint_reuse=lambda **kwargs: (
            fingerprint_calls.append(kwargs)
            or FingerprintReuseDecisionStub(fingerprint_seen=False, same_fingerprint_retry_allowed=False)
        ),
        compute_submit_backoff=lambda **kwargs: 3.25,
        seen_fingerprints=set(),
        run_state={"state": "value"},
        code_fingerprint="code-fp",
        save_run_state=lambda updates: None,
        on_message=messages.append,
    )

    assert resolution.classification.reason == "network_or_timeout"
    assert resolution.fallback_application.retry_as_notebook is False
    assert resolution.fingerprint == "fp:stdout:network down"
    assert resolution.error_action is not None
    assert resolution.error_action.action == "retry"
    assert resolution.error_action.wait_seconds == 3.25
    assert len(fingerprint_calls) == 1
    assert fingerprint_calls[0]["fingerprint"] == "fp:stdout:network down"
    assert fingerprint_calls[0]["seen_fingerprints"] == set()
    assert fingerprint_calls[0]["run_state"] == {"state": "value"}
    assert fingerprint_calls[0]["code_fingerprint"] == "code-fp"
    assert callable(fingerprint_calls[0]["save_run_state"])
    assert messages == [
        "[yellow]submit retry[/yellow]: transient submit error (reason=network_or_timeout, attempt=1/2, wait=3.2s)"
    ]


def test_resolve_submit_cli_error_for_run_binds_run_state_save(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    saved_updates: list[tuple[Path, dict[str, object]]] = []

    def decide_fingerprint_reuse(**kwargs: object) -> FingerprintReuseDecisionStub:
        kwargs["save_run_state"]({"last_submit_fingerprint": kwargs["fingerprint"]})
        return FingerprintReuseDecisionStub(fingerprint_seen=True, same_fingerprint_retry_allowed=True)

    resolution = resolve_submit_cli_error_for_run(
        run_dir=run_dir,
        state=_runtime_state(),
        stdout="stdout",
        stderr="network down",
        output="",
        exit_code=1,
        attempt=1,
        max_attempts=2,
        backoff_base_seconds=3.0,
        classify_submit_error=lambda stdout, stderr, exit_code: {
            "kind": "transient",
            "reason": "network_or_timeout",
        },
        should_use_notebook_fallback=lambda **kwargs: False,
        code_competition=False,
        sample_submission_path=tmp_path / "sample_submission.csv",
        fallback_sample_submission_path=tmp_path / "data_sample_submission.csv",
        submission_path=tmp_path / "submission.csv",
        resolve_notebook_submit_artifact_mode=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"notebook resolver should not be used: {kwargs}")
        ),
        decide_notebook_submit_artifact_mode_for_paths=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"artifact resolver should not be used: {kwargs}")
        ),
        count_tabular_data_rows=lambda path: 3,
        compute_error_fingerprint=lambda stdout, stderr: f"fp:{stdout}:{stderr}",
        decide_submit_fingerprint_reuse=decide_fingerprint_reuse,
        compute_submit_backoff=lambda **kwargs: 3.25,
        seen_fingerprints={"fp:stdout:network down"},
        run_state={"state": "value"},
        code_fingerprint="code-fp",
        save_run_state_for_run=lambda path, updates: saved_updates.append((path, updates)),
        on_message=lambda _message: None,
    )

    assert resolution.error_action is not None
    assert resolution.error_action.action == "retry"
    assert saved_updates == [(run_dir, {"last_submit_fingerprint": "fp:stdout:network down"})]


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
