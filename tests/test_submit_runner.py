from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from kagglebot.submit_cli_error_resolution import SubmitStageRuntimeState
from kagglebot.submit_runner import SubmitRunnerDependencies, SubmitRunnerLimits, attempt_submit_for_run


def _deps(tmp_path: Path) -> SubmitRunnerDependencies:
    return SubmitRunnerDependencies(
        load_competition_rule_constraints=lambda _paths: SimpleNamespace(notebook_submissions_only=False),
        env_truthy=lambda _name: False,
        load_run_state=lambda _run_dir: {},
        save_run_state=lambda _run_dir, _updates: None,
        compute_submit_code_fingerprint=lambda **_kwargs: "code-fp",
        compute_submission_sha256=lambda _path: "sha",
        now_iso=lambda: "2026-06-26T00:00:00+00:00",
        now_datetime=lambda: datetime(2026, 6, 26, tzinfo=UTC),
        normalize_error_text=lambda text, **_kwargs: str(text),
        record_error_fix_insight=lambda **_kwargs: None,
        build_error=RuntimeError,
        check_rules_accepted=lambda *_args, **_kwargs: True,
        infer_code_competition_from_paths=lambda _paths: False,
        collect_duplicate_submission_sources=lambda **_kwargs: [],
        decide_duplicate_submission_action=lambda **_kwargs: SimpleNamespace(action="proceed"),
        decide_same_submission_path_action=lambda **_kwargs: SimpleNamespace(action="retry"),
        resolve_notebook_submit_artifact_mode=lambda **_kwargs: "wrapper",
        decide_notebook_submit_artifact_mode_for_paths=lambda **_kwargs: SimpleNamespace(mode="wrapper", message=""),
        count_csv_data_rows=lambda _path: 1,
        resolve_kaggle_username=lambda **_kwargs: "user",
        run_submit_kernel=lambda **_kwargs: None,
        run_kaggle_submit_kernel=lambda **_kwargs: None,
        copy_submission_artifact_to_iteration_dir=lambda **_kwargs: None,
        classify_submit_error=lambda *_args, **_kwargs: {"kind": "permanent", "reason": "bad_request"},
        should_retry_ambiguous_notebook_submit_error=lambda **_kwargs: False,
        should_use_notebook_submit_fallback=lambda **_kwargs: False,
        compute_error_fingerprint=lambda *_args, **_kwargs: "fp",
        decide_submit_fingerprint_reuse=lambda **_kwargs: SimpleNamespace(
            fingerprint_seen=False,
            same_fingerprint_retry_allowed=False,
        ),
        compute_submit_backoff=lambda **_kwargs: 0.0,
        is_missing_kaggle_credentials_error=lambda _error: False,
        deliverable_mode=lambda _paths: "leaderboard",
        list_competition_submissions=lambda *_args, **_kwargs: [],
        sleep=lambda _seconds: None,
        on_message=lambda _message: None,
    )


def _limits() -> SubmitRunnerLimits:
    return SubmitRunnerLimits(
        stdout_tail_chars=100,
        stderr_tail_chars=100,
        max_transient_retries=2,
        backoff_base_sec=1.0,
        poll_max_attempts=1,
        poll_interval_sec=0.0,
        poll_max_fetch_errors=0,
    )


def test_attempt_submit_for_run_short_circuits_when_disabled(tmp_path: Path) -> None:
    config = SimpleNamespace(submit=False, dry_run=False)

    assert (
        attempt_submit_for_run(
            config=config,
            run_id="run-1",
            submission_path=tmp_path / "submission.csv",
            best_score=None,
            problem_types=[],
            deps=_deps(tmp_path),
            limits=_limits(),
        )
        is None
    )


def test_attempt_submit_for_run_composes_submit_stage_boundaries(monkeypatch, tmp_path: Path) -> None:
    from kagglebot import submit_runner

    paths = SimpleNamespace(
        kernel_source_dir=tmp_path / "kernel",
        context_dir=tmp_path / "context",
        data_dir=tmp_path / "data",
        sample_submission_path=tmp_path / "sample_submission.csv",
        submission_ledger_path=tmp_path / "ledger.jsonl",
        run_dir=lambda run_id: tmp_path / "runs" / run_id,
    )
    config = SimpleNamespace(
        submit=True,
        dry_run=False,
        slug="demo",
        paths=paths,
        knowledge_paths=object(),
        force_submit=False,
        message="submit message",
        campaign_mode="standard",
        target_direction="maximize",
        kaggle_username=None,
        kernel_name=None,
        accelerator="cpu",
        strict_accelerator=False,
        time_budget_min=None,
    )
    submission_path = tmp_path / "submission.csv"
    prepared_path = tmp_path / "prepared.csv"
    calls: list[str] = []
    recorder = SimpleNamespace(record_payloads=lambda _payloads: None)

    monkeypatch.setattr(
        submit_runner._submit_stage,
        "build_submit_run_context",
        lambda **kwargs: calls.append("run_context")
        or SimpleNamespace(
            submit_attempt_recorder=recorder,
            run_state={},
            latest_submit_attempt={},
            submit_code_fingerprint="code-fp",
            allow_force=False,
            input_submission_path=submission_path,
            submit_aborter=object(),
            submit_retry_recorder=object(),
        ),
    )
    monkeypatch.setattr(
        submit_runner._submit_stage,
        "build_submit_runtime_context",
        lambda **kwargs: calls.append("runtime_context")
        or SimpleNamespace(
            message="submit message",
            submission_service=SimpleNamespace(validate_and_prepare_submission=lambda path: prepared_path),
            submitted_at=datetime(2026, 6, 26, tzinfo=UTC),
        ),
    )
    monkeypatch.setattr(
        submit_runner._submit_stage,
        "prepare_and_resolve_submit_preflight_for_run_or_abort",
        lambda **kwargs: calls.append("preflight")
        or SimpleNamespace(
            prepared_context=SimpleNamespace(prepared_submission_path=prepared_path, prepared_submission_sha="sha"),
            preflight_context=SimpleNamespace(
                duplicate_skip_result=None,
                same_submission_path_skipped=False,
                submit_stage_state=SubmitStageRuntimeState(False, False, "wrapper"),
                code_competition=False,
                seen_fingerprints=set(),
            ),
        ),
    )
    monkeypatch.setattr(
        submit_runner,
        "_build_notebook_submit_runner",
        lambda **kwargs: SimpleNamespace(submit=lambda **_kwargs: ("notebook", "ref", None)),
    )
    monkeypatch.setattr(
        submit_runner._submit_stage,
        "run_submit_stage_attempts_until_success_or_abort",
        lambda **kwargs: calls.append("attempt_loop")
        or SimpleNamespace(
            submit_stage_state=kwargs["state"],
            submission_reference=str(prepared_path),
            submission_artifact_path=prepared_path,
            submission_result=SimpleNamespace(stdout="", stderr="", exit_code=0),
        ),
    )
    monkeypatch.setattr(
        submit_runner._submit_outcome,
        "finalize_submit_outcome_for_run_or_abort",
        lambda **kwargs: calls.append("finalize") or {"ok": True, "submission_ref": kwargs["submission_ref"]},
    )

    result = attempt_submit_for_run(
        config=config,
        run_id="run-1",
        submission_path=submission_path,
        best_score=0.42,
        problem_types=["tabular"],
        deps=_deps(tmp_path),
        limits=_limits(),
    )

    assert result == {"ok": True, "submission_ref": str(prepared_path)}
    assert calls == ["run_context", "runtime_context", "preflight", "attempt_loop", "finalize"]
