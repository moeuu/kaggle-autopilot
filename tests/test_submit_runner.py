from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        count_tabular_data_rows=lambda _path: 1,
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


def test_resolve_fallback_sample_submission_path_uses_context_non_csv_when_data_missing(tmp_path: Path) -> None:
    from kagglebot.submit_runner import _resolve_fallback_sample_submission_path

    sample_path = tmp_path / "context" / "sample_submission.jsonl"
    sample_path.parent.mkdir()
    sample_path.write_text('{"id": 1, "target": 0}\n', encoding="utf-8")
    paths = SimpleNamespace(
        data_dir=tmp_path / "data",
        sample_submission_path=sample_path,
    )

    assert _resolve_fallback_sample_submission_path(paths) == sample_path


def test_resolve_fallback_sample_submission_path_discovers_context_alias_when_primary_missing(
    tmp_path: Path,
) -> None:
    from kagglebot.submit_runner import _resolve_fallback_sample_submission_path

    sample_path = tmp_path / "context" / "AnswerTemplate.csv"
    sample_path.parent.mkdir()
    sample_path.write_text("id,target\n1,0\n", encoding="utf-8")
    paths = SimpleNamespace(
        context_dir=tmp_path / "context",
        data_dir=tmp_path / "data",
        sample_submission_path=tmp_path / "context" / "sample_submission.csv",
    )

    assert _resolve_fallback_sample_submission_path(paths) == sample_path


def test_resolve_fallback_sample_submission_path_prefers_context_over_data_alias(
    tmp_path: Path,
) -> None:
    from kagglebot.submit_runner import _resolve_fallback_sample_submission_path

    context_sample = tmp_path / "context" / "sample_submission.jsonl"
    data_sample = tmp_path / "data" / "sample_submission.csv"
    context_sample.parent.mkdir()
    data_sample.parent.mkdir()
    context_sample.write_text('{"id": 1, "target": 0}\n', encoding="utf-8")
    data_sample.write_text("id,target\n1,0\n", encoding="utf-8")
    paths = SimpleNamespace(
        context_dir=tmp_path / "context",
        data_dir=tmp_path / "data",
        sample_submission_path=tmp_path / "context" / "sample_submission.csv",
    )

    assert _resolve_fallback_sample_submission_path(paths) == context_sample


def test_resolve_fallback_sample_submission_path_preserves_missing_primary_suffix(tmp_path: Path) -> None:
    from kagglebot.submit_runner import _resolve_fallback_sample_submission_path

    paths = SimpleNamespace(
        context_dir=tmp_path / "context",
        data_dir=tmp_path / "data",
        sample_submission_path=tmp_path / "context" / "sample_submission.jsonl",
    )

    assert _resolve_fallback_sample_submission_path(paths) == tmp_path / "data" / "sample_submission.jsonl"


def test_resolve_fallback_sample_submission_path_preserves_missing_structured_primary_suffix(tmp_path: Path) -> None:
    from kagglebot.submit_runner import _resolve_fallback_sample_submission_path

    paths = SimpleNamespace(
        context_dir=tmp_path / "context",
        data_dir=tmp_path / "data",
        sample_submission_path=tmp_path / "context" / "sample_submission.xlsx",
    )

    assert _resolve_fallback_sample_submission_path(paths) == tmp_path / "data" / "sample_submission.xlsx"


@pytest.mark.parametrize("suffix", [".html", ".html.zst"])
def test_resolve_fallback_sample_submission_path_preserves_missing_html_primary_suffix(
    tmp_path: Path,
    suffix: str,
) -> None:
    from kagglebot.submit_runner import _resolve_fallback_sample_submission_path

    paths = SimpleNamespace(
        context_dir=tmp_path / "context",
        data_dir=tmp_path / "data",
        sample_submission_path=tmp_path / "context" / f"sample_submission{suffix}",
    )

    assert _resolve_fallback_sample_submission_path(paths) == tmp_path / "data" / f"sample_submission{suffix}"


@pytest.mark.parametrize("suffix", [".orc", ".hdf", ".hdf5"])
def test_resolve_fallback_sample_submission_path_preserves_missing_binary_primary_suffix(
    tmp_path: Path,
    suffix: str,
) -> None:
    from kagglebot.submit_runner import _resolve_fallback_sample_submission_path

    paths = SimpleNamespace(
        context_dir=tmp_path / "context",
        data_dir=tmp_path / "data",
        sample_submission_path=tmp_path / "context" / f"sample_submission{suffix}",
    )

    assert _resolve_fallback_sample_submission_path(paths) == tmp_path / "data" / f"sample_submission{suffix}"


def test_resolve_fallback_sample_submission_path_preserves_missing_sqlite_primary_suffix(
    tmp_path: Path,
) -> None:
    from kagglebot.submit_runner import _resolve_fallback_sample_submission_path

    paths = SimpleNamespace(
        context_dir=tmp_path / "context",
        data_dir=tmp_path / "data",
        sample_submission_path=tmp_path / "context" / "sample_submission.sqlite",
    )

    assert _resolve_fallback_sample_submission_path(paths) == tmp_path / "data" / "sample_submission.sqlite"


@pytest.mark.parametrize("suffix", [".tar.xz", ".tar.zst"])
def test_expected_notebook_submit_output_file_uses_format_archive_suffix(tmp_path: Path, suffix: str) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nSubmit a `submission{suffix}` archive containing model weights.\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / "sample_submission.csv",
    )

    assert _expected_notebook_submit_output_file(paths) == f"submission{suffix}"


def test_expected_notebook_submit_output_file_uses_format_rar_suffix(tmp_path: Path) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a single `submission.rar` archive.\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / "sample_submission.csv",
    )

    assert _expected_notebook_submit_output_file(paths) == "submission.rar"


def test_expected_notebook_submit_output_file_uses_format_model_suffix(tmp_path: Path) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a single ONNX file named submission.onnx.\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / "sample_submission.csv",
    )

    assert _expected_notebook_submit_output_file(paths) == "submission.onnx"


def test_expected_notebook_submit_output_file_uses_explicit_non_tabular_filename(tmp_path: Path) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a single file named `answers.nii.gz`.\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / "sample_submission.csv",
    )

    assert _expected_notebook_submit_output_file(paths) == "answers.nii.gz"


def test_expected_notebook_submit_output_file_uses_explicit_model_artifact_filename(tmp_path: Path) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `model.safetensors.index.json` as the final output.\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / "sample_submission.csv",
    )

    assert _expected_notebook_submit_output_file(paths) == "model.safetensors.index.json"


def test_expected_notebook_submit_output_file_uses_model_directory_format_suffix(tmp_path: Path) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a Hugging Face model directory for scoring.\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / "sample_submission.csv",
    )

    assert _expected_notebook_submit_output_file(paths) == "submission.hfmodel"


def test_expected_notebook_submit_output_file_uses_explicit_directory_array_filename(tmp_path: Path) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nThe required output is a Zarr store called predictions.zarr.\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / "sample_submission.csv",
    )

    assert _expected_notebook_submit_output_file(paths) == "predictions.zarr"


def test_expected_notebook_submit_output_file_ignores_sample_filename_when_explicit_format_mentions_it(
    tmp_path: Path,
) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUse `sample_submission.csv` as the template, then upload `answers.csv`.\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / "sample_submission.csv",
    )

    assert _expected_notebook_submit_output_file(paths) == "answers.csv"


@pytest.mark.parametrize(
    ("description", "filename"),
    [
        ("Participants must upload an EPUB document for scoring.", "submission.epub"),
        ("Participants must upload a zstd-compressed LaTeX file for scoring.", "submission.tex.zst"),
        ("Participants must upload COCO annotations for scoring.", "submission.json"),
        ("Participants must upload a safetensors model file for scoring.", "submission.safetensors"),
    ],
)
def test_expected_notebook_submit_output_file_uses_non_tabular_prose_suffix(
    tmp_path: Path,
    description: str,
    filename: str,
) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "submission_format.md").write_text(f"## Submission Format\n{description}\n", encoding="utf-8")
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / "sample_submission.csv",
    )

    assert _expected_notebook_submit_output_file(paths) == filename


@pytest.mark.parametrize("suffix", [".orc", ".hdf", ".hdf5"])
def test_expected_notebook_submit_output_file_uses_binary_sample_suffix(tmp_path: Path, suffix: str) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / f"sample_submission{suffix}",
    )

    assert _expected_notebook_submit_output_file(paths) == f"submission{suffix}"


@pytest.mark.parametrize("suffix", [".orc", ".hdf", ".hdf5"])
def test_expected_notebook_submit_output_file_uses_binary_format_suffix(tmp_path: Path, suffix: str) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` with columns id,target.\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / "sample_submission.csv",
    )

    assert _expected_notebook_submit_output_file(paths) == f"submission{suffix}"


@pytest.mark.parametrize(
    ("description", "filename"),
    [
        ("Upload a zstd-compressed NDJSON file with columns id,target.", "submission.ndjson.zst"),
        ("Upload a bzip2-compressed HTML file with columns id,target.", "submission.html.bz2"),
        ("Upload an xz-compressed PSV file with columns id,target.", "submission.psv.xz"),
    ],
)
def test_expected_notebook_submit_output_file_uses_compressed_tabular_format_keywords(
    tmp_path: Path,
    description: str,
    filename: str,
) -> None:
    from kagglebot.submit_runner import _expected_notebook_submit_output_file

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "submission_format.md").write_text(f"## Submission Format\n{description}\n", encoding="utf-8")
    paths = SimpleNamespace(
        context_dir=context_dir,
        data_dir=tmp_path / "data",
        sample_submission_path=context_dir / "sample_submission.csv",
    )

    assert _expected_notebook_submit_output_file(paths) == filename


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
    paths.data_dir.mkdir()
    fallback_sample_path = paths.data_dir / "sample_submission.tsv"
    fallback_sample_path.write_text("id\ttarget\n1\t0.0\n", encoding="utf-8")
    calls: list[str] = []
    captured: dict[str, Path] = {}
    recorder = SimpleNamespace(record_payloads=lambda _payloads: None)

    monkeypatch.setattr(
        submit_runner._submit_context,
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
        submit_runner._submit_context,
        "build_submit_runtime_context",
        lambda **kwargs: calls.append("runtime_context")
        or SimpleNamespace(
            message="submit message",
            submission_service=SimpleNamespace(validate_and_prepare_submission=lambda path: prepared_path),
            submitted_at=datetime(2026, 6, 26, tzinfo=UTC),
        ),
    )

    def _prepare_preflight(**kwargs):
        calls.append("preflight")
        captured["preflight_fallback_sample_submission_path"] = kwargs["fallback_sample_submission_path"]
        return SimpleNamespace(
            prepared_context=SimpleNamespace(prepared_submission_path=prepared_path, prepared_submission_sha="sha"),
            preflight_context=SimpleNamespace(
                duplicate_skip_result=None,
                same_submission_path_skipped=False,
                submit_stage_state=SubmitStageRuntimeState(False, False, "wrapper"),
                code_competition=False,
                seen_fingerprints=set(),
            ),
        )

    monkeypatch.setattr(
        submit_runner._submit_stage,
        "prepare_and_resolve_submit_preflight_for_run_or_abort",
        _prepare_preflight,
    )
    monkeypatch.setattr(
        submit_runner,
        "_build_notebook_submit_runner",
        lambda **kwargs: SimpleNamespace(submit=lambda **_kwargs: ("notebook", "ref", None)),
    )

    def _run_attempt_loop(**kwargs):
        calls.append("attempt_loop")
        captured["attempt_loop_fallback_sample_submission_path"] = kwargs["fallback_sample_submission_path"]
        return SimpleNamespace(
            submit_stage_state=kwargs["state"],
            submission_reference=str(prepared_path),
            submission_artifact_path=prepared_path,
            submission_result=SimpleNamespace(stdout="", stderr="", exit_code=0),
        )

    monkeypatch.setattr(
        submit_runner._submit_attempt_loop,
        "run_submit_stage_attempts_until_success_or_abort",
        _run_attempt_loop,
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
    assert captured == {
        "preflight_fallback_sample_submission_path": fallback_sample_path,
        "attempt_loop_fallback_sample_submission_path": fallback_sample_path,
    }
