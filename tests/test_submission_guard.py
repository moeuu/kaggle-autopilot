from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from kagglebot.exceptions import SubmissionCliError, SubmissionValidationError
from kagglebot.submission.guard import (
    classify_submit_error,
    compute_error_fingerprint,
    normalize_error_text,
    run_kaggle_submit,
    run_kaggle_submit_kernel,
)
from kagglebot.submission.validate import validate_submission


def _write_sample_and_submission(tmp_path: Path) -> tuple[Path, Path]:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": [0.0, 0.0, 0.0]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    return sample, submission


def test_validate_submission_columns_mismatch(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2, 3], "score": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="columns mismatch"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_row_count_mismatch(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="row count mismatch"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_id_nan(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, None, 3], "target": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="id column 'id' contains NaN"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_id_duplicate(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 1, 3], "target": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="duplicate values"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_pred_nan_or_non_numeric(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, "abc", 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="NaN/non-numeric"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_pred_inf(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, float("inf"), 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="contains \\+/-inf"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_categorical_target_passes(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": ["Absence", "Presence", "Absence"]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [1, 2, 3], "target": ["Presence", "Absence", "Presence"]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_categorical_target_allows_unknown_values(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": ["Absence", "Presence", "Absence"]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_requires_rna_anchor_columns_to_match_sample(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "ID": ["RNA1_1", "RNA1_2"],
            "resname": ["A", "C"],
            "resid": [1, 2],
            "x_1": [0.0, 0.0],
            "y_1": [0.0, 0.0],
            "z_1": [0.0, 0.0],
        }
    ).to_csv(sample, index=False)
    pd.DataFrame(
        {
            "ID": ["RNA1_1", "RNA1_2"],
            "resname": ["G", "C"],
            "resid": [1, 2],
            "x_1": [0.1, 0.2],
            "y_1": [0.3, 0.4],
            "z_1": [0.5, 0.6],
        }
    ).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError, match="anchor column 'resname'"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_reports_multiple_problems(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, None], "target": [0.1, "bad"]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(str(submission), str(sample))
    message = str(exc.value)
    assert "Submission validation failed:" in message
    assert "- row count mismatch:" in message
    assert "- id column 'id' contains NaN values:" in message
    assert "- prediction column 'target' contains NaN/non-numeric values:" in message


def test_validate_submission_uses_overview_hint_when_sample_is_header_only(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = data_dir / "context"
    cache_dir = data_dir / ".kagglebot_cache"
    context_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample = cache_dir / "sample_submission_synth.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    (context_dir / "overview.md").write_text(
        "## Submission Format\n\n```csv\nfilename,right_place,prediction_string\n```\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"filename": ["0.jpg"], "right_place": [0], "prediction_string": ["-"]}).to_csv(
        submission, index=False
    )

    validate_submission(str(submission), str(sample))


def test_validate_submission_uses_data_md_hint_when_sample_is_header_only(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = data_dir / "context"
    cache_dir = data_dir / ".kagglebot_cache"
    context_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample = cache_dir / "sample_submission_synth.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "data.md").write_text(
        "## Submission Format\n\n"
        "A CSV file with the following columns:\n"
        "* `Id`: The filename\n"
        "* `Category`: The predicted class\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"Id": ["x"], "Category": ["Health"]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_header_only_sample_validates_against_icpr_evaluation_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    eval_root = data_dir / "ICPR02" / "kaggle" / "evaluation"
    for sample_id in ("a0", "b1", "c2"):
        sample_dir = eval_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "B2.tif").write_bytes(b"TIFF")

    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["a0", "b1"], "prediction": [0.1, 0.2]}).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(str(submission), str(sample), data_dir=data_dir)
    message = str(exc.value)
    assert "row count mismatch" in message
    assert "id values mismatch (header-only sample detected; validated against evaluation directory ids)" in message


def test_validate_submission_header_only_sample_accepts_icpr_evaluation_id_set(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    eval_root = data_dir / "ICPR02" / "kaggle" / "evaluation"
    for sample_id in ("a0", "b1", "c2"):
        sample_dir = eval_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "B2.tif").write_bytes(b"TIFF")

    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["c2", "a0", "b1"], "prediction": [0.3, 0.1, 0.2]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_rejects_missing_required_id_suffix_when_inferred(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    (data_dir / "Kaggle_Prepared" / "val" / "MS").mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "data.md").write_text(
        "## Submission Format\n\n"
        "A CSV file with the following columns:\n"
        "* `Id`: The filename (e.g., `val_a1b2c3d4.tif`)\n"
        "* `Category`: The predicted class\n",
        encoding="utf-8",
    )
    for stem in ("val_0001", "val_0002"):
        (data_dir / "Kaggle_Prepared" / "val" / "MS" / f"{stem}.tif").write_bytes(b"TIFF")

    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["val_0001", "val_0002"], "prediction": ["Health", "Rust"]}).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError, match="require '\\.tif' suffix"):
        validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_accepts_inferred_required_id_suffix_when_present(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    (data_dir / "Kaggle_Prepared" / "val" / "MS").mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "data.md").write_text(
        "## Submission Format\n\n"
        "A CSV file with the following columns:\n"
        "* `Id`: The filename (e.g., `val_a1b2c3d4.tif`)\n"
        "* `Category`: The predicted class\n",
        encoding="utf-8",
    )
    for stem in ("val_0001", "val_0002"):
        (data_dir / "Kaggle_Prepared" / "val" / "MS" / f"{stem}.tif").write_bytes(b"TIFF")

    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["val_0001.tif", "val_0002.tif"], "prediction": ["Health", "Rust"]}).to_csv(
        submission, index=False
    )

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_does_not_infer_suffix_when_real_sample_ids_are_suffixless(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "test_set").mkdir(parents=True, exist_ok=True)
    for stem in ("0", "1", "2"):
        (data_dir / "test_set" / f"{stem}.png").write_bytes(b"PNG")

    sample = tmp_path / "sample_submission.csv"
    sample.write_text(
        "id,image_id,prediction_string\n0,0,0.9 1 2 3 4\n1,1, \n2,2,0.8 5 6 7 8\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    submission.write_text(
        "id,image_id,prediction_string\n0,0,0.9 1 2 3 4\n1,1, \n2,2,0.8 5 6 7 8\n",
        encoding="utf-8",
    )

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_checks_overview_hint_not_only_sample(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = data_dir / "context"
    cache_dir = data_dir / ".kagglebot_cache"
    context_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample = cache_dir / "sample_submission_synth.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    (context_dir / "overview.md").write_text(
        "## Submission Format\n\n```csv\nfilename,right_place,prediction_string\n```\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1], "target": [0.5]}).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError, match="expected \\(submission_format/overview hint\\)"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_ignores_overview_rules_text_with_commas(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "overview.md").write_text(
        "## Submission Code Requirements\n"
        "a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or "
        "Competition Specific Rules above, during the Competition Period, you are not allowed to privately share "
        "source or executable code developed in connection with or based upon the Competition Data.</h5>\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["X"], "prediction": [0.1]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_ignores_submission_format_rules_text_with_commas(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "#### 6. SUBMISSION CODE REQUIREMENTS\n"
        "a. Private Code Sharing, during the Competition Period, you are not allowed to privately share "
        "source or executable code developed in connection with or based upon the Competition Data.\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["X"], "prediction": [0.1]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_prefers_real_submission_section_over_rules_heading(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = data_dir / "context"
    cache_dir = data_dir / ".kagglebot_cache"
    context_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample = cache_dir / "sample_submission_synth.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    (context_dir / "overview.md").write_text(
        "#### 6. SUBMISSION CODE REQUIREMENTS\n"
        "a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or "
        "Competition Specific Rules above, during the Competition Period, you are not allowed to privately share "
        "source or executable code developed in connection with or based upon the Competition Data.</h5>\n\n"
        "## Submission\n\n"
        "```csv\n"
        "id,prediction\n"
        "```\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["X"], "prediction": [0.1]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_sniffs_tab_delimiter_and_flags_missing_header(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"

    pd.DataFrame({"col0": ["P1", "P2"], "col1": ["T1", "T2"], "col2": [0.0, 0.0]}).to_csv(sample, index=False)
    submission.write_text("P1\tT1\t0.9\nP2\tT2\t0.8\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(str(submission), str(sample))
    message = str(exc.value)
    assert "columns mismatch" in message
    assert "missing a header row" in message


@pytest.mark.parametrize(
    ("stderr_text", "expected_kind", "expected_reason"),
    [
        (
            "Submission not allowed: This competition only accepts Submissions from Notebooks.",
            "permanent",
            "notebook_only_submission_required",
        ),
        ("400 Client Error: Bad Request for url: https://www.kaggle.com/api/v1/...", "permanent", "bad_request"),
        ("You must accept the rules before submitting", "permanent", "rules_not_accepted"),
        ("No Kaggle API credentials found", "permanent", "authentication"),
        ("Unauthorized (401)", "permanent", "authentication"),
        ("Kernel push error: Notebook not found", "permanent", "kernel_push_failed"),
        ("Kaggle kernel not found after push; aborting.", "permanent", "kernel_push_failed"),
        ("Competition is not accepting submissions", "permanent", "competition_unavailable"),
        ("Submission limit reached: maximum number of submissions", "permanent", "submission_limit"),
        ("ConnectionError: temporarily unavailable (503)", "transient", "network_or_timeout"),
        ("Bad Gateway (502)", "transient", "network_or_timeout"),
        ("Gateway Timeout 504", "transient", "network_or_timeout"),
    ],
)
def test_classify_submit_error_examples(stderr_text: str, expected_kind: str, expected_reason: str) -> None:
    classified = classify_submit_error("", stderr_text, 1)
    assert classified["kind"] == expected_kind
    assert classified["reason"] == expected_reason
    if expected_kind == "transient":
        assert classified["retry_after_seconds"] == 2
    else:
        assert classified["retry_after_seconds"] is None


def test_classify_submit_error_unknown() -> None:
    classified = classify_submit_error("", "some uncategorized cli message", 3)
    assert classified["kind"] == "unknown"
    assert classified["reason"] == "unclassified_submit_error"
    assert classified["retry_after_seconds"] is None


def test_classify_submit_error_ambiguous_notebook_bad_request() -> None:
    classified = classify_submit_error(
        "",
        (
            "400 Client Error: Bad Request for url: "
            "https://www.kaggle.com/api/v1/competitions/submissions/submit-notebook/"
            "deep-past-initiative-machine-translation"
        ),
        1,
    )
    assert classified["kind"] == "unknown"
    assert classified["reason"] == "ambiguous_notebook_bad_request"
    assert classified["retry_after_seconds"] == 3


def test_normalize_and_fingerprint_are_stable() -> None:
    a = (
        "Error at /home/user/repo/artifacts/demo/runs/20260101T000000Z-abcd1234: "
        "timeout 2026-02-15T12:00:00Z on 2026-02-15"
    )
    b = (
        "Error at /home/other/repo/artifacts/demo/runs/20260101T000000Z-efef2222: "
        "timeout 2026-02-16T12:00:00Z on 2026-02-16"
    )
    na = normalize_error_text(a)
    normalize_error_text(b)
    assert "<PATH>" in na or "<ARTIFACT_PATH>" in na
    assert "<DATETIME>" in na
    assert "<DATE>" in na
    assert compute_error_fingerprint(a, "") == compute_error_fingerprint(b, "")


def test_run_kaggle_submit_captures_stdout_stderr(monkeypatch) -> None:
    def fake_subprocess_run(*args, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(
            args=["kaggle", "competitions", "submit"],
            returncode=0,
            stdout="submit ok",
            stderr="warning line",
        )

    monkeypatch.setattr("kagglebot.submission.guard.subprocess.run", fake_subprocess_run)
    result = run_kaggle_submit(slug="demo", submission_file=Path("submission.csv"), message="m")
    assert result.returncode == 0
    assert result.stdout == "submit ok"
    assert result.stderr == "warning line"
    assert result.command[:3] == ["kaggle", "competitions", "submit"]
    assert "-q" in result.command
    assert result.duration_sec >= 0.0


def test_run_kaggle_submit_failure_includes_tails_and_returncode(monkeypatch) -> None:
    long_stdout = "X" * 8000
    long_stderr = "\n".join(f"stderr line {idx}" for idx in range(300))

    def fake_subprocess_run(*args, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(
            args=["kaggle", "competitions", "submit"],
            returncode=2,
            stdout=long_stdout,
            stderr=long_stderr,
        )

    monkeypatch.setattr("kagglebot.submission.guard.subprocess.run", fake_subprocess_run)
    with pytest.raises(SubmissionCliError) as exc:
        run_kaggle_submit(slug="demo", submission_file=Path("submission.csv"), message="m")
    err = exc.value
    assert err.exit_code == 2
    assert err.command[:3] == ["kaggle", "competitions", "submit"]
    assert "-q" in err.command
    assert len(err.stdout) <= 6000
    assert "stderr line 299" in err.stderr
    assert "stderr line 0" not in err.stderr


def test_run_kaggle_submit_kernel_uses_kernel_flag(monkeypatch) -> None:
    def fake_subprocess_run(*args, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(
            args=["kaggle", "competitions", "submit"],
            returncode=0,
            stdout="submit ok",
            stderr="",
        )

    monkeypatch.setattr("kagglebot.submission.guard.subprocess.run", fake_subprocess_run)
    result = run_kaggle_submit_kernel(slug="demo", kernel="user/demo-kernel", message="m")
    assert result.returncode == 0
    assert result.command[:3] == ["kaggle", "competitions", "submit"]
    assert "-k" in result.command
    assert "user/demo-kernel" in result.command


def test_run_kaggle_submit_kernel_supports_output_and_version(monkeypatch) -> None:
    def fake_subprocess_run(*args, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(
            args=["kaggle", "competitions", "submit"],
            returncode=0,
            stdout="submit ok",
            stderr="",
        )

    monkeypatch.setattr("kagglebot.submission.guard.subprocess.run", fake_subprocess_run)
    result = run_kaggle_submit_kernel(
        slug="demo",
        kernel="user/demo-kernel",
        message="m",
        output_file="submission.csv",
        version="3",
    )
    assert "-k" in result.command
    assert "user/demo-kernel" in result.command
    assert "-f" in result.command
    assert "submission.csv" in result.command
    assert "-v" in result.command
    assert "3" in result.command
