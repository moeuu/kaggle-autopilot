from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.submission_semantics import (
    analyze_submission_semantics,
    discover_submission_metrics_path,
    semantic_finding_messages,
    validate_autopilot_submission_semantics,
)


def _finding_codes(report: dict[str, object]) -> set[str]:
    findings = report.get("findings")
    return {str(row.get("code")) for row in findings if isinstance(findings, list) and isinstance(row, dict)}


def test_semantic_preflight_allows_diverse_predictions(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    sample = tmp_path / "sample_submission.csv"
    submission.write_text("id,target\na,0.1\nb,0.4\nc,0.8\n", encoding="utf-8")
    sample.write_text("id,target\na,0.0\nb,0.0\nc,0.0\n", encoding="utf-8")

    report = analyze_submission_semantics(
        submission_path=submission,
        sample_submission_path=sample,
    )

    assert report["applicable"] is True
    assert report["block_submit"] is False
    assert report["findings"] == []


def test_semantic_preflight_treats_class_id_as_prediction_not_identifier(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("row_id,class_id\na,0\nb,1\nc,2\n", encoding="utf-8")

    report = analyze_submission_semantics(submission_path=submission)

    assert report["prediction_columns"] == ["class_id"]
    assert report["unique_prediction_rows"] == 3
    assert report["block_submit"] is False


def test_semantic_preflight_blocks_row_constant_predictions(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("id,target\na,0.5\nb,0.5\nc,0.5\n", encoding="utf-8")

    report = analyze_submission_semantics(submission_path=submission)

    assert "row_constant_predictions" in _finding_codes(report)
    assert report["block_submit"] is True


def test_semantic_preflight_blocks_constant_single_column_without_id(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("prediction\nanswer\nanswer\nanswer\n", encoding="utf-8")

    messages = semantic_finding_messages(submission_path=submission)

    assert any("identical predictions" in message for message in messages)


def test_semantic_preflight_blocks_identical_multioutput_heads(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text(
        "row_id,class_a,class_b,class_c\na,0.1,0.1,0.1\nb,0.4,0.4,0.4\nc,0.8,0.8,0.8\n",
        encoding="utf-8",
    )

    report = analyze_submission_semantics(submission_path=submission)

    assert "identical_prediction_heads" in _finding_codes(report)


def test_semantic_preflight_blocks_unchanged_sample_template(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    sample = tmp_path / "sample_submission.csv"
    payload = "id,target\na,0.1\nb,0.9\n"
    submission.write_text(payload, encoding="utf-8")
    sample.write_text(payload, encoding="utf-8")

    report = analyze_submission_semantics(
        submission_path=submission,
        sample_submission_path=sample,
    )

    assert "sample_template_unchanged" in _finding_codes(report)


def test_semantic_preflight_blocks_placeholder_text(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text(
        "id,translation\na,placeholder\nb,placeholder\nc,placeholder\n",
        encoding="utf-8",
    )

    report = analyze_submission_semantics(submission_path=submission)

    assert "placeholder_text_predictions" in _finding_codes(report)


def test_semantic_preflight_blocks_metrics_fallback_and_pipeline_mismatch(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("id,target\na,0.1\nb,0.4\nc,0.8\n", encoding="utf-8")

    report = analyze_submission_semantics(
        submission_path=submission,
        metrics_payload={
            "selected_pipeline": "trained_model",
            "emitted_pipeline": "dummy_fallback",
            "fallback_only": True,
            "submission_row_count": 2,
            "submission_filename": "wrong.csv",
        },
    )

    assert {
        "fallback_submission_output",
        "selected_emitted_pipeline_mismatch",
        "metrics_submission_row_count_mismatch",
        "metrics_submission_filename_mismatch",
    }.issubset(_finding_codes(report))


def test_validate_semantics_persists_machine_readable_report(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    report_path = tmp_path / "run" / "submission_semantic_preflight.json"
    submission.write_text("id,target\na,0.5\nb,0.5\nc,0.5\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="semantic submission preflight"):
        validate_autopilot_submission_semantics(
            submission_path=submission,
            sample_submission_path=None,
            data_dir=None,
            metrics_path=None,
            report_path=report_path,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["block_submit"] is True
    assert report["findings"][0]["code"] == "row_constant_predictions"


def test_discover_submission_metrics_prefers_same_iteration(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    submission = run_dir / "iter-2" / "submission.csv"
    metrics = run_dir / "iter-2" / "metrics.json"
    submission.parent.mkdir(parents=True)
    submission.write_text("id,target\na,0.1\n", encoding="utf-8")
    metrics.write_text('{"selected_pipeline":"model"}', encoding="utf-8")

    resolved = discover_submission_metrics_path(
        submission_path=submission,
        run_dir=run_dir,
    )

    assert resolved == metrics
