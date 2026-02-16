from __future__ import annotations

from pathlib import Path

import pandas as pd

from kagglebot.submission_service import SubmissionConfig, SubmissionService


def test_submission_service_prefers_discovered_sample_over_header_only_context_sample(tmp_path: Path) -> None:
    comp_root = tmp_path / "comp"
    data_dir = comp_root / "data"
    context_dir = comp_root / "context"
    images_test_dir = data_dir / "images" / "test"
    images_test_dir.mkdir(parents=True, exist_ok=True)
    (images_test_dir / "0.jpg").write_bytes(b"")
    (images_test_dir / "1.jpg").write_bytes(b"")

    pd.DataFrame({"filename": ["0.jpg", "1.jpg"], "right_place": [0, 1]}).to_csv(
        data_dir / "train_labels.csv", index=False
    )

    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission File\n\n```csv\nfilename,right_place,prediction_string\n0.jpg,0,-\n```\n",
        encoding="utf-8",
    )
    placeholder_sample = context_dir / "sample_submission.csv"
    placeholder_sample.write_text("id,target\n", encoding="utf-8")

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"filename": ["0.jpg", "1.jpg"], "right_place": [0, 0], "prediction_string": ["-", "-"]}).to_csv(
        submission_path, index=False
    )

    config = SubmissionConfig(
        slug="demo",
        data_dir=data_dir,
        sample_submission_path=placeholder_sample,
        submission_ledger_path=tmp_path / "ledger.jsonl",
        dry_run=True,
        force_submit=True,
    )
    service = SubmissionService(config)
    resolved_sample = service._resolve_sample_submission()
    assert resolved_sample.name == "sample_submission_synth.csv"
    assert resolved_sample != placeholder_sample
    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared.exists()


def test_submission_service_synthesizes_sample_for_non_tabular_competitions(tmp_path: Path) -> None:
    comp_root = tmp_path / "comp"
    data_dir = comp_root / "data"
    context_dir = comp_root / "context"
    images_test_dir = data_dir / "images" / "test"
    images_test_dir.mkdir(parents=True, exist_ok=True)
    (images_test_dir / "0.jpg").write_bytes(b"")
    (images_test_dir / "1.jpg").write_bytes(b"")

    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission File\n\n```csv\nfilename,right_place,prediction_string\n0.jpg,0,-\n```\n",
        encoding="utf-8",
    )
    placeholder_sample = context_dir / "sample_submission.csv"
    placeholder_sample.write_text("id,target\n", encoding="utf-8")

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"filename": ["0.jpg", "1.jpg"], "right_place": [0, 0], "prediction_string": ["-", "-"]}).to_csv(
        submission_path, index=False
    )

    config = SubmissionConfig(
        slug="demo",
        data_dir=data_dir,
        sample_submission_path=placeholder_sample,
        submission_ledger_path=tmp_path / "ledger.jsonl",
        dry_run=True,
        force_submit=True,
    )
    service = SubmissionService(config)
    resolved_sample = service._resolve_sample_submission()
    assert resolved_sample.name == "sample_submission_synth.csv"
    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared.exists()
