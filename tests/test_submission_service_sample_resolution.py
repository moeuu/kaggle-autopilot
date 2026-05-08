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


def test_submission_service_prefers_real_data_sample_when_placeholders_exist(tmp_path: Path) -> None:
    comp_root = tmp_path / "comp"
    data_dir = comp_root / "data"
    context_dir = comp_root / "context"
    data_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    placeholder_context_sample = context_dir / "sample_submission.csv"
    placeholder_context_sample.write_text("id,prediction\n", encoding="utf-8")

    placeholder_data_sample = data_dir / "sample_submission.csv"
    placeholder_data_sample.write_text("id,prediction\n", encoding="utf-8")

    real_sample = data_dir / "SampleSubmission.csv"
    pd.DataFrame(
        {
            "ID": [1, 2],
            "LabelA": [0, 0],
            "LabelB": [0, 0],
        }
    ).to_csv(real_sample, index=False)

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "ID": [1, 2],
            "LabelA": [1, 0],
            "LabelB": [0, 1],
        }
    ).to_csv(submission_path, index=False)

    config = SubmissionConfig(
        slug="demo",
        data_dir=data_dir,
        sample_submission_path=placeholder_context_sample,
        submission_ledger_path=tmp_path / "ledger.jsonl",
        dry_run=True,
        force_submit=True,
    )
    service = SubmissionService(config)
    resolved_sample = service._resolve_sample_submission()
    assert resolved_sample == real_sample

    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared == submission_path


def test_submission_service_prefers_stage2_sample_over_stage1_by_default(tmp_path: Path) -> None:
    comp_root = tmp_path / "comp"
    data_dir = comp_root / "data"
    context_dir = comp_root / "context"
    data_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    placeholder_context_sample = context_dir / "sample_submission.csv"
    placeholder_context_sample.write_text("id,prediction\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,prediction\n", encoding="utf-8")

    stage1 = data_dir / "SampleSubmissionStage1.csv"
    pd.DataFrame({"ID": ["2022_1_2", "2022_1_3", "2022_2_3"], "Pred": [0.5, 0.5, 0.5]}).to_csv(stage1, index=False)
    stage2 = data_dir / "SampleSubmissionStage2.csv"
    pd.DataFrame({"ID": ["2026_1_2", "2026_1_3"], "Pred": [0.5, 0.5]}).to_csv(stage2, index=False)

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"ID": ["2026_1_2", "2026_1_3"], "Pred": [0.51, 0.49]}).to_csv(submission_path, index=False)

    config = SubmissionConfig(
        slug="demo",
        data_dir=data_dir,
        sample_submission_path=placeholder_context_sample,
        submission_ledger_path=tmp_path / "ledger.jsonl",
        dry_run=True,
        force_submit=True,
    )
    service = SubmissionService(config)
    resolved_sample = service._resolve_sample_submission()
    assert resolved_sample == stage2

    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared == submission_path


def test_submission_service_prefers_valid_higher_stage_over_context_sample_when_available(tmp_path: Path) -> None:
    comp_root = tmp_path / "comp"
    data_dir = comp_root / "data"
    context_dir = comp_root / "context"
    data_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    context_sample = context_dir / "sample_submission.csv"
    pd.DataFrame({"ID": ["2022_1_2", "2022_1_3", "2022_2_3"], "Pred": [0.5, 0.5, 0.5]}).to_csv(
        context_sample, index=False
    )

    stage1 = data_dir / "SampleSubmissionStage1.csv"
    pd.DataFrame({"ID": ["2022_1_2", "2022_1_3", "2022_2_3"], "Pred": [0.5, 0.5, 0.5]}).to_csv(stage1, index=False)
    stage2 = data_dir / "SampleSubmissionStage2.csv"
    pd.DataFrame({"ID": ["2026_1_2", "2026_1_3"], "Pred": [0.5, 0.5]}).to_csv(stage2, index=False)

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"ID": ["2026_1_2", "2026_1_3"], "Pred": [0.51, 0.49]}).to_csv(submission_path, index=False)

    config = SubmissionConfig(
        slug="demo",
        data_dir=data_dir,
        sample_submission_path=context_sample,
        submission_ledger_path=tmp_path / "ledger.jsonl",
        dry_run=True,
        force_submit=True,
    )
    service = SubmissionService(config)
    resolved_sample = service._resolve_sample_submission()
    assert resolved_sample == stage2

    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared == submission_path


def test_submission_service_allows_stage_override_via_env(tmp_path: Path, monkeypatch) -> None:
    comp_root = tmp_path / "comp"
    data_dir = comp_root / "data"
    context_dir = comp_root / "context"
    data_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    placeholder_context_sample = context_dir / "sample_submission.csv"
    placeholder_context_sample.write_text("id,prediction\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,prediction\n", encoding="utf-8")

    stage1 = data_dir / "SampleSubmissionStage1.csv"
    pd.DataFrame({"ID": ["2022_1_2", "2022_1_3", "2022_2_3"], "Pred": [0.5, 0.5, 0.5]}).to_csv(stage1, index=False)
    stage2 = data_dir / "SampleSubmissionStage2.csv"
    pd.DataFrame({"ID": ["2026_1_2", "2026_1_3"], "Pred": [0.5, 0.5]}).to_csv(stage2, index=False)

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"ID": ["2026_1_2", "2026_1_3"], "Pred": [0.51, 0.49]}).to_csv(submission_path, index=False)

    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_STAGE", "2")
    config = SubmissionConfig(
        slug="demo",
        data_dir=data_dir,
        sample_submission_path=placeholder_context_sample,
        submission_ledger_path=tmp_path / "ledger.jsonl",
        dry_run=True,
        force_submit=True,
    )
    service = SubmissionService(config)
    resolved_sample = service._resolve_sample_submission()
    assert resolved_sample == stage2

    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared == submission_path


def test_submission_service_uses_alternate_sample_when_primary_would_force_bad_autofix(tmp_path: Path) -> None:
    comp_root = tmp_path / "comp"
    data_dir = comp_root / "data"
    context_dir = comp_root / "context"
    data_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    stage1 = data_dir / "SampleSubmissionStage1.csv"
    pd.DataFrame({"ID": ["2022_1_2", "2022_1_3", "2022_2_3"], "Pred": [0.5, 0.5, 0.5]}).to_csv(stage1, index=False)
    stage2 = data_dir / "SampleSubmissionStage2.csv"
    pd.DataFrame({"ID": ["2026_1_2", "2026_1_3"], "Pred": [0.5, 0.5]}).to_csv(stage2, index=False)

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"ID": ["2022_1_2", "2022_1_3", "2022_2_3"], "Pred": [0.51, 0.49, 0.52]}).to_csv(
        submission_path, index=False
    )

    # Primary sample points to a mismatched stage. Service should still avoid autofixing
    # into that wrong shape when another in-data sample already validates the submission.
    config = SubmissionConfig(
        slug="demo",
        data_dir=data_dir,
        sample_submission_path=stage2,
        submission_ledger_path=tmp_path / "ledger.jsonl",
        dry_run=True,
        force_submit=True,
    )
    service = SubmissionService(config)
    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared == submission_path
