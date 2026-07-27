from __future__ import annotations

import json
from pathlib import Path

from kagglebot.deliverable_artifacts import resolve_deliverable_artifact_contract


def test_resolves_required_writeup_notebook_output_from_competition_context(tmp_path: Path) -> None:
    competition_dir = tmp_path / "demo"
    context_dir = competition_dir / "context"
    context_dir.mkdir(parents=True)
    (competition_dir / "plan.json").write_text(
        json.dumps({"deliverable_mode": "writeup", "submit_mode": "notebook"}),
        encoding="utf-8",
    )
    (context_dir / "overview.md").write_text(
        "Each team must submit one Kaggle Notebook (required).\nThe notebook outputs a file named **features.csv**.\n",
        encoding="utf-8",
    )

    contract = resolve_deliverable_artifact_contract(competition_dir)

    assert contract.deliverable_mode == "writeup"
    assert contract.submit_mode == "notebook"
    assert contract.requires_notebook is True
    assert contract.required_output_names == ("features.csv",)


def test_explicit_required_outputs_are_deduplicated_and_helper_files_are_excluded(tmp_path: Path) -> None:
    competition_dir = tmp_path / "demo"
    competition_dir.mkdir()
    (competition_dir / "plan.json").write_text(
        json.dumps(
            {
                "deliverable_mode": "writeup",
                "required_output_files": ["result.parquet", "result.parquet", "metrics.json"],
            }
        ),
        encoding="utf-8",
    )

    contract = resolve_deliverable_artifact_contract(competition_dir)

    assert contract.required_output_names == ("result.parquet",)


def test_leaderboard_contract_does_not_infer_arbitrary_data_filenames(tmp_path: Path) -> None:
    competition_dir = tmp_path / "demo"
    context_dir = competition_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "overview.md").write_text(
        "Train with train.csv and evaluate against test.csv. See data.json for details.",
        encoding="utf-8",
    )

    contract = resolve_deliverable_artifact_contract(competition_dir)

    assert contract.deliverable_mode == "leaderboard"
    assert contract.required_output_names == ()


def test_writeup_detects_attached_public_notebook_requirement(tmp_path: Path) -> None:
    competition_dir = tmp_path / "demo"
    context_dir = competition_dir / "context"
    context_dir.mkdir(parents=True)
    (competition_dir / "plan.json").write_text(
        json.dumps({"deliverable_mode": "writeup", "submit_mode": "file"}),
        encoding="utf-8",
    )
    (context_dir / "overview.md").write_text(
        "A valid submission must contain an Attached Public Notebook.",
        encoding="utf-8",
    )

    contract = resolve_deliverable_artifact_contract(competition_dir)

    assert contract.requires_notebook is True
