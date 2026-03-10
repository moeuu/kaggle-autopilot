from __future__ import annotations

import json
from pathlib import Path

from kagglebot.paths import CompetitionPaths
from kagglebot.solver.evaluate import EvaluationResult
from kagglebot.writeup import (
    build_writeup_bundle,
    infer_deliverable_mode,
    infer_deliverable_mode_from_paths,
    infer_submit_mode_from_paths,
)


def test_infer_deliverable_mode_from_paths_detects_writeup(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_md_path.write_text("This hackathon is judged by a panel.\n", encoding="utf-8")
    paths.overview_md_path.write_text("Documentation and writeup quality are part of the rubric.\n", encoding="utf-8")
    assert infer_deliverable_mode_from_paths(paths) == "writeup"


def test_infer_deliverable_mode_ignores_negative_writeup_mentions() -> None:
    mode = infer_deliverable_mode(
        "deliverable_mode=csv rather than writeup\n"
        "This is a normal leaderboard CSV competition, not a judged/writeup competition.\n"
        "Submissions must contain id,target probability predictions.\n"
    )
    assert mode == "leaderboard"


def test_infer_deliverable_mode_from_paths_prefers_csv_evidence_over_negative_writeup_mentions(
    tmp_path: Path,
) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_md_path.write_text("You may select up to two Final Submissions for judging.\n", encoding="utf-8")
    paths.context_dir.joinpath("eval_advisor").mkdir(parents=True, exist_ok=True)
    paths.context_dir.joinpath("eval_advisor", "sources_summary.md").write_text(
        "This supports deliverable_mode=csv rather than writeup.\n"
        "This is a normal leaderboard CSV competition, not a judged/writeup competition.\n",
        encoding="utf-8",
    )

    assert infer_deliverable_mode_from_paths(paths) == "leaderboard"


def test_infer_submit_mode_from_paths_detects_notebook_only_rules(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_md_path.write_text(
        "Submissions to this competition must be made through Notebooks.\n",
        encoding="utf-8",
    )

    assert infer_submit_mode_from_paths(paths) == "notebook"


def test_build_writeup_bundle_creates_report_and_metadata(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_md_path.write_text("Clinical Relevance (25)\nTechnical Quality (30)\n", encoding="utf-8")
    paths.overview_md_path.write_text("This judged hackathon requires a writeup.\n", encoding="utf-8")
    evaluation = EvaluationResult(
        score_source="cv",
        metric="accuracy",
        direction="maximize",
        value=0.91,
        std=0.02,
        train_score=None,
        val_score=None,
        fold_scores=[0.9, 0.92],
    )
    bundle = build_writeup_bundle(
        paths=paths,
        run_id="run-1",
        iteration=3,
        resolved={"deliverable_mode": "writeup", "target_metric": "accuracy", "target_score": 0.9},
        evaluation=evaluation,
        metrics_payload={"chosen_pipeline": "primary_blend::demo"},
        top1_info={"score": None},
    )

    report_path = Path(str(bundle["report_path"]))
    metadata_path = Path(str(bundle["report_path"])).parent / "writeup_metadata.json"
    assert report_path.exists()
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["status"] == "manual_finalization_required"
    assert "Clinical Relevance" in report_path.read_text(encoding="utf-8")
