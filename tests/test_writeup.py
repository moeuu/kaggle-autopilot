from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from kagglebot.paths import CompetitionPaths
from kagglebot.solver.evaluate import EvaluationResult
from kagglebot.writeup import (
    build_writeup_bundle,
    infer_code_competition_from_paths,
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


def test_infer_code_competition_from_paths_ignores_invalid_evaluation_spec(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.joinpath("evaluation_spec.json").write_text("{", encoding="utf-8")
    paths.data_dir.joinpath("test.csv").write_text("id\n1\n", encoding="utf-8")
    paths.data_dir.joinpath("sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    assert infer_code_competition_from_paths(paths) is False


def test_infer_code_competition_from_plan_notebook_tiny_public_contract(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.plan_path.parent.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text(json.dumps({"submit_mode": "notebook"}), encoding="utf-8")
    paths.data_dir.joinpath("test.csv").write_text("id,text\n1,a\n2,b\n3,c\n", encoding="utf-8")
    paths.data_dir.joinpath("sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    assert infer_code_competition_from_paths(paths) is True


def test_infer_code_competition_from_plan_notebook_tiny_public_jsonl_contract(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.plan_path.parent.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text(json.dumps({"submit_mode": "notebook"}), encoding="utf-8")
    paths.data_dir.joinpath("test.jsonl").write_text(
        '{"id": 1, "text": "a"}\n{"id": 2, "text": "b"}\n{"id": 3, "text": "c"}\n',
        encoding="utf-8",
    )
    paths.data_dir.joinpath("sample_submission.jsonl").write_text(
        '{"id": 1, "target": 0}\n{"id": 2, "target": 0}\n{"id": 3, "target": 0}\n',
        encoding="utf-8",
    )

    assert infer_code_competition_from_paths(paths) is True


def test_infer_code_competition_from_plan_notebook_tiny_public_alias_contract(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.plan_path.parent.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text(json.dumps({"submit_mode": "notebook"}), encoding="utf-8")
    paths.data_dir.joinpath("PublicTest.jsonl").write_text(
        '{"id": 1, "text": "a"}\n{"id": 2, "text": "b"}\n{"id": 3, "text": "c"}\n',
        encoding="utf-8",
    )
    pd.DataFrame({"id": [1, 2, 3], "target": [0, 0, 0]}).to_excel(
        paths.data_dir / "SampleSubmission.xlsx",
        index=False,
    )

    assert infer_code_competition_from_paths(paths) is True


def test_infer_code_competition_from_plan_uses_context_sample_alias_for_tiny_public_contract(
    tmp_path: Path,
) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.plan_path.parent.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text(json.dumps({"submit_mode": "notebook"}), encoding="utf-8")
    paths.data_dir.joinpath("test.csv").write_text("id,text\n1,a\n2,b\n3,c\n", encoding="utf-8")
    paths.context_dir.joinpath("AnswerTemplate.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    assert infer_code_competition_from_paths(paths) is True


def test_infer_code_competition_from_plan_notebook_tiny_public_compressed_and_excel_contract(
    tmp_path: Path,
) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.plan_path.parent.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text(json.dumps({"submit_mode": "notebook"}), encoding="utf-8")
    with gzip.open(paths.data_dir / "test.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id,text\n1,a\n2,b\n3,c\n")
    pd.DataFrame({"id": [1, 2, 3], "target": [0, 0, 0]}).to_excel(
        paths.data_dir / "sample_submission.xlsx",
        index=False,
    )

    assert infer_code_competition_from_paths(paths) is True


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
    evidence_path = Path(str(bundle["evidence_path"]))
    assert report_path.exists()
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["status"] == "manual_finalization_required"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["proxy_metric"] == "accuracy"
    assert evidence["proxy_value"] == 0.91
    assert "Clinical Relevance" in report_path.read_text(encoding="utf-8")
