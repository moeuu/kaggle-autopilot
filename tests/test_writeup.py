from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from kagglebot.paths import CompetitionPaths
from kagglebot.solver.evaluate import EvaluationResult
from kagglebot.writeup import (
    attach_published_writeup_notebook,
    build_writeup_bundle,
    extract_writeup_constraints,
    infer_code_competition_from_paths,
    infer_deliverable_mode,
    infer_deliverable_mode_from_paths,
    infer_submit_mode_from_paths,
    normalize_deliverable_mode,
)


def test_infer_deliverable_mode_from_paths_detects_writeup(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_md_path.write_text("This hackathon is judged by a panel.\n", encoding="utf-8")
    paths.overview_md_path.write_text("Documentation and writeup quality are part of the rubric.\n", encoding="utf-8")
    assert infer_deliverable_mode_from_paths(paths) == "writeup"


@pytest.mark.parametrize(
    "text",
    [
        "Submissions to this competition must be made through Writeups.",
        "Submit a Writeup describing your approach and supporting evidence.",
        "Writeups will be judged according to the published criteria.",
        "All submissions are judged based on the following rubric.",
        "Your submission should include a Kaggle writeup documenting the agent and a link to its code.",
    ],
)
def test_infer_deliverable_mode_detects_kaggle_writeups_wording(text: str) -> None:
    assert infer_deliverable_mode(text) == "writeup"


def test_normalize_deliverable_mode_accepts_plural_writeups() -> None:
    assert normalize_deliverable_mode("Writeups") == "writeup"


def test_infer_deliverable_mode_ignores_negative_writeup_mentions() -> None:
    mode = infer_deliverable_mode(
        "deliverable_mode=csv rather than writeup\n"
        "This is a normal leaderboard CSV competition, not a judged/writeup competition.\n"
        "Submissions must contain id,target probability predictions.\n"
    )
    assert mode == "leaderboard"


def test_infer_deliverable_mode_prefers_explicit_code_submission_contract_over_rules_boilerplate() -> None:
    mode = infer_deliverable_mode(
        "For hackathons, each team may submit one submission only.\n"
        "Each submission will be ranked by the evaluation metric or evaluation rubric "
        "in the case of hackathon competitions.\n"
        "A panel may require documentation from a winner.\n",
        "Your submission CSV must contain one row per predicted track.\n"
        "Submissions to this competition must be made through Notebooks.\n"
        "The submission file must be named submission.csv.\n"
        "See the Code Competition FAQ.\n",
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
    assert payload["status"] == "ready_for_submit"
    assert payload["validation"]["valid"] is True
    assert payload["content_sha256"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["proxy_metric"] == "accuracy"
    assert evidence["proxy_value"] == 0.91
    report_text = report_path.read_text(encoding="utf-8")
    assert "Clinical Relevance" in report_text
    assert "Use this section to" not in report_text


def test_build_writeup_bundle_seals_external_evaluation_report_and_archive(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="skill-demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text(
        json.dumps({"deliverable_mode": "writeup", "submit_mode": "file", "track": "static_skills"}),
        encoding="utf-8",
    )
    paths.overview_md_path.write_text(
        "This judged hackathon requires a Writeup. Submit a single `.zip` with this structure:\n"
        "```\nsubmission.zip\n└── skills/example/SKILL.md\n```\n"
        "The Writeup must be no more than 2000 words.\n",
        encoding="utf-8",
    )
    iter_dir = paths.iter_dir("run-1", 1)
    iter_dir.mkdir(parents=True)
    archive = iter_dir / "submission.zip"
    archive.write_bytes(b"validated archive")
    source_report = tmp_path / "kernel-writeup.md"
    source_report.write_text(
        "# Skill Demo\n\n" + "Evidence-backed external evaluation report. " * 100 + "\n",
        encoding="utf-8",
    )

    bundle = build_writeup_bundle(
        paths=paths,
        run_id="run-1",
        iteration=1,
        resolved={"deliverable_mode": "writeup", "target_direction": "maximize", "track": "static_skills"},
        evaluation=None,
        metrics_payload={
            "selected_pipeline": "static_skill_portfolio",
            "cv_metric_name": "routing_screening_objective",
            "cv_metric_value": 0.75,
        },
        top1_info=None,
        source_report_path=source_report,
    )

    assert bundle["status"] == "ready_for_submit"
    assert bundle["external_evaluation_required"] is True
    assert bundle["track"] == "static_skills"
    assert bundle["artifact_contract"]["requires_resource_attachment"] is True
    assert bundle["required_artifacts"][0]["name"] == "submission.zip"
    assert Path(str(bundle["report_path"])).read_text(encoding="utf-8") == source_report.read_text(encoding="utf-8")


def test_writeup_notebook_artifact_is_validated_before_submit(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text(
        json.dumps({"deliverable_mode": "writeup", "submit_mode": "notebook"}),
        encoding="utf-8",
    )
    paths.overview_md_path.write_text(
        "This judged hackathon requires a writeup and one Kaggle Notebook (required).\n"
        "The notebook outputs a file named features.csv.\n",
        encoding="utf-8",
    )
    iter_dir = paths.iter_dir("run-1", 1)
    iter_dir.mkdir(parents=True)
    (iter_dir / "features.csv").write_text("matchid,teamid,feature\n1,2,3\n", encoding="utf-8")
    evaluation = EvaluationResult(
        score_source="contract",
        metric="contract_score",
        direction="maximize",
        value=1.0,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )

    bundle = build_writeup_bundle(
        paths=paths,
        run_id="run-1",
        iteration=1,
        resolved={"deliverable_mode": "writeup"},
        evaluation=evaluation,
        metrics_payload={"chosen_pipeline": "feature_builder"},
        top1_info=None,
    )

    assert bundle["status"] == "ready_for_notebook_publish"
    assert bundle["required_artifacts"][0]["name"] == "features.csv"
    assert bundle["notebook"]["status"] == "publish_required"

    remote_output = tmp_path / "remote-output"
    remote_output.mkdir()
    (remote_output / "features.csv").write_text("matchid,teamid,feature\n1,2,3\n", encoding="utf-8")
    finalized = attach_published_writeup_notebook(
        bundle,
        kernel_id="owner/private-demo-notebook",
        output_dir=remote_output,
    )

    assert finalized["status"] == "ready_for_submit"
    assert finalized["notebook"]["private"] is True
    assert "owner/private-demo-notebook" in Path(str(finalized["report_path"])).read_text(encoding="utf-8")


def test_writeup_notebook_publication_rejects_output_drift(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    report_path.write_text("# Demo\n\n## Appendix\n", encoding="utf-8")
    local_artifact = tmp_path / "local-features.csv"
    local_artifact.write_text("a\n1\n", encoding="utf-8")
    remote_output = tmp_path / "remote-output"
    remote_output.mkdir()
    (remote_output / "features.csv").write_text("a\n2\n", encoding="utf-8")
    bundle = {
        "status": "ready_for_notebook_publish",
        "report_path": str(report_path),
        "artifact_contract": {"required_output_names": ["features.csv"]},
        "required_artifacts": [
            {
                "name": "features.csv",
                "path": str(local_artifact),
                "sha256": hashlib.sha256(local_artifact.read_bytes()).hexdigest(),
            }
        ],
        "validation": {"valid": True, "errors": []},
    }

    finalized = attach_published_writeup_notebook(
        bundle,
        kernel_id="owner/private-demo-notebook",
        output_dir=remote_output,
    )

    assert finalized["status"] == "validation_failed"
    assert any("differs" in error for error in finalized["validation"]["errors"])


def test_extract_writeup_constraints_and_block_short_report(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.overview_md_path.write_text(
        "The formal Kaggle Writeup should be between 1,500 and 4,000 words.",
        encoding="utf-8",
    )

    assert extract_writeup_constraints(paths) == {"min_words": 1500, "max_words": 4000}
