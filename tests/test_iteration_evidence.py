from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kagglebot.iteration_evidence import (
    EVIDENCE_FILENAME,
    IterationEvidenceIntegrityError,
    prepare_iteration_evidence,
    verify_iteration_evidence_bundle,
)
from kagglebot.paths import CompetitionPaths


def _evaluation(*, value: float, source: str = "cv") -> SimpleNamespace:
    return SimpleNamespace(
        metric="auc",
        direction="maximize",
        value=value,
        score_source=source,
        std=0.01,
        fold_scores=[value - 0.01, value + 0.01],
    )


def _write_iteration_metrics(
    iter_dir: Path,
    *,
    value: float,
    source: str,
    trusted: bool,
) -> None:
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metric": "auc",
                "direction": "maximize",
                "score_source": source,
                "offline_value": value,
                "offline_std": 0.01,
                "accuracy_potential": {
                    "trusted": trusted,
                    "quality_reasons": [] if trusted else ["untrusted_score_source"],
                },
                "competition_faithfulness": {
                    "faithful": trusted,
                    "reasons": [] if trusted else ["competition_score_source_mismatch"],
                },
            }
        ),
        encoding="utf-8",
    )
    (iter_dir / "evaluation_report.json").write_text(
        json.dumps(
            {
                "metric_name": "auc",
                "direction": "maximize",
                "score_source": source,
                "metric_value": value,
                "split_strategy": "stratified_kfold",
                "n_splits": 2,
                "per_fold_scores": [value - 0.01, value + 0.01],
            }
        ),
        encoding="utf-8",
    )


def _make_paths(tmp_path: Path) -> CompetitionPaths:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.kernel_source_dir.mkdir(parents=True, exist_ok=True)
    paths.kernel_source_dir.joinpath("kernel.py").write_text("MODEL = 'baseline'\n", encoding="utf-8")
    paths.plan_path.parent.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text(json.dumps({"model": "baseline", "folds": 3}), encoding="utf-8")
    return paths


def test_iteration_evidence_attributes_material_change_across_comparable_scores(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    run_id = "run-1"
    iter1 = paths.iter_dir(run_id, 1)
    _write_iteration_metrics(iter1, value=0.60, source="cv", trusted=True)

    first = prepare_iteration_evidence(
        paths=paths,
        slug="demo",
        run_id=run_id,
        iteration=1,
        evaluation=_evaluation(value=0.60),
        target_score=0.80,
        current_score=0.60,
        current_score_source="cv",
        delta_offline=None,
        pending_problem_insights=[],
        previous_submission_history=None,
    )
    assert first.path == iter1 / EVIDENCE_FILENAME

    strategy_dir = iter1 / "agent" / "improve_strategy-01"
    strategy_dir.mkdir(parents=True)
    strategy_dir.joinpath("strategy_last_message.txt").write_text(
        "Hypothesis: missing categorical interactions. Add CatBoost and compare OOF AUC.\n",
        encoding="utf-8",
    )
    iter1.joinpath("agent", "codex_last_message.txt").write_text(
        "Added a CatBoost candidate and OOF selection.\n",
        encoding="utf-8",
    )
    paths.kernel_source_dir.joinpath("kernel.py").write_text("MODEL = 'catboost'\n", encoding="utf-8")
    paths.plan_path.write_text(json.dumps({"model": "catboost", "folds": 5}), encoding="utf-8")
    prepare_iteration_evidence(
        paths=paths,
        slug="demo",
        run_id=run_id,
        iteration=1,
        evaluation=_evaluation(value=0.60),
        target_score=0.80,
        current_score=0.60,
        current_score_source="cv",
        delta_offline=None,
        pending_problem_insights=[],
        previous_submission_history=None,
    )
    assert iter1.joinpath("kernel_before_improvement.py").read_text(encoding="utf-8") == "MODEL = 'baseline'\n"
    assert json.loads(iter1.joinpath("plan_before_improvement.json").read_text(encoding="utf-8"))["model"] == (
        "baseline"
    )

    iter2 = paths.iter_dir(run_id, 2)
    _write_iteration_metrics(iter2, value=0.70, source="cv", trusted=True)
    second = prepare_iteration_evidence(
        paths=paths,
        slug="demo",
        run_id=run_id,
        iteration=2,
        evaluation=_evaluation(value=0.70),
        target_score=0.80,
        current_score=0.70,
        current_score_source="cv",
        delta_offline=0.10,
        pending_problem_insights=[],
        previous_submission_history=None,
    )

    transition = second.payload["transitions_observed"][0]
    assert transition["score_comparison"]["comparable"] is True
    assert transition["score_comparison"]["improvement_delta"] == pytest.approx(0.10)
    assert transition["hypothesis_outcome"] == "supported"
    assert transition["material_change_detected"] is True
    assert "baseline" in transition["kernel_change"]["diff_excerpt"]
    assert "catboost" in transition["kernel_change"]["diff_excerpt"]
    assert any("model" in field for field in transition["plan_changed_fields"])
    assert "missing categorical interactions" in transition["strategy_that_requested_change"]["excerpt"]
    assert "iter-1→iter-2: comparable=True" in second.prompt_summary


def test_iteration_evidence_refuses_delta_for_untrusted_or_incomparable_score(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    run_id = "run-2"
    iter1 = paths.iter_dir(run_id, 1)
    _write_iteration_metrics(iter1, value=0.32, source="cv", trusted=True)
    prepare_iteration_evidence(
        paths=paths,
        slug="demo",
        run_id=run_id,
        iteration=1,
        evaluation=_evaluation(value=0.32),
        target_score=0.97,
        current_score=0.32,
        current_score_source="cv",
        delta_offline=None,
        pending_problem_insights=[],
        previous_submission_history=None,
    )

    iter2 = paths.iter_dir(run_id, 2)
    _write_iteration_metrics(iter2, value=0.897, source="public_lb_reference", trusted=False)
    bundle = prepare_iteration_evidence(
        paths=paths,
        slug="demo",
        run_id=run_id,
        iteration=2,
        evaluation=_evaluation(value=0.897, source="public_lb_reference"),
        target_score=0.97,
        current_score=0.897,
        current_score_source="public_lb_reference",
        delta_offline=0.577,
        pending_problem_insights=[],
        previous_submission_history=None,
    )

    transition = bundle.payload["transitions_observed"][0]
    comparison = transition["score_comparison"]
    assert comparison["comparable"] is False
    assert comparison["raw_delta"] is None
    assert comparison["improvement_delta"] is None
    assert "score_source_mismatch:cv!=public_lb_reference" in comparison["reasons"]
    assert "current_score_untrusted" in comparison["reasons"]
    assert transition["hypothesis_outcome"] == "not_assessable"
    assert any("Current score is untrusted" in gap for gap in bundle.payload["decision_requirements"]["evidence_gaps"])


def test_iteration_evidence_recovery_requires_exact_persisted_digest(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    run_id = "run-3"
    iter1 = paths.iter_dir(run_id, 1)
    _write_iteration_metrics(iter1, value=0.60, source="cv", trusted=True)
    bundle = prepare_iteration_evidence(
        paths=paths,
        slug="demo",
        run_id=run_id,
        iteration=1,
        evaluation=_evaluation(value=0.60),
        target_score=0.80,
        current_score=0.60,
        current_score_source="cv",
        delta_offline=None,
        pending_problem_insights=[],
        previous_submission_history=None,
    )

    recovered = prepare_iteration_evidence(
        paths=paths,
        slug="demo",
        run_id=run_id,
        iteration=1,
        evaluation=_evaluation(value=0.60),
        target_score=0.80,
        current_score=0.60,
        current_score_source="cv",
        delta_offline=None,
        pending_problem_insights=[],
        previous_submission_history=None,
        expected_path=bundle.path,
        expected_sha256=bundle.sha256,
    )
    assert recovered.sha256 == bundle.sha256

    bundle.path.write_text("{}", encoding="utf-8")
    with pytest.raises(IterationEvidenceIntegrityError, match="digest changed"):
        prepare_iteration_evidence(
            paths=paths,
            slug="demo",
            run_id=run_id,
            iteration=1,
            evaluation=_evaluation(value=0.60),
            target_score=0.80,
            current_score=0.60,
            current_score_source="cv",
            delta_offline=None,
            pending_problem_insights=[],
            previous_submission_history=None,
            expected_path=bundle.path,
            expected_sha256=bundle.sha256,
        )


def test_iteration_evidence_detects_snapshot_tampering_during_implementation(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    run_id = "run-4"
    iter1 = paths.iter_dir(run_id, 1)
    _write_iteration_metrics(iter1, value=0.60, source="cv", trusted=True)
    bundle = prepare_iteration_evidence(
        paths=paths,
        slug="demo",
        run_id=run_id,
        iteration=1,
        evaluation=_evaluation(value=0.60),
        target_score=0.80,
        current_score=0.60,
        current_score_source="cv",
        delta_offline=None,
        pending_problem_insights=[],
        previous_submission_history=None,
    )

    verify_iteration_evidence_bundle(bundle)
    iter1.joinpath("kernel_before_improvement.py").write_text("MODEL = 'tampered'\n", encoding="utf-8")
    with pytest.raises(IterationEvidenceIntegrityError, match="kernel_snapshot was modified"):
        verify_iteration_evidence_bundle(bundle)
