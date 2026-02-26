from __future__ import annotations

import json
from pathlib import Path

from kagglebot.eval import EvaluationAdvisor, validate_advisor_payload
from kagglebot.paths import CompetitionPaths


def _valid_payload() -> dict[str, object]:
    return {
        "evaluation_spec": {
            "metric_name": "auc",
            "direction": "maximize",
            "split_strategy": "stratified_kfold",
            "n_splits": 5,
            "seeds": [42, 1337],
            "repeats": 2,
            "ci_method": "normal",
            "ci_alpha": 0.05,
            "readiness_rule": {
                "method": "ci_bound",
                "k": 1.0,
                "target_score": 0.8,
                "submission_gate": "always",
            },
            "drift_check": {"enabled": False, "drift_weight": 1.0},
            "stop_policy": {
                "min_delta": 0.001,
                "no_improve_patience": 2,
                "same_config_patience": 2,
            },
        },
        "sources_summary_md": "- https://example.com/source\n",
        "search_queries": ["kaggle auc stratified kfold best practice"],
    }


def test_validate_advisor_payload_schema_passes() -> None:
    spec, sources_summary, queries, issues = validate_advisor_payload(_valid_payload())
    assert issues == []
    assert spec is not None
    assert spec["metric_name"] == "auc"
    assert spec["split_strategy"] == "stratified_kfold"
    assert sources_summary is not None
    assert queries is not None


def test_validate_advisor_payload_schema_rejects_extra_keys() -> None:
    payload = _valid_payload()
    payload["evaluation_spec"]["unsupported"] = True  # type: ignore[index]
    spec, _, _, issues = validate_advisor_payload(payload)
    assert spec is None
    assert issues
    assert any("unexpected keys" in item for item in issues)


def test_evaluation_advisor_fallback_writes_spec_when_unavailable(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"metric": "rmse", "task": "regression", "modality": "tabular"}, indent=2),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("Evaluation metric is RMSE.\n", encoding="utf-8")
    paths.overview_md_path.write_text("Tabular regression competition.\n", encoding="utf-8")
    paths.submission_format_md_path.write_text("id,target\n", encoding="utf-8")

    advisor = EvaluationAdvisor(
        paths=paths,
        slug="demo",
        dry_run=False,
        force=False,
        search_capability_check=lambda: False,
    )
    spec, source = advisor.ensure_spec()
    assert source == "fallback"
    assert spec["metric_name"] == "rmse"
    assert (paths.context_dir / "evaluation_spec.json").exists()
    assert (paths.context_dir / "eval_advisor" / "sources_summary.md").exists()


def test_evaluation_advisor_fallback_prefers_context_metric_over_profile_metric(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"metric": "accuracy", "task": "classification", "modality": "tabular"}, indent=2),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("Model submissions are ranked by the official metric.\n", encoding="utf-8")
    paths.overview_md_path.write_text("Evaluation metric: ROC-AUC.\n", encoding="utf-8")
    paths.submission_format_md_path.write_text("id,target\n", encoding="utf-8")

    advisor = EvaluationAdvisor(
        paths=paths,
        slug="demo",
        dry_run=False,
        force=False,
        search_capability_check=lambda: False,
    )
    spec, source = advisor.ensure_spec()
    assert source == "fallback"
    assert spec["metric_name"] == "auc"


def test_evaluation_advisor_refreshes_stale_frozen_spec(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"metric": "accuracy", "task": "classification", "modality": "tabular"}, indent=2),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("Competition ranking uses AUC.\n", encoding="utf-8")
    paths.overview_md_path.write_text("Evaluation: AUC.\n", encoding="utf-8")
    paths.submission_format_md_path.write_text("id,target\n", encoding="utf-8")
    stale = _valid_payload()["evaluation_spec"] | {"metric_name": "accuracy", "direction": "maximize"}
    paths.context_dir.joinpath("evaluation_spec.json").write_text(json.dumps(stale, indent=2), encoding="utf-8")

    advisor = EvaluationAdvisor(
        paths=paths,
        slug="demo",
        dry_run=False,
        force=False,
        search_capability_check=lambda: False,
    )
    spec, source = advisor.ensure_spec()
    assert source == "fallback"
    assert spec["metric_name"] == "auc"


def test_evaluation_advisor_preserves_frozen_spec_without_force(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    frozen = _valid_payload()["evaluation_spec"]
    paths.context_dir.joinpath("evaluation_spec.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    called = {"runner": 0}

    def _never_called(prompt_path: Path, output_dir: Path):  # noqa: ARG001
        called["runner"] += 1
        raise AssertionError("runner must not be called when frozen spec exists")

    advisor = EvaluationAdvisor(
        paths=paths,
        slug="demo",
        dry_run=False,
        force=False,
        strategy_runner=_never_called,
        search_capability_check=lambda: True,
    )
    spec, source = advisor.ensure_spec()
    assert source == "frozen"
    assert called["runner"] == 0
    assert spec["metric_name"] == "auc"
