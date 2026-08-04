from __future__ import annotations

import json
from pathlib import Path

import kagglebot.eval.advisor as advisor_module
from kagglebot.eval import EvaluationAdvisor, validate_advisor_payload
from kagglebot.exec_utils import CommandResult
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


def test_validate_advisor_payload_accepts_brier_score_metric() -> None:
    payload = _valid_payload()
    payload["evaluation_spec"]["metric_name"] = "brier_score"
    payload["evaluation_spec"]["direction"] = "minimize"
    spec, _, _, issues = validate_advisor_payload(payload)
    assert issues == []
    assert spec is not None
    assert spec["metric_name"] == "brier_score"
    assert spec["direction"] == "minimize"


def test_validate_advisor_payload_accepts_aurc_as_minimize_metric() -> None:
    payload = _valid_payload()
    payload["evaluation_spec"]["metric_name"] = "aurc"
    payload["evaluation_spec"]["direction"] = "minimize"

    spec, _, _, issues = validate_advisor_payload(payload)

    assert issues == []
    assert spec is not None
    assert spec["metric_name"] == "aurc"
    assert spec["direction"] == "minimize"


def test_parse_json_response_drops_only_malformed_search_queries() -> None:
    payload = _valid_payload()
    valid_prefix = json.dumps(
        {
            "evaluation_spec": payload["evaluation_spec"],
            "sources_summary_md": payload["sources_summary_md"],
        }
    )[:-1]
    malformed = f'{valid_prefix},\n"search_queries": ["site:example.com "AURC""]}}'

    parsed = advisor_module._parse_json_response(malformed)

    assert parsed is not None
    assert parsed["evaluation_spec"] == payload["evaluation_spec"]
    assert parsed["sources_summary_md"] == payload["sources_summary_md"]
    assert parsed["search_queries"] == []


def test_validate_advisor_payload_schema_rejects_extra_keys() -> None:
    payload = _valid_payload()
    payload["evaluation_spec"]["unsupported"] = True  # type: ignore[index]
    spec, _, _, issues = validate_advisor_payload(payload)
    assert spec is None
    assert issues
    assert any("unexpected keys" in item for item in issues)


def test_validate_advisor_payload_schema_accepts_optional_faithfulness() -> None:
    payload = _valid_payload()
    payload["evaluation_spec"]["faithfulness"] = {
        "accepted_score_sources": ["cv", "holdout"],
        "require_metric_match": True,
        "require_split_match": True,
        "require_trusted_score_source": True,
        "require_competition_faithful": True,
        "require_full_dataset": False,
    }
    spec, _, _, issues = validate_advisor_payload(payload)
    assert issues == []
    assert spec is not None
    faithfulness = spec.get("faithfulness")
    assert isinstance(faithfulness, dict)
    assert faithfulness["accepted_score_sources"] == ["cv", "holdout"]


def test_validate_advisor_payload_accepts_deliverable_mode_writeup() -> None:
    payload = _valid_payload()
    payload["evaluation_spec"]["deliverable_mode"] = "writeup"
    spec, _, _, issues = validate_advisor_payload(payload)
    assert issues == []
    assert spec is not None
    assert spec["deliverable_mode"] == "writeup"


def test_validate_advisor_payload_canonicalizes_legacy_csv_deliverable_mode() -> None:
    payload = _valid_payload()
    payload["evaluation_spec"]["deliverable_mode"] = "csv"
    spec, _, _, issues = validate_advisor_payload(payload)
    assert issues == []
    assert spec is not None
    assert spec["deliverable_mode"] == "leaderboard"


def test_validate_advisor_payload_accepts_medal_target_fields() -> None:
    payload = _valid_payload()
    payload["evaluation_spec"]["target_medal"] = "bronze"
    payload["evaluation_spec"]["target_rank_percentile"] = 0.1
    spec, _, _, issues = validate_advisor_payload(payload)
    assert issues == []
    assert spec is not None
    assert spec["target_medal"] == "bronze"
    assert spec["target_rank_percentile"] == 0.1


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


def test_evaluation_advisor_fallback_recognizes_aurc_before_generic_accuracy_text(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"metric": "accuracy", "task": "classification", "modality": "image"}, indent=2),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("Submissions are ranked on the leaderboard.\n", encoding="utf-8")
    paths.overview_md_path.write_text(
        "The official metric is AURC (Area Under the Risk-Coverage Curve); lower is better. "
        "It measures transcription accuracy and confidence calibration.\n",
        encoding="utf-8",
    )
    paths.submission_format_md_path.write_text("image_file,text,confidence\n", encoding="utf-8")

    advisor = EvaluationAdvisor(
        paths=paths,
        slug="demo",
        search_capability_check=lambda: False,
    )

    spec, source = advisor.ensure_spec()

    assert source == "fallback"
    assert spec["metric_name"] == "aurc"
    assert spec["direction"] == "minimize"


def test_evaluation_advisor_fallback_applies_competition_policy_overrides(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"metric": "auc", "task": "classification", "modality": "tabular"}, indent=2),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("Evaluation metric is ROC-AUC.\n", encoding="utf-8")
    paths.overview_md_path.write_text("Leaderboard competition.\n", encoding="utf-8")
    paths.submission_format_md_path.write_text("id,target\n", encoding="utf-8")
    paths.competition_policy_path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "fallback_overrides": {
                        "seeds": [42, 2025],
                        "repeats": 2,
                        "ci_method": "bootstrap",
                    },
                    "search_stop_rank_percentile": 0.08,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    advisor = EvaluationAdvisor(
        paths=paths,
        slug="demo",
        dry_run=False,
        force=False,
        search_capability_check=lambda: False,
    )
    spec, source = advisor.ensure_spec()
    assert source == "fallback"
    assert spec["seeds"] == [42, 2025]
    assert spec["repeats"] == 2
    assert spec["ci_method"] == "bootstrap"
    assert spec["target_rank_percentile"] == 0.001


def test_evaluation_advisor_fallback_infers_writeup_mode_from_context(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"metric": "accuracy", "task": "classification", "modality": "tabular"}, indent=2),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("This hackathon is judged by a panel and requires a writeup.\n", encoding="utf-8")
    paths.overview_md_path.write_text(
        "Scoring follows a rubric with documentation and writeup quality.\n",
        encoding="utf-8",
    )
    paths.submission_format_md_path.write_text(
        "Writeup submission details are described in the overview.\n",
        encoding="utf-8",
    )

    advisor = EvaluationAdvisor(
        paths=paths,
        slug="demo",
        dry_run=False,
        force=False,
        search_capability_check=lambda: False,
    )
    spec, source = advisor.ensure_spec()
    assert source == "fallback"
    assert spec["deliverable_mode"] == "writeup"
    assert spec["submit_mode"] == "file"


def test_evaluation_advisor_fallback_keeps_csv_mode_when_writeup_terms_are_negative(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"metric": "auc", "task": "classification", "modality": "tabular"}, indent=2),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("You may select up to two Final Submissions for judging.\n", encoding="utf-8")
    paths.overview_md_path.write_text(
        "This is a normal leaderboard CSV competition, not a judged/writeup competition.\n",
        encoding="utf-8",
    )
    paths.submission_format_md_path.write_text(
        "Submissions must contain id,target probability predictions.\n"
        "This supports deliverable_mode=csv rather than writeup.\n",
        encoding="utf-8",
    )

    advisor = EvaluationAdvisor(
        paths=paths,
        slug="demo",
        dry_run=False,
        force=False,
        search_capability_check=lambda: False,
    )
    spec, source = advisor.ensure_spec()
    assert source == "fallback"
    assert spec["deliverable_mode"] == "leaderboard"
    assert spec["submit_mode"] == "file"
    assert spec["target_medal"] == "winner"
    assert spec["target_rank_percentile"] == 0.001


def test_evaluation_advisor_fallback_infers_notebook_submit_mode_from_context(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"metric": "auc", "task": "classification", "modality": "tabular"}, indent=2),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text(
        "Submissions to this competition must be made through Notebooks.\n",
        encoding="utf-8",
    )
    paths.overview_md_path.write_text("Leaderboard competition.\n", encoding="utf-8")
    paths.submission_format_md_path.write_text("Upload predictions after running a notebook.\n", encoding="utf-8")

    advisor = EvaluationAdvisor(
        paths=paths,
        slug="demo",
        dry_run=False,
        force=False,
        search_capability_check=lambda: False,
    )
    spec, source = advisor.ensure_spec()
    assert source == "fallback"
    assert spec["deliverable_mode"] == "leaderboard"
    assert spec["submit_mode"] == "notebook"


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


def test_evaluation_advisor_refreshes_structurally_invalid_frozen_spec(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"metric": "rmse", "task": "regression", "modality": "tabular"}),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("Evaluation metric is RMSE.\n", encoding="utf-8")
    paths.context_dir.joinpath("evaluation_spec.json").write_text("{}\n", encoding="utf-8")

    advisor = EvaluationAdvisor(
        paths=paths,
        slug="demo",
        search_capability_check=lambda: False,
    )
    spec, source = advisor.ensure_spec()

    assert source == "fallback"
    assert spec["metric_name"] == "rmse"
    assert spec["direction"] == "minimize"


def test_evaluation_advisor_refreshes_frozen_spec_with_wrong_metric_direction(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"metric": "auc", "task": "classification", "modality": "tabular"}),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("Evaluation metric is AUC.\n", encoding="utf-8")
    invalid = _valid_payload()["evaluation_spec"] | {"direction": "minimize"}
    paths.context_dir.joinpath("evaluation_spec.json").write_text(json.dumps(invalid), encoding="utf-8")

    advisor = EvaluationAdvisor(
        paths=paths,
        slug="demo",
        search_capability_check=lambda: False,
    )
    spec, source = advisor.ensure_spec()

    assert source == "fallback"
    assert spec["metric_name"] == "auc"
    assert spec["direction"] == "maximize"


def test_evaluation_advisor_refreshes_frozen_writeup_mode_when_code_submission_contract_conflicts(
    tmp_path: Path,
) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"metric": "accuracy", "task": "classification", "modality": "video"}, indent=2),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text(
        "For hackathons, a submission may be judged by a panel using a rubric.\n",
        encoding="utf-8",
    )
    paths.overview_md_path.write_text(
        "Your submission CSV must contain one row per predicted track.\n"
        "Submissions to this competition must be made through Notebooks.\n"
        "The submission file must be named submission.csv.\n"
        "See the Code Competition FAQ.\n",
        encoding="utf-8",
    )
    stale = _valid_payload()["evaluation_spec"] | {
        "deliverable_mode": "writeup",
        "submit_mode": "file",
        "metric_name": "accuracy",
    }
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
    assert spec["deliverable_mode"] == "leaderboard"
    assert spec["submit_mode"] == "notebook"


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


def test_supports_live_search_allows_modern_codex_without_search_flag(monkeypatch) -> None:
    def _fake_run_command(args: list[str], **kwargs: object) -> CommandResult:  # noqa: ARG001
        assert args == ["codex", "exec", "--help"]
        return CommandResult(
            args=args,
            returncode=0,
            stdout="Usage: codex exec [OPTIONS]\n      --enable <FEATURE>\n",
            stderr="",
            duration_sec=0.01,
        )

    monkeypatch.setattr(advisor_module, "run_command", _fake_run_command)
    assert advisor_module._supports_live_search() is True


def test_supports_live_search_returns_false_when_codex_missing(monkeypatch) -> None:
    def _raise_file_not_found(args: list[str], **kwargs: object) -> CommandResult:  # noqa: ARG001
        raise FileNotFoundError("codex")

    monkeypatch.setattr(advisor_module, "run_command", _raise_file_not_found)
    assert advisor_module._supports_live_search() is False
