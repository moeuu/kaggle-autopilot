"""Tests for knowledge base tagging and search."""

from __future__ import annotations

import json
import sqlite3

import kagglebot.knowledge as knowledge_mod
import kagglebot.knowledge_init as knowledge_init_mod
from kagglebot.knowledge import (
    build_dataset_profile,
    build_plan_and_initial_prompt,
    derive_problem_types,
    ensure_taxonomy,
    format_error_fix_insights,
    knowledge_search,
    load_taxonomy,
    record_competition_profile,
    record_error_fix_insight,
    record_improvement,
    record_iteration,
    record_problem_type_insight,
    record_research_artifacts,
    record_run,
    resolve_error_fix_insights,
    resolve_problem_type_insights,
    resolve_research_artifacts,
)
from kagglebot.knowledge.repositories import InsightRepository
from kagglebot.knowledge_context import (
    load_problem_type_knowledge_text,
    refresh_knowledge_hints,
    resolve_problem_types_from_profile,
)
from kagglebot.paths import CompetitionPaths, KnowledgePaths


def test_knowledge_search_orders_by_overlap(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    taxonomy = ensure_taxonomy(knowledge_paths)

    record_competition_profile(
        knowledge_paths=knowledge_paths,
        taxonomy=taxonomy,
        slug="comp-a",
        competition_url=None,
        profile={"metric": "accuracy", "task": "classification", "tags": ["tabular", "binary"]},
    )
    record_competition_profile(
        knowledge_paths=knowledge_paths,
        taxonomy=taxonomy,
        slug="comp-b",
        competition_url=None,
        profile={"metric": "rmse", "task": "regression", "tags": ["tabular"]},
    )

    results = knowledge_search(knowledge_paths, ["tabular", "binary"], limit=5)
    assert results[0]["slug"] == "comp-a"


def test_load_taxonomy_yaml(tmp_path) -> None:
    content = """
data_modality:
  - tabular
  - text
aliases:
  bin: binary
"""
    path = tmp_path / "taxonomy.yml"
    path.write_text(content, encoding="utf-8")
    data = load_taxonomy(path)
    assert "tabular" in data["tags"]
    assert data["aliases"]["bin"] == "binary"


def test_build_dataset_profile_samples_oversized_tables(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,feature,target\n1,10,0\n2,20,1\n3,30,0\n4,40,1\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n5,50\n6,60\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n5,0\n6,0\n", encoding="utf-8")

    monkeypatch.setenv("KAGGLEBOT_PROFILE_MAX_TABLE_BYTES", "1")
    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["train_rows"] == 4
    sampling = profile["profile_sampling"]
    assert sampling["enabled"] is True
    assert sampling["train"] is True
    assert sampling["test"] is True
    assert sampling["sample_submission"] is True


def test_profile_max_table_bytes_env_uses_shared_number_parsing(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_PROFILE_MAX_TABLE_BYTES", "nan")
    assert knowledge_mod._profile_max_table_bytes() == 256 * 1024 * 1024  # noqa: SLF001
    assert knowledge_init_mod._profile_max_table_bytes() == 256 * 1024 * 1024  # noqa: SLF001

    monkeypatch.setenv("KAGGLEBOT_PROFILE_MAX_TABLE_BYTES", "0")
    assert knowledge_mod._profile_max_table_bytes() == 256 * 1024 * 1024  # noqa: SLF001

    monkeypatch.setenv("KAGGLEBOT_PROFILE_MAX_TABLE_BYTES", "1024")
    assert knowledge_mod._profile_max_table_bytes() == 1024  # noqa: SLF001


def test_build_dataset_profile_handles_json_list_features(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.json").write_text(
        json.dumps(
            [
                {"id": "a", "grid": [[1, 2], [3, 4]], "target": 0},
                {"id": "b", "grid": [[4, 3], [2, 1]], "target": 1},
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "test.json").write_text(
        json.dumps(
            [
                {"id": "c", "grid": [[1, 1], [1, 1]]},
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "sample_submission.json").write_text(
        json.dumps(
            [
                {"id": "c", "target": 0},
            ]
        ),
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["file_extension_counts"] == {".json": 3}


def test_build_dataset_profile_does_not_treat_times_suffix_as_timeseries(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "id,Total asset turnover rate (Times),target\n1,0.1,0\n2,0.2,1\n3,0.3,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,Total asset turnover rate (Times)\n4,0.4\n5,0.5\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "tabular"


def test_build_dataset_profile_handles_march_mania_submission_only_format(tmp_path) -> None:
    data_dir = tmp_path / "march-machine-learning-mania-2026" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "MRegularSeasonCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2025,10,1101,70,1102,65,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "WRegularSeasonCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2025,10,3101,80,3102,72,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "MNCAATourneyCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2024,136,1101,75,1102,70,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "WNCAATourneyCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2024,136,3101,77,3102,71,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "SampleSubmissionStage1.csv").write_text(
        "ID,Pred\n2026_1101_1102,0.5\n2026_3101_3102,0.5\n",
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["sample_submission_file"] == "SampleSubmissionStage1.csv"
    assert profile["id_column"] == "ID"
    assert profile["target_column"] == "Pred"
    assert profile["task"] == "classification"
    assert profile["metric"] == "brier_score"
    assert profile["split_strategy_hint"] == "group_kfold"
    assert "binary" in profile["tags"]


def test_build_plan_and_initial_prompt_handles_unknown_train_dimensions(tmp_path) -> None:
    data_dir = tmp_path / "march-machine-learning-mania-2026" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "MRegularSeasonCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2025,10,1101,70,1102,65,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "WRegularSeasonCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2025,10,3101,80,3102,72,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "MNCAATourneyCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2024,136,1101,75,1102,70,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "WNCAATourneyCompactResults.csv").write_text(
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n2024,136,3101,77,3102,71,N,0\n",
        encoding="utf-8",
    )
    (data_dir / "SampleSubmissionStage1.csv").write_text(
        "ID,Pred\n2026_1101_1102,0.5\n2026_3101_3102,0.5\n",
        encoding="utf-8",
    )

    profile = build_dataset_profile(data_dir)

    prompt = build_plan_and_initial_prompt(
        slug="march-machine-learning-mania-2026",
        rules_url="https://www.kaggle.com/competitions/march-machine-learning-mania-2026",
        profile=profile,
        taxonomy={},
        similar_improvements=[],
    )

    assert "**Dataset**: train table unavailable; sample/test view: 2 rows × 2 columns" in prompt


def test_problem_type_insight_record_and_resolve(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    profile = {"modality": "tabular", "task": "regression", "tags": ["tabular", "regression"]}
    problem_types = derive_problem_types(profile)
    assert "tabular:regression" in problem_types

    record_problem_type_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-a",
        run_id="run-1",
        iteration=1,
        problem_types=problem_types,
        why_poor="Model was underfit with weak features and high validation error.",
        how_improved="Added CatBoost and richer feature engineering with longer training.",
        delta_offline=0.12,
        outcome_bucket="good",
        submission_score=0.8123,
    )

    insights = resolve_problem_type_insights(knowledge_paths, ["tabular:regression"], limit=5)
    assert insights
    first = insights[0]
    assert first["problem_type"] == "tabular:regression"
    assert first["cause_category"] != ""
    assert first["fix_category"] != ""
    assert first["outcome_bucket"] == "good"
    assert first["submission_score"] == 0.8123


def test_error_fix_insight_record_and_resolve(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    problem_types = ["tabular:binary", "tabular"]

    record_error_fix_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-b",
        run_id="run-2",
        iteration=2,
        problem_types=problem_types,
        error_message="ModuleNotFoundError: No module named 'featurewiz'",
        fix_summary="Removed featurewiz import and added sklearn fallback.",
        resolved=True,
        outcome_bucket="low",
        submission_score=0.731,
    )

    insights = resolve_error_fix_insights(knowledge_paths, ["tabular:binary"], limit=5)
    assert insights
    first = insights[0]
    assert first["error_category"] == "dependency_missing"
    assert bool(first["resolved"]) is True
    assert first["outcome_bucket"] == "low"
    assert first["submission_score"] == 0.731

    rendered = format_error_fix_insights(insights, limit=5)
    assert "dependency_missing" in rendered
    assert "featurewiz" in rendered


def test_load_problem_type_knowledge_text_renders_shared_context(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    profile_path = tmp_path / "dataset_profile.json"
    profile_path.write_text(
        json.dumps({"modality": "tabular", "task": "binary", "tags": ["classification"]}),
        encoding="utf-8",
    )
    problem_types = ["tabular:binary", "tabular", "binary", "classification"]
    record_problem_type_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-a",
        run_id="run-1",
        iteration=1,
        problem_types=problem_types,
        why_poor="Validation did not match public leaderboard.",
        how_improved="Use stratified folds and check prediction distribution.",
        delta_offline=0.012,
        outcome_bucket="high",
        submission_score=0.88,
    )
    record_error_fix_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-a",
        run_id="run-1",
        iteration=1,
        problem_types=problem_types,
        error_message="Submission Scoring Error: incorrect format",
        fix_summary="Align columns and ids to sample_submission.",
        resolved=True,
        outcome_bucket="high",
        submission_score=0.88,
    )
    record_research_artifacts(
        knowledge_paths=knowledge_paths,
        slug="comp-a",
        problem_types=problem_types,
        research_sources_jsonl='{"url":"https://example.com"}',
        research_summary_md="Research summary.",
    )

    text = load_problem_type_knowledge_text(
        dataset_profile_path=profile_path,
        knowledge_paths=knowledge_paths,
        include_research=True,
    )

    assert "Problem-type knowledge" in text
    assert "Error-fix knowledge" in text
    assert "Cross-competition research artifacts" in text
    assert "Submission Scoring Error" in text


def test_load_problem_type_knowledge_text_can_skip_research(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    profile_path = tmp_path / "dataset_profile.json"
    profile_path.write_text(json.dumps({"modality": "tabular", "task": "binary"}), encoding="utf-8")

    text = load_problem_type_knowledge_text(
        dataset_profile_path=profile_path,
        knowledge_paths=knowledge_paths,
        include_research=False,
    )

    assert "No prior problem-type insights available." in text
    assert "No prior error-fix insights available." in text
    assert "Cross-competition research artifacts" not in text


def test_resolve_research_artifacts_ignores_invalid_problem_types_json(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    record_research_artifacts(
        knowledge_paths=knowledge_paths,
        slug="comp-a",
        problem_types=["tabular", "binary"],
        research_sources_jsonl='{"url":"https://example.com"}',
        research_summary_md="Research summary.",
    )
    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        conn.execute(
            "UPDATE competition_research SET problem_types_json = ? WHERE slug = ?",
            ("{", "comp-a"),
        )

    records = resolve_research_artifacts(knowledge_paths=knowledge_paths, problem_types=["tabular"])

    assert records
    assert records[0]["problem_types"] == []


def test_resolve_problem_types_from_profile_handles_json_profile(tmp_path) -> None:
    profile_path = tmp_path / "dataset_profile.json"
    profile_path.write_text(json.dumps({"modality": "tabular", "task": "regression"}), encoding="utf-8")

    assert resolve_problem_types_from_profile(dataset_profile_path=profile_path) == [
        "tabular:regression",
        "tabular",
        "regression",
    ]


def test_resolve_problem_types_from_profile_handles_invalid_profile(tmp_path) -> None:
    profile_path = tmp_path / "dataset_profile.json"
    profile_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    assert resolve_problem_types_from_profile(dataset_profile_path=profile_path) == ["unknown"]


def test_refresh_knowledge_hints_writes_similar_competition_context(tmp_path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path / "knowledge")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(json.dumps({"tags": ["tabular", "binary"]}), encoding="utf-8")
    taxonomy = ensure_taxonomy(knowledge_paths)
    record_competition_profile(
        knowledge_paths=knowledge_paths,
        taxonomy=taxonomy,
        slug="prior-comp",
        competition_url=None,
        profile={"tags": ["tabular", "binary"], "metric": "auc"},
    )
    record_run(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        slug="prior-comp",
        compute="local",
        goal_metric="auc",
        goal_score=0.8,
        direction="maximize",
    )
    record_improvement(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=1,
        summary="Stratified folds improved public score.",
        delta_offline=0.02,
    )

    refresh_knowledge_hints(paths=paths, knowledge_paths=knowledge_paths)

    hints = paths.knowledge_hints_path.read_text(encoding="utf-8")
    assert "Similar competitions and what improved score" in hints
    assert "prior-comp" in hints
    assert "Stratified folds improved public score." in hints
    assert "No self-improvement context available yet." in hints


def test_knowledge_classifies_external_signal_and_online_mismatch(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)

    record_error_fix_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-c",
        run_id="run-3",
        iteration=1,
        problem_types=["tabular:binary"],
        error_message="ORIG_proba constant_fallback: original_data_found=false and reference inputs missing.",
        fix_summary="Recover original data from reference inputs and disable constant_fallback.",
        resolved=False,
        outcome_bucket="unknown",
        submission_score=None,
    )
    error_insights = resolve_error_fix_insights(knowledge_paths, ["tabular:binary"], limit=5)
    assert error_insights[0]["error_category"] == "external_signal_missing"

    record_problem_type_insight(
        knowledge_paths=knowledge_paths,
        slug="comp-c",
        run_id="run-3",
        iteration=2,
        problem_types=["tabular:binary"],
        why_poor="Offline improved but public leaderboard regressed, indicating an online mismatch.",
        how_improved="Ban same-family-only tuning and increase model-family diversity with blending.",
        delta_offline=None,
        outcome_bucket="low",
        submission_score=0.812,
    )
    problem_insights = resolve_problem_type_insights(knowledge_paths, ["tabular:binary"], limit=5)
    assert problem_insights[0]["cause_category"] == "online_mismatch"
    assert problem_insights[0]["fix_category"] == "model_diversification"


def test_record_iteration_upserts_on_duplicate_run_iteration(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)

    record_iteration(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=1,
        score_source="holdout",
        offline_value=0.42,
        offline_std=0.02,
        top1_public_score=0.5,
        met_target=False,
        git_commit="aaa111",
    )
    record_iteration(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=1,
        score_source="cv",
        offline_value=0.35,
        offline_std=0.01,
        top1_public_score=0.45,
        met_target=True,
        git_commit="bbb222",
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT score_source, offline_value, offline_std, top1_public_score, met_target, git_commit
            FROM iterations
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 1),
        ).fetchone()

    assert row == ("cv", 0.35, 0.01, 0.45, 1, "bbb222")


def test_record_iteration_falls_back_to_update_on_unique_violation(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    state = {"insert_calls": 0, "raised": False}

    class _FlakyConnection:
        def __init__(self, path, state_map):
            self._conn = sqlite3.connect(path)
            self._state = state_map

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._conn.__exit__(exc_type, exc, tb)

        def execute(self, sql, params=()):
            if "INSERT INTO iterations" in sql:
                self._state["insert_calls"] += 1
                if self._state["insert_calls"] == 2 and not self._state["raised"]:
                    self._state["raised"] = True
                    raise sqlite3.IntegrityError("UNIQUE constraint failed: iterations.run_id, iterations.iter")
            return self._conn.execute(sql, params)

    repo = InsightRepository(
        knowledge_paths,
        ensure_db=knowledge_mod._ensure_db,
        connect=lambda path: _FlakyConnection(path, state),
    )

    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="holdout",
        offline_value=0.42,
        offline_std=0.02,
        top1_public_score=0.5,
        met_target=False,
        git_commit="aaa111",
    )
    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="cv",
        offline_value=0.35,
        offline_std=0.01,
        top1_public_score=0.45,
        met_target=True,
        git_commit="bbb222",
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT score_source, offline_value, offline_std, top1_public_score, met_target, git_commit
            FROM iterations
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 1),
        ).fetchone()

    assert state["raised"] is True
    assert row == ("cv", 0.35, 0.01, 0.45, 1, "bbb222")


def test_record_iteration_falls_back_to_update_on_unique_violation_code_name(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    state = {"insert_calls": 0, "raised": False}

    class _FlakyConnection:
        def __init__(self, path, state_map):
            self._conn = sqlite3.connect(path)
            self._state = state_map

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._conn.__exit__(exc_type, exc, tb)

        def execute(self, sql, params=()):
            if "INSERT INTO iterations" in sql:
                self._state["insert_calls"] += 1
                if self._state["insert_calls"] == 2 and not self._state["raised"]:
                    self._state["raised"] = True
                    exc = sqlite3.IntegrityError("constraint failed")
                    exc.sqlite_errorcode = sqlite3.SQLITE_CONSTRAINT_UNIQUE
                    exc.sqlite_errorname = "SQLITE_CONSTRAINT_UNIQUE"
                    raise exc
            return self._conn.execute(sql, params)

    repo = InsightRepository(
        knowledge_paths,
        ensure_db=knowledge_mod._ensure_db,
        connect=lambda path: _FlakyConnection(path, state),
    )

    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="holdout",
        offline_value=0.42,
        offline_std=0.02,
        top1_public_score=0.5,
        met_target=False,
        git_commit="aaa111",
    )
    repo.record_iteration(
        run_id="run-1",
        iteration=1,
        score_source="cv",
        offline_value=0.35,
        offline_std=0.01,
        top1_public_score=0.45,
        met_target=True,
        git_commit="bbb222",
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT score_source, offline_value, offline_std, top1_public_score, met_target, git_commit
            FROM iterations
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 1),
        ).fetchone()

    assert state["raised"] is True
    assert row == ("cv", 0.35, 0.01, 0.45, 1, "bbb222")


def test_record_improvement_upserts_on_duplicate_run_iteration(tmp_path) -> None:
    knowledge_paths = KnowledgePaths(workdir=tmp_path)

    record_improvement(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=2,
        summary="first",
        delta_offline=0.01,
    )
    record_improvement(
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        iteration=2,
        summary="second",
        delta_offline=0.08,
    )

    with sqlite3.connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            """
            SELECT summary, delta_offline
            FROM improvements
            WHERE run_id = ? AND iter = ?
            """,
            ("run-1", 2),
        ).fetchone()

    assert row == ("second", 0.08)
