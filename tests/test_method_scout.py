from __future__ import annotations

import json
from pathlib import Path

from kagglebot.method_scout import (
    build_method_candidates,
    build_method_scout_queries,
    build_validation_registry,
    classify_source,
    method_registry_path,
    render_method_registry_for_prompt,
    run_method_scout,
    source_registry_path,
    unsafe_method_reason,
)
from kagglebot.paths import CompetitionPaths


def test_query_builder_uses_modality_metric_and_validation_regression() -> None:
    queries = build_method_scout_queries(
        slug="demo-vision",
        problem_types=["image-classification"],
        dataset_profile={"target_metric": "accuracy"},
        metric="accuracy",
        campaign_state={
            "direction": "maximize",
            "latest_submission_score": 0.72,
            "champion_score": 0.80,
        },
        max_sources=8,
    )

    query_text = "\n".join(str(item["query"]) for item in queries)
    assert "validation split public leaderboard mismatch" in query_text
    assert "image" in query_text
    assert "accuracy" in query_text
    assert any(item["purpose"] == "validation_redesign" for item in queries)


def test_source_quality_blocks_unsafe_leaderboard_proxy_method() -> None:
    source = {
        "url": "https://example.com/solution",
        "title": "LB proxy exact test match solution",
        "extracted_technique": "Use leaderboard proxy and exact matching on test rows.",
        "takeaway": "unsafe",
    }

    assert unsafe_method_reason("Use leaderboard proxy and exact matching on test rows")
    methods = build_method_candidates(
        slug="demo",
        problem_types=["tabular:binary"],
        dataset_profile={},
        metric="auc",
        sources=[source],
    )

    blocked = [method for method in methods if method.status == "blocked"]
    assert blocked
    assert blocked[0].blocked_reason


def test_method_registry_ranks_competition_specific_above_generic(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="playground-series-s6e2", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    (paths.context_dir / "research_sources.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "url": "https://www.kaggle.com/competitions/playground-series-s6e2/code",
                        "title": "Top S6E2 CatBoost XGBoost LightGBM blend",
                        "extracted_technique": "Train CatBoost, XGBoost, LightGBM and logit blend OOF predictions.",
                        "takeaway": "Competition-specific GBDT blend is strong.",
                        "query": "playground-series-s6e2 Kaggle winning solution",
                    }
                ),
                json.dumps(
                    {
                        "url": "https://example.com/generic-blog",
                        "title": "Generic tabular tips",
                        "extracted_technique": "Try random forest.",
                        "takeaway": "Generic baseline.",
                        "query": "tabular tips",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    registry = run_method_scout(
        paths=paths,
        slug=paths.slug,
        problem_types=["tabular:binary"],
        dataset_profile={"n_rows": 100000},
        metric="auc",
        mode="refresh",
    )

    assert registry["active_method_ids"]
    first_method = registry["methods"][0]
    assert isinstance(first_method["implementation_adapter"], dict)
    assert isinstance(first_method["dependency_check"], dict)
    assert first_method["source_type"] in {"competition_specific", "generic"}
    assert any(method["source_type"] == "competition_specific" for method in registry["methods"])
    assert method_registry_path(paths.context_dir).exists()
    source_registry = json.loads(source_registry_path(paths.context_dir).read_text(encoding="utf-8"))
    assert source_registry["active_source_ids"]
    assert source_registry["planned_query_ids"]


def test_method_registry_prompt_lists_active_and_blocked_methods(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    registry = run_method_scout(
        paths=paths,
        slug="demo",
        problem_types=["tabular:binary"],
        dataset_profile={},
        metric="auc",
        mode="refresh",
    )

    prompt = render_method_registry_for_prompt(registry)

    assert "Competition-specific method scout is active" in prompt
    assert "Active method candidates" in prompt
    assert "adapter=" in prompt


def test_classify_source_types() -> None:
    assert (
        classify_source(
            {"url": "https://www.kaggle.com/competitions/demo/code/ref", "title": "Demo top notebook"},
            slug="demo",
        )
        == "competition_specific"
    )
    assert classify_source({"url": "https://arxiv.org/abs/2501.12345"}, slug="demo") == "paper"
    assert classify_source({"url": "https://github.com/example/method"}, slug="demo") == "official_repo"


def test_validation_registry_records_public_regression_split_candidates() -> None:
    registry = build_validation_registry(
        slug="demo",
        problem_types=["tabular:binary"],
        campaign_state={
            "direction": "maximize",
            "latest_submission_score": 0.75,
            "champion_score": 0.82,
            "offline_online_correlation": 0.1,
        },
        sources=[],
    )

    assert registry["priority"] is True
    assert registry["public_regression_signal"] is True
    assert registry["next_action"] == "validation_redesign"
    assert registry["active_profile"] == "group_or_proxy_cv"
    assert all("run_status" in item for item in registry["profiles"])
    assert any(item["profile_id"] == "adversarial_proxy_cv" for item in registry["profiles"])


def test_research_scout_off_skips_source_registry_sources(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    (paths.context_dir / "research_sources.jsonl").write_text(
        json.dumps({"url": "https://arxiv.org/abs/2501.1", "title": "Strong paper"}) + "\n",
        encoding="utf-8",
    )

    registry = run_method_scout(
        paths=paths,
        slug="demo",
        problem_types=["tabular:binary"],
        dataset_profile={},
        metric="auc",
        mode="refresh",
        research_mode="off",
    )

    source_registry = json.loads(source_registry_path(paths.context_dir).read_text(encoding="utf-8"))
    assert registry["research_mode"] == "off"
    assert registry["source_count"] == 0
    assert source_registry["source_count"] == 0
