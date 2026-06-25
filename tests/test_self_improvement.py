from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kagglebot.knowledge import build_plan_and_initial_prompt
from kagglebot.paths import KnowledgePaths
from kagglebot.self_improvement import (
    SelfImprovementConfig,
    _best_iteration_value,
    _best_online_score,
    _read_json_object,
    _score_gap,
    load_self_improvement_context,
    run_self_improvement_cycle,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_json_object_returns_empty_for_missing_invalid_or_non_object_payload(tmp_path: Path) -> None:
    assert _read_json_object(tmp_path / "missing.json") == {}

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert _read_json_object(invalid) == {}

    array_payload = tmp_path / "array.json"
    array_payload.write_text("[]", encoding="utf-8")
    assert _read_json_object(array_payload) == {}


def test_self_improvement_score_helpers_use_shared_direction_policy() -> None:
    iterations = [{"value": 0.5}, {"value": "bad"}, {"value": 0.4}]
    outcomes = [{"score": 0.7}, {"score": 0.8}]

    assert _best_iteration_value(iterations=iterations, direction="minimize") == 0.4
    assert _best_iteration_value(iterations=iterations, direction=None) == 0.5
    assert _best_online_score(outcomes=outcomes, direction="maximize") == 0.8
    assert _score_gap(best_score=0.75, top1_score=0.9, direction="maximize") == 0.15000000000000002
    assert _score_gap(best_score=0.75, top1_score=0.6, direction="minimize") == 0.15000000000000002
    assert _score_gap(best_score=0.95, top1_score=0.9, direction="maximize") == 0.0


def test_self_improvement_report_detects_top1_gap_and_submit_failure(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    slug = "demo"
    run_id = "run-1"
    run_dir = artifacts / slug / "runs" / run_id
    _write_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "slug": slug,
            "status": "submit_failed",
            "config": {"target_direction": "maximize", "target_metric": "auc"},
        },
    )
    _write_json(run_dir / "iter-1" / "metrics.json", {"offline_value": 0.7, "score_source": "cv"})
    (run_dir / "submit_attempts.jsonl").write_text(
        "\n".join(
            [
                "not-json",
                json.dumps(["not", "a", "dict"]),
                json.dumps({"action_taken": "failed", "reason": "submit error", "iteration": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(artifacts / slug / "context" / "top1_public.json", {"score": 0.9})
    ledger = artifacts / slug / "submissions" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "event": "outcome",
                "run_id": run_id,
                "outcome": {"status": "complete", "score": 0.75},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_self_improvement_cycle(
        SelfImprovementConfig(
            artifacts_dir=artifacts,
            knowledge_paths=KnowledgePaths(workdir=tmp_path),
            invoke_codex=False,
            force=True,
        )
    )

    assert result["status"] == "written"
    report = json.loads((artifacts / "_self_improvement" / "latest.json").read_text(encoding="utf-8"))
    assert report["cause_counts"]["submit_failed"] == 1
    assert report["cause_counts"]["online_far_from_top1"] == 1
    assert report["largest_top1_gaps"][0]["top1_gap"] == 0.15000000000000002
    assert (artifacts / "_self_improvement" / "strategy_context.md").exists()
    assert (artifacts / "_self_improvement" / "experiment_backlog.json").exists()
    assert (artifacts / "_self_improvement" / "outcomes.jsonl").exists()
    assert (tmp_path / "knowledge" / "playbooks" / "global.md").exists()
    assert "submit_failed" in load_self_improvement_context(artifacts)
    backlog = json.loads((artifacts / "_self_improvement" / "experiment_backlog.json").read_text(encoding="utf-8"))
    assert "Architectural changes are allowed" in backlog[0]["architecture_scope"]


def test_self_improvement_includes_campaign_method_outcomes(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    slug = "demo"
    run_id = "run-1"
    run_dir = artifacts / slug / "runs" / run_id
    _write_json(
        run_dir / "run.json",
        {"run_id": run_id, "status": "completed", "config": {"target_direction": "maximize"}},
    )
    _write_json(run_dir / "iter-1" / "metrics.json", {"offline_value": 0.8})
    context_dir = artifacts / slug / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "campaign_outcomes.jsonl").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "method_id": "tabular-gbdt-portfolio",
                "validation_profile_id": "group_or_proxy_cv",
                "candidate_category": "strong_single",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_self_improvement_cycle(
        SelfImprovementConfig(
            artifacts_dir=artifacts,
            knowledge_paths=KnowledgePaths(workdir=tmp_path),
            invoke_codex=False,
            force=True,
        )
    )

    report = json.loads((artifacts / "_self_improvement" / "latest.json").read_text(encoding="utf-8"))
    assert report["campaign_method_counts"]["tabular-gbdt-portfolio"] == 1
    assert report["campaign_validation_profile_counts"]["group_or_proxy_cv"] == 1
    context = (artifacts / "_self_improvement" / "strategy_context.md").read_text(encoding="utf-8")
    assert "Campaign Method Outcomes" in context


def test_self_improvement_calls_codex_when_enabled_and_clean(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    _write_json(run_dir / "run.json", {"run_id": "run-1", "status": "completed", "config": {}})
    _write_json(run_dir / "iter-1" / "metrics.json", {"offline_value": 0.1})
    calls: dict[str, object] = {}

    def fake_run_codex(prompt_path, output_dir, **kwargs):  # noqa: ANN001, ANN003
        calls["prompt"] = prompt_path.read_text(encoding="utf-8")
        calls["output_dir"] = output_dir
        return SimpleNamespace(
            returncode=0,
            transcript_path=output_dir / "codex_exec.jsonl",
            last_message_path=output_dir / "codex_last_message.txt",
        )

    monkeypatch.setattr("kagglebot.self_improvement._git_dirty", lambda workdir: False)
    monkeypatch.setattr("kagglebot.self_improvement.run_codex", fake_run_codex)

    result = run_self_improvement_cycle(
        SelfImprovementConfig(
            artifacts_dir=artifacts,
            knowledge_paths=KnowledgePaths(workdir=tmp_path),
            invoke_codex=True,
            force=True,
        )
    )

    assert result["status"] == "written"
    assert result["codex_improvement"]["status"] == "completed"
    assert "Kagglebot Self-Improvement Task" in str(calls["prompt"])
    assert "Architectural changes are allowed" in str(calls["prompt"])
    assert "first-place Kaggle leaderboard performance" in str(calls["prompt"])
    assert result["codex_improvement"]["publish"]["status"] == "disabled"


def test_self_improvement_skips_codex_when_worktree_dirty(monkeypatch, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    _write_json(run_dir / "run.json", {"run_id": "run-1", "status": "completed", "config": {}})
    _write_json(run_dir / "iter-1" / "metrics.json", {"offline_value": 0.1})
    monkeypatch.setattr("kagglebot.self_improvement._git_dirty", lambda workdir: True)

    result = run_self_improvement_cycle(
        SelfImprovementConfig(
            artifacts_dir=artifacts,
            knowledge_paths=KnowledgePaths(workdir=tmp_path),
            invoke_codex=True,
            force=True,
        )
    )

    assert result["codex_improvement"]["status"] == "skipped_dirty_worktree"


def test_self_improvement_passes_publish_policy_to_controller(monkeypatch, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    _write_json(run_dir / "run.json", {"run_id": "run-1", "status": "completed", "config": {}})
    _write_json(run_dir / "iter-1" / "metrics.json", {"offline_value": 0.1})
    published: dict[str, object] = {}

    def fake_run_codex(prompt_path, output_dir, **kwargs):  # noqa: ANN001, ANN003
        return SimpleNamespace(
            returncode=0,
            transcript_path=output_dir / "codex_exec.jsonl",
            last_message_path=output_dir / "codex_last_message.txt",
        )

    def fake_publish(*, config, codex_returncode):  # noqa: ANN001
        published["enabled"] = config.publish_codex_changes
        published["returncode"] = codex_returncode
        return {"status": "pushed", "commit": "abc123"}

    monkeypatch.setattr("kagglebot.self_improvement._git_dirty", lambda workdir: False)
    monkeypatch.setattr("kagglebot.self_improvement.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.self_improvement._maybe_publish_codex_changes", fake_publish)

    result = run_self_improvement_cycle(
        SelfImprovementConfig(
            artifacts_dir=artifacts,
            knowledge_paths=KnowledgePaths(workdir=tmp_path),
            invoke_codex=True,
            publish_codex_changes=True,
            force=True,
        )
    )

    assert result["codex_improvement"]["publish"]["status"] == "pushed"
    assert published == {"enabled": True, "returncode": 0}


def test_initial_prompt_includes_self_improvement_context() -> None:
    prompt = build_plan_and_initial_prompt(
        slug="demo",
        rules_url="https://www.kaggle.com/c/demo/rules",
        profile={"tags": ["tabular"], "task": "regression", "metric": "rmse"},
        taxonomy={},
        similar_improvements=[],
        self_improvement_context="Force broader model-family search when top1_gap repeats.",
    )

    assert "## System Self-Improvement Directives" in prompt
    assert "Force broader model-family search" in prompt
