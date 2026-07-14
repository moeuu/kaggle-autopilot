from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kagglebot.knowledge import build_plan_and_initial_prompt
from kagglebot.knowledge.event_store import search_agent_events
from kagglebot.knowledge.skill_registry import search_skills, upsert_skill
from kagglebot.paths import KnowledgePaths
from kagglebot.repository_transaction import RepositoryBaseline
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


def _mock_repository_baseline(monkeypatch, tmp_path: Path) -> RepositoryBaseline:
    baseline = RepositoryBaseline(
        workdir=tmp_path,
        branch="main",
        upstream="origin/main",
        remote="origin",
        remote_branch="main",
        repository_url="https://github.com/example/kaggle-autopilot",
        head_sha="a" * 40,
        remote_sha="a" * 40,
    )
    monkeypatch.setattr("kagglebot.self_improvement.verify_clean_pushed_repository", lambda workdir: baseline)
    monkeypatch.setattr("kagglebot.self_improvement.revalidate_repository_baseline", lambda current: None)
    monkeypatch.setattr(
        "kagglebot.self_improvement.validate_repository_oracle_response",
        lambda text, current: {
            "repository_url": baseline.repository_url,
            "baseline_sha": baseline.head_sha,
            "proposed_files": ["src/kagglebot/example.py"],
            "acceptance_tests": ["uv run pytest -q"],
            "rollback_strategy": "revert",
        },
    )
    return baseline


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
    assert (artifacts / "_self_improvement" / "skill_candidates.json").exists()
    assert (artifacts / "_self_improvement" / "outcomes.jsonl").exists()
    assert (tmp_path / "knowledge" / "playbooks" / "global.md").exists()
    assert (tmp_path / "knowledge" / "skills" / "submit_failure_recovery.md").exists()
    assert "submit_failed" in load_self_improvement_context(artifacts)
    backlog = json.loads((artifacts / "_self_improvement" / "experiment_backlog.json").read_text(encoding="utf-8"))
    assert "Architectural changes are allowed" in backlog[0]["architecture_scope"]
    candidates = json.loads((artifacts / "_self_improvement" / "skill_candidates.json").read_text(encoding="utf-8"))
    assert candidates[0]["skill_id"] == "submit_failure_recovery"
    assert report["consolidated_knowledge"]["lesson_count"] == 1
    assert search_agent_events(knowledge_paths=KnowledgePaths(workdir=tmp_path), query="submit_failed", limit=5)
    skills = search_skills(
        knowledge_paths=KnowledgePaths(workdir=tmp_path),
        problem_types=["submission"],
        query="submit failure",
        limit=5,
    )
    assert skills[0]["skill_id"] == "submit_failure_recovery"


def test_self_improvement_records_only_implemented_skill_outcomes(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    slug = "demo"
    run_id = "run-1"
    run_dir = artifacts / slug / "runs" / run_id
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    _write_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "slug": slug,
            "status": "completed",
            "config": {"target_direction": "maximize", "target_metric": "auc"},
        },
    )
    _write_json(run_dir / "iter-1" / "metrics.json", {"offline_value": 0.8, "score_source": "cv"})
    _write_json(artifacts / slug / "context" / "top1_public.json", {"score": 0.9})
    (artifacts / slug / "context" / "relevant_skills.json").write_text(
        json.dumps(
            [
                {"skill_id": "tabular_binary_oof_blend"},
                {"skill_id": "tabular_binary_oof_blend"},
                {"skill_id": "suggested_but_unused"},
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "iter-1" / "applied_knowledge.json",
        {
            "skills": [
                {
                    "skill_id": "tabular_binary_oof_blend",
                    "lifecycle": "implemented",
                    "evidence": "OOF blend was used by the selected pipeline.",
                }
            ]
        },
    )
    ledger = artifacts / slug / "submissions" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "event": "outcome",
                "run_id": run_id,
                "outcome": {"status": "complete", "score": 0.91},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    upsert_skill(
        knowledge_paths=knowledge_paths,
        skill_id="tabular_binary_oof_blend",
        title="Tabular Binary OOF Blend",
        summary="Use leak-free OOF blending for tabular binary tasks.",
        body="Build OOF predictions and blend diverse GBDT families.",
        tags=["tabular", "binary"],
        problem_types=["tabular", "binary"],
        status="active",
        source="test",
    )
    upsert_skill(
        knowledge_paths=knowledge_paths,
        skill_id="suggested_but_unused",
        title="Suggested But Unused",
        summary="This skill was suggested but not applied.",
        body="Do not attribute an outcome without implementation evidence.",
        tags=["tabular"],
        problem_types=["tabular"],
        status="active",
        source="test",
    )

    run_self_improvement_cycle(
        SelfImprovementConfig(
            artifacts_dir=artifacts,
            knowledge_paths=knowledge_paths,
            invoke_codex=False,
            force=True,
        )
    )

    outcomes = [
        json.loads(line)
        for line in (artifacts / "_self_improvement" / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    skills = search_skills(knowledge_paths=knowledge_paths, problem_types=["tabular"], query="OOF blend", limit=5)
    assert outcomes[0]["used_skills"] == ["tabular_binary_oof_blend"]
    assert skills[0]["skill_id"] == "tabular_binary_oof_blend"
    assert skills[0]["usage_count"] == 1
    assert skills[0]["success_count"] == 1
    unused = search_skills(
        knowledge_paths=knowledge_paths,
        problem_types=["tabular"],
        query="Suggested But Unused",
        limit=5,
    )
    unused_skill = next(item for item in unused if item["skill_id"] == "suggested_but_unused")
    assert unused_skill["usage_count"] == 0


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
    _mock_repository_baseline(monkeypatch, tmp_path)

    def fake_run_strategy(prompt_path, output_dir, **kwargs):  # noqa: ANN001, ANN003
        calls["strategy_prompt"] = prompt_path.read_text(encoding="utf-8")
        calls["strategy_output_dir"] = output_dir
        calls["strategy_engine"] = kwargs.get("engine")
        last_message_path = output_dir / "strategy_last_message.txt"
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text("Prioritize submit diagnostics and retry classification.\n", encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout="Prioritize submit diagnostics and retry classification.",
            stderr="",
            transcript_path=output_dir / "strategy_exec.txt",
            last_message_path=last_message_path,
            engine="oracle",
        )

    def fake_run_codex(prompt_path, output_dir, **kwargs):  # noqa: ANN001, ANN003
        calls["prompt"] = prompt_path.read_text(encoding="utf-8")
        calls["output_dir"] = output_dir
        return SimpleNamespace(
            returncode=0,
            transcript_path=output_dir / "codex_exec.jsonl",
            last_message_path=output_dir / "codex_last_message.txt",
        )

    monkeypatch.setattr("kagglebot.self_improvement._git_dirty", lambda workdir: False)
    monkeypatch.setattr("kagglebot.self_improvement.run_strategy", fake_run_strategy)
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
    assert result["codex_improvement"]["strategy_engine"] == "oracle"
    assert calls["strategy_engine"] == "oracle"
    assert "Kagglebot Self-Improvement Strategy" in str(calls["strategy_prompt"])
    assert "Architectural changes are allowed" in str(calls["strategy_prompt"])
    assert "Kagglebot Self-Improvement Implementation" in str(calls["prompt"])
    assert "Prioritize submit diagnostics and retry classification." in str(calls["prompt"])
    assert "first-place Kaggle leaderboard performance" in " ".join(str(calls["prompt"]).split())
    assert result["codex_improvement"]["publish"]["status"] == "disabled"
    assert result["codex_improvement"]["implementation_profile"]["cli_profile"] == "sol-ultra"


def test_self_improvement_skips_codex_when_worktree_dirty(monkeypatch, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    _write_json(run_dir / "run.json", {"run_id": "run-1", "status": "completed", "config": {}})
    _write_json(run_dir / "iter-1" / "metrics.json", {"offline_value": 0.1})
    calls: dict[str, object] = {}

    def fake_run_strategy(prompt_path, output_dir, **kwargs):  # noqa: ANN001, ANN003
        calls["strategy_prompt"] = prompt_path.read_text(encoding="utf-8")
        calls["strategy_engine"] = kwargs.get("engine")
        last_message_path = output_dir / "strategy_last_message.txt"
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text("Use Oracle to plan the next improvement.\n", encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout="Use Oracle to plan the next improvement.",
            stderr="",
            transcript_path=output_dir / "strategy_exec.txt",
            last_message_path=last_message_path,
            engine="oracle",
        )

    def fail_run_codex(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Codex implementation should be skipped for a dirty worktree.")

    monkeypatch.setattr("kagglebot.self_improvement._git_dirty", lambda workdir: True)
    monkeypatch.setattr("kagglebot.self_improvement.run_strategy", fake_run_strategy)
    monkeypatch.setattr("kagglebot.self_improvement.run_codex", fail_run_codex)

    result = run_self_improvement_cycle(
        SelfImprovementConfig(
            artifacts_dir=artifacts,
            knowledge_paths=KnowledgePaths(workdir=tmp_path),
            invoke_codex=True,
            force=True,
        )
    )

    assert result["codex_improvement"]["status"] == "blocked_dirty"
    assert result["codex_improvement"]["publish"]["status"] == "disabled"
    assert calls == {}


def test_self_improvement_publishes_pending_changes_before_codex(monkeypatch, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    _write_json(run_dir / "run.json", {"run_id": "run-1", "status": "completed", "config": {}})
    _write_json(run_dir / "iter-1" / "metrics.json", {"offline_value": 0.1})
    dirty_states = iter([True, False])
    publish_messages: list[str] = []
    calls: dict[str, object] = {}
    _mock_repository_baseline(monkeypatch, tmp_path)

    def fake_run_strategy(prompt_path, output_dir, **kwargs):  # noqa: ANN001, ANN003, ARG001
        last_message_path = output_dir / "strategy_last_message.txt"
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text("Publish pending work, then implement.\n", encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout="Publish pending work, then implement.",
            stderr="",
            transcript_path=output_dir / "strategy_exec.txt",
            last_message_path=last_message_path,
            engine="oracle",
        )

    def fake_publish(*, config, codex_returncode, commit_message):  # noqa: ANN001
        publish_messages.append(commit_message)
        return {"status": "pushed", "commit": "abc123", "returncode": codex_returncode}

    def fake_run_codex(prompt_path, output_dir, **kwargs):  # noqa: ANN001, ANN003
        calls["prompt"] = prompt_path.read_text(encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            transcript_path=output_dir / "codex_exec.jsonl",
            last_message_path=output_dir / "codex_last_message.txt",
        )

    monkeypatch.setattr("kagglebot.self_improvement._git_dirty", lambda workdir: next(dirty_states))
    monkeypatch.setattr("kagglebot.self_improvement.run_strategy", fake_run_strategy)
    monkeypatch.setattr("kagglebot.self_improvement._publish_codex_changes", fake_publish)
    monkeypatch.setattr("kagglebot.self_improvement.run_codex", fake_run_codex)

    result = run_self_improvement_cycle(
        SelfImprovementConfig(
            artifacts_dir=artifacts,
            knowledge_paths=KnowledgePaths(workdir=tmp_path),
            invoke_codex=True,
            publish_codex_changes=True,
            force=True,
        )
    )

    assert result["codex_improvement"]["status"] == "completed"
    assert publish_messages == [
        "Publish pending autopilot changes before self-improvement",
        "Self-improve autopilot from report",
    ]
    assert "Kagglebot Self-Improvement Implementation" in str(calls["prompt"])


def test_self_improvement_passes_publish_policy_to_controller(monkeypatch, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    _write_json(run_dir / "run.json", {"run_id": "run-1", "status": "completed", "config": {}})
    _write_json(run_dir / "iter-1" / "metrics.json", {"offline_value": 0.1})
    published: dict[str, object] = {}
    _mock_repository_baseline(monkeypatch, tmp_path)

    def fake_run_strategy(prompt_path, output_dir, **kwargs):  # noqa: ANN001, ANN003, ARG001
        last_message_path = output_dir / "strategy_last_message.txt"
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text("Implement the highest-value reusable fix.\n", encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout="Implement the highest-value reusable fix.",
            stderr="",
            transcript_path=output_dir / "strategy_exec.txt",
            last_message_path=last_message_path,
            engine="oracle",
        )

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
    monkeypatch.setattr("kagglebot.self_improvement.run_strategy", fake_run_strategy)
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


def test_self_improvement_invalid_oracle_response_blocks_codex(monkeypatch, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    _write_json(run_dir / "run.json", {"run_id": "run-1", "status": "completed", "config": {}})
    _write_json(run_dir / "iter-1" / "metrics.json", {"offline_value": 0.1})
    baseline = RepositoryBaseline(
        workdir=tmp_path,
        branch="main",
        upstream="origin/main",
        remote="origin",
        remote_branch="main",
        repository_url="https://github.com/example/kaggle-autopilot",
        head_sha="a" * 40,
        remote_sha="a" * 40,
    )

    def fake_strategy(prompt_path, output_dir, **kwargs):  # noqa: ANN001, ANN003, ARG001
        return SimpleNamespace(
            returncode=0,
            stdout="This response has no required delimiters.",
            stderr="",
            transcript_path=output_dir / "strategy_exec.txt",
            last_message_path=output_dir / "strategy_last_message.txt",
            engine="oracle",
        )

    monkeypatch.setattr("kagglebot.self_improvement._git_dirty", lambda workdir: False)
    monkeypatch.setattr("kagglebot.self_improvement.verify_clean_pushed_repository", lambda workdir: baseline)
    monkeypatch.setattr("kagglebot.self_improvement.run_strategy", fake_strategy)
    monkeypatch.setattr(
        "kagglebot.self_improvement.run_codex",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Codex must remain blocked")),
    )

    result = run_self_improvement_cycle(
        SelfImprovementConfig(
            artifacts_dir=artifacts,
            knowledge_paths=KnowledgePaths(workdir=tmp_path),
            invoke_codex=True,
            force=True,
        )
    )

    assert result["codex_improvement"]["status"] == "oracle_invalid"
    assert "missing required sections" in result["codex_improvement"]["reason"]
    scheduler = json.loads((artifacts / "_self_improvement" / "scheduler.json").read_text(encoding="utf-8"))
    assert scheduler["last_status"] == "oracle_invalid"
    assert scheduler["retry_at"] is not None


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
