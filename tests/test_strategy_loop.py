"""Tests for the codex -> claude -> codex planning pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from kagglebot.orchestrator.strategy_loop import StrategyConfig, run_strategy_pipeline
from kagglebot.paths import CompetitionPaths


class DummyCodexResult:
    def __init__(self, output_dir: Path) -> None:
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""
        self.last_message_path = output_dir / "codex_last_message.txt"
        self.last_message_path.write_text("brief summary\n", encoding="utf-8")


class DummyClaudeResult:
    def __init__(self, output: str) -> None:
        self.returncode = 0
        self.stdout = output
        self.stderr = ""


def _write_context(paths: CompetitionPaths) -> None:
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_url_path.write_text("https://example.com/rules\n", encoding="utf-8")
    paths.dataset_profile_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    paths.overview_md_path.write_text("overview\n", encoding="utf-8")
    paths.data_md_path.write_text("data\n", encoding="utf-8")
    paths.rules_md_path.write_text("rules\n", encoding="utf-8")
    paths.sample_submission_head_path.write_text("id,target\n1,0.1\n", encoding="utf-8")


def test_strategy_pipeline_runs_all_stages(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)
    paths.context_agent_dir.mkdir(parents=True, exist_ok=True)

    codex_calls: list[Path] = []

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyCodexResult:  # noqa: ARG001
        codex_calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(codex_calls) == 1:
            (paths.context_agent_dir / "brief_for_claude.md").write_text("brief\n", encoding="utf-8")
            (paths.context_agent_dir / "brief_for_claude.json").write_text("{}", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_claude(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyClaudeResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(
            [
                "===CLAUDE_STRATEGY===",
                "strategy text",
                "===CODEX_IMPLEMENTATION_INSTRUCTIONS===",
                "do the thing",
                "===REFERENCES===",
                "ref1",
            ]
        )
        return DummyClaudeResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.strategy_loop.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.strategy_loop.run_claude", fake_run_claude)
    monkeypatch.setattr("kagglebot.orchestrator.strategy_loop._git_status_paths", lambda *args, **kwargs: [])

    config = StrategyConfig(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        compute="local_cpu",
        accelerator="cpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )
    run_strategy_pipeline(paths=paths, config=config)

    agent_dir = paths.context_agent_dir
    assert (agent_dir / "brief_for_claude.md").exists()
    assert (agent_dir / "brief_for_claude.json").exists()
    assert (agent_dir / "claude_strategy.md").exists()
    assert (agent_dir / "codex_implementation_instructions.md").exists()
    assert (agent_dir / "references.md").exists()
    assert (agent_dir / "claude_transcript.txt").exists()
    assert len(codex_calls) == 2
    brief_prompt = (agent_dir / "brief" / "prompt.md").read_text(encoding="utf-8")
    assert str(paths.overview_md_path) in brief_prompt
    assert str(paths.data_md_path) in brief_prompt
    assert str(paths.rules_md_path) in brief_prompt


def test_strategy_pipeline_blocks_code_changes_in_brief(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)
    paths.context_agent_dir.mkdir(parents=True, exist_ok=True)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyCodexResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        (paths.context_agent_dir / "brief_for_claude.md").write_text("brief\n", encoding="utf-8")
        (paths.context_agent_dir / "brief_for_claude.json").write_text("{}", encoding="utf-8")
        return DummyCodexResult(output_dir)

    monkeypatch.setattr("kagglebot.orchestrator.strategy_loop.run_codex", fake_run_codex)

    def fake_claude(*args, **kwargs):  # noqa: ARG001
        return DummyClaudeResult("")

    monkeypatch.setattr("kagglebot.orchestrator.strategy_loop.run_claude", fake_claude)
    status_calls = {"count": 0}

    def fake_status(*args, **kwargs):  # noqa: ARG001
        status_calls["count"] += 1
        if status_calls["count"] == 1:
            return []
        return ["src/app.py"]

    monkeypatch.setattr("kagglebot.orchestrator.strategy_loop._git_status_paths", fake_status)

    config = StrategyConfig(
        slug="demo",
        competition_url=None,
        compute="local_cpu",
        accelerator="cpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    try:
        run_strategy_pipeline(paths=paths, config=config)
    except Exception as exc:  # noqa: BLE001
        assert "Codex brief modified files unexpectedly" in str(exc)
    else:
        raise AssertionError("Expected error for code changes during brief stage.")
