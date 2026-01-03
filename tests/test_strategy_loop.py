"""Tests for the codex -> claude -> codex agent pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from kagglebot.orchestrator.agent_pipeline import AgentPipelineConfig, run_agent_pipeline
from kagglebot.paths import CompetitionPaths


class DummyCodexResult:
    def __init__(self, output_dir: Path, message: str = "brief summary") -> None:
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""
        self.last_message_path = output_dir / "codex_last_message.txt"
        self.last_message_path.write_text(message + "\n", encoding="utf-8")


class DummyClaudeResult:
    def __init__(self, output: str) -> None:
        self.returncode = 0
        self.stdout = output
        self.stderr = ""


def _write_context(paths: CompetitionPaths) -> None:
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_url_path.write_text("https://example.com/rules\n", encoding="utf-8")
    paths.dataset_profile_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    paths.overview_md_path.write_text("overview text\n", encoding="utf-8")
    paths.data_md_path.write_text("data text\n", encoding="utf-8")
    paths.rules_md_path.write_text("rules text\n", encoding="utf-8")
    paths.sample_submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")


def test_agent_pipeline_runs_all_stages(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    codex_calls: list[Path] = []

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyCodexResult:  # noqa: ARG001
        codex_calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(codex_calls) == 2:
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('kernel')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_claude(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyClaudeResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(
            [
                "===STRATEGY===",
                "strategy text",
                "===CODEX_INSTRUCTIONS===",
                "update kernel.py",
            ]
        )
        return DummyClaudeResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_claude", fake_run_claude)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        compute="local_cpu",
        accelerator="cpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )
    run_agent_pipeline(paths=paths, config=config)

    agent_dir = paths.context_agent_dir
    assert (agent_dir / "brief_for_claude.md").exists()
    assert (agent_dir / "claude_strategy.md").exists()
    assert (agent_dir / "codex_instructions.md").exists()
    assert (agent_dir / "claude_transcript.txt").exists()
    assert len(codex_calls) == 2
    brief_prompt = (agent_dir / "brief" / "prompt.md").read_text(encoding="utf-8")
    assert "overview text" in brief_prompt
    assert "data text" in brief_prompt
    assert "rules text" in brief_prompt


def test_agent_pipeline_write_guard_blocks_outside_kernel(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyCodexResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.name == "implement":
            (tmp_path / "oops.txt").write_text("nope", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_claude(*args, **kwargs):  # noqa: ARG001
        text = "\n".join(
            [
                "===STRATEGY===",
                "strategy text",
                "===CODEX_INSTRUCTIONS===",
                "update kernel.py",
            ]
        )
        return DummyClaudeResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_claude", fake_run_claude)

    config = AgentPipelineConfig(
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
        run_agent_pipeline(paths=paths, config=config)
    except Exception as exc:  # noqa: BLE001
        assert "Agent write-guard failed" in str(exc)
    else:
        raise AssertionError("Expected write-guard failure for out-of-allowlist write.")


def test_agent_pipeline_allows_kernel_write(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyCodexResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.name == "implement":
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('ok')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_claude(*args, **kwargs):  # noqa: ARG001
        text = "\n".join(
            [
                "===STRATEGY===",
                "strategy text",
                "===CODEX_INSTRUCTIONS===",
                "update kernel.py",
            ]
        )
        return DummyClaudeResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_claude", fake_run_claude)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_cpu",
        accelerator="cpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    run_agent_pipeline(paths=paths, config=config)
    assert (paths.kernel_source_dir / "kernel.py").exists()
