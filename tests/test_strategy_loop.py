"""Tests for the codex -> gpt -> codex agent pipeline."""

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


class DummyStrategyResult:
    def __init__(self, output: str) -> None:
        self.returncode = 0
        self.stdout = output
        self.stderr = ""


def _long_strategy_text() -> str:
    base = "\n".join(
        [
            "## Problem",
            "This section frames the problem and target metric.",
            "## Data",
            "We summarize data structure and submission format.",
            "## Candidate Approaches",
            "Candidate models and alternatives with pros/cons.",
            "## Final Approach",
            "Final model choice and rationale.",
            "## Training & Evaluation",
            "Train/validation plan and evaluation strategy.",
            "## Compute Plan",
            "GPU usage plan and time budget.",
            "## Error Analysis & Ablation",
            "Ablation plan and error analysis steps.",
            "## Risks",
            "Risks and rule constraints.",
            "## Search Notes",
            "Search queries and key findings.",
            "## Sources",
            "- Source A (example.com)",
            "- Source B (example.org)",
            "- Source C (example.net)",
        ]
    )
    filler = "Details and rationale. " * 80
    return base + "\n" + filler


def _long_instructions_text() -> str:
    steps = [
        "1) Update kernel.py with the new model pipeline.",
        "2) Add helper modules under kernel/ for parsing and preprocessing.",
        "3) Ensure training/evaluation flow uses the chosen metric.",
        "4) Write outputs to submission.csv with correct format.",
        "5) Validate against sample submission and log metrics.",
    ]
    return "\n".join(steps) + "\n" + ("More detail. " * 40)


def _plan_json_text() -> str:
    return "\n".join(
        [
            '{"target_metric": "accuracy",',
            ' "target_direction": "maximize",',
            ' "target_score": 0.9,',
            ' "score_source": "holdout",',
            ' "holdout_frac": 0.2,',
            ' "cv_folds": 5,',
            ' "seed": 42,',
            ' "max_iterations": 1,',
            ' "patience": 2,',
            ' "min_improvement": 0.0}',
        ]
    )


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

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        codex_calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(codex_calls) == 2:
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('kernel')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_strategy(prompt_path: Path, output_dir: Path, dry_run: bool) -> DummyStrategyResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(
            [
                "===STRATEGY===",
                _long_strategy_text(),
                "===PLAN_JSON===",
                _plan_json_text(),
                "===CODEX_INSTRUCTIONS===",
                _long_instructions_text(),
            ]
        )
        return DummyStrategyResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)
    monkeypatch.setattr(
        "kagglebot.orchestrator.agent_pipeline._load_problem_type_knowledge_text",
        lambda *args, **kwargs: "Problem-type knowledge (test fixture)",
    )

    config = AgentPipelineConfig(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )
    run_agent_pipeline(paths=paths, config=config)

    agent_dir = paths.context_agent_dir
    assert (agent_dir / "brief_for_strategy.md").exists()
    assert (agent_dir / "strategy_plan.md").exists()
    assert (agent_dir / "codex_instructions.md").exists()
    assert (agent_dir / "strategy_transcript.txt").exists()
    assert len(codex_calls) == 2
    brief_prompt = (agent_dir / "brief" / "prompt.md").read_text(encoding="utf-8")
    assert str(paths.overview_md_path) in brief_prompt
    assert str(paths.data_md_path) in brief_prompt
    assert str(paths.rules_md_path) in brief_prompt
    assert "Problem-type knowledge (test fixture)" in brief_prompt


def test_agent_pipeline_write_guard_blocks_outside_kernel(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.name == "implement":
            (tmp_path / "oops.txt").write_text("nope", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_strategy(*args, **kwargs):  # noqa: ARG001
        text = "\n".join(
            [
                "===STRATEGY===",
                _long_strategy_text(),
                "===PLAN_JSON===",
                _plan_json_text(),
                "===CODEX_INSTRUCTIONS===",
                _long_instructions_text(),
            ]
        )
        return DummyStrategyResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    run_agent_pipeline(paths=paths, config=config)
    assert not (tmp_path / "oops.txt").exists()


def test_agent_pipeline_allows_kernel_write(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_context(paths)

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs) -> DummyCodexResult:  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.name == "implement":
            kernel_path = paths.kernel_source_dir / "kernel.py"
            kernel_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_path.write_text("print('ok')\n", encoding="utf-8")
        return DummyCodexResult(output_dir)

    def fake_run_strategy(*args, **kwargs):  # noqa: ARG001
        text = "\n".join(
            [
                "===STRATEGY===",
                _long_strategy_text(),
                "===PLAN_JSON===",
                _plan_json_text(),
                "===CODEX_INSTRUCTIONS===",
                _long_instructions_text(),
            ]
        )
        return DummyStrategyResult(text)

    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_codex", fake_run_codex)
    monkeypatch.setattr("kagglebot.orchestrator.agent_pipeline.run_strategy", fake_run_strategy)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    run_agent_pipeline(paths=paths, config=config)
    assert (paths.kernel_source_dir / "kernel.py").exists()
