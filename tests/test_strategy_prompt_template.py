from __future__ import annotations

from pathlib import Path

import kagglebot.orchestrator.agent_pipeline as agent_pipeline
from kagglebot.knowledge import build_improve_template
from kagglebot.orchestrator.agent_pipeline import AgentPipelineConfig
from kagglebot.paths import CompetitionPaths


def test_strategy_plan_prompt_requires_kernel_py_mention() -> None:
    base_dir = Path(agent_pipeline.__file__).resolve().parents[1] / "prompts"
    template = (base_dir / "strategy_plan.md").read_text(encoding="utf-8")
    assert "kernel.py" in template


def test_codex_kernel_impl_prompt_requires_safe_pipeline_lookup() -> None:
    base_dir = Path(agent_pipeline.__file__).resolve().parents[1] / "prompts"
    template = (base_dir / "codex_kernel_impl.md").read_text(encoding="utf-8")
    assert "missing pipeline names must NOT raise" in template


def test_strategy_plan_prompt_includes_quality_gate_checklist() -> None:
    base_dir = Path(agent_pipeline.__file__).resolve().parents[1] / "prompts"
    template = (base_dir / "strategy_plan.md").read_text(encoding="utf-8")
    assert "QUALITY GATE REQUIREMENTS" in template
    assert "===RESEARCH_SOURCES_JSONL===" in template
    assert ">=1200 characters" in template
    assert "{{code_snapshot}}" in template
    assert "{{models_snapshot}}" in template
    assert "{{discussion_snapshot}}" in template


def test_codex_brief_prompt_mentions_community_context_files() -> None:
    base_dir = Path(agent_pipeline.__file__).resolve().parents[1] / "prompts"
    template = (base_dir / "codex_brief.md").read_text(encoding="utf-8")
    assert "{{code_path}}" in template
    assert "{{code_index_path}}" in template
    assert "{{models_path}}" in template
    assert "{{discussion_path}}" in template
    assert "{{discussion_threads_dir}}" in template
    assert "{{discussion_index_path}}" in template


def test_improve_template_uses_loop_decision_language() -> None:
    template = build_improve_template()
    assert "{current_score_source}" in template
    assert "loop-decision" in template


def test_strategy_prompt_includes_code_models_discussion_context(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_url_path.write_text("https://www.kaggle.com/competitions/demo/rules\n", encoding="utf-8")
    paths.dataset_profile_path.write_text('{"task": "classification"}', encoding="utf-8")
    paths.submission_format_md_path.write_text("submission format text", encoding="utf-8")
    paths.sample_submission_head_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    paths.code_md_path.write_text("code snapshot token", encoding="utf-8")
    paths.models_md_path.write_text("models snapshot token", encoding="utf-8")
    paths.discussion_md_path.write_text("discussion snapshot token", encoding="utf-8")

    config = AgentPipelineConfig(
        slug="demo",
        competition_url="https://www.kaggle.com/competitions/demo",
        compute="local_gpu",
        accelerator="gpu",
        internet="on",
        run_id="run-1",
        dry_run=True,
        repo_root=tmp_path,
    )
    template = (Path(agent_pipeline.__file__).resolve().parents[1] / "prompts" / "strategy_plan.md").read_text(
        encoding="utf-8"
    )
    prompt = agent_pipeline._build_strategy_prompt(  # noqa: SLF001
        template=template,
        config=config,
        paths=paths,
        brief_content="brief",
        compact=False,
    )

    assert "code snapshot token" in prompt
    assert "models snapshot token" in prompt
    assert "discussion snapshot token" in prompt
