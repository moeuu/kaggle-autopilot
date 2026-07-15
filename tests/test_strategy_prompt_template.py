from __future__ import annotations

from pathlib import Path

import kagglebot.orchestrator.agent_pipeline as agent_pipeline
from kagglebot.agents.identity import BRIEF_AGENT
from kagglebot.knowledge import build_improve_template, build_kernel_fix_template
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
    assert "uv add <package>" in template
    assert "maximum score ceiling over guaranteed submitability" in template
    assert "OOF predictions" in template
    assert "explicit blend candidate" in template
    assert "geospatial" in template
    assert "bio/sequence" in template
    assert "model-artifact" in template
    assert "submission_manifest.json" in template
    assert "disguising tabular CSV bytes" in template
    assert "competition-specific output schema and semantic invariants" in template
    assert "several eligible repairs compete for the same constrained source/target" in template
    assert "one GPU's floating-point boundary behavior" in template


def test_prompts_require_fold_intermediate_submissions() -> None:
    base_dir = Path(agent_pipeline.__file__).resolve().parents[1]
    kernel_template = (base_dir / "prompts" / "codex_kernel_impl.md").read_text(encoding="utf-8")
    improve_template = (base_dir / "templates" / "improve_iteration.md").read_text(encoding="utf-8")
    combined = kernel_template + "\n" + improve_template

    assert "submission_<name>_fold<N>.<suffix>" in kernel_template
    assert "candidate_<name>_fold<N>.json" in kernel_template
    assert "submission_<candidate>_fold<N>.<suffix>" in improve_template
    assert "candidate_<candidate>_fold<N>.json" in improve_template
    assert "KAGGLEBOT_SUBMISSION_FILENAME" in kernel_template
    assert "/kaggle/working/<submission filename>" in kernel_template
    assert "sample_submission.*" in combined
    assert "runtime test ids" in kernel_template
    assert "never emit a 3-row public placeholder submission" in kernel_template
    assert "completed folds only in memory" in combined


def test_strategy_plan_prompt_includes_quality_gate_checklist() -> None:
    base_dir = Path(agent_pipeline.__file__).resolve().parents[1] / "prompts"
    template = (base_dir / "strategy_plan.md").read_text(encoding="utf-8")
    assert "QUALITY GATE REQUIREMENTS" in template
    assert "===RESEARCH_SOURCES_JSONL===" in template
    assert ">=1200 characters" in template
    assert "{{code_snapshot}}" in template
    assert "{{models_snapshot}}" in template
    assert "{{discussion_snapshot}}" in template
    assert "{{competition_url}}" in template
    assert "{{strategy_context_bundle}}" in template
    assert "{{strategy_context_bundle_path}}" in template
    assert "oracle_context_manifest.md" in template
    assert "Authorized benign use" in template
    assert ">=8000 characters" in template
    assert "12000-25000 characters" in template
    assert "There is no 1200-character cap" in template
    assert "Sample submission preview" in template
    assert "Sample submission head (CSV" not in template


def test_legacy_strategy_template_uses_required_format_preview() -> None:
    base_dir = Path(agent_pipeline.__file__).resolve().parents[1] / "prompts" / "templates"
    template = (base_dir / "strategy_plan.md").read_text(encoding="utf-8")

    assert "Sample submission preview (required format)" in template
    assert "not as proof that the artifact must be CSV" in template
    assert "Sample submission head:" not in template


def test_codex_brief_prompt_mentions_community_context_files() -> None:
    base_dir = Path(agent_pipeline.__file__).resolve().parents[1] / "prompts"
    template = (base_dir / "codex_brief.md").read_text(encoding="utf-8")
    assert "{{code_path}}" in template
    assert "{{code_index_path}}" in template
    assert "{{models_path}}" in template
    assert "{{discussion_path}}" in template
    assert "{{discussion_threads_dir}}" in template
    assert "{{discussion_index_path}}" in template


def test_codex_brief_prompt_renders_current_agent_identity() -> None:
    rendered = agent_pipeline._load_template("codex_brief.md")  # noqa: SLF001
    assert f"# {BRIEF_AGENT.display_name} Brief Extraction" in rendered
    assert f"You are {BRIEF_AGENT.display_name}." in rendered
    assert "You are Codex." not in rendered


def test_improve_template_uses_loop_decision_language() -> None:
    template = build_improve_template()
    assert "{implementation_agent_name}" in template
    assert "Kagglebot Codex" not in template
    assert "{current_score_source}" in template
    assert "loop-decision" in template
    assert "{code_md}" in template
    assert "{code_index}" in template
    assert "{code_reference_status}" in template
    assert "highest realistic score ceiling" in template


def test_implementation_templates_use_generalized_table_io() -> None:
    base_dir = Path(agent_pipeline.__file__).resolve().parents[1]
    initial_template = (base_dir / "templates" / "initial_plan_and_implement.md").read_text(encoding="utf-8")
    improve_template = (base_dir / "templates" / "improve_iteration.md").read_text(encoding="utf-8")

    assert "from kagglebot.solver.io import read_table" in initial_template
    assert "from kagglebot.solver.io import read_table, write_table" in initial_template
    assert 'train = read_table(Path("{train_path}"))' in initial_template
    assert 'sample = read_table(Path("{sample_submission_path}"))' in initial_template
    assert 'write_table(submission, Path("{submission_path}"))' in initial_template
    assert "pd.read_csv" not in initial_template
    assert "submission.to_csv" not in initial_template
    assert "diff <(head" not in initial_template
    assert "diff <(head" not in improve_template
    assert 'sample_path = Path("{sample_submission_path}")' in improve_template
    assert "non-tabular or manifest-based submission check required" in initial_template
    assert "non-tabular or manifest-based submission check required" in improve_template
    for template in (initial_template, improve_template):
        assert "geospatial" in template
        assert "bio/sequence" in template
        assert "graph" in template
        assert "signal" in template
        assert "annotation" in template
        assert "model-artifact" in template


def test_strategy_plan_prompt_prioritizes_accuracy_over_submitability() -> None:
    base_dir = Path(agent_pipeline.__file__).resolve().parents[1] / "prompts"
    template = (base_dir / "strategy_plan.md").read_text(encoding="utf-8")
    assert "Prioritize maximum achievable accuracy over submitability." in template
    assert "target_medal" in template
    assert "target_rank_percentile" in template
    assert "OOF blend" in template
    assert "geospatial/bio/sequence/graph/signal/annotation/model-artifact" in template
    assert "required artifact/manifest handling" in template


def test_legacy_strategy_template_lists_broad_non_tabular_modalities() -> None:
    base_dir = Path(agent_pipeline.__file__).resolve().parents[1] / "prompts" / "templates"
    template = (base_dir / "strategy_plan.md").read_text(encoding="utf-8")

    assert "geospatial/bio/sequence/graph/signal/annotation/model-artifact" in template
    assert "geometric/geospatial/structure-feature" in template


def test_kernel_fix_template_mentions_dependency_add_path() -> None:
    template = build_kernel_fix_template()
    assert "{implementation_agent_name}" in template
    assert "Kagglebot Codex" not in template
    assert "uv add <package>" in template
    assert "pyproject.toml" in template
    assert "uv.lock" in template


def test_strategy_prompt_includes_code_models_discussion_context(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_url_path.write_text("https://www.kaggle.com/competitions/demo/rules\n", encoding="utf-8")
    paths.dataset_profile_path.write_text('{"task": "classification"}', encoding="utf-8")
    paths.overview_md_path.write_text("overview body token", encoding="utf-8")
    paths.data_md_path.write_text("data page token", encoding="utf-8")
    paths.rules_md_path.write_text("rules body token", encoding="utf-8")
    paths.submission_format_md_path.write_text("submission format text", encoding="utf-8")
    paths.sample_submission_head_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    paths.code_md_path.write_text("code snapshot token", encoding="utf-8")
    paths.models_md_path.write_text("models snapshot token", encoding="utf-8")
    paths.discussion_md_path.write_text("discussion snapshot token", encoding="utf-8")
    (paths.data_dir / "train.csv").write_text("id,feature,target\n1,a,0\n2,b,1\n", encoding="utf-8")

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
    assert "Competition URL: https://www.kaggle.com/competitions/demo" in prompt
    assert "overview body token" in prompt
    assert "data page token" in prompt
    assert "rules body token" in prompt
    assert "Data File Structure and Representative Samples" in prompt
    assert "train.csv" in prompt
    assert "id,feature,target" in prompt
    assert (paths.context_agent_dir / "strategy_context_bundle.md").exists()


def test_strategy_prompt_includes_selected_hardware_profile(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_url_path.write_text("https://www.kaggle.com/competitions/demo/rules\n", encoding="utf-8")
    paths.dataset_profile_path.write_text('{"task": "text"}', encoding="utf-8")
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
        hardware_profile="rtx3060",
        time_budget_min=1200,
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

    assert "NVIDIA GeForce RTX 3060 12GB" in prompt
    assert "RTX3060-class accuracy-first rule" in prompt
    assert "rtx5090" in prompt.lower()


def test_strategy_prompt_appends_high_accuracy_tabular_policy(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.rules_url_path.write_text("https://www.kaggle.com/competitions/demo/rules\n", encoding="utf-8")
    dataset_profile = (
        '{"task": "classification", "modality": "tabular", "tags": ["tabular", "binary"], '
        '"train_rows": 12000, "categorical_columns": ["a", "b", "c"], '
        '"high_cardinality_columns": ["c"]}'
    )
    paths.dataset_profile_path.write_text(dataset_profile, encoding="utf-8")
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

    assert "[HIGH_ACCURACY_TABULAR_POLICY]" in prompt
    assert "CatBoost raw categorical" in prompt
    assert "OOF blend candidate" in prompt
    assert "suite-aware ablations" in prompt
