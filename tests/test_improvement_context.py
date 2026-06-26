from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kagglebot.improvement_context import (
    build_improvement_implementation_prompt,
    build_improvement_prompt_plan,
)
from kagglebot.paths import CompetitionPaths, KnowledgePaths


def _write_prompt_template(paths: CompetitionPaths) -> None:
    paths.prompts_dir.mkdir(parents=True, exist_ok=True)
    paths.codex_improve_template.write_text(
        "\n".join(
            [
                "slug={slug}",
                "iteration={iteration}",
                "metric={metric}",
                "direction={direction}",
                "current={current_score}",
                "mode={improvement_mode}",
                "kernel={kernel_main}",
                "code_ref={code_reference_score}",
                "code_status={code_reference_status}",
            ]
        ),
        encoding="utf-8",
    )


def _make_config(tmp_path: Path):
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    _write_prompt_template(paths)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.kernel_source_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"task": "binary_classification", "modality": "tabular"}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        slug="demo",
        compute="local_gpu",
        accelerator="gpu",
        hardware_profile=None,
        time_budget_min=120,
        paths=paths,
        knowledge_paths=KnowledgePaths(workdir=tmp_path),
    )


def test_build_improvement_prompt_plan_renders_policy_context(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.paths.method_registry_path.write_text(
        json.dumps({"methods": [{"name": "blend", "summary": "OOF logit blend"}]}),
        encoding="utf-8",
    )
    config.paths.competition_policy_path.write_text(
        json.dumps(
            {
                "required_capabilities": ["heterogeneous_tabular_ensemble", "requires_oof_blend"],
                "prompt": {"min_model_families_before_stop": 3},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kagglebot.knowledge_context.load_problem_type_knowledge_text",
        lambda **kwargs: "Prior knowledge",
    )

    plan = build_improvement_prompt_plan(
        config=config,
        run_id="run-1",
        iteration=2,
        iter_dir=config.paths.iter_dir("run-1", 2),
        agent_dir=config.paths.iter_dir("run-1", 2) / "agent",
        evaluation=SimpleNamespace(metric="auc", direction="maximize", value=0.62),
        top1_info={"score": 0.78, "source": "leaderboard"},
        target_score=0.78,
        delta_offline=-0.01,
        current_score=None,
        current_score_source="cv",
        minimum_improvement_mode="validation_redesign",
        minimum_improvement_reason="stagnation",
        target_medal="gold",
        target_rank_percentile=0.05,
        forced_improvement_mode=None,
        forced_improvement_reason=None,
        extra_policy_notes=["repair pseudo labels"],
        enforce_code_reference_implementation=True,
        code_reference_enforcement_reason="baseline policy",
        best_score_so_far=0.66,
        previous_submission_history={"best_score": 0.7, "direction": "maximize"},
        prompt_identity_args={},
    )

    assert plan.prompt_path == config.paths.iter_dir("run-1", 2) / "agent" / "prompt.md"
    assert plan.strategy_dir == config.paths.iter_dir("run-1", 2) / "agent" / "improve_strategy-02"
    assert plan.code_reference_mandatory is True
    assert plan.mode_notices[0].kind == "floor"
    assert plan.improvement_mode == "validation_redesign"
    assert "Minimum improvement mode policy is active." in plan.base_prompt_text
    assert "Medal-aware search policy:" in plan.base_prompt_text
    assert "Competition policy override is active." in plan.base_prompt_text
    assert "Minimum model families before stop: 3" in plan.base_prompt_text
    assert "Additional repair targets:" in plan.base_prompt_text
    assert "Regression Guard Policy:" in plan.base_prompt_text
    assert "## Code Reference Gate" in plan.base_prompt_text
    assert "Code reference implementation is policy-mandatory" in plan.base_prompt_text
    assert "Prior knowledge" in plan.strategy_prompt


def test_build_improvement_implementation_prompt_wraps_strategy() -> None:
    text = build_improvement_implementation_prompt(
        base_prompt_text="local context",
        strategy_text="1. add stronger validation",
    )

    assert "Improvement Implementation" in text
    assert "Extra-High Improvement Prompt" in text
    assert "1. add stronger validation" in text
    assert "local context" in text
    assert build_improvement_implementation_prompt(base_prompt_text="base", strategy_text="") == "base"
