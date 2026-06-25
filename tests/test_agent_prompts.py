from __future__ import annotations

from pathlib import Path

from kagglebot.agent_prompts import (
    build_code_reference_repair_prompt,
    build_error_strategy_prompt,
    build_improvement_strategy_prompt,
)
from kagglebot.code_reference import CodeReferenceNotebook, code_reference_marker


def test_build_improvement_strategy_prompt_renders_scores_and_required_sections() -> None:
    text = build_improvement_strategy_prompt(
        slug="demo-comp",
        run_id="run-1",
        iteration=2,
        metric="log_loss",
        direction="minimize",
        current_score=0.456789,
        current_score_source="cv",
        target_score=0.123456,
        top1_score=None,
        top1_source="leaderboard",
        top1_gap=None,
        delta_offline=-0.01,
        improvement_mode="accuracy",
        hardware_constraints="GPU 12GB",
        codex_prompt="improve kernel.py",
        problem_type_knowledge="past fixes",
    )

    assert "# Kagglebot Improvement Strategy" in text
    assert "Competition: demo-comp" in text
    assert "Current score: 0.456789 (source: cv)" in text
    assert "Top1 score: unavailable" in text
    assert "Delta vs previous best: -0.010000" in text
    assert "GPU 12GB" in text
    assert "improve kernel.py" in text
    assert "past fixes" in text
    assert "Do not include chain-of-thought." in text


def test_build_error_strategy_prompt_renders_failure_context() -> None:
    text = build_error_strategy_prompt(
        stage="kernel_fix",
        slug="demo-comp",
        run_id="run-1",
        attempt=3,
        compute="local_gpu",
        accelerator="gpu",
        hardware_constraints="No wall-clock cap",
        error_text="CUDA out of memory",
        codex_prompt="repair training loop",
    )

    assert "# Kagglebot" in text
    assert "Stage: kernel_fix" in text
    assert "Attempt: 3" in text
    assert "Compute: local_gpu (gpu)" in text
    assert "CUDA out of memory" in text
    assert "repair training loop" in text
    assert "Root cause hypothesis." in text


def test_build_code_reference_repair_prompt_includes_marker_and_tabicl_requirement() -> None:
    reference = CodeReferenceNotebook(
        kernel_id="owner/tabicl-demo",
        title="TabICL reference",
    )

    text = build_code_reference_repair_prompt(
        base_prompt_text="original context",
        reference=reference,
        issues=["missing marker", "no TabICL path"],
        kernel_path=Path("artifacts/demo/kernel/kernel.py"),
    )

    assert "Mandatory Code Reference Implementation" in text
    assert "missing marker, no TabICL path" in text
    assert "owner/tabicl-demo (TabICL reference)" in text
    assert code_reference_marker(reference) in text
    assert "MUST include a real TabICL path" in text
    assert "artifacts/demo/kernel/kernel.py" in text
    assert "original context" in text
