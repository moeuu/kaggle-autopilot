from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kagglebot.kernel_fix_context import append_kernel_fix_strategy, build_kernel_fix_prompt_plan
from kagglebot.runtime_fixes import save_blocked_modules


class DummyPaths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.context_dir = root / "context"
        self.kernel_source_dir = root / "kernel"
        self.codex_kernel_fix_template = root / "kernel_fix_template.md"
        self.rules_url_path = root / "rules.url"
        self.rules_md_path = root / "rules.md"
        self.overview_md_path = root / "overview.md"
        self.data_md_path = root / "data.md"
        self.submission_format_md_path = root / "submission_format.md"
        self.dataset_profile_path = root / "dataset_profile.json"
        self.sample_submission_path = root / "sample_submission.csv"
        self.context_dir.mkdir(parents=True, exist_ok=True)
        self.kernel_source_dir.mkdir(parents=True, exist_ok=True)
        self.codex_kernel_fix_template.write_text(
            "\n".join(
                [
                    "slug={slug}",
                    "run={run_id}",
                    "iteration={iteration}",
                    "compute={compute}",
                    "accelerator={accelerator}",
                    "error={error_message}",
                    "blocked={blocked_modules}",
                    "logs={logs_dir}",
                    "kernel={kernel_main}",
                    "script={kernel_script}",
                    "sample={sample_submission}",
                ]
            ),
            encoding="utf-8",
        )

    def kernel_run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id / "kernel"


def test_kernel_fix_prompt_plan_handles_missing_module_and_blocked_modules(tmp_path: Path) -> None:
    paths = DummyPaths(tmp_path)
    save_blocked_modules(paths.context_dir, ["foo", "bar"])
    config = SimpleNamespace(slug="demo", compute="local_gpu", accelerator="gpu", paths=paths)
    iter_dir = tmp_path / "runs" / "run-1" / "iter-1"
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True)

    plan = build_kernel_fix_prompt_plan(
        config=config,
        run_id="run-1",
        iteration=1,
        iter_dir=iter_dir,
        agent_dir=agent_dir,
        error_message="ModuleNotFoundError: No module named 'foo'",
        attempt=2,
        prompt_prefix="Repair prefix",
        use_gpt_strategy=True,
        prompt_identity_args={},
        hardware_constraints="GPU P100",
    )

    assert plan.prompt_path == agent_dir / "kernel_fix_prompt.md"
    assert plan.attempt_path == agent_dir / "kernel_fix_prompt-02.md"
    assert plan.strategy_skip_reason is None
    assert plan.strategy_prompt is not None
    assert plan.prompt_text.startswith("Repair prefix")
    assert "Missing dependency detected: foo" in plan.prompt_text
    assert "blocked=- bar" in plan.prompt_text
    assert "blocked=- foo" not in plan.prompt_text


def test_kernel_fix_prompt_plan_adds_subgroup_context_and_strategy(monkeypatch, tmp_path: Path) -> None:
    paths = DummyPaths(tmp_path)
    config = SimpleNamespace(slug="demo", compute="local_cpu", accelerator="cpu", paths=paths)
    iter_dir = tmp_path / "runs" / "run-1" / "iter-1"
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text('{"metric":"rmse"}', encoding="utf-8")
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "kagglebot.kernel_quality.detect_subgroup_collapse_signal",
        lambda **kwargs: {"note": "node_type collapse detected"},
    )

    plan = build_kernel_fix_prompt_plan(
        config=config,
        run_id="run-1",
        iteration=1,
        iter_dir=iter_dir,
        agent_dir=agent_dir,
        error_message="RuntimeError: training failed",
        attempt=1,
        prompt_prefix="",
        use_gpt_strategy=True,
        prompt_identity_args={},
        hardware_constraints="CPU only",
    )

    assert plan.strategy_skip_reason is None
    assert plan.strategy_prompt is not None
    assert "Subgroup repair target:" in plan.prompt_text
    assert "node_type collapse detected" in plan.prompt_text
    assert "Stage: kernel_fix" in plan.strategy_prompt
    assert "CPU only" in plan.strategy_prompt


def test_append_kernel_fix_strategy_appends_named_strategy_section() -> None:
    text = append_kernel_fix_strategy(
        prompt_text="base prompt",
        strategy_text="1. fix the root cause",
        strategy_agent_display_name="GPT",
    )

    assert "base prompt" in text
    assert "## GPT Extra-High Error-Fix Strategy" in text
    assert "1. fix the root cause" in text
