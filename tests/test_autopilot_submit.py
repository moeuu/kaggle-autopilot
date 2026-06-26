from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kagglebot import autopilot_submit
from kagglebot.exceptions import SubmitAbortedError


def test_attempt_submit_for_autopilot_run_supplies_runner_dependencies(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    config = SimpleNamespace(
        submit=True,
        dry_run=False,
        slug="demo",
        paths=SimpleNamespace(run_dir=lambda run_id: tmp_path / run_id),
        knowledge_paths=object(),
        force_submit=False,
        message="submit",
        campaign_mode="single",
        target_direction="maximize",
        kaggle_username=None,
        kernel_name=None,
        accelerator="cpu",
        strict_accelerator=False,
        time_budget_min=None,
    )
    submission_path = tmp_path / "submission.csv"

    def fake_attempt_submit_for_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(
        "kagglebot.autopilot_submit._submit_runner.attempt_submit_for_run",
        fake_attempt_submit_for_run,
    )

    result = autopilot_submit.attempt_submit_for_autopilot_run(
        config=config,
        run_id="run-1",
        submission_path=submission_path,
        best_score=0.42,
        problem_types=["tabular"],
        submit_mode="notebook",
        notebook_submit_artifact_mode="inference",
    )

    assert result == {"status": "ok"}
    assert captured["config"] is config
    assert captured["run_id"] == "run-1"
    assert captured["submission_path"] == submission_path
    assert captured["best_score"] == 0.42
    assert captured["problem_types"] == ["tabular"]
    assert captured["submit_mode"] == "notebook"
    assert captured["notebook_submit_artifact_mode"] == "inference"
    deps = captured["deps"]
    limits = captured["limits"]
    assert deps.build_error is SubmitAbortedError
    assert callable(deps.run_submit_kernel)
    assert callable(deps.run_kaggle_submit_kernel)
    assert limits.max_transient_retries == 3
    assert limits.stdout_tail_chars == 1200


def test_build_autopilot_submit_dependencies_resolves_defaults_at_call_time(monkeypatch) -> None:
    def fake_check_rules(*_args: object, **_kwargs: object) -> bool:
        return False

    def fake_resolve_username(*_args: object, **_kwargs: object) -> str:
        return "patched-user"

    monkeypatch.setattr(autopilot_submit, "check_rules_accepted", fake_check_rules)
    monkeypatch.setattr(autopilot_submit, "resolve_kaggle_username", fake_resolve_username)

    deps = autopilot_submit.build_autopilot_submit_dependencies()

    assert deps.check_rules_accepted is fake_check_rules
    assert deps.resolve_kaggle_username is fake_resolve_username
