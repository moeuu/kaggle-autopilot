from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kagglebot.autopilot_session import AutopilotSession, SubmissionPhase


def test_submission_phase_delegates_to_public_autopilot_submit(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    config = SimpleNamespace(slug="demo")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.2\n", encoding="utf-8")

    def fake_attempt_submit(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("kagglebot.autopilot_submit.attempt_submit_for_autopilot_run", fake_attempt_submit)

    result = SubmissionPhase(
        config=config,
        run_id="run-1",
        problem_types=["tabular"],
        submit_mode="notebook",
        notebook_submit_artifact_mode="inference",
    ).attempt(submission_path=submission_path, best_score=0.42)

    assert result == {"ok": True}
    assert captured == {
        "config": config,
        "run_id": "run-1",
        "submission_path": submission_path,
        "best_score": 0.42,
        "problem_types": ["tabular"],
        "submit_mode": "notebook",
        "notebook_submit_artifact_mode": "inference",
    }


def test_autopilot_session_run_delegates_to_public_autopilot_core(monkeypatch) -> None:
    captured: dict[str, object] = {}
    config = SimpleNamespace(slug="demo")

    def fake_run_autopilot_core(config_arg, run_id_arg, *, resume_run):  # noqa: ANN001
        captured["config"] = config_arg
        captured["run_id"] = run_id_arg
        captured["resume_run"] = resume_run

    monkeypatch.setattr("kagglebot.autopilot.run_autopilot_core", fake_run_autopilot_core)

    AutopilotSession(config=config, run_id="run-1", resume_run=True).run()

    assert captured == {"config": config, "run_id": "run-1", "resume_run": True}


def test_autopilot_keeps_session_class_compatibility() -> None:
    from kagglebot import autopilot as autopilot_mod

    assert autopilot_mod.AutopilotSession is AutopilotSession
    assert autopilot_mod.SubmissionPhase is SubmissionPhase
