from __future__ import annotations

from kagglebot.submit_stage import decide_initial_submit_stage_mode


def test_decide_initial_submit_stage_mode_keeps_file_submit() -> None:
    decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=False,
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        resolved_notebook_artifact_mode="inference",
    )

    assert decision.notebook_submit_required is False
    assert decision.notebook_fallback_activated is False
    assert decision.submission_artifact_mode == "wrapper"
    assert decision.messages == ()


def test_decide_initial_submit_stage_mode_uses_requested_notebook_mode() -> None:
    decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=True,
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        resolved_notebook_artifact_mode="inference",
    )

    assert decision.notebook_submit_required is True
    assert decision.notebook_fallback_activated is True
    assert decision.submission_artifact_mode == "inference"
    assert decision.messages == ("[yellow]submit mode[/yellow]: using notebook submit",)


def test_decide_initial_submit_stage_mode_forces_notebook_only_competition() -> None:
    decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=False,
        notebook_submissions_only=True,
        notebook_submit_artifact_mode="wrapper",
        resolved_notebook_artifact_mode="inference",
    )

    assert decision.notebook_submit_required is True
    assert decision.notebook_fallback_activated is True
    assert decision.submission_artifact_mode == "inference"
    assert decision.messages == (
        "[yellow]submit mode[/yellow]: notebook-only competition detected; forcing notebook submit",
        "[yellow]submit mode[/yellow]: using notebook submit",
    )
