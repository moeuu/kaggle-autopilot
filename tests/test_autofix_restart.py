from __future__ import annotations

import json
from pathlib import Path

from kagglebot.autofix_restart import (
    kernel_regenerate_marker_path,
    kernel_regeneration_already_marked,
    maybe_restart_for_src_changes,
    write_kernel_regeneration_marker,
    write_kernel_regeneration_note,
)


def test_kernel_regeneration_marker_and_notes_round_trip(tmp_path: Path) -> None:
    agent_dir = tmp_path / "iter-1" / "agent"

    assert kernel_regeneration_already_marked(agent_dir) is False
    marker_path = write_kernel_regeneration_marker(
        agent_dir=agent_dir,
        run_id="run-1",
        iteration=2,
        attempt=3,
        trigger_reason="repeated_error",
    )

    assert marker_path == kernel_regenerate_marker_path(agent_dir)
    assert kernel_regeneration_already_marked(agent_dir) is True
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["run_id"] == "run-1"
    assert marker["iteration"] == 2
    assert marker["attempt"] == 3
    assert marker["trigger_reason"] == "repeated_error"
    assert marker["created_at"]

    success_note = write_kernel_regeneration_note(
        agent_dir=agent_dir,
        attempt=3,
        trigger_reason="repeated_error",
    )
    assert "kernel_regen_applied" in success_note.read_text(encoding="utf-8")

    failure_note = write_kernel_regeneration_note(
        agent_dir=agent_dir,
        attempt=4,
        trigger_reason="repeated_error",
        error=RuntimeError("boom"),
    )
    failure_text = failure_note.read_text(encoding="utf-8")
    assert "kernel_regen_failed" in failure_text
    assert "error: boom" in failure_text


def test_maybe_restart_for_src_changes_allows_new_stage_family_after_legacy_restart(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)

    state_path = run_dir / "autofix_restart.json"
    state_path.write_text(
        json.dumps({"count": 1, "last_stage": "kernel_fix_attempt_1"}, indent=2),
        encoding="utf-8",
    )

    execv_calls: list[tuple[str, list[str]]] = []

    def fake_execv(executable: str, argv: list[str]) -> None:
        execv_calls.append((executable, argv))

    monkeypatch.setattr("kagglebot.autofix_restart.os.execv", fake_execv)

    restarted = maybe_restart_for_src_changes(
        dry_run=False,
        run_dir=run_dir,
        run_id="run-1",
        slug="demo",
        changed=["src/kagglebot/solver/io.py"],
        stage="autofix_attempt_1",
        max_restarts=1,
    )

    assert restarted is True
    assert len(execv_calls) == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_stage"] == "autofix_attempt_1"
    assert state["last_stage_family"] == "autofix"
    assert state["counts_by_stage"]["kernel_fix"] == 1
    assert state["counts_by_stage"]["autofix"] == 1


def test_maybe_restart_for_src_changes_blocks_second_restart_in_same_stage_family(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)

    state_path = run_dir / "autofix_restart.json"
    state_path.write_text(
        json.dumps({"counts_by_stage": {"autofix": 1}, "count": 1, "last_stage": "autofix_attempt_1"}, indent=2),
        encoding="utf-8",
    )

    execv_calls: list[tuple[str, list[str]]] = []

    def fake_execv(executable: str, argv: list[str]) -> None:
        execv_calls.append((executable, argv))

    monkeypatch.setattr("kagglebot.autofix_restart.os.execv", fake_execv)

    restarted = maybe_restart_for_src_changes(
        dry_run=False,
        run_dir=run_dir,
        run_id="run-1",
        slug="demo",
        changed=["src/kagglebot/autopilot.py"],
        stage="autofix_attempt_2",
        max_restarts=1,
    )

    assert restarted is False
    assert execv_calls == []
