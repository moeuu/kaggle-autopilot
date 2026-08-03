from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.autofix_restart import (
    SourceReloadLoopError,
    kernel_regenerate_marker_path,
    kernel_regeneration_already_marked,
    maybe_regenerate_kernel_sources_once,
    maybe_restart_for_src_changes,
    source_tree_fingerprint,
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


def test_maybe_regenerate_kernel_sources_once_runs_callback_and_writes_note(tmp_path: Path) -> None:
    agent_dir = tmp_path / "iter-1" / "agent"
    calls: list[str] = []
    messages: list[str] = []

    regenerated = maybe_regenerate_kernel_sources_once(
        dry_run=False,
        agent_dir=agent_dir,
        run_id="run-1",
        iteration=1,
        attempt=2,
        trigger_reason="repeated_error",
        regenerate_kernel_sources=lambda: calls.append("regenerate"),
        on_message=messages.append,
    )

    assert regenerated is True
    assert calls == ["regenerate"]
    assert messages
    assert kernel_regeneration_already_marked(agent_dir) is True
    assert "kernel_regen_applied" in (agent_dir / "kernel_regen_note-02.txt").read_text(encoding="utf-8")


def test_maybe_regenerate_kernel_sources_once_skips_after_marker(tmp_path: Path) -> None:
    agent_dir = tmp_path / "iter-1" / "agent"
    write_kernel_regeneration_marker(
        agent_dir=agent_dir,
        run_id="run-1",
        iteration=1,
        attempt=1,
        trigger_reason="prior",
    )
    calls: list[str] = []

    regenerated = maybe_regenerate_kernel_sources_once(
        dry_run=False,
        agent_dir=agent_dir,
        run_id="run-1",
        iteration=1,
        attempt=2,
        trigger_reason="repeated_error",
        regenerate_kernel_sources=lambda: calls.append("regenerate"),
    )

    assert regenerated is False
    assert calls == []


def test_maybe_regenerate_kernel_sources_once_records_failure_note(tmp_path: Path) -> None:
    agent_dir = tmp_path / "iter-1" / "agent"

    def fail() -> None:
        raise RuntimeError("boom")

    regenerated = maybe_regenerate_kernel_sources_once(
        dry_run=False,
        agent_dir=agent_dir,
        run_id="run-1",
        iteration=1,
        attempt=3,
        trigger_reason="repeated_error",
        regenerate_kernel_sources=fail,
    )

    assert regenerated is False
    assert kernel_regeneration_already_marked(agent_dir) is True
    note = (agent_dir / "kernel_regen_note-03.txt").read_text(encoding="utf-8")
    assert "kernel_regen_failed" in note
    assert "error: boom" in note


def test_kernel_regeneration_marker_is_bounded_per_kernel_sha(tmp_path: Path) -> None:
    agent_dir = tmp_path / "iter-1" / "agent"
    current_sha = ["a" * 64]
    calls: list[str] = []

    def regenerate() -> None:
        calls.append(current_sha[0])
        current_sha[0] = "b" * 64

    assert maybe_regenerate_kernel_sources_once(
        dry_run=False,
        agent_dir=agent_dir,
        run_id="run-1",
        iteration=1,
        attempt=1,
        trigger_reason="no_changes",
        regenerate_kernel_sources=regenerate,
        get_kernel_sha256=lambda: current_sha[0],
    )
    assert not maybe_regenerate_kernel_sources_once(
        dry_run=False,
        agent_dir=agent_dir,
        run_id="run-1",
        iteration=1,
        attempt=2,
        trigger_reason="same_kernel",
        regenerate_kernel_sources=lambda: calls.append("unexpected"),
        get_kernel_sha256=lambda: current_sha[0],
    )

    current_sha[0] = "c" * 64
    assert maybe_regenerate_kernel_sources_once(
        dry_run=False,
        agent_dir=agent_dir,
        run_id="run-1",
        iteration=1,
        attempt=3,
        trigger_reason="source_changed",
        regenerate_kernel_sources=lambda: calls.append(current_sha[0]),
        get_kernel_sha256=lambda: current_sha[0],
    )
    assert calls == ["a" * 64, "c" * 64]


def test_maybe_restart_for_src_changes_allows_new_stage_family_after_legacy_restart(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    source_path = tmp_path / "src" / "kagglebot" / "solver" / "io.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("VALUE = 1\n", encoding="utf-8")

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
        repo_root=tmp_path,
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


def test_maybe_restart_for_src_changes_fails_closed_for_same_source_generation(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    source_path = tmp_path / "src" / "kagglebot" / "autopilot.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    fingerprint = source_tree_fingerprint(tmp_path)

    state_path = run_dir / "autofix_restart.json"
    state_path.write_text(
        json.dumps(
            {
                "counts_by_stage": {"autofix": 1},
                "counts_by_source": {fingerprint: 1},
                "count": 1,
                "last_stage": "autofix_attempt_1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    execv_calls: list[tuple[str, list[str]]] = []

    def fake_execv(executable: str, argv: list[str]) -> None:
        execv_calls.append((executable, argv))

    monkeypatch.setattr("kagglebot.autofix_restart.os.execv", fake_execv)

    with pytest.raises(SourceReloadLoopError, match="refusing to continue in a stale process"):
        maybe_restart_for_src_changes(
            dry_run=False,
            run_dir=run_dir,
            repo_root=tmp_path,
            run_id="run-1",
            slug="demo",
            changed=["src/kagglebot/autopilot.py"],
            stage="autofix_attempt_2",
            max_restarts=1,
        )

    assert execv_calls == []


def test_maybe_restart_for_src_changes_restarts_each_distinct_source_generation(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    source_path = tmp_path / "src" / "kagglebot" / "autopilot.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    first_fingerprint = source_tree_fingerprint(tmp_path)
    (run_dir / "autofix_restart.json").write_text(
        json.dumps({"counts_by_source": {first_fingerprint: 1}}),
        encoding="utf-8",
    )
    source_path.write_text("VALUE = 2\n", encoding="utf-8")
    execv_calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        "kagglebot.autofix_restart.os.execv",
        lambda executable, argv: execv_calls.append((executable, argv)),
    )

    assert maybe_restart_for_src_changes(
        dry_run=False,
        run_dir=run_dir,
        repo_root=tmp_path,
        run_id="run-1",
        slug="demo",
        changed=["src/kagglebot/autopilot.py"],
        stage="autofix_attempt_2",
        max_restarts=1,
    )
    assert len(execv_calls) == 1
    state = json.loads((run_dir / "autofix_restart.json").read_text(encoding="utf-8"))
    assert state["last_source_fingerprint"] != first_fingerprint
