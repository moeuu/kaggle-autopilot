from __future__ import annotations

import json
from pathlib import Path

from kagglebot.autofix_restart import maybe_restart_for_src_changes


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
