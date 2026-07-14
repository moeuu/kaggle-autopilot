from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.oracle_workflow_state import (
    OracleWorkflowStateError,
    PendingOracleWorkflowConflictError,
    load_pending_oracle_workflow,
    oracle_workflow_checkpoint,
    oracle_workflow_state_path,
)


def test_oracle_workflow_checkpoint_tracks_oracle_codex_completion(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    response_path = run_dir / "agent" / "strategy_last_message.txt"

    with oracle_workflow_checkpoint(
        run_dir=run_dir,
        workflow_id="kernel-fix-iter-1-attempt-1",
        workflow_kind="kernel_fix",
        recovery_payload={"iteration": 1, "attempt": 1},
    ) as checkpoint:
        checkpoint.mark_oracle_complete(response_path=response_path)
        pending = load_pending_oracle_workflow(run_dir)
        assert pending is not None
        assert pending["status"] == "pending_codex"

    payload = json.loads(oracle_workflow_state_path(run_dir).read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["oracle_response_path"] == str(response_path)
    assert load_pending_oracle_workflow(run_dir) is None


def test_oracle_workflow_checkpoint_leaves_interrupted_base_exception_pending(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    with pytest.raises(KeyboardInterrupt):
        with oracle_workflow_checkpoint(
            run_dir=run_dir,
            workflow_id="improvement-iter-1",
            workflow_kind="improvement",
            recovery_payload={"iteration": 1},
        ):
            raise KeyboardInterrupt

    pending = load_pending_oracle_workflow(run_dir)
    assert pending is not None
    assert pending["status"] == "pending_oracle"


def test_oracle_workflow_checkpoint_marks_regular_failure_without_recovery(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="oracle invalid"):
        with oracle_workflow_checkpoint(
            run_dir=run_dir,
            workflow_id="autofix-attempt-1",
            workflow_kind="autofix",
            recovery_payload={"attempt": 1},
        ):
            raise RuntimeError("oracle invalid")

    payload = json.loads(oracle_workflow_state_path(run_dir).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert load_pending_oracle_workflow(run_dir) is None


def test_oracle_workflow_checkpoint_blocks_different_pending_workflow(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        with oracle_workflow_checkpoint(
            run_dir=run_dir,
            workflow_id="kernel-fix-iter-1-attempt-1",
            workflow_kind="kernel_fix",
            recovery_payload={"iteration": 1},
        ):
            raise KeyboardInterrupt

    with pytest.raises(PendingOracleWorkflowConflictError):
        with oracle_workflow_checkpoint(
            run_dir=run_dir,
            workflow_id="improvement-iter-1",
            workflow_kind="improvement",
            recovery_payload={"iteration": 1},
        ):
            pass


def test_load_pending_oracle_workflow_blocks_corrupt_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    oracle_workflow_state_path(run_dir).write_text('{"status": "pending_oracle"', encoding="utf-8")

    with pytest.raises(OracleWorkflowStateError, match="Cannot read Oracle workflow state"):
        load_pending_oracle_workflow(run_dir)


def test_load_pending_oracle_workflow_blocks_incomplete_pending_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    oracle_workflow_state_path(run_dir).write_text(
        json.dumps({"status": "pending_codex", "workflow_id": "fix-1"}),
        encoding="utf-8",
    )

    with pytest.raises(OracleWorkflowStateError, match="workflow_kind"):
        load_pending_oracle_workflow(run_dir)
