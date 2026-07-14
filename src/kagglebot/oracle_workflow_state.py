from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ORACLE_WORKFLOW_STATE_FILENAME = "oracle_workflow_state.json"
PENDING_ORACLE_WORKFLOW_STATUSES = frozenset({"pending_oracle", "pending_codex"})


class OracleWorkflowStateError(RuntimeError):
    """The persisted Oracle workflow state cannot be recovered safely."""


class PendingOracleWorkflowConflictError(RuntimeError):
    """A different interrupted Oracle workflow must be recovered first."""


@dataclass(frozen=True)
class OracleWorkflowCheckpoint:
    path: Path
    workflow_id: str

    def mark_oracle_complete(self, *, response_path: Path | None = None) -> None:
        updates: dict[str, object] = {"status": "pending_codex", "oracle_completed_at": _now()}
        if response_path is not None:
            updates["oracle_response_path"] = str(response_path)
        self._update(updates)

    def mark_completed(self) -> None:
        self._update({"status": "completed", "completed_at": _now()})

    def mark_failed(self, error: Exception) -> None:
        self._update(
            {
                "status": "failed",
                "failed_at": _now(),
                "error_type": type(error).__name__,
                "error": str(error)[:4000],
            }
        )

    def _update(self, updates: dict[str, object]) -> None:
        payload = _load_state(self.path)
        if str(payload.get("workflow_id") or "") != self.workflow_id:
            raise PendingOracleWorkflowConflictError(
                f"Oracle workflow checkpoint changed while running: expected {self.workflow_id}."
            )
        payload.update(updates)
        payload["updated_at"] = _now()
        _write_state(self.path, payload)


def oracle_workflow_state_path(run_dir: Path) -> Path:
    return run_dir / ORACLE_WORKFLOW_STATE_FILENAME


def load_pending_oracle_workflow(run_dir: Path) -> dict[str, object] | None:
    path = oracle_workflow_state_path(run_dir)
    if not path.exists():
        return None
    payload = _load_state(path)
    if str(payload.get("status") or "") not in PENDING_ORACLE_WORKFLOW_STATUSES:
        return None
    if not str(payload.get("workflow_id") or "").strip():
        raise OracleWorkflowStateError(f"Pending Oracle workflow state has no workflow_id: {path}")
    if not str(payload.get("workflow_kind") or "").strip():
        raise OracleWorkflowStateError(f"Pending Oracle workflow state has no workflow_kind: {path}")
    if not isinstance(payload.get("recovery_payload"), dict):
        raise OracleWorkflowStateError(f"Pending Oracle workflow state has no recovery_payload object: {path}")
    return payload


def begin_oracle_workflow(
    *,
    run_dir: Path,
    workflow_id: str,
    workflow_kind: str,
    recovery_payload: dict[str, object],
) -> OracleWorkflowCheckpoint:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = oracle_workflow_state_path(run_dir)
    previous = _load_state(path) if path.exists() else {}
    previous_status = str(previous.get("status") or "")
    previous_id = str(previous.get("workflow_id") or "")
    if previous_status in PENDING_ORACLE_WORKFLOW_STATUSES and previous_id != workflow_id:
        raise PendingOracleWorkflowConflictError(
            f"Interrupted Oracle workflow {previous_id!r} must be recovered before starting {workflow_id!r}."
        )
    recovery_count = int(previous.get("recovery_count") or 0)
    started_at = _now()
    if previous_status in PENDING_ORACLE_WORKFLOW_STATUSES and previous_id == workflow_id:
        recovery_count += 1
        started_at = str(previous.get("started_at") or started_at)
    payload: dict[str, object] = {
        "version": 1,
        "workflow_id": workflow_id,
        "workflow_kind": workflow_kind,
        "status": "pending_oracle",
        "started_at": started_at,
        "updated_at": _now(),
        "recovery_count": recovery_count,
        "recovery_payload": recovery_payload,
    }
    _write_state(path, payload)
    return OracleWorkflowCheckpoint(path=path, workflow_id=workflow_id)


@contextmanager
def oracle_workflow_checkpoint(
    *,
    run_dir: Path,
    workflow_id: str,
    workflow_kind: str,
    recovery_payload: dict[str, object],
) -> Iterator[OracleWorkflowCheckpoint]:
    checkpoint = begin_oracle_workflow(
        run_dir=run_dir,
        workflow_id=workflow_id,
        workflow_kind=workflow_kind,
        recovery_payload=recovery_payload,
    )
    try:
        yield checkpoint
    except Exception as exc:
        checkpoint.mark_failed(exc)
        raise
    else:
        checkpoint.mark_completed()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_state(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OracleWorkflowStateError(f"Cannot read Oracle workflow state {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OracleWorkflowStateError(f"Oracle workflow state must be a JSON object: {path}")
    return payload


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
