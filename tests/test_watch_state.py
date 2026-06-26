from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from kagglebot.watch_state import (
    WATCH_STATE_ENV,
    active_run_stale_hours,
    active_state_is_stale,
    load_watch_state,
    safe_state_scope,
    set_resume_env,
    update_watch_phase,
    write_watch_state,
)


def test_update_watch_phase_writes_active_state(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setenv(WATCH_STATE_ENV, str(state_path))
    config = SimpleNamespace(slug="demo", compute="local_gpu")

    update_watch_phase(
        config,
        "run-1",
        "gpt_planning",
        detail="drafting plan",
        iteration=2,
    )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["active_slug"] == "demo"
    assert payload["active_run_id"] == "run-1"
    assert payload["last_status"] == "running"
    assert payload["phase"] == "gpt_planning"
    assert payload["phase_detail"] == "drafting plan"
    assert payload["iteration"] == 2
    assert payload["compute"] == "local_gpu"
    assert payload["updated_at"]


def test_update_watch_phase_ignores_different_active_run(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"active_slug": "demo", "active_run_id": "run-other", "last_status": "running"}),
        encoding="utf-8",
    )
    monkeypatch.setenv(WATCH_STATE_ENV, str(state_path))
    config = SimpleNamespace(slug="demo", compute="local_gpu")

    update_watch_phase(config, "run-1", "resolving_plan")

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload == {"active_slug": "demo", "active_run_id": "run-other", "last_status": "running"}


def test_load_watch_state_returns_empty_for_missing_invalid_or_non_object_payload(tmp_path: Path) -> None:
    assert load_watch_state(tmp_path / "missing.json") == {}

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert load_watch_state(invalid) == {}

    array_payload = tmp_path / "array.json"
    array_payload.write_text("[]", encoding="utf-8")
    assert load_watch_state(array_payload) == {}


def test_write_watch_state_adds_timestamp(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"

    write_watch_state(state_path, {"active_slug": "demo"})

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["active_slug"] == "demo"
    assert payload["updated_at"]


def test_safe_state_scope_replaces_unsafe_characters() -> None:
    assert safe_state_scope(" local gpu / rtx 3060 ") == "local-gpu-rtx-3060"


def test_active_run_stale_hours_falls_back_and_clamps(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_WATCH_ACTIVE_RUN_STALE_HOURS", "bad")
    assert active_run_stale_hours() == 24.0

    monkeypatch.setenv("KAGGLEBOT_WATCH_ACTIVE_RUN_STALE_HOURS", "-1")
    assert active_run_stale_hours() == 0.0


def test_active_state_is_stale_uses_updated_timestamp(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_WATCH_ACTIVE_RUN_STALE_HOURS", "1")
    old = datetime.now(UTC) - timedelta(hours=2)
    fresh = datetime.now(UTC)

    assert active_state_is_stale({"active_slug": "demo", "active_run_id": "run-1", "updated_at": old.isoformat()})
    assert not active_state_is_stale({"active_slug": "demo", "active_run_id": "run-1", "updated_at": fresh.isoformat()})
    assert not active_state_is_stale({"active_slug": "demo", "updated_at": old.isoformat()})


def test_set_resume_env(monkeypatch) -> None:
    monkeypatch.delenv("KAGGLEBOT_RESUME_RUN_ID", raising=False)
    monkeypatch.delenv("KAGGLEBOT_RESUME_SLUG", raising=False)

    set_resume_env(slug="demo", run_id="run-1")

    assert os.environ["KAGGLEBOT_RESUME_RUN_ID"] == "run-1"
    assert os.environ["KAGGLEBOT_RESUME_SLUG"] == "demo"
