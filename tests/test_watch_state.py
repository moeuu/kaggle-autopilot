from __future__ import annotations

import json
from types import SimpleNamespace

from kagglebot.watch_state import WATCH_STATE_ENV, update_watch_phase


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
