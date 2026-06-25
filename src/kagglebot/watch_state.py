from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.json_utils import load_json_object_or_empty, write_json_object

WATCH_STATE_ENV = "KAGGLEBOT_WATCH_STATE_PATH"


def update_watch_phase(
    config: object,
    run_id: str,
    phase: str,
    *,
    detail: str | None = None,
    iteration: int | None = None,
) -> None:
    state_raw = os.environ.get(WATCH_STATE_ENV)
    if not state_raw:
        return
    state_path = Path(state_raw)
    payload = load_json_object_or_empty(state_path)
    config_slug = str(getattr(config, "slug", "") or "").strip()
    active_slug = str(payload.get("active_slug") or "").strip()
    active_run_id = str(payload.get("active_run_id") or "").strip()
    if active_slug and active_slug != config_slug:
        return
    if active_run_id and active_run_id != run_id:
        return
    payload.update(
        {
            "active_slug": config_slug,
            "active_run_id": run_id,
            "last_status": "running",
            "phase": phase,
            "compute": str(getattr(config, "compute", "") or ""),
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )
    if detail:
        payload["phase_detail"] = detail
    else:
        payload.pop("phase_detail", None)
    if iteration is not None:
        payload["iteration"] = iteration
    try:
        write_json_object(state_path, payload, sort_keys=True)
    except OSError:
        return
