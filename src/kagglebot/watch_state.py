from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kagglebot.datetime_utils import parse_iso_datetime_utc
from kagglebot.env_utils import parse_float_value
from kagglebot.json_utils import load_json_object_or_empty, write_json_object

WATCH_STATE_ENV = "KAGGLEBOT_WATCH_STATE_PATH"
DEFAULT_ACTIVE_RUN_STALE_HOURS = 24.0


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


def safe_state_scope(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")


def load_watch_state(path: Path) -> dict[str, object]:
    return load_json_object_or_empty(path)


def write_watch_state(path: Path, payload: dict[str, object]) -> None:
    payload = dict(payload)
    payload.setdefault("updated_at", datetime.now(UTC).isoformat())
    write_json_object(path, payload, sort_keys=True)


def set_resume_env(*, slug: str, run_id: str) -> None:
    os.environ["KAGGLEBOT_RESUME_RUN_ID"] = run_id
    os.environ["KAGGLEBOT_RESUME_SLUG"] = slug


def active_state_is_stale(state: dict[str, object]) -> bool:
    slug = str(state.get("active_slug") or "").strip()
    run_id = str(state.get("active_run_id") or "").strip()
    if not slug or not run_id:
        return False
    timestamp = _parse_ts(state.get("updated_at")) or _parse_ts(state.get("started_at"))
    if timestamp is None:
        return False
    max_age_hours = active_run_stale_hours()
    if max_age_hours <= 0:
        return False
    return timestamp + timedelta(hours=max_age_hours) <= datetime.now(UTC)


def active_run_stale_hours() -> float:
    value = parse_float_value(os.environ.get("KAGGLEBOT_WATCH_ACTIVE_RUN_STALE_HOURS"))
    if value is None:
        return DEFAULT_ACTIVE_RUN_STALE_HOURS
    return max(0.0, value)


def _parse_ts(value: object) -> datetime | None:
    return parse_iso_datetime_utc(value)
