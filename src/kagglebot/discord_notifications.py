from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rich import print

from kagglebot.datetime_utils import parse_iso_datetime_utc
from kagglebot.json_utils import (
    load_json_object_or_empty,
    load_jsonl_records,
    parse_json_object_bytes,
    write_json_object,
)
from kagglebot.kaggle_api import leaderboard_rank_for_score
from kagglebot.metric_matching import metrics_equivalent as _metrics_equivalent

SOURCE = "kaggle-autopilot"
DEFAULT_ACCOUNT = "lab_rdp"
STATE_FILENAME = "discord_notifier_state.json"
DELIVERY_RECEIPTS_FILENAME = "discord_delivery_receipts.jsonl"
LEDGER_OFFSET_KEY = "watch_ledger_offset"
LEDGER_CURSOR_INITIALIZED_KEY = "watch_ledger_cursor_initialized"
_LIFECYCLE_EVENT_TYPES = {
    "started": "autopilot.started",
    "finished": "autopilot.finished",
    "failed": "autopilot.failed",
}


@dataclass(frozen=True)
class DiscordEventConfig:
    api_url: str
    api_token: str
    account: str = DEFAULT_ACCOUNT
    source: str = SOURCE
    timeout_sec: float = 10.0


@dataclass(frozen=True)
class DiscordEventReceipt:
    accepted: bool
    event_id: str = ""
    event_type: str = ""
    dedupe_key: str = ""
    occurred_at: str = ""
    matched_routes: int = 0
    response_event_id: str = ""
    response_status: str = ""
    error: str = ""

    def __bool__(self) -> bool:
        return self.accepted

    def as_record(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "dedupe_key": self.dedupe_key,
            "occurred_at": self.occurred_at,
            "matched_routes": self.matched_routes,
            "response_event_id": self.response_event_id,
            "response_status": self.response_status,
            "error": self.error,
        }


class DiscordEventNotifier:
    def __init__(self, config: DiscordEventConfig | None) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config is not None

    def emit(
        self,
        *,
        event_type: str,
        severity: str,
        dedupe_key: str,
        payload: dict[str, object],
        occurred_at: datetime | None = None,
    ) -> DiscordEventReceipt:
        if self.config is None:
            return DiscordEventReceipt(
                accepted=False,
                event_type=event_type,
                dedupe_key=dedupe_key,
                error="notifier_disabled",
            )
        occurred = occurred_at or datetime.now(UTC)
        event_id = f"evt-kaggle-autopilot-{occurred.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        occurred_text = occurred.isoformat().replace("+00:00", "Z")
        body = {
            "id": event_id,
            "source": self.config.source,
            "type": event_type,
            "occurred_at": occurred_text,
            "account": self.config.account,
            "severity": severity,
            "dedupe_key": dedupe_key,
            "payload": payload,
        }
        data = json.dumps(body, ensure_ascii=True).encode("utf-8")
        request = urllib.request.Request(
            self.config.api_url,
            data=data,
            headers={
                "authorization": f"Bearer {self.config.api_token}",
                "content-type": "application/json",
                "user-agent": "kaggle-autopilot-discord-notifier/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                raw = response.read()
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            print(f"[yellow]discord notify failed[/yellow]: {exc}")
            return DiscordEventReceipt(
                accepted=False,
                event_id=event_id,
                event_type=event_type,
                dedupe_key=dedupe_key,
                occurred_at=occurred_text,
                error=f"{type(exc).__name__}: {exc}",
            )
        parsed = parse_json_object_bytes(raw) or {}
        matched_routes = _int_or_none(parsed.get("matched_routes")) or 0
        response_event_id = _response_text(parsed, "event_id", "id")
        response_status = _response_text(parsed, "delivery_status", "status")
        if matched_routes <= 0:
            print("[yellow]discord notify accepted but matched no routes[/yellow]")
            return DiscordEventReceipt(
                accepted=False,
                event_id=event_id,
                event_type=event_type,
                dedupe_key=dedupe_key,
                occurred_at=occurred_text,
                matched_routes=matched_routes,
                response_event_id=response_event_id,
                response_status=response_status,
                error="accepted_without_matching_route",
            )
        return DiscordEventReceipt(
            accepted=True,
            event_id=event_id,
            event_type=event_type,
            dedupe_key=dedupe_key,
            occurred_at=occurred_text,
            matched_routes=matched_routes,
            response_event_id=response_event_id,
            response_status=response_status,
        )


def notifier_from_env() -> DiscordEventNotifier:
    api_url = _env_first("KAGGLEBOT_DISCORD_EVENT_API_URL", "DISCORD_EVENT_API_URL")
    token = _env_first("KAGGLEBOT_DISCORD_EVENT_API_TOKEN", "DISCORD_EVENT_API_TOKEN", "EVENT_API_TOKEN")
    if not api_url or not token:
        return DiscordEventNotifier(None)
    account = _env_first("KAGGLEBOT_DISCORD_EVENT_ACCOUNT") or DEFAULT_ACCOUNT
    timeout_raw = _env_first("KAGGLEBOT_DISCORD_EVENT_TIMEOUT_SEC")
    timeout = _float_or_default(timeout_raw, 10.0)
    return DiscordEventNotifier(
        DiscordEventConfig(
            api_url=api_url,
            api_token=token,
            account=account,
            timeout_sec=timeout,
        )
    )


def run_discord_notifier_once(
    *,
    artifacts_dir: Path,
    heartbeat_sec: int,
    force: bool = False,
    notifier: DiscordEventNotifier | None = None,
    now: datetime | None = None,
) -> bool:
    notifier = notifier or notifier_from_env()
    current_time = now or datetime.now(UTC)
    sent_any = False
    for watch_state_path in _watch_state_paths(artifacts_dir):
        sent_any = (
            _run_discord_notifier_for_watch_state(
                artifacts_dir=artifacts_dir,
                watch_state_path=watch_state_path,
                heartbeat_sec=heartbeat_sec,
                force=force,
                notifier=notifier,
                current_time=current_time,
            )
            or sent_any
        )
    return sent_any


def _run_discord_notifier_for_watch_state(
    *,
    artifacts_dir: Path,
    watch_state_path: Path,
    heartbeat_sec: int,
    force: bool,
    notifier: DiscordEventNotifier,
    current_time: datetime,
) -> bool:
    snapshot = build_autopilot_status_payload(
        artifacts_dir=artifacts_dir,
        watch_state_path=watch_state_path,
        now=current_time,
    )
    state_path = watch_state_path.parent / STATE_FILENAME
    state = _read_json_object(state_path)
    snapshot_key = _snapshot_key(snapshot)
    lifecycle_sent, lifecycle_blocked = _replay_watch_lifecycle_events(
        artifacts_dir=artifacts_dir,
        watch_state_path=watch_state_path,
        state_path=state_path,
        state=state,
        snapshot=snapshot,
        notifier=notifier,
        current_time=current_time,
    )
    if lifecycle_blocked:
        return lifecycle_sent
    if lifecycle_sent and _is_idle_snapshot(snapshot):
        state.update(
            {
                "last_snapshot_key": snapshot_key,
                "last_sent_at": current_time.isoformat(),
                "last_run_id": _clean_str(snapshot.get("run_id")),
            }
        )
        write_json_object(state_path, state, sort_keys=True)
        return True

    event_type = _event_type_for_snapshot(snapshot)
    current_run_id = _clean_str(snapshot.get("run_id"))
    last_run_id = _clean_str(state.get("last_run_id"))
    if current_run_id and current_run_id != last_run_id:
        event_type = "autopilot.started"
    last_key = str(state.get("last_snapshot_key") or "")
    last_sent_at = _parse_datetime(state.get("last_sent_at"))
    heartbeat_due = last_sent_at is None or (current_time - last_sent_at).total_seconds() >= max(1, heartbeat_sec)
    heartbeat_send_allowed = heartbeat_due and not _is_idle_snapshot(snapshot)
    should_send = force or lifecycle_sent or snapshot_key != last_key or heartbeat_send_allowed
    if not should_send:
        if current_run_id and current_run_id != last_run_id:
            state["last_run_id"] = current_run_id
            write_json_object(state_path, state, sort_keys=True)
        print(f"[cyan]discord notifier[/cyan]: unchanged ({snapshot_key})")
        return False
    if not notifier.enabled:
        print("[yellow]discord notifier[/yellow]: disabled; set KAGGLEBOT_DISCORD_EVENT_API_URL and token")
        return False
    snapshot = dict(snapshot)
    snapshot["discord_update_key"] = _discord_event_update_key(snapshot=snapshot, event_type=event_type)
    receipt = notifier.emit(
        event_type=event_type,
        severity=_severity_for_snapshot(snapshot),
        dedupe_key=_dedupe_key(snapshot=snapshot, event_type=event_type, now=current_time),
        payload=snapshot,
        occurred_at=current_time,
    )
    _persist_delivery_receipt(
        watch_dir=watch_state_path.parent,
        receipt=receipt,
        recorded_at=current_time,
        source="status_snapshot",
    )
    if receipt:
        state.update(
            {
                "last_snapshot_key": snapshot_key,
                "last_sent_at": current_time.isoformat(),
                "last_event_type": event_type,
                "last_run_id": current_run_id,
            }
        )
        write_json_object(state_path, state, sort_keys=True)
        print(f"[green]discord notifier[/green]: sent {event_type} ({snapshot_key})")
    return bool(receipt)


def _replay_watch_lifecycle_events(
    *,
    artifacts_dir: Path,
    watch_state_path: Path,
    state_path: Path,
    state: dict[str, object],
    snapshot: dict[str, object],
    notifier: DiscordEventNotifier,
    current_time: datetime,
) -> tuple[bool, bool]:
    if not notifier.enabled:
        return False, False

    ledger_path = watch_state_path.with_name("ledger.jsonl")
    if not state.get(LEDGER_CURSOR_INITIALIZED_KEY):
        state[LEDGER_OFFSET_KEY] = ledger_path.stat().st_size if ledger_path.exists() else 0
        state[LEDGER_CURSOR_INITIALIZED_KEY] = True
        write_json_object(state_path, state, sort_keys=True)
        return False, False

    offset = max(0, _int_or_none(state.get(LEDGER_OFFSET_KEY)) or 0)
    if ledger_path.exists() and offset > ledger_path.stat().st_size:
        offset = 0
    saved_offset = offset
    sent_any = False
    for next_offset, record in _watch_ledger_records_after(ledger_path, offset):
        event_name = _clean_str(record.get("event")) or ""
        event_type = _LIFECYCLE_EVENT_TYPES.get(event_name)
        if event_type is not None:
            payload = _lifecycle_payload(
                artifacts_dir=artifacts_dir,
                snapshot=snapshot,
                record=record,
                event_name=event_name,
                event_type=event_type,
                current_time=current_time,
            )
            occurred_at = _parse_datetime(record.get("ts")) or current_time
            dedupe_key = _lifecycle_dedupe_key(
                payload=payload,
                event_type=event_type,
                ledger_offset=offset,
            )
            receipt = notifier.emit(
                event_type=event_type,
                severity="error" if event_name == "failed" else "info",
                dedupe_key=dedupe_key,
                payload=payload,
                occurred_at=occurred_at,
            )
            _persist_delivery_receipt(
                watch_dir=watch_state_path.parent,
                receipt=receipt,
                recorded_at=current_time,
                source="watch_lifecycle",
                ledger_offset=offset,
                ledger_next_offset=next_offset,
            )
            if not receipt:
                state[LEDGER_OFFSET_KEY] = offset
                write_json_object(state_path, state, sort_keys=True)
                return sent_any, True
            sent_any = True
            offset = next_offset
            state["last_event_type"] = event_type
            state["last_run_id"] = _clean_str(payload.get("run_id"))
            state["last_sent_at"] = current_time.isoformat()
            state[LEDGER_OFFSET_KEY] = offset
            write_json_object(state_path, state, sort_keys=True)
            print(f"[green]discord notifier[/green]: replayed {event_type} at ledger offset {offset}")
            saved_offset = offset
            continue
        offset = next_offset
    if offset != saved_offset:
        state[LEDGER_OFFSET_KEY] = offset
        write_json_object(state_path, state, sort_keys=True)
    return sent_any, False


def _watch_ledger_records_after(path: Path, offset: int) -> list[tuple[int, dict[str, object]]]:
    if not path.exists():
        return []
    size = path.stat().st_size
    start = offset if offset <= size else 0
    records: list[tuple[int, dict[str, object]]] = []
    with path.open("rb") as handle:
        handle.seek(start)
        while True:
            raw = handle.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                break
            next_offset = handle.tell()
            parsed = parse_json_object_bytes(raw)
            if parsed is not None:
                records.append((next_offset, parsed))
            else:
                records.append((next_offset, {}))
    return records


def _persist_delivery_receipt(
    *,
    watch_dir: Path,
    receipt: object,
    recorded_at: datetime,
    source: str,
    ledger_offset: int | None = None,
    ledger_next_offset: int | None = None,
) -> None:
    """Append an API acknowledgement before advancing any lifecycle cursor."""
    if not isinstance(receipt, DiscordEventReceipt):
        return
    record = receipt.as_record()
    record.update(
        {
            "recorded_at": recorded_at.isoformat(),
            "source": source,
        }
    )
    _put_if_not_none(record, "ledger_offset", ledger_offset)
    _put_if_not_none(record, "ledger_next_offset", ledger_next_offset)
    path = watch_dir / DELIVERY_RECEIPTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _lifecycle_payload(
    *,
    artifacts_dir: Path,
    snapshot: dict[str, object],
    record: dict[str, object],
    event_name: str,
    event_type: str,
    current_time: datetime,
) -> dict[str, object]:
    state_scope = str(snapshot.get("state_scope") or "local_gpu")
    compute = _compute_for_scope(state_scope)
    slug = _clean_str(record.get("slug"))
    run_id = _clean_str(record.get("run_id"))
    same_run = bool(slug and run_id and slug == snapshot.get("competition") and run_id == snapshot.get("run_id"))
    payload = (
        dict(snapshot)
        if same_run
        else {
            "host": socket.gethostname(),
            "compute": compute,
            "state_scope": state_scope,
            "artifact_root": str(artifacts_dir / slug) if slug else str(artifacts_dir),
            "observed_at": (_parse_datetime(record.get("ts")) or current_time).isoformat(),
        }
    )
    if slug:
        payload["competition"] = slug
        payload["slug"] = slug
    if run_id:
        payload["run_id"] = run_id
        if slug:
            payload["run_dir"] = str(artifacts_dir / slug / "runs" / run_id)
    payload["status"] = "running" if event_name == "started" else event_name
    payload["phase"] = event_name
    payload["message"] = _lifecycle_message(event_name=event_name, slug=slug, compute=compute)
    for key in (
        "reason",
        "error",
        "resume",
        "submission_status",
        "submission_url",
        "submission_score",
        "submission_rank",
        "submission_total_teams",
        "writeup_title",
    ):
        _put_if_not_none(payload, key, record.get(key))
    payload["discord_update_key"] = _discord_event_update_key(snapshot=payload, event_type=event_type)
    return payload


def _lifecycle_message(*, event_name: str, slug: str | None, compute: str) -> str:
    competition = slug or "an unknown competition"
    verb = {"started": "started", "finished": "finished", "failed": "failed"}.get(event_name, event_name)
    return f"Compute: {compute}\nKaggle autopilot {verb} for {competition}."


def _is_idle_snapshot(snapshot: dict[str, object]) -> bool:
    return str(snapshot.get("phase") or "").strip().lower() == "idle"


def run_discord_notifier_forever(
    *,
    artifacts_dir: Path,
    interval_sec: int,
    heartbeat_sec: int,
) -> None:
    while True:
        try:
            run_discord_notifier_once(
                artifacts_dir=artifacts_dir,
                heartbeat_sec=heartbeat_sec,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[yellow]discord notifier cycle failed[/yellow]: {type(exc).__name__}: {exc}")
        time.sleep(max(1, interval_sec))


def build_autopilot_status_payload(
    *,
    artifacts_dir: Path,
    now: datetime | None = None,
    watch_state_path: Path | None = None,
) -> dict[str, object]:
    current_time = now or datetime.now(UTC)
    state_path = watch_state_path or artifacts_dir / "_watch" / "state.json"
    state_scope = _watch_state_scope(artifacts_dir=artifacts_dir, state_path=state_path)
    compute = _compute_for_scope(state_scope)
    watch_state = _read_json_object(state_path)
    slug = _clean_str(watch_state.get("active_slug"))
    run_id = _clean_str(watch_state.get("active_run_id"))
    host = socket.gethostname()
    account = _env_first("KAGGLEBOT_DISCORD_EVENT_ACCOUNT") or DEFAULT_ACCOUNT
    if not slug or not run_id:
        return {
            "message": f"Kaggle autopilot is idle on {compute}.",
            "host": host,
            "compute": compute,
            "state_scope": state_scope,
            "discord_update_key": f"{_discord_update_key(account=account, state_scope=state_scope)}:idle",
            "status": _clean_str(watch_state.get("last_status")) or "idle",
            "phase": "idle",
            "artifact_root": str(artifacts_dir),
            "observed_at": current_time.isoformat(),
        }

    run_dir = artifacts_dir / slug / "runs" / run_id
    run_json = _read_json_object(run_dir / "run.json")
    run_config = run_json.get("config") if isinstance(run_json.get("config"), dict) else {}
    assert isinstance(run_config, dict)
    max_iterations = _int_or_none(run_config.get("max_iterations"))
    iter_dirs = _iteration_dirs(run_dir)
    current_iteration = iter_dirs[-1][0] if iter_dirs else None
    completed_iteration, marker = _latest_completed_iteration(iter_dirs)
    latest_metrics = _latest_metrics(iter_dirs)
    latest_score = _metric_score(latest_metrics)
    latest_metric = _clean_str(latest_metrics.get("metric"))
    best_score = _best_metric_score(
        iter_dirs,
        direction=_clean_str(run_config.get("target_direction")) or _clean_str(latest_metrics.get("direction")),
        preferred_metric=latest_metric or _clean_str(run_config.get("target_metric")),
    )
    submission_scores = _submission_score_summary(
        artifacts_dir=artifacts_dir,
        slug=slug,
        run_id=run_id,
        iter_dirs=iter_dirs,
        direction=_clean_str(run_config.get("target_direction")) or _clean_str(latest_metrics.get("direction")),
    )
    run_status = _clean_str(run_json.get("status"))
    watch_status = _clean_str(watch_state.get("last_status"))
    # The run record can retain a failed inner-stage status while the supervisor is
    # already recovering it. The active watch state is the orchestration authority.
    status = watch_status or run_status or "running"
    resolved_phase = _resolve_phase(
        status=status,
        current_iteration=current_iteration,
        completed_iteration=completed_iteration,
        marker=marker,
    )
    explicit_phase = _clean_str(watch_state.get("phase"))
    phase = explicit_phase if status.strip().lower() in {"", "running"} and explicit_phase else resolved_phase
    payload: dict[str, object] = {
        "message": _status_message(
            slug=slug,
            phase=phase,
            current_iteration=current_iteration,
            max_iterations=max_iterations,
            compute=compute,
        ),
        "host": host,
        "compute": compute,
        "state_scope": state_scope,
        "competition": slug,
        "slug": slug,
        "run_id": run_id,
        "discord_update_key": (f"{_discord_update_key(account=account, state_scope=state_scope)}:run:{run_id}"),
        "status": status,
        "phase": phase,
        "artifact_root": str(artifacts_dir / slug),
        "run_dir": str(run_dir),
        "observed_at": current_time.isoformat(),
    }
    _put_if_not_none(payload, "current_iteration", current_iteration)
    _put_if_not_none(payload, "iteration", current_iteration)
    _put_if_not_none(payload, "latest_completed_iteration", completed_iteration)
    _put_if_not_none(payload, "max_iterations", max_iterations)
    _put_if_not_none(payload, "latest_score", latest_score)
    _put_if_not_none(payload, "score", latest_score)
    _put_if_not_none(payload, "best_score", best_score)
    _put_if_not_none(payload, "latest_submission_score", submission_scores.get("latest_submission_score"))
    _put_if_not_none(payload, "best_submission_score", submission_scores.get("best_submission_score"))
    _put_if_not_none(payload, "submission_score_source", submission_scores.get("submission_score_source"))
    _put_if_not_none(payload, "latest_submission_iteration", submission_scores.get("latest_submission_iteration"))
    _put_if_not_none(payload, "best_submission_iteration", submission_scores.get("best_submission_iteration"))
    _put_if_not_none(payload, "latest_submission_rank", submission_scores.get("latest_submission_rank"))
    _put_if_not_none(payload, "latest_submission_total_teams", submission_scores.get("latest_submission_total_teams"))
    _put_if_not_none(
        payload, "latest_submission_rank_percentile", submission_scores.get("latest_submission_rank_percentile")
    )
    _put_if_not_none(payload, "latest_submission_rank_source", submission_scores.get("latest_submission_rank_source"))
    _put_if_not_none(payload, "best_submission_rank", submission_scores.get("best_submission_rank"))
    _put_if_not_none(payload, "best_submission_total_teams", submission_scores.get("best_submission_total_teams"))
    _put_if_not_none(
        payload, "best_submission_rank_percentile", submission_scores.get("best_submission_rank_percentile")
    )
    _put_if_not_none(payload, "best_submission_rank_source", submission_scores.get("best_submission_rank_source"))
    _put_if_not_none(
        payload, "latest_submission_estimated_rank", submission_scores.get("latest_submission_estimated_rank")
    )
    _put_if_not_none(
        payload,
        "latest_submission_estimated_total_teams",
        submission_scores.get("latest_submission_estimated_total_teams"),
    )
    _put_if_not_none(
        payload,
        "latest_submission_estimated_rank_percentile",
        submission_scores.get("latest_submission_estimated_rank_percentile"),
    )
    _put_if_not_none(
        payload,
        "latest_submission_rank_estimate_source",
        submission_scores.get("latest_submission_rank_estimate_source"),
    )
    _put_if_not_none(payload, "best_submission_estimated_rank", submission_scores.get("best_submission_estimated_rank"))
    _put_if_not_none(
        payload,
        "best_submission_estimated_total_teams",
        submission_scores.get("best_submission_estimated_total_teams"),
    )
    _put_if_not_none(
        payload,
        "best_submission_estimated_rank_percentile",
        submission_scores.get("best_submission_estimated_rank_percentile"),
    )
    _put_if_not_none(
        payload, "best_submission_rank_estimate_source", submission_scores.get("best_submission_rank_estimate_source")
    )
    _put_if_not_none(payload, "metric", latest_metric)
    _put_if_not_none(payload, "target_metric", _clean_str(run_config.get("target_metric")))
    _put_if_not_none(
        payload, "score_source", _clean_str(latest_metrics.get("score_source") or run_config.get("score_source"))
    )
    _put_if_not_none(
        payload,
        "submit_phase_state",
        _status_submit_phase_state(
            status=status,
            current_iteration=current_iteration,
            completed_iteration=completed_iteration,
            completed_marker=marker,
            run_config=run_config,
        ),
    )
    _put_if_not_none(payload, "phase_detail", _clean_str(watch_state.get("phase_detail")))
    _put_if_not_none(payload, "run_record_status", run_status)
    _put_if_not_none(payload, "readiness_score", _nested_number(latest_metrics, "readiness", "score"))
    return payload


def _event_type_for_snapshot(snapshot: dict[str, object]) -> str:
    status = str(snapshot.get("status") or "").lower()
    phase = str(snapshot.get("phase") or "").lower()
    if status in {"failed", "submit_failed"}:
        return "autopilot.failed"
    if status in {"finished", "completed", "submitted"}:
        return "autopilot.finished"
    if phase == "idle":
        return "autopilot.status"
    current_iteration = snapshot.get("current_iteration")
    completed_phase = str(snapshot.get("submit_phase_state") or "iteration_completed").strip().lower()
    if (
        current_iteration is not None
        and snapshot.get("latest_completed_iteration") == current_iteration
        and phase == completed_phase
    ):
        return "autopilot.iteration_completed"
    return "autopilot.status"


def _severity_for_snapshot(snapshot: dict[str, object]) -> str:
    status = str(snapshot.get("status") or "").lower()
    if status in {"failed", "submit_failed"}:
        return "error"
    return "info"


def _watch_state_paths(artifacts_dir: Path) -> list[Path]:
    watch_dir = artifacts_dir / "_watch"
    paths = [watch_dir / "state.json"]
    if watch_dir.exists():
        paths.extend(sorted(path for path in watch_dir.glob("*/state.json") if path.is_file()))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists() or path == watch_dir / "state.json":
            unique.append(path)
    return unique


def _watch_state_scope(*, artifacts_dir: Path, state_path: Path) -> str:
    watch_dir = artifacts_dir / "_watch"
    try:
        relative = state_path.resolve().relative_to(watch_dir.resolve())
    except ValueError:
        return "local_gpu"
    if len(relative.parts) >= 2 and relative.parts[-1] == "state.json":
        return relative.parts[0]
    return "local_gpu"


def _compute_for_scope(state_scope: str) -> str:
    return "local_gpu" if state_scope in {"", "local", "local_gpu"} else state_scope


def _discord_update_key(*, account: str, state_scope: str) -> str:
    normalized_scope = state_scope or "local_gpu"
    return f"kaggle-autopilot:{account}:{normalized_scope}"


def _discord_event_update_key(*, snapshot: dict[str, object], event_type: str) -> str:
    account = _env_first("KAGGLEBOT_DISCORD_EVENT_ACCOUNT") or DEFAULT_ACCOUNT
    state_scope = str(snapshot.get("state_scope") or "local_gpu")
    base = _discord_update_key(account=account, state_scope=state_scope)
    run_id = _clean_str(snapshot.get("run_id"))
    if not run_id:
        return f"{base}:idle"
    run_key = f"{base}:run:{run_id}"
    if event_type in {"autopilot.started", "autopilot.status"}:
        return run_key
    if event_type == "autopilot.iteration_completed":
        iteration = _int_or_none(snapshot.get("current_iteration"))
        return f"{run_key}:iteration:{iteration if iteration is not None else 'unknown'}"
    return f"{run_key}:{event_type.rsplit('.', 1)[-1]}"


def _dedupe_key(*, snapshot: dict[str, object], event_type: str, now: datetime) -> str:
    state_scope = str(snapshot.get("state_scope") or "local_gpu")
    slug = str(snapshot.get("competition") or "idle")
    run_id = str(snapshot.get("run_id") or "none")
    iteration = str(snapshot.get("current_iteration") or "none")
    phase = str(snapshot.get("phase") or "unknown")
    if event_type in {"autopilot.started", "autopilot.finished", "autopilot.failed"} and run_id != "none":
        bucket = "lifecycle"
    else:
        bucket = now.strftime("%Y%m%dT%H%M%S")
    return f"kaggle-autopilot:{state_scope}:{event_type}:{slug}:{run_id}:{iteration}:{phase}:{bucket}"


def _lifecycle_dedupe_key(
    *,
    payload: dict[str, object],
    event_type: str,
    ledger_offset: int,
) -> str:
    state_scope = str(payload.get("state_scope") or "local_gpu")
    slug = str(payload.get("competition") or "unknown")
    run_id = str(payload.get("run_id") or "none")
    return f"kaggle-autopilot:{state_scope}:{event_type}:{slug}:{run_id}:ledger:{ledger_offset}"


def _snapshot_key(snapshot: dict[str, object]) -> str:
    parts = [
        str(snapshot.get("state_scope") or "local_gpu"),
        str(snapshot.get("competition") or "idle"),
        str(snapshot.get("run_id") or "none"),
        str(snapshot.get("status") or "unknown"),
        str(snapshot.get("phase") or "unknown"),
        str(snapshot.get("current_iteration") or "none"),
        str(snapshot.get("latest_completed_iteration") or "none"),
        str(snapshot.get("latest_score") or "none"),
        str(snapshot.get("latest_submission_score") or "none"),
        str(snapshot.get("best_submission_score") or "none"),
        str(snapshot.get("latest_submission_rank") or snapshot.get("latest_submission_estimated_rank") or "none"),
        str(
            snapshot.get("latest_submission_total_teams")
            or snapshot.get("latest_submission_estimated_total_teams")
            or "none"
        ),
        str(snapshot.get("best_submission_rank") or snapshot.get("best_submission_estimated_rank") or "none"),
        str(
            snapshot.get("best_submission_total_teams")
            or snapshot.get("best_submission_estimated_total_teams")
            or "none"
        ),
        str(snapshot.get("submit_phase_state") or "none"),
    ]
    return "|".join(parts)


def _status_message(
    *,
    slug: str,
    phase: str,
    current_iteration: int | None,
    max_iterations: int | None,
    compute: str,
) -> str:
    iter_text = ""
    if current_iteration is not None and max_iterations is not None:
        iter_text = f" iteration {current_iteration}/{max_iterations}"
    elif current_iteration is not None:
        iter_text = f" iteration {current_iteration}"
    return f"Compute: {compute}\nKaggle autopilot is {phase.replace('_', ' ')} for {slug}{iter_text}."


def _resolve_phase(
    *,
    status: str,
    current_iteration: int | None,
    completed_iteration: int | None,
    marker: dict[str, object],
) -> str:
    normalized_status = status.strip().lower()
    if normalized_status not in {"running", ""}:
        return normalized_status
    if current_iteration is None:
        return "selecting_competition"
    if completed_iteration == current_iteration:
        submit_state = _clean_str(marker.get("submit_phase_state"))
        if submit_state:
            return submit_state
        return "iteration_completed"
    return "kernel_running"


def _status_submit_phase_state(
    *,
    status: str,
    current_iteration: int | None,
    completed_iteration: int | None,
    completed_marker: dict[str, object],
    run_config: dict[str, object],
) -> str | None:
    """Describe the active iteration without leaking the previous iteration's submit state."""
    if current_iteration is not None and completed_iteration == current_iteration:
        return _clean_str(completed_marker.get("submit_phase_state"))
    if current_iteration is None or status.strip().lower() not in {"", "running"}:
        return None
    submit_enabled = run_config.get("submit")
    if submit_enabled is True:
        return "pending"
    if submit_enabled is False:
        return "disabled"
    return None


def _iteration_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    if not run_dir.exists():
        return result
    for path in run_dir.glob("iter-*"):
        if not path.is_dir():
            continue
        try:
            iteration = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        result.append((iteration, path))
    result.sort(key=lambda item: item[0])
    return result


def _latest_completed_iteration(iter_dirs: list[tuple[int, Path]]) -> tuple[int | None, dict[str, object]]:
    latest_iteration: int | None = None
    latest_marker: dict[str, object] = {}
    for iteration, path in iter_dirs:
        marker = _read_json_object(path / "iteration_state.json")
        if marker.get("iteration_complete") is True:
            latest_iteration = iteration
            latest_marker = marker
    return latest_iteration, latest_marker


def _latest_metrics(iter_dirs: list[tuple[int, Path]]) -> dict[str, object]:
    for _iteration, path in reversed(iter_dirs):
        metrics = _read_json_object(path / "metrics.json")
        if metrics:
            return metrics
    return {}


def _best_metric_score(
    iter_dirs: list[tuple[int, Path]],
    *,
    direction: str | None,
    preferred_metric: str | None,
) -> float | None:
    best: float | None = None
    normalized_direction = (direction or "maximize").strip().lower()
    for _iteration, path in iter_dirs:
        metrics = _read_json_object(path / "metrics.json")
        if preferred_metric and not _metrics_equivalent(_clean_str(metrics.get("metric")), preferred_metric):
            continue
        value = _metric_score(metrics)
        if value is None:
            continue
        if best is None:
            best = value
            continue
        if normalized_direction == "minimize":
            if value < best:
                best = value
        elif value > best:
            best = value
    return best


def _submission_score_summary(
    *,
    artifacts_dir: Path,
    slug: str,
    run_id: str,
    iter_dirs: list[tuple[int, Path]],
    direction: str | None,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    ledger_path = artifacts_dir / slug / "submissions" / "ledger.jsonl"
    for index, record in enumerate(load_jsonl_records(ledger_path)):
        if record.get("run_id") != run_id:
            continue
        outcome = record.get("outcome")
        if not isinstance(outcome, dict):
            continue
        score = _float_or_none(outcome.get("score"))
        if score is None:
            continue
        candidate = {
            "score": score,
            "iteration": _iteration_from_submission_record(record),
            "timestamp": _submission_timestamp(record, fallback_index=index),
            "source": "submission_public_score",
        }
        candidate.update(_submission_rank_fields(outcome))
        _refresh_cached_rank_estimate(
            candidate,
            artifacts_dir=artifacts_dir,
            slug=slug,
            direction=direction,
        )
        candidates.append(candidate)

    for iteration, path in iter_dirs:
        metrics = _read_json_object(path / "metrics.json")
        score = _float_or_none(metrics.get("submission_score"))
        if score is None:
            continue
        candidate = {
            "score": score,
            "iteration": iteration,
            "timestamp": _float_or_none(metrics.get("timestamp")) or float(iteration),
            "source": "submission_public_score",
        }
        candidate.update(_submission_rank_fields(metrics))
        _refresh_cached_rank_estimate(
            candidate,
            artifacts_dir=artifacts_dir,
            slug=slug,
            direction=direction,
        )
        candidates.append(candidate)

    if not candidates:
        return {}

    latest = max(candidates, key=lambda item: float(item.get("timestamp") or 0.0))
    normalized_direction = (direction or "maximize").strip().lower()
    if normalized_direction == "minimize":
        best = min(
            candidates,
            key=lambda item: (
                float(item["score"]),
                -_submission_rank_availability(item),
                -float(item.get("timestamp") or 0.0),
            ),
        )
    else:
        best = max(
            candidates,
            key=lambda item: (
                float(item["score"]),
                _submission_rank_availability(item),
                float(item.get("timestamp") or 0.0),
            ),
        )
    summary: dict[str, object] = {
        "latest_submission_score": latest["score"],
        "best_submission_score": best["score"],
        "submission_score_source": "submission_public_score",
        "latest_submission_iteration": latest.get("iteration"),
        "best_submission_iteration": best.get("iteration"),
    }
    _add_submission_rank_summary(summary, "latest", latest)
    _add_submission_rank_summary(summary, "best", best)
    return summary


def _refresh_cached_rank_estimate(
    candidate: dict[str, object],
    *,
    artifacts_dir: Path,
    slug: str,
    direction: str | None,
) -> None:
    score = _float_or_none(candidate.get("score"))
    normalized_direction = (direction or "").strip().lower()
    if score is None or normalized_direction not in {"maximize", "minimize"}:
        return
    try:
        estimate = leaderboard_rank_for_score(
            slug,
            artifacts_dir / slug / "context",
            score=score,
            direction=normalized_direction,
            dry_run=True,
        )
    except Exception:  # noqa: BLE001
        return
    estimated_rank = _int_or_none(estimate.get("rank"))
    estimated_total = _int_or_none(estimate.get("total_teams"))
    if estimated_rank is None or estimated_total is None:
        return
    candidate["estimated_rank"] = estimated_rank
    candidate["estimated_total_teams"] = estimated_total
    percentile = _float_or_none(estimate.get("rank_percentile"))
    if percentile is not None:
        candidate["estimated_rank_percentile"] = percentile
    candidate["rank_estimate_source"] = "cached_leaderboard_score_estimate"


def _add_submission_rank_summary(summary: dict[str, object], prefix: str, candidate: dict[str, object]) -> None:
    for key in (
        "rank",
        "total_teams",
        "rank_percentile",
        "rank_source",
        "estimated_rank",
        "estimated_total_teams",
        "estimated_rank_percentile",
        "rank_estimate_source",
    ):
        value = candidate.get(key)
        if value is not None:
            summary[f"{prefix}_submission_{key}"] = value


def _submission_rank_availability(candidate: dict[str, object]) -> int:
    if candidate.get("rank") is not None and candidate.get("total_teams") is not None:
        return 2
    if candidate.get("estimated_rank") is not None and candidate.get("estimated_total_teams") is not None:
        return 1
    return 0


def _submission_rank_fields(payload: dict[str, object]) -> dict[str, object]:
    fields: dict[str, object] = {}
    _put_if_not_none(
        fields,
        "rank",
        _int_from_keys(payload, ("rank", "submission_rank", "public_rank", "leaderboard_rank")),
    )
    _put_if_not_none(
        fields,
        "total_teams",
        _int_from_keys(
            payload, ("total_teams", "submission_total_teams", "public_total_teams", "leaderboard_total_teams")
        ),
    )
    _put_if_not_none(
        fields, "rank_percentile", _float_from_keys(payload, ("rank_percentile", "submission_rank_percentile"))
    )
    _put_if_not_none(fields, "rank_source", _clean_str(payload.get("rank_source")))
    _put_if_not_none(
        fields,
        "estimated_rank",
        _int_from_keys(payload, ("estimated_rank", "submission_estimated_rank", "estimated_submission_rank")),
    )
    _put_if_not_none(
        fields,
        "estimated_total_teams",
        _int_from_keys(
            payload,
            ("estimated_total_teams", "submission_estimated_total_teams", "estimated_submission_total_teams"),
        ),
    )
    _put_if_not_none(
        fields,
        "estimated_rank_percentile",
        _float_from_keys(payload, ("estimated_rank_percentile", "submission_estimated_rank_percentile")),
    )
    _put_if_not_none(fields, "rank_estimate_source", _clean_str(payload.get("rank_estimate_source")))

    raw = payload.get("raw")
    if isinstance(raw, dict):
        fields.setdefault(
            "rank",
            _int_from_keys(raw, ("rank", "publicRank", "public_rank", "leaderboardRank", "leaderboard_rank")),
        )
        fields.setdefault(
            "total_teams",
            _int_from_keys(raw, ("totalTeams", "total_teams", "teamCount", "team_count", "leaderboardTotalTeams")),
        )

    rank = _int_or_none(fields.get("rank"))
    total_teams = _int_or_none(fields.get("total_teams"))
    if rank is not None and total_teams is not None and total_teams > 0:
        fields.setdefault("rank_percentile", rank / total_teams)
    estimated_rank = _int_or_none(fields.get("estimated_rank"))
    estimated_total_teams = _int_or_none(fields.get("estimated_total_teams"))
    if estimated_rank is not None and estimated_total_teams is not None and estimated_total_teams > 0:
        fields.setdefault("estimated_rank_percentile", estimated_rank / estimated_total_teams)

    return {key: value for key, value in fields.items() if value is not None}


def _iteration_from_submission_record(record: dict[str, object]) -> int | None:
    for key in ("iteration", "iter", "current_iteration"):
        value = _int_or_none(record.get(key))
        if value is not None:
            return value
    message = _clean_str(record.get("message")) or ""
    for marker in (" i=", "iter-", "iteration "):
        if marker not in message:
            continue
        tail = message.split(marker, 1)[1]
        digits = []
        for char in tail:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if digits:
            return int("".join(digits))
    path_text = _clean_str(record.get("submission_path")) or ""
    for part in Path(path_text).parts:
        if part.startswith("iter-"):
            return _int_or_none(part.removeprefix("iter-"))
    return None


def _submission_timestamp(record: dict[str, object], *, fallback_index: int) -> float:
    for key in ("ts", "checked_at", "submitted_at", "created_at"):
        parsed = _parse_datetime(record.get(key))
        if parsed is not None:
            return parsed.timestamp()
    outcome = record.get("outcome")
    if isinstance(outcome, dict):
        for key in ("checked_at", "submitted_at", "created_at"):
            parsed = _parse_datetime(outcome.get(key))
            if parsed is not None:
                return parsed.timestamp()
    return float(fallback_index)


def _metric_score(metrics: dict[str, object]) -> float | None:
    for key in ("offline_value", "value", "score", "readiness_score"):
        value = _float_or_none(metrics.get(key))
        if value is not None:
            return value
    readiness = metrics.get("readiness")
    if isinstance(readiness, dict):
        return _float_or_none(readiness.get("score"))
    return None


def _nested_number(payload: dict[str, object], outer: str, inner: str) -> float | None:
    nested = payload.get(outer)
    if not isinstance(nested, dict):
        return None
    return _float_or_none(nested.get(inner))


def _read_json_object(path: Path) -> dict[str, object]:
    return load_json_object_or_empty(path)


def _parse_datetime(value: object) -> datetime | None:
    return parse_iso_datetime_utc(value)


def _put_if_not_none(payload: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        payload[key] = value


def _clean_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _response_text(payload: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _int_from_keys(payload: dict[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        parsed = _int_or_none(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _float_from_keys(payload: dict[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        parsed = _float_or_none(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def _float_or_default(value: object, default: float) -> float:
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None
