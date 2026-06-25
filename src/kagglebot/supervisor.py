from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO
from urllib.parse import unquote

from rich import print

from kagglebot.autopilot import AutopilotConfig, run_autopilot
from kagglebot.bootstrap import bootstrap_competition
from kagglebot.datetime_utils import parse_iso_datetime_utc
from kagglebot.env_utils import parse_float_value, parse_int_value
from kagglebot.eval import EvaluationAdvisor
from kagglebot.exceptions import (
    KaggleCliResourceError,
    KernelCapacityError,
    RulesNotAcceptedError,
    SubmitAbortedError,
)
from kagglebot.history import new_run_id
from kagglebot.json_utils import append_jsonl_record, load_json_object, load_jsonl_records, write_json_object
from kagglebot.kaggle_api import (
    EnteredCompetition,
    competition_total_size_bytes,
    leaderboard_rank_for_score,
    list_entered_competitions,
)
from kagglebot.paths import CompetitionPaths, KnowledgePaths
from kagglebot.self_improvement import SelfImprovementConfig, run_self_improvement_cycle
from kagglebot.solver.metrics import infer_direction

_TERMINAL_RUN_STATUSES = {
    "completed",
    "submitted",
    "stopped",
    "missing_target",
    "manual_finalization_required",
}

_REWARD_AMOUNT_RE = re.compile(
    r"(?:\$\s*(?P<dollar>[0-9][0-9,]*(?:\.[0-9]+)?)|(?P<usd>[0-9][0-9,]*(?:\.[0-9]+)?)\s*usd)",
    flags=re.IGNORECASE,
)
_DEFAULT_KAGGLE_GPU_QUOTA_FILE_MAX_AGE_HOURS = 24.0
_DEFAULT_RESOURCE_BLOCK_TTL_HOURS = 168.0
_DEFAULT_ACTIVE_RUN_STALE_HOURS = 24.0


@dataclass(frozen=True)
class SubmissionHistory:
    autopilot_started: bool
    submitted: bool
    rank_percentile: float | None
    rank: int | None
    total_teams: int | None
    outcome_score: float | None


@dataclass(frozen=True)
class WatchConfig:
    workdir: Path
    artifacts_dir: Path
    compute: str
    accelerator: str
    strict_accelerator: bool
    kaggle_username: str | None
    kernel_name: str | None
    internet: str | None
    time_budget_min: int | None
    seed: int | None
    score_source: str | None
    holdout_frac: float | None
    cv_folds: int | None
    max_iterations: int
    max_total_min: int | None
    patience: int | None
    min_improvement: float | None
    submit_policy: str
    verify_cmd: str
    auto_eval_spec: bool
    page_limit: int
    allow_slugs: tuple[str, ...]
    block_slugs: tuple[str, ...]
    cooldown_hours: float
    dry_run: bool
    force: bool
    state_scope: str = ""
    lightweight_only: bool = False
    lightweight_max_data_bytes: int | None = None
    lightweight_max_training_min: int | None = 120
    lightweight_preferred_categories: tuple[str, ...] = ("gettingstarted", "playground")
    kaggle_gpu_min_available_minutes_for_new_competition: int | None = None
    kaggle_gpu_quota_web_lookup: bool = False
    self_improvement_interval_hours: float | None = 6.0
    self_improvement_codex: bool = True
    self_improvement_publish: bool = False
    top1_exhaustive: bool = True
    top1_submit_policy: str = "value_only"
    hardware_profile: str | None = "auto"

    @property
    def root_watch_dir(self) -> Path:
        return self.artifacts_dir / "_watch"

    @property
    def watch_dir(self) -> Path:
        scope = _safe_state_scope(self.state_scope)
        return self.root_watch_dir / scope if scope else self.root_watch_dir

    @property
    def ledger_path(self) -> Path:
        return self.watch_dir / "ledger.jsonl"

    @property
    def state_path(self) -> Path:
        return self.watch_dir / "state.json"


@dataclass(frozen=True)
class WatchCycleResult:
    status: str
    slug: str | None
    run_id: str | None
    reason: str | None = None


@dataclass(frozen=True)
class KaggleGpuQuotaStatus:
    available_minutes: int | None
    total_minutes: int | None = None
    used_minutes: int | None = None
    reserved_minutes: int | None = None
    refresh_time: str | None = None
    source: str = "unknown"


class WatchLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: str, **payload: object) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            **payload,
        }
        append_jsonl_record(self.path, record, sort_keys=True)

    def records(self) -> list[dict[str, object]]:
        return load_jsonl_records(self.path)


def _try_acquire_watch_resource_lock(config: WatchConfig, ledger: WatchLedger) -> TextIO | None:
    lock_path = _watch_resource_lock_path(config)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        if exc.errno not in {errno.EACCES, errno.EAGAIN} and not isinstance(exc, BlockingIOError):
            raise
        ledger.append(
            "locked",
            reason="watch_resource_locked",
            compute=config.compute,
            hardware_profile=config.hardware_profile,
            lock_path=str(lock_path),
        )
        return None

    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "compute": config.compute,
                "hardware_profile": config.hardware_profile,
                "state_scope": config.state_scope,
                "acquired_at": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    return handle


def _release_watch_resource_lock(handle: TextIO) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _watch_resource_lock_path(config: WatchConfig) -> Path:
    lock_name = _watch_resource_lock_name(config)
    return config.root_watch_dir / "locks" / f"{lock_name}.lock"


def _watch_resource_lock_name(config: WatchConfig) -> str:
    compute = _safe_state_scope(config.compute or "default").lower() or "default"
    if compute == "local_gpu":
        return compute
    scope = _safe_state_scope(config.state_scope).lower()
    if scope:
        return f"{compute}-{scope}"
    profile = _safe_state_scope(str(config.hardware_profile or "")).lower()
    if profile and profile != "auto":
        return f"{compute}-{profile}"
    return compute


def run_watch_forever(
    config: WatchConfig,
    *,
    sleep_empty_sec: int,
    sleep_error_sec: int,
) -> None:
    while True:
        try:
            result = run_watch_once(config)
        except Exception as exc:  # noqa: BLE001
            WatchLedger(config.ledger_path).append("failed", reason=type(exc).__name__, error=str(exc))
            print(f"[red]watch failed[/red]: {exc}")
            _maybe_run_self_improvement(config)
            time.sleep(max(1, sleep_error_sec))
            continue
        _maybe_run_self_improvement(config)
        if result.status in {"no_candidates", "dry_run", "no_capacity", "locked"}:
            time.sleep(max(1, sleep_empty_sec))
            continue
        if result.status in {"failed", "skipped"}:
            time.sleep(max(1, sleep_error_sec))


def run_watch_once(config: WatchConfig) -> WatchCycleResult:
    ledger = WatchLedger(config.ledger_path)
    lock_handle = _try_acquire_watch_resource_lock(config, ledger)
    if lock_handle is None:
        reason = "watch_resource_locked"
        print(f"[yellow]watch[/yellow]: {config.compute} resource is already in use; skipping this cycle")
        return WatchCycleResult(status="locked", slug=None, run_id=None, reason=reason)
    try:
        return _run_watch_once_unlocked(config, ledger)
    finally:
        _release_watch_resource_lock(lock_handle)


def _run_watch_once_unlocked(config: WatchConfig, ledger: WatchLedger) -> WatchCycleResult:
    state = _load_state(config.state_path)
    active_slug = str(state.get("active_slug") or "").strip()
    active_run_id = str(state.get("active_run_id") or "").strip()

    if active_slug and active_run_id and _run_can_resume(config, active_slug, active_run_id, state=state):
        candidate = _candidate_from_slug(active_slug)
        run_id = active_run_id
        resume = True
        ledger.append("selected", slug=active_slug, run_id=run_id, reason="resume_active")
    else:
        if active_slug and active_run_id and _active_state_is_stale(state):
            ledger.append("stale_active_cleared", slug=active_slug, run_id=active_run_id, reason="stale_watch_state")
            _write_state(
                config.state_path,
                {
                    "active_slug": None,
                    "active_run_id": None,
                    "last_status": "failed",
                    "phase": "stale_active_cleared",
                    "stale_slug": active_slug,
                    "stale_run_id": active_run_id,
                },
            )
        quota_block = _new_kaggle_gpu_competition_quota_block(config=config, ledger=ledger)
        if quota_block is not None:
            state = {
                "active_slug": None,
                "active_run_id": None,
                "last_status": "no_capacity",
                "phase": quota_block["phase"],
                "reason": quota_block["reason"],
                "updated_at": datetime.now(UTC).isoformat(),
            }
            state.update({key: value for key, value in quota_block.items() if value is not None})
            _write_state(config.state_path, state)
            print(f"[yellow]watch[/yellow]: {quota_block['message']}")
            return WatchCycleResult(status="no_capacity", slug=None, run_id=None, reason=str(quota_block["reason"]))

        candidates = select_next_competition(config, ledger=ledger)
        if not candidates:
            ledger.append("skipped", reason="no_candidates")
            _write_state(
                config.state_path,
                {"active_slug": None, "active_run_id": None, "last_status": "no_candidates"},
            )
            print("[yellow]watch[/yellow]: no entered competitions are eligible right now")
            return WatchCycleResult(status="no_candidates", slug=None, run_id=None, reason="no_candidates")
        candidate = candidates[0]
        run_id = new_run_id()
        resume = False
        ledger.append("selected", slug=candidate.slug, run_id=run_id, reason="best_candidate")

    if config.dry_run:
        ledger.append("dry_run", slug=candidate.slug, run_id=run_id)
        print(f"[yellow]DRY RUN[/yellow]: would run autopilot for {candidate.slug} ({run_id})")
        return WatchCycleResult(status="dry_run", slug=candidate.slug, run_id=run_id)

    _write_state(
        config.state_path,
        {
            "active_slug": candidate.slug,
            "active_run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "last_status": "running",
            "phase": "bootstrapping",
        },
    )
    ledger.append("started", slug=candidate.slug, run_id=run_id, resume=resume)

    paths = CompetitionPaths(slug=candidate.slug, artifacts_dir=config.artifacts_dir)
    knowledge_paths = KnowledgePaths(workdir=config.workdir)
    previous_state_path = os.environ.get("KAGGLEBOT_WATCH_STATE_PATH")
    os.environ["KAGGLEBOT_WATCH_STATE_PATH"] = str(config.state_path)
    try:
        _prepare_competition(config=config, candidate=candidate, paths=paths, knowledge_paths=knowledge_paths)
        _write_state(
            config.state_path,
            {
                "active_slug": candidate.slug,
                "active_run_id": run_id,
                "started_at": datetime.now(UTC).isoformat(),
                "last_status": "running",
                "phase": "autopilot_starting",
            },
        )
        autopilot_config = _build_autopilot_config(
            config=config,
            candidate=candidate,
            paths=paths,
            knowledge_paths=knowledge_paths,
            run_id=None if resume else run_id,
        )
        if resume:
            _set_resume_env(slug=candidate.slug, run_id=run_id)
        run_autopilot(autopilot_config)
    except RulesNotAcceptedError as exc:
        ledger.append("skipped", slug=candidate.slug, run_id=run_id, reason="rules_not_accepted")
        _write_state(config.state_path, {"active_slug": None, "active_run_id": None, "last_status": "skipped"})
        return WatchCycleResult(status="skipped", slug=candidate.slug, run_id=run_id, reason=str(exc))
    except SubmitAbortedError as exc:
        ledger.append("failed", slug=candidate.slug, run_id=run_id, reason="submit_aborted", error=str(exc))
        _write_state(config.state_path, {"active_slug": None, "active_run_id": None, "last_status": "failed"})
        return WatchCycleResult(status="failed", slug=candidate.slug, run_id=run_id, reason=str(exc))
    except KernelCapacityError as exc:
        ledger.append("no_capacity", slug=candidate.slug, run_id=run_id, reason="kaggle_gpu_capacity", error=str(exc))
        _write_state(config.state_path, {"active_slug": None, "active_run_id": None, "last_status": "no_capacity"})
        print("[yellow]watch[/yellow]: kaggle_gpu capacity unavailable; leaving local_gpu as the only active runner")
        return WatchCycleResult(status="no_capacity", slug=candidate.slug, run_id=run_id, reason=str(exc))
    except KaggleCliResourceError as exc:
        reason = "kaggle_cli_resource_limit"
        ledger.append("resource_blocked", slug=candidate.slug, run_id=run_id, reason=reason, error=str(exc))
        _write_state(
            config.state_path,
            {
                "active_slug": None,
                "active_run_id": None,
                "last_status": "skipped",
                "phase": "resource_blocked",
                "reason": reason,
            },
        )
        print(f"[yellow]watch[/yellow]: resource guard blocked {candidate.slug}; skipping retry for now")
        return WatchCycleResult(status="skipped", slug=candidate.slug, run_id=run_id, reason=reason)
    except Exception as exc:  # noqa: BLE001
        ledger.append("failed", slug=candidate.slug, run_id=run_id, reason=type(exc).__name__, error=str(exc))
        _write_state(config.state_path, {"active_slug": None, "active_run_id": None, "last_status": "failed"})
        return WatchCycleResult(status="failed", slug=candidate.slug, run_id=run_id, reason=str(exc))
    finally:
        if previous_state_path is None:
            os.environ.pop("KAGGLEBOT_WATCH_STATE_PATH", None)
        else:
            os.environ["KAGGLEBOT_WATCH_STATE_PATH"] = previous_state_path

    ledger.append("finished", slug=candidate.slug, run_id=run_id)
    _write_state(config.state_path, {"active_slug": None, "active_run_id": None, "last_status": "finished"})
    return WatchCycleResult(status="finished", slug=candidate.slug, run_id=run_id)


def run_watch_self_improvement(config: WatchConfig, *, force: bool = False) -> dict[str, object]:
    return _maybe_run_self_improvement(config, force=force) or {"status": "disabled"}


def _maybe_run_self_improvement(config: WatchConfig, *, force: bool = False) -> dict[str, object] | None:
    interval = config.self_improvement_interval_hours
    if not force and (interval is None or interval <= 0):
        return None
    try:
        result = run_self_improvement_cycle(
            SelfImprovementConfig(
                artifacts_dir=config.artifacts_dir,
                knowledge_paths=KnowledgePaths(workdir=config.workdir),
                min_interval_hours=interval,
                invoke_codex=config.self_improvement_codex,
                publish_codex_changes=config.self_improvement_publish,
                force=force,
                dry_run=config.dry_run,
            )
        )
    except Exception as exc:  # noqa: BLE001
        result = {"status": "failed", "error": str(exc), "error_type": type(exc).__name__}
    if result.get("status") == "written":
        print(f"[cyan]self-improvement[/cyan]: {result.get('report_path')}")
    elif result.get("status") == "failed":
        print(f"[yellow]self-improvement failed[/yellow]: {result.get('error')}")
    return result


def _new_kaggle_gpu_competition_quota_block(
    *,
    config: WatchConfig,
    ledger: WatchLedger,
) -> dict[str, object] | None:
    threshold = config.kaggle_gpu_min_available_minutes_for_new_competition
    if config.compute != "kaggle_gpu" or threshold is None or threshold <= 0:
        return None

    quota = _resolve_kaggle_gpu_quota_status(config)
    threshold_text = _format_minutes(threshold)
    if quota is None or quota.available_minutes is None:
        ledger.append(
            "new_competition_blocked",
            reason="kaggle_gpu_quota_unavailable",
            threshold_minutes=threshold,
        )
        return {
            "phase": "kaggle_gpu_quota_unavailable",
            "reason": "kaggle_gpu_quota_unavailable",
            "threshold_minutes": threshold,
            "message": (
                "Kaggle GPU quota is unavailable; not starting a new competition below safety policy "
                f"({threshold_text} required)."
            ),
        }

    if quota.available_minutes < threshold:
        available_text = _format_minutes(quota.available_minutes)
        ledger.append(
            "new_competition_blocked",
            reason="kaggle_gpu_quota_low",
            available_minutes=quota.available_minutes,
            total_minutes=quota.total_minutes,
            used_minutes=quota.used_minutes,
            reserved_minutes=quota.reserved_minutes,
            threshold_minutes=threshold,
            source=quota.source,
        )
        return {
            "phase": "kaggle_gpu_quota_low",
            "reason": "kaggle_gpu_quota_low",
            "available_minutes": quota.available_minutes,
            "total_minutes": quota.total_minutes,
            "used_minutes": quota.used_minutes,
            "reserved_minutes": quota.reserved_minutes,
            "threshold_minutes": threshold,
            "quota_source": quota.source,
            "quota_refresh_time": quota.refresh_time,
            "message": (
                f"Kaggle GPU quota is low ({available_text} available; {threshold_text} required); "
                "not starting a new competition."
            ),
        }
    return None


def _resolve_kaggle_gpu_quota_status(config: WatchConfig) -> KaggleGpuQuotaStatus | None:
    explicit_minutes = _env_int("KAGGLEBOT_KAGGLE_GPU_AVAILABLE_MINUTES")
    if explicit_minutes is not None:
        return KaggleGpuQuotaStatus(
            available_minutes=explicit_minutes,
            total_minutes=_env_int("KAGGLEBOT_KAGGLE_GPU_TOTAL_MINUTES"),
            source="env:KAGGLEBOT_KAGGLE_GPU_AVAILABLE_MINUTES",
        )

    for name in ("KAGGLEBOT_KAGGLE_GPU_QUOTA_TEXT", "KAGGLE_GPU_QUOTA_TEXT"):
        quota = _parse_kaggle_gpu_quota_text(os.environ.get(name), source=f"env:{name}")
        if quota is not None:
            return quota

    if config.kaggle_gpu_quota_web_lookup:
        quota = _fetch_kaggle_gpu_quota_from_web_cookie()
        if quota is not None:
            return quota

    for path in _kaggle_gpu_quota_file_candidates(config):
        quota = _read_kaggle_gpu_quota_file(path)
        if quota is not None:
            return quota
    return None


def _kaggle_gpu_quota_file_candidates(config: WatchConfig) -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("KAGGLEBOT_KAGGLE_GPU_QUOTA_FILE")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(config.watch_dir / "quota.json")
    candidates.append(config.root_watch_dir / "kaggle_gpu_quota.json")
    return list(dict.fromkeys(candidates))


def _read_kaggle_gpu_quota_file(path: Path) -> KaggleGpuQuotaStatus | None:
    payload = load_json_object(path)
    if payload is None:
        return None
    expires_at = _parse_ts(payload.get("expires_at"))
    if expires_at is not None and expires_at <= datetime.now(UTC):
        return None
    if expires_at is None and _kaggle_gpu_quota_file_is_stale(path=path, payload=payload):
        return None
    source = f"file:{path}"
    quota = _parse_kaggle_gpu_quota_text(payload.get("text") or payload.get("quota_text"), source=source)
    if quota is not None:
        return quota
    available_minutes = _coerce_minutes(payload.get("available_minutes"))
    total_minutes = _coerce_minutes(payload.get("total_minutes"))
    used_minutes = _coerce_minutes(payload.get("used_minutes"))
    reserved_minutes = _coerce_minutes(payload.get("reserved_minutes"))
    if available_minutes is None and total_minutes is not None and used_minutes is not None:
        available_minutes = max(0, total_minutes - used_minutes)
    if available_minutes is None:
        return None
    return KaggleGpuQuotaStatus(
        available_minutes=available_minutes,
        total_minutes=total_minutes,
        used_minutes=used_minutes,
        reserved_minutes=reserved_minutes,
        refresh_time=str(payload.get("refresh_time") or payload.get("quota_refresh_time") or "") or None,
        source=source,
    )


def _kaggle_gpu_quota_file_is_stale(*, path: Path, payload: dict[str, object]) -> bool:
    max_age_hours = _kaggle_gpu_quota_file_max_age_hours()
    if max_age_hours <= 0:
        return False
    cache_ts = _kaggle_gpu_quota_file_timestamp(path=path, payload=payload)
    if cache_ts is None:
        return True
    return cache_ts + timedelta(hours=max_age_hours) <= datetime.now(UTC)


def _kaggle_gpu_quota_file_timestamp(*, path: Path, payload: dict[str, object]) -> datetime | None:
    for key in ("updated_at", "refresh_time", "quota_refresh_time"):
        parsed = _parse_ts(payload.get(key))
        if parsed is not None:
            return parsed
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None


def _kaggle_gpu_quota_file_max_age_hours() -> float:
    value = parse_float_value(os.environ.get("KAGGLEBOT_KAGGLE_GPU_QUOTA_FILE_MAX_AGE_HOURS"))
    return _DEFAULT_KAGGLE_GPU_QUOTA_FILE_MAX_AGE_HOURS if value is None else value


def _fetch_kaggle_gpu_quota_from_web_cookie() -> KaggleGpuQuotaStatus | None:
    cookie = _read_env_or_file("KAGGLEBOT_KAGGLE_WEB_COOKIE", "KAGGLEBOT_KAGGLE_WEB_COOKIE_FILE")
    if not cookie:
        cookie = _read_env_or_file("KAGGLE_WEB_COOKIE", "KAGGLE_WEB_COOKIE_FILE")
    if not cookie:
        return None
    try:
        import requests
    except ImportError:
        return None
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Cookie": cookie,
        "Origin": "https://www.kaggle.com",
        "Referer": "https://www.kaggle.com/settings",
        "User-Agent": "Mozilla/5.0 kagglebot-quota-check",
    }
    xsrf_token = _cookie_value(cookie, "XSRF-TOKEN") or _cookie_value(cookie, "CSRF-TOKEN")
    if xsrf_token:
        headers["X-XSRF-TOKEN"] = unquote(xsrf_token)
    try:
        response = requests.post(
            "https://www.kaggle.com/api/i/kernels.KernelsService/GetAcceleratorQuotaStatistics",
            headers=headers,
            json={},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return _quota_status_from_web_payload(payload, source="kaggle_web:GetAcceleratorQuotaStatistics")


def _quota_status_from_web_payload(payload: dict[str, object], *, source: str) -> KaggleGpuQuotaStatus | None:
    raw_gpu = payload.get("gpuQuota")
    if not isinstance(raw_gpu, dict):
        return None
    total = _duration_to_minutes(raw_gpu.get("totalTimeAllowed"))
    used = _duration_to_minutes(raw_gpu.get("timeUsed"))
    reserved = _duration_to_minutes(raw_gpu.get("timeReserved"))
    if total is None or used is None:
        return None
    return KaggleGpuQuotaStatus(
        available_minutes=max(0, total - used),
        total_minutes=total,
        used_minutes=used,
        reserved_minutes=reserved,
        refresh_time=str(payload.get("quotaRefreshTime") or "") or None,
        source=source,
    )


def _parse_kaggle_gpu_quota_text(value: object, *, source: str = "text") -> KaggleGpuQuotaStatus | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(
        r"(?P<available>\d+\s*h(?:ours?)?(?:\s+\d+\s*m(?:in(?:utes?)?)?)?|\d+\s*m(?:in(?:utes?)?)?)\s+available\s+of\s+(?P<total>\d+\s*h(?:ours?)?(?:\s+\d+\s*m(?:in(?:utes?)?)?)?|\d+\s*m(?:in(?:utes?)?)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    available = _parse_hours_minutes_text(match.group("available"))
    total = _parse_hours_minutes_text(match.group("total"))
    if available is None:
        return None
    return KaggleGpuQuotaStatus(
        available_minutes=available,
        total_minutes=total,
        used_minutes=max(0, total - available) if total is not None else None,
        source=source,
    )


def _parse_hours_minutes_text(value: str) -> int | None:
    text = value.strip().lower()
    hours_match = re.search(r"(\d+)\s*h", text)
    minutes_match = re.search(r"(\d+)\s*m", text)
    if not hours_match and not minutes_match:
        return None
    hours = int(hours_match.group(1)) if hours_match else 0
    minutes = int(minutes_match.group(1)) if minutes_match else 0
    return hours * 60 + minutes


def _duration_to_minutes(value: object) -> int | None:
    seconds = _duration_to_seconds(value)
    if seconds is None:
        return None
    return max(0, int(seconds // 60))


def _duration_to_seconds(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if "seconds" in value:
            try:
                return float(value.get("seconds") or 0) + float(value.get("nanos") or 0) / 1_000_000_000
            except (TypeError, ValueError):
                return None
        days = _to_float(value.get("days")) or 0.0
        hours = _to_float(value.get("hours")) or 0.0
        minutes = _to_float(value.get("minutes")) or 0.0
        seconds = _to_float(value.get("seconds")) or 0.0
        if days or hours or minutes or seconds:
            return days * 86400 + hours * 3600 + minutes * 60 + seconds
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("s"):
        try:
            return float(text[:-1])
        except ValueError:
            return None
    iso_match = re.fullmatch(
        r"P(?:(?P<days>\d+(?:\.\d+)?)D)?(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?",
        text,
        flags=re.IGNORECASE,
    )
    if iso_match:
        days = _to_float(iso_match.group("days")) or 0.0
        hours = _to_float(iso_match.group("hours")) or 0.0
        minutes = _to_float(iso_match.group("minutes")) or 0.0
        seconds = _to_float(iso_match.group("seconds")) or 0.0
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    hm = _parse_hours_minutes_text(text)
    if hm is not None:
        return hm * 60
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_minutes(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _env_int(name: str) -> int | None:
    return parse_int_value(os.environ.get(name), allow_float=True)


def _read_env_or_file(env_name: str, file_env_name: str) -> str | None:
    direct = os.environ.get(env_name)
    if direct and direct.strip():
        return direct.strip()
    file_value = os.environ.get(file_env_name)
    if not file_value:
        return None
    try:
        return Path(file_value).expanduser().read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _cookie_value(cookie: str, name: str) -> str | None:
    prefix = f"{name}="
    for part in cookie.split(";"):
        item = part.strip()
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def _format_minutes(value: int) -> str:
    hours, minutes = divmod(max(0, int(value)), 60)
    if minutes:
        return f"{hours}h {minutes}m"
    return f"{hours}h"


def _to_float(value: object) -> float | None:
    return parse_float_value(value)


def select_next_competition(config: WatchConfig, *, ledger: WatchLedger | None = None) -> list[EnteredCompetition]:
    ledger = ledger or WatchLedger(config.ledger_path)
    allow = {slug.strip().lower() for slug in config.allow_slugs if slug.strip()}
    block = {slug.strip().lower() for slug in config.block_slugs if slug.strip()}
    records = _all_watch_records(config, primary=ledger)
    cooldown = _cooldown_slugs(records, hours=config.cooldown_hours)
    resource_blocked = _resource_blocked_slugs(records)
    active = _active_slugs(config)
    candidates = list_entered_competitions(page_limit=config.page_limit, dry_run=config.dry_run)
    filtered: list[tuple[bool, bool, float, int, EnteredCompetition]] = []
    for index, candidate in enumerate(candidates):
        slug = candidate.slug.lower()
        if slug in active:
            continue
        if allow and slug not in allow:
            continue
        if slug in block or slug in cooldown:
            continue
        if slug in resource_blocked:
            ledger.append("candidate_skipped", slug=candidate.slug, reason="resource_blocked")
            continue
        if candidate.submissions_disabled:
            continue
        eligibility = _candidate_eligibility(config=config, candidate=candidate)
        if eligibility is not None:
            ledger.append("candidate_skipped", slug=candidate.slug, reason=eligibility)
            continue
        data_size_bytes = _candidate_data_size_bytes(config=config, candidate=candidate)
        if config.lightweight_only and not _is_lightweight_candidate(
            config=config,
            candidate=candidate,
        ):
            estimated_training_min = _estimate_training_minutes(candidate)
            ledger.append(
                "candidate_skipped",
                slug=candidate.slug,
                reason="not_lightweight",
                estimated_training_min=estimated_training_min,
                data_size_bytes=data_size_bytes,
            )
            continue
        history = _load_submission_history(config=config, slug=candidate.slug)
        history = _enrich_submission_history_from_leaderboard(config=config, slug=candidate.slug, history=history)
        ledger.append(
            "candidate_seen",
            slug=candidate.slug,
            title=candidate.title,
            category=candidate.category,
            reward=candidate.reward,
            deadline=_aware_datetime(candidate.deadline).isoformat() if _aware_datetime(candidate.deadline) else None,
            medal_candidate=_is_medal_candidate(candidate),
            autopilot_started=history.autopilot_started,
            submitted=history.submitted,
            rank_percentile=history.rank_percentile,
            rank=history.rank,
            total_teams=history.total_teams,
            estimated_training_min=_estimate_training_minutes(candidate),
            data_size_bytes=data_size_bytes,
        )
        score = (
            _lightweight_candidate_score(candidate, history=history)
            if config.lightweight_only
            else _candidate_score(candidate, history=history)
        )
        filtered.append((history.autopilot_started, history.submitted, score, index, candidate))
    filtered.sort(key=lambda item: (item[0], item[1], -item[2], item[3], item[4].slug))
    return [candidate for _started, _submitted, _score, _index, candidate in filtered]


def _prepare_competition(
    *,
    config: WatchConfig,
    candidate: EnteredCompetition,
    paths: CompetitionPaths,
    knowledge_paths: KnowledgePaths,
) -> None:
    print(f"[cyan]watch[/cyan]: bootstrapping {candidate.slug}")
    bootstrap_competition(
        slug=candidate.slug,
        competition_url=candidate.url,
        paths=paths,
        knowledge_paths=knowledge_paths,
        rules_source="url",
        download=True,
        force=False,
        dry_run=config.dry_run,
    )
    if config.auto_eval_spec:
        advisor = EvaluationAdvisor(paths=paths, slug=candidate.slug, dry_run=config.dry_run, force=config.force)
        advisor.ensure_spec()


def _build_autopilot_config(
    *,
    config: WatchConfig,
    candidate: EnteredCompetition,
    paths: CompetitionPaths,
    knowledge_paths: KnowledgePaths,
    run_id: str | None,
) -> AutopilotConfig:
    max_iterations = _watch_max_iterations_for_candidate(config=config, paths=paths)
    return AutopilotConfig(
        run_id=run_id,
        slug=candidate.slug,
        competition_url=candidate.url,
        paths=paths,
        knowledge_paths=knowledge_paths,
        agent="codex",
        compute=config.compute,
        accelerator=config.accelerator,
        strict_accelerator=config.strict_accelerator,
        kaggle_username=config.kaggle_username,
        kernel_name=config.kernel_name,
        internet=config.internet,
        time_budget_min=config.time_budget_min,
        seed=config.seed,
        score_source=config.score_source,
        holdout_frac=config.holdout_frac,
        cv_folds=config.cv_folds,
        target_metric=None,
        target_score=None,
        target_direction=None,
        max_iterations=max_iterations,
        max_total_min=config.max_total_min,
        patience=config.patience,
        min_improvement=config.min_improvement,
        submit=config.submit_policy != "none",
        force_submit=False,
        message=None,
        verify_cmd=config.verify_cmd,
        dry_run=config.dry_run,
        submit_policy=config.submit_policy,
        campaign_mode="top1",
        method_scout="refresh" if config.top1_exhaustive else "auto",
        research_scout="refresh" if config.top1_exhaustive else "auto",
        method_scout_max_sources=12,
        portfolio_execution="budgeted" if config.top1_exhaustive else "serial",
        validation_lab="force" if config.top1_exhaustive else "auto",
        candidate_budget_min=config.time_budget_min if config.top1_exhaustive else None,
        max_candidates_per_iteration=3 if config.top1_exhaustive else None,
        top1_exhaustive=config.top1_exhaustive,
        top1_submit_policy=config.top1_submit_policy,
        hardware_profile=config.hardware_profile,
    )


def _watch_max_iterations_for_candidate(*, config: WatchConfig, paths: CompetitionPaths) -> int:
    plan_max = _plan_max_iterations(paths.plan_path)
    if plan_max is None:
        return config.max_iterations
    return min(config.max_iterations, plan_max)


def _plan_max_iterations(plan_path: Path) -> int | None:
    payload = load_json_object(plan_path)
    if payload is None:
        return None
    raw = payload.get("max_iterations")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        parsed = int(raw)
    elif isinstance(raw, str):
        try:
            parsed = int(raw.strip())
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed > 0 else None


def _candidate_score(candidate: EnteredCompetition, *, history: SubmissionHistory | None = None) -> float:
    category = candidate.category.strip().lower()
    score = 0.0
    if _is_medal_candidate(candidate):
        score += 3000.0
    score += _reward_priority_score(candidate.reward)
    score += _deadline_priority_score(candidate)
    if history is not None and not history.submitted:
        score += 500.0
    elif history is not None:
        score += 100.0
        if history.rank_percentile is not None:
            score += max(0.0, min(1.0, history.rank_percentile)) * 400.0
        elif history.rank is not None and history.total_teams is not None and history.total_teams > 0:
            score += max(0.0, min(1.0, history.rank / history.total_teams)) * 400.0
    if category in {"featured", "research", "community"}:
        score += 120.0
    elif category == "playground":
        score += 40.0
    elif category == "gettingstarted":
        score -= 150.0
    if candidate.is_kernels_submissions_only:
        score -= 25.0
    if candidate.team_count is not None:
        score += min(candidate.team_count, 5000) / 1000.0
    if candidate.max_daily_submissions is not None:
        score += min(candidate.max_daily_submissions, 5) * 0.2
    return score


def _lightweight_candidate_score(
    candidate: EnteredCompetition,
    *,
    history: SubmissionHistory | None = None,
) -> float:
    score = 0.0
    if _is_medal_candidate(candidate):
        score += 2500.0
    score += _reward_priority_score(candidate.reward)
    score += _deadline_priority_score(candidate)
    if history is not None and not history.submitted:
        score += 250.0
    elif history is not None:
        score += 50.0
        if history.rank_percentile is not None:
            score += max(0.0, min(1.0, history.rank_percentile)) * 250.0
    category = candidate.category.strip().lower()
    if category in {"featured", "research", "community"}:
        score += 150.0
    elif category == "playground":
        score += 25.0
    elif category == "gettingstarted":
        score -= 150.0
    estimated_training_min = _estimate_training_minutes(candidate)
    if estimated_training_min is not None:
        score += max(0.0, 250.0 - estimated_training_min)
    if candidate.team_count is not None:
        score += min(candidate.team_count, 5000) / 100.0
    if candidate.is_kernels_submissions_only:
        score -= 25.0
    return score


def _is_lightweight_candidate(
    *,
    config: WatchConfig,
    candidate: EnteredCompetition,
) -> bool:
    estimated_training_min = _estimate_training_minutes(candidate)
    if estimated_training_min is None:
        return False
    if config.lightweight_max_training_min is not None and estimated_training_min > config.lightweight_max_training_min:
        return False
    return not candidate.submissions_disabled


def _candidate_eligibility(*, config: WatchConfig, candidate: EnteredCompetition) -> str | None:
    now = datetime.now(UTC)
    deadline = _aware_datetime(candidate.deadline)
    if deadline is not None and deadline <= now:
        return "late_submit"
    # `watch` only consumes Kaggle's group=entered list. A passed new-entrant
    # deadline blocks new teams from joining; it does not make an already-entered
    # team ineligible to train or submit before the competition deadline.
    # Do not hard-skip unfamiliar task families here. Autopilot's planner can
    # inspect rules/data and either build a kernel/writeup path or fail with a
    # concrete actionable reason; watch selection should only enforce hard
    # submission eligibility.
    return None


def _candidate_data_size_bytes(*, config: WatchConfig, candidate: EnteredCompetition) -> int | None:
    if not config.lightweight_only or config.lightweight_max_data_bytes is None:
        return None
    try:
        return competition_total_size_bytes(candidate.slug, dry_run=config.dry_run)
    except Exception as exc:  # noqa: BLE001
        WatchLedger(config.ledger_path).append(
            "candidate_size_unavailable",
            slug=candidate.slug,
            reason=type(exc).__name__,
            error=str(exc),
        )
        return None


def _estimate_training_minutes(candidate: EnteredCompetition) -> int | None:
    text = " ".join(
        [
            candidate.slug,
            candidate.title,
            candidate.category,
            candidate.evaluation_metric,
        ]
    ).lower()
    metric = candidate.evaluation_metric.lower()
    category = candidate.category.strip().lower()

    complex_training = any(
        marker in text for marker in ("reasoning", "simulation", "optimization", "orbit", "wars", "golf")
    )

    minutes = 360 if complex_training else 90
    if not complex_training and any(
        marker in metric for marker in ("auc", "accuracy", "log loss", "rmse", "rmsle", "mse", "mae", "f1")
    ):
        minutes = 45
    if not complex_training and any(
        marker in text for marker in ("tabular", "stock", "returns", "classification", "regression", "prediction")
    ):
        minutes = min(minutes, 60)
    if any(marker in text for marker in ("image", "detection", "segmentation", "ultrasound", "mri", "xray")):
        minutes = max(minutes, 180)
    if any(marker in metric for marker in ("map", "iou", "dice")):
        minutes = max(minutes, 180)
    if any(marker in text for marker in ("rna", "structure", "translation", "nlp", "text")):
        minutes = max(minutes, 240)
    if not complex_training and category == "gettingstarted":
        minutes = min(minutes, 45)
    if not complex_training and category == "playground":
        minutes = min(minutes, 75)
    if candidate.is_kernels_submissions_only:
        minutes += 30
    if candidate.team_count is not None and candidate.team_count >= 2000:
        minutes += 30
    return minutes


def _reward_priority_score(reward: str) -> float:
    amount = _reward_amount_usd(reward)
    if amount is None or amount <= 0:
        return 0.0
    return 900.0 + min(amount, 250_000.0) / 250.0


def _deadline_priority_score(candidate: EnteredCompetition) -> float:
    deadline = _aware_datetime(candidate.deadline)
    if deadline is None:
        return 0.0
    days_left = max(0.0, (deadline - datetime.now(UTC)).total_seconds() / 86_400.0)
    if days_left <= 1:
        return 1200.0
    if days_left <= 3:
        return 1000.0
    if days_left <= 7:
        return 800.0
    if days_left <= 14:
        return 600.0
    if days_left <= 30:
        return 400.0
    if days_left <= 90:
        return 150.0
    return 0.0


def _is_medal_candidate(candidate: EnteredCompetition) -> bool:
    category = candidate.category.strip().lower()
    if category not in {"featured", "research", "community"}:
        return False
    if _reward_amount_usd(candidate.reward) is None:
        return False
    if candidate.team_count is not None and candidate.team_count < 50:
        return False
    return True


def _reward_amount_usd(reward: str) -> float | None:
    if not reward:
        return None
    amounts: list[float] = []
    for match in _REWARD_AMOUNT_RE.finditer(reward):
        raw_amount = (match.group("dollar") or match.group("usd") or "").replace(",", "")
        try:
            amounts.append(float(raw_amount))
        except ValueError:
            continue
    if not amounts:
        return None
    return max(amounts)


def _load_submission_history(*, config: WatchConfig, slug: str) -> SubmissionHistory:
    autopilot_started = _has_autopilot_history(config=config, slug=slug)
    ledger_path = CompetitionPaths(slug=slug, artifacts_dir=config.artifacts_dir).submission_ledger_path
    if not ledger_path.exists():
        return SubmissionHistory(
            autopilot_started=autopilot_started,
            submitted=False,
            rank_percentile=None,
            rank=None,
            total_teams=None,
            outcome_score=None,
        )
    submitted = False
    best_percentile: float | None = None
    best_rank: int | None = None
    best_total: int | None = None
    best_score: float | None = None
    for record in load_jsonl_records(ledger_path):
        event = record.get("event")
        if event in (None, "submit"):
            submitted = True
        if event != "outcome":
            continue
        submitted = True
        outcome = record.get("outcome")
        if not isinstance(outcome, dict):
            continue
        rank = _optional_int(outcome.get("rank"))
        total = _optional_int(outcome.get("total_teams"))
        percentile = _optional_float(outcome.get("rank_percentile"))
        score = _optional_float(outcome.get("score"))
        if percentile is None and rank is not None and total is not None and total > 0:
            percentile = rank / total
        if percentile is None:
            if best_score is None and score is not None:
                best_score = score
            continue
        if best_percentile is None or percentile > best_percentile:
            best_percentile = percentile
            best_rank = rank
            best_total = total
            best_score = score
    return SubmissionHistory(
        autopilot_started=autopilot_started,
        submitted=submitted,
        rank_percentile=best_percentile,
        rank=best_rank,
        total_teams=best_total,
        outcome_score=best_score,
    )


def _enrich_submission_history_from_leaderboard(
    *,
    config: WatchConfig,
    slug: str,
    history: SubmissionHistory,
) -> SubmissionHistory:
    if not history.submitted or history.rank_percentile is not None or history.outcome_score is None or config.dry_run:
        return history
    paths = CompetitionPaths(slug=slug, artifacts_dir=config.artifacts_dir)
    direction = _load_metric_direction(paths)
    if direction is None:
        return history
    try:
        rank_info = leaderboard_rank_for_score(
            slug,
            paths.context_dir,
            score=history.outcome_score,
            direction=direction,
            dry_run=config.dry_run,
        )
    except Exception:
        return history
    rank_percentile = _optional_float(rank_info.get("rank_percentile"))
    rank = _optional_int(rank_info.get("rank"))
    total_teams = _optional_int(rank_info.get("total_teams"))
    if rank_percentile is None and rank is not None and total_teams is not None and total_teams > 0:
        rank_percentile = rank / total_teams
    if rank_percentile is None:
        return history
    return SubmissionHistory(
        autopilot_started=history.autopilot_started,
        submitted=history.submitted,
        rank_percentile=rank_percentile,
        rank=rank,
        total_teams=total_teams,
        outcome_score=history.outcome_score,
    )


def _has_autopilot_history(*, config: WatchConfig, slug: str) -> bool:
    runs_dir = CompetitionPaths(slug=slug, artifacts_dir=config.artifacts_dir).runs_dir
    if not runs_dir.exists():
        return False
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "run.json").exists() or (child / "run_state.json").exists():
            return True
        if any(child.glob("iter-*")):
            return True
    return False


def _load_metric_direction(paths: CompetitionPaths) -> str | None:
    spec_path = paths.context_dir / "evaluation_spec.json"
    payload = load_json_object(spec_path)
    if payload is None:
        return None
    explicit = payload.get("direction")
    if isinstance(explicit, str) and explicit.strip() and explicit.strip().lower() != "auto":
        direction = explicit.strip().lower()
        if direction in {"minimize", "maximize"}:
            return direction
    metric = payload.get("metric_name") or payload.get("target_metric") or payload.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        return None
    direction = infer_direction(metric.strip(), explicit if isinstance(explicit, str) else None)
    if direction in {"minimize", "maximize"}:
        return direction
    return None


def _optional_int(value: object) -> int | None:
    return parse_int_value(value, allow_float=True)


def _optional_float(value: object) -> float | None:
    return parse_float_value(value)


def _aware_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _cooldown_slugs(records: list[dict[str, object]], *, hours: float) -> set[str]:
    if hours <= 0:
        return set()
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    slugs: set[str] = set()
    for record in records:
        if record.get("event") not in {"finished", "failed", "skipped"}:
            continue
        ts = _parse_ts(record.get("ts"))
        slug = str(record.get("slug") or "").strip().lower()
        if slug and ts is not None and ts >= cutoff:
            slugs.add(slug)
    return slugs


def _resource_blocked_slugs(records: list[dict[str, object]]) -> set[str]:
    hours = _resource_block_ttl_hours()
    if hours <= 0:
        return set()
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    slugs: set[str] = set()
    for record in records:
        if record.get("event") != "resource_blocked":
            continue
        ts = _parse_ts(record.get("ts"))
        slug = str(record.get("slug") or "").strip().lower()
        if slug and ts is not None and ts >= cutoff:
            slugs.add(slug)
    return slugs


def _resource_block_ttl_hours() -> float:
    value = parse_float_value(os.environ.get("KAGGLEBOT_RESOURCE_BLOCK_TTL_HOURS"))
    if value is None:
        return _DEFAULT_RESOURCE_BLOCK_TTL_HOURS
    return max(0.0, value)


def _all_watch_records(config: WatchConfig, *, primary: WatchLedger) -> list[dict[str, object]]:
    records = primary.records()
    seen = {primary.path.resolve()}
    ledger_paths = [config.root_watch_dir / "ledger.jsonl"]
    if config.root_watch_dir.exists():
        ledger_paths.extend(path for path in config.root_watch_dir.glob("*/ledger.jsonl") if path.is_file())
    for path in ledger_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        records.extend(WatchLedger(path).records())
    return records


def _active_slugs(config: WatchConfig) -> set[str]:
    state_paths = [config.root_watch_dir / "state.json"]
    if config.root_watch_dir.exists():
        state_paths.extend(path for path in config.root_watch_dir.glob("*/state.json") if path.is_file())
    active: set[str] = set()
    seen: set[Path] = set()
    for path in state_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        state = _load_state(path)
        slug = str(state.get("active_slug") or "").strip().lower()
        run_id = str(state.get("active_run_id") or "").strip()
        if slug and run_id and _run_can_resume(config, slug, run_id, state=state):
            active.add(slug)
    return active


def _run_can_resume(config: WatchConfig, slug: str, run_id: str, *, state: dict[str, object] | None = None) -> bool:
    if state is not None and _active_state_is_stale(state):
        return False
    run_dir = CompetitionPaths(slug=slug, artifacts_dir=config.artifacts_dir).run_dir(run_id)
    if not run_dir.exists():
        return False
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return True
    payload = load_json_object(run_json)
    if payload is None:
        return True
    status = str(payload.get("status") or "").strip().lower()
    return status not in _TERMINAL_RUN_STATUSES


def _active_state_is_stale(state: dict[str, object]) -> bool:
    slug = str(state.get("active_slug") or "").strip()
    run_id = str(state.get("active_run_id") or "").strip()
    if not slug or not run_id:
        return False
    timestamp = _parse_ts(state.get("updated_at")) or _parse_ts(state.get("started_at"))
    if timestamp is None:
        return False
    max_age_hours = _active_run_stale_hours()
    if max_age_hours <= 0:
        return False
    return timestamp + timedelta(hours=max_age_hours) <= datetime.now(UTC)


def _active_run_stale_hours() -> float:
    value = parse_float_value(os.environ.get("KAGGLEBOT_WATCH_ACTIVE_RUN_STALE_HOURS"))
    if value is None:
        return _DEFAULT_ACTIVE_RUN_STALE_HOURS
    return max(0.0, value)


def _safe_state_scope(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")


def _candidate_from_slug(slug: str) -> EnteredCompetition:
    return EnteredCompetition(
        slug=slug,
        title=slug,
        url=f"https://www.kaggle.com/competitions/{slug}",
        category="entered",
        reward="",
        evaluation_metric="",
        deadline=None,
        enabled_date=None,
        new_entrant_deadline=None,
        merger_deadline=None,
        team_count=None,
        max_daily_submissions=None,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        source="state",
    )


def _load_state(path: Path) -> dict[str, object]:
    return load_json_object(path) or {}


def _write_state(path: Path, payload: dict[str, object]) -> None:
    payload = dict(payload)
    payload.setdefault("updated_at", datetime.now(UTC).isoformat())
    write_json_object(path, payload, sort_keys=True)


def _set_resume_env(*, slug: str, run_id: str) -> None:
    import os

    os.environ["KAGGLEBOT_RESUME_RUN_ID"] = run_id
    os.environ["KAGGLEBOT_RESUME_SLUG"] = slug


def _parse_ts(value: object) -> datetime | None:
    return parse_iso_datetime_utc(value)
