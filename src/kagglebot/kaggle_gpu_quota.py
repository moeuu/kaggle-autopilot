from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kagglebot.datetime_utils import parse_iso_datetime_utc
from kagglebot.env_utils import parse_float_value
from kagglebot.json_utils import load_json_object

DEFAULT_KAGGLE_GPU_QUOTA_FILE_MAX_AGE_HOURS = 24.0


@dataclass(frozen=True)
class KaggleGpuQuotaStatus:
    available_minutes: int | None
    total_minutes: int | None = None
    used_minutes: int | None = None
    reserved_minutes: int | None = None
    refresh_time: str | None = None
    source: str = "unknown"


def read_kaggle_gpu_quota_file(path: Path) -> KaggleGpuQuotaStatus | None:
    payload = load_json_object(path)
    if payload is None:
        return None
    expires_at = _parse_ts(payload.get("expires_at"))
    if expires_at is not None and expires_at <= datetime.now(UTC):
        return None
    if expires_at is None and kaggle_gpu_quota_file_is_stale(path=path, payload=payload):
        return None
    source = f"file:{path}"
    quota = parse_kaggle_gpu_quota_text(payload.get("text") or payload.get("quota_text"), source=source)
    if quota is not None:
        return quota
    available_minutes = coerce_minutes(payload.get("available_minutes"))
    total_minutes = coerce_minutes(payload.get("total_minutes"))
    used_minutes = coerce_minutes(payload.get("used_minutes"))
    reserved_minutes = coerce_minutes(payload.get("reserved_minutes"))
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


def kaggle_gpu_quota_file_is_stale(*, path: Path, payload: dict[str, object]) -> bool:
    max_age_hours = kaggle_gpu_quota_file_max_age_hours()
    if max_age_hours <= 0:
        return False
    cache_ts = kaggle_gpu_quota_file_timestamp(path=path, payload=payload)
    if cache_ts is None:
        return True
    return cache_ts + timedelta(hours=max_age_hours) <= datetime.now(UTC)


def kaggle_gpu_quota_file_timestamp(*, path: Path, payload: dict[str, object]) -> datetime | None:
    for key in ("updated_at", "refresh_time", "quota_refresh_time"):
        parsed = _parse_ts(payload.get(key))
        if parsed is not None:
            return parsed
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None


def kaggle_gpu_quota_file_max_age_hours() -> float:
    value = parse_float_value(os.environ.get("KAGGLEBOT_KAGGLE_GPU_QUOTA_FILE_MAX_AGE_HOURS"))
    return DEFAULT_KAGGLE_GPU_QUOTA_FILE_MAX_AGE_HOURS if value is None else value


def quota_status_from_web_payload(payload: dict[str, object], *, source: str) -> KaggleGpuQuotaStatus | None:
    raw_gpu = payload.get("gpuQuota")
    if not isinstance(raw_gpu, dict):
        return None
    total = duration_to_minutes(raw_gpu.get("totalTimeAllowed"))
    used = duration_to_minutes(raw_gpu.get("timeUsed"))
    reserved = duration_to_minutes(raw_gpu.get("timeReserved"))
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


def parse_kaggle_gpu_quota_text(value: object, *, source: str = "text") -> KaggleGpuQuotaStatus | None:
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
    available = parse_hours_minutes_text(match.group("available"))
    total = parse_hours_minutes_text(match.group("total"))
    if available is None:
        return None
    return KaggleGpuQuotaStatus(
        available_minutes=available,
        total_minutes=total,
        used_minutes=max(0, total - available) if total is not None else None,
        source=source,
    )


def parse_hours_minutes_text(value: str) -> int | None:
    text = value.strip().lower()
    hours_match = re.search(r"(\d+)\s*h", text)
    minutes_match = re.search(r"(\d+)\s*m", text)
    if not hours_match and not minutes_match:
        return None
    hours = int(hours_match.group(1)) if hours_match else 0
    minutes = int(minutes_match.group(1)) if minutes_match else 0
    return hours * 60 + minutes


def duration_to_minutes(value: object) -> int | None:
    seconds = duration_to_seconds(value)
    if seconds is None:
        return None
    return max(0, int(seconds // 60))


def duration_to_seconds(value: object) -> float | None:
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
        days = parse_float_value(value.get("days")) or 0.0
        hours = parse_float_value(value.get("hours")) or 0.0
        minutes = parse_float_value(value.get("minutes")) or 0.0
        seconds = parse_float_value(value.get("seconds")) or 0.0
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
        days = parse_float_value(iso_match.group("days")) or 0.0
        hours = parse_float_value(iso_match.group("hours")) or 0.0
        minutes = parse_float_value(iso_match.group("minutes")) or 0.0
        seconds = parse_float_value(iso_match.group("seconds")) or 0.0
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    hm = parse_hours_minutes_text(text)
    if hm is not None:
        return hm * 60
    try:
        return float(text)
    except ValueError:
        return None


def coerce_minutes(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_minutes(value: int) -> str:
    hours, minutes = divmod(max(0, int(value)), 60)
    if minutes:
        return f"{hours}h {minutes}m"
    return f"{hours}h"


def _parse_ts(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return parse_iso_datetime_utc(text)
