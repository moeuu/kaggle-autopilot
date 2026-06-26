from __future__ import annotations

import json
from pathlib import Path

from kagglebot.kaggle_gpu_quota import (
    kaggle_gpu_quota_file_max_age_hours,
    parse_kaggle_gpu_quota_text,
    quota_status_from_web_payload,
    read_kaggle_gpu_quota_file,
)


def test_parse_kaggle_gpu_quota_text_available_of_total() -> None:
    quota = parse_kaggle_gpu_quota_text("14h 36m available of 30h", source="test")

    assert quota is not None
    assert quota.available_minutes == 876
    assert quota.total_minutes == 1800
    assert quota.used_minutes == 924
    assert quota.source == "test"


def test_quota_status_from_web_payload_parses_duration_shapes() -> None:
    quota = quota_status_from_web_payload(
        {
            "gpuQuota": {
                "totalTimeAllowed": {"seconds": 7200},
                "timeUsed": "PT1H",
                "timeReserved": "30m",
            },
            "quotaRefreshTime": "tomorrow",
        },
        source="test-web",
    )

    assert quota is not None
    assert quota.available_minutes == 60
    assert quota.total_minutes == 120
    assert quota.used_minutes == 60
    assert quota.reserved_minutes == 30
    assert quota.refresh_time == "tomorrow"
    assert quota.source == "test-web"


def test_read_kaggle_gpu_quota_file_ignores_missing_invalid_or_non_object_payload(tmp_path: Path) -> None:
    assert read_kaggle_gpu_quota_file(tmp_path / "missing.json") is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert read_kaggle_gpu_quota_file(invalid) is None

    array_payload = tmp_path / "array.json"
    array_payload.write_text("[]", encoding="utf-8")
    assert read_kaggle_gpu_quota_file(array_payload) is None


def test_read_kaggle_gpu_quota_file_ignores_stale_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLEBOT_KAGGLE_GPU_QUOTA_FILE_MAX_AGE_HOURS", raising=False)
    quota_path = tmp_path / "quota.json"
    quota_path.write_text(
        json.dumps(
            {
                "available_minutes": 876,
                "total_minutes": 1800,
                "updated_at": "2000-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    assert read_kaggle_gpu_quota_file(quota_path) is None


def test_kaggle_gpu_quota_file_max_age_falls_back_for_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_KAGGLE_GPU_QUOTA_FILE_MAX_AGE_HOURS", "nan")
    assert kaggle_gpu_quota_file_max_age_hours() == 24.0


def test_read_kaggle_gpu_quota_file_honors_explicit_expiry(tmp_path: Path) -> None:
    quota_path = tmp_path / "quota.json"
    quota_path.write_text(
        json.dumps(
            {
                "available_minutes": 1800,
                "total_minutes": 1800,
                "updated_at": "2000-01-01T00:00:00+00:00",
                "expires_at": "2999-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    quota = read_kaggle_gpu_quota_file(quota_path)

    assert quota is not None
    assert quota.available_minutes == 1800
