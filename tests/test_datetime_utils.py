from __future__ import annotations

from datetime import UTC, datetime

from kagglebot.datetime_utils import parse_datetime_utc, parse_iso_datetime_utc


def test_parse_iso_datetime_utc_accepts_z_suffix_and_offsets() -> None:
    assert parse_iso_datetime_utc("2026-06-25T12:00:00Z") == datetime(2026, 6, 25, 12, tzinfo=UTC)
    assert parse_iso_datetime_utc("2026-06-25T21:00:00+09:00") == datetime(2026, 6, 25, 12, tzinfo=UTC)


def test_parse_iso_datetime_utc_treats_naive_values_as_utc() -> None:
    assert parse_iso_datetime_utc("2026-06-25T12:00:00") == datetime(2026, 6, 25, 12, tzinfo=UTC)


def test_parse_iso_datetime_utc_rejects_blank_and_invalid_values() -> None:
    assert parse_iso_datetime_utc(None) is None
    assert parse_iso_datetime_utc("") is None
    assert parse_iso_datetime_utc("not a date") is None


def test_parse_datetime_utc_accepts_explicit_formats_and_utc_suffix() -> None:
    assert parse_datetime_utc(
        "2026-05-09 06:16:21.527000",
        formats=("%Y-%m-%d %H:%M:%S.%f",),
    ) == datetime(2026, 5, 9, 6, 16, 21, 527000, tzinfo=UTC)
    assert parse_datetime_utc(
        "2026/05/09 06:16:21 UTC",
        formats=("%Y/%m/%d %H:%M:%S",),
    ) == datetime(2026, 5, 9, 6, 16, 21, tzinfo=UTC)
