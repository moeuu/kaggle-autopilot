from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable


@dataclass(frozen=True)
class SubmissionOutcomeService:
    fetch_rows: Callable[[str, bool], list[dict[str, str]]]
    max_attempts: int = 20
    poll_interval_sec: float = 30.0

    def wait_for_outcome(
        self,
        *,
        slug: str,
        message: str,
        submitted_at: datetime,
    ) -> dict[str, object] | None:
        for attempt in range(1, self.max_attempts + 1):
            try:
                rows = self.fetch_rows(slug, False)
            except Exception:  # noqa: BLE001
                return None
            match = self._select_submission_row(rows=rows, message=message, submitted_at=submitted_at)
            if match is not None:
                status = self._extract_submission_status(match)
                score = self._extract_submission_score(match)
                if score is not None:
                    return {
                        "status": status,
                        "score": score,
                        "raw": match,
                        "checked_at": datetime.now(UTC).isoformat(),
                    }
                if status in {"complete", "completed", "error", "failed", "cancelled"}:
                    return {
                        "status": status,
                        "score": None,
                        "raw": match,
                        "checked_at": datetime.now(UTC).isoformat(),
                    }
            if attempt < self.max_attempts:
                time.sleep(self.poll_interval_sec)
        return None

    def _select_submission_row(
        self,
        *,
        rows: list[dict[str, str]],
        message: str,
        submitted_at: datetime,
    ) -> dict[str, str] | None:
        if not rows:
            return None
        target = message.strip()
        with_message = [row for row in rows if self._row_matches_submission_message(row, target)]
        candidates = with_message if with_message else rows
        rows_with_ts: list[tuple[datetime, dict[str, str]]] = []
        for row in candidates:
            ts = self._parse_submission_row_time(row)
            if ts is None:
                continue
            rows_with_ts.append((ts, row))
        if rows_with_ts:
            window_start = submitted_at.timestamp() - 3600
            recent = [item for item in rows_with_ts if item[0].timestamp() >= window_start]
            source = recent or rows_with_ts
            source.sort(key=lambda item: item[0], reverse=True)
            return source[0][1]
        return candidates[0]

    @staticmethod
    def _row_matches_submission_message(row: dict[str, str], message: str) -> bool:
        if not message:
            return False
        for key in ("description", "message", "comments", "comment"):
            value = SubmissionOutcomeService._get_row_value_ci(row, key)
            if value and value.strip() == message:
                return True
        return False

    @staticmethod
    def _parse_submission_row_time(row: dict[str, str]) -> datetime | None:
        for key in ("date", "submittedDate", "submitted_date", "createdAt", "created_at", "timestamp"):
            value = SubmissionOutcomeService._get_row_value_ci(row, key)
            if not value:
                continue
            parsed = SubmissionOutcomeService._parse_datetime(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _extract_submission_status(row: dict[str, str]) -> str:
        for key in ("status", "state"):
            value = SubmissionOutcomeService._get_row_value_ci(row, key)
            if value:
                return value.strip().lower()
        return "unknown"

    @staticmethod
    def _extract_submission_score(row: dict[str, str]) -> float | None:
        for key in ("publicScore", "public_score", "score", "privateScore", "private_score"):
            value = SubmissionOutcomeService._get_row_value_ci(row, key)
            parsed = SubmissionOutcomeService._to_float(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _get_row_value_ci(row: dict[str, str], key: str) -> str | None:
        target = key.strip().lower()
        for current_key, value in row.items():
            if current_key.strip().lower() == target:
                return value
        return None

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        raw = str(value).strip()
        if not raw:
            return None
        normalized = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y/%m/%d %H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    @staticmethod
    def _to_float(value: object) -> float | None:
        if value is None:
            return None
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
