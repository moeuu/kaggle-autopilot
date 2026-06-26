from __future__ import annotations

import builtins
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from kagglebot.datetime_utils import parse_datetime_utc
from kagglebot.scalar_utils import tolerant_finite_float, tolerant_int

_RANK_PAIR_RE = re.compile(r"(?P<rank>\d+)\s*/\s*(?P<total>\d+)")
_TERMINAL_SUBMISSION_STATUSES = {"complete", "completed", "error", "failed", "cancelled", "canceled"}


class SubmissionOutcomePollingError(RuntimeError):
    def __init__(self, message: str, *, attempt: int, consecutive_errors: int, detail: str) -> None:
        super().__init__(message)
        self.attempt = attempt
        self.consecutive_errors = consecutive_errors
        self.detail = detail


@dataclass(frozen=True)
class SubmissionOutcomeService:
    fetch_rows: Callable[[str], list[dict[str, str]]]
    max_attempts: int | None = None
    poll_interval_sec: float = 30.0
    max_fetch_errors: int = 3

    def wait_for_outcome(
        self,
        *,
        slug: str,
        message: str,
        submitted_at: datetime,
    ) -> dict[str, object] | None:
        attempt = 0
        consecutive_fetch_errors = 0
        while True:
            attempt += 1
            try:
                rows = self.fetch_rows(slug)
                consecutive_fetch_errors = 0
            except Exception as exc:  # noqa: BLE001
                consecutive_fetch_errors += 1
                builtins.print(f"submission poll: attempt={attempt} status=fetch_error waiting", flush=True)
                if self.max_fetch_errors > 0 and consecutive_fetch_errors >= self.max_fetch_errors:
                    detail = f"{type(exc).__name__}: {exc}"
                    raise SubmissionOutcomePollingError(
                        (
                            "Submission polling failed after consecutive fetch errors "
                            f"(attempt={attempt}, consecutive_errors={consecutive_fetch_errors}): {detail}"
                        ),
                        attempt=attempt,
                        consecutive_errors=consecutive_fetch_errors,
                        detail=detail,
                    ) from exc
                if self.max_attempts is not None and self.max_attempts > 0 and attempt >= self.max_attempts:
                    return None
                time.sleep(self.poll_interval_sec)
                continue
            match = self._select_submission_row(rows=rows, message=message, submitted_at=submitted_at)
            if match is not None:
                status = self._extract_submission_status(match)
                score = self._extract_submission_score(match)
                rank, total_teams = self._extract_submission_rank(match)
                rank_payload = self._build_rank_payload(rank=rank, total_teams=total_teams)
                if score is not None:
                    payload: dict[str, object] = {
                        "status": status,
                        "score": score,
                        "raw": match,
                        "checked_at": datetime.now(UTC).isoformat(),
                    }
                    payload.update(rank_payload)
                    return payload
                if status in _TERMINAL_SUBMISSION_STATUSES:
                    payload = {
                        "status": status,
                        "score": None,
                        "raw": match,
                        "checked_at": datetime.now(UTC).isoformat(),
                    }
                    payload.update(rank_payload)
                    return payload
                builtins.print(
                    f"submission poll: attempt={attempt} status={status or 'unknown'} waiting",
                    flush=True,
                )
            else:
                builtins.print(f"submission poll: attempt={attempt} status=not_found waiting", flush=True)
            if self.max_attempts is not None and self.max_attempts > 0 and attempt >= self.max_attempts:
                return None
            time.sleep(self.poll_interval_sec)

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
                return SubmissionOutcomeService._normalize_submission_status(value)
        return "unknown"

    @staticmethod
    def _normalize_submission_status(value: object) -> str:
        raw = str(value).strip().lower()
        if not raw:
            return "unknown"
        if "." in raw:
            prefix, _, suffix = raw.rpartition(".")
            if suffix and "status" in prefix:
                return suffix.strip()
        return raw

    @staticmethod
    def _extract_submission_score(row: dict[str, str]) -> float | None:
        for key in ("publicScore", "public_score", "score", "privateScore", "private_score"):
            value = SubmissionOutcomeService._get_row_value_ci(row, key)
            parsed = tolerant_finite_float(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _extract_submission_rank(row: dict[str, str]) -> tuple[int | None, int | None]:
        rank: int | None = None
        total_teams: int | None = None

        rank_keys = (
            "publicLeaderboardRank",
            "public_rank",
            "publicRank",
            "leaderboardRank",
            "rank",
            "position",
        )
        total_keys = (
            "publicLeaderboardTotalTeams",
            "publicLeaderboardSize",
            "totalTeams",
            "teamCount",
            "total_teams",
            "leaderboardSize",
            "participants",
        )

        for key in rank_keys:
            value = SubmissionOutcomeService._get_row_value_ci(row, key)
            if not value:
                continue
            parsed_rank, parsed_total = SubmissionOutcomeService._parse_rank_value(value)
            if parsed_rank is not None:
                rank = parsed_rank
            if parsed_total is not None:
                total_teams = parsed_total
            if rank is not None:
                break

        for key in total_keys:
            value = SubmissionOutcomeService._get_row_value_ci(row, key)
            parsed_total = tolerant_int(value)
            if parsed_total is not None:
                total_teams = parsed_total
                break

        if rank is not None and total_teams is None:
            for raw_value in row.values():
                parsed_rank, parsed_total = SubmissionOutcomeService._parse_rank_value(raw_value)
                if parsed_rank == rank and parsed_total is not None:
                    total_teams = parsed_total
                    break

        return rank, total_teams

    @staticmethod
    def _build_rank_payload(*, rank: int | None, total_teams: int | None) -> dict[str, object]:
        payload: dict[str, object] = {}
        if rank is not None:
            payload["rank"] = rank
            payload["rank_source"] = "submission_row"
        if total_teams is not None:
            payload["total_teams"] = total_teams
        if rank is not None and total_teams is not None and total_teams > 0:
            payload["rank_percentile"] = rank / total_teams
        return payload

    @staticmethod
    def _parse_rank_value(value: object) -> tuple[int | None, int | None]:
        parsed_int = tolerant_int(value)
        if parsed_int is not None:
            return parsed_int, None
        if value is None:
            return None, None
        text = str(value).strip()
        if not text:
            return None, None
        match = _RANK_PAIR_RE.search(text)
        if match is None:
            return None, None
        rank = tolerant_int(match.group("rank"))
        total = tolerant_int(match.group("total"))
        return rank, total

    @staticmethod
    def _get_row_value_ci(row: dict[str, str], key: str) -> str | None:
        target = key.strip().lower()
        for current_key, value in row.items():
            if current_key.strip().lower() == target:
                return value
        return None

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        return parse_datetime_utc(
            value,
            formats=(
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y/%m/%d %H:%M:%S",
            ),
        )
