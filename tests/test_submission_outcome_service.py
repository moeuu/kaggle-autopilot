from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kagglebot.submission.outcome_service import (
    SubmissionOutcomePollingError,
    SubmissionOutcomeService,
    submission_poll_delay_seconds,
)


def test_submission_poll_delay_backs_off_pending_and_fetch_errors() -> None:
    assert submission_poll_delay_seconds(attempt=1, base_interval_sec=30.0) == 30.0
    assert submission_poll_delay_seconds(attempt=11, base_interval_sec=30.0) == 60.0
    assert submission_poll_delay_seconds(attempt=31, base_interval_sec=30.0) == 120.0
    assert (
        submission_poll_delay_seconds(
            attempt=2,
            base_interval_sec=30.0,
            consecutive_fetch_errors=2,
        )
        == 120.0
    )
    assert submission_poll_delay_seconds(attempt=100, base_interval_sec=0.0) == 0.0


def test_submission_outcome_service_extracts_rank_pair_from_row() -> None:
    rows = [
        {
            "description": "demo",
            "status": "complete",
            "publicScore": "0.9532",
            "rank": "1300 / 2700",
            "date": "2026-02-16T10:00:00Z",
        }
    ]
    service = SubmissionOutcomeService(fetch_rows=lambda slug: rows, max_attempts=1, poll_interval_sec=0.0)
    outcome = service.wait_for_outcome(
        slug="demo",
        message="demo",
        submitted_at=datetime.now(UTC),
    )
    assert isinstance(outcome, dict)
    assert outcome.get("rank") == 1300
    assert outcome.get("total_teams") == 2700
    assert outcome.get("rank_percentile") == 1300 / 2700
    assert outcome.get("rank_source") == "submission_row"


def test_submission_outcome_service_extracts_rank_and_total_keys() -> None:
    rows = [
        {
            "description": "demo",
            "status": "complete",
            "publicScore": "0.9532",
            "publicLeaderboardRank": "88",
            "publicLeaderboardTotalTeams": "2700",
            "date": "2026-02-16T10:00:00Z",
        }
    ]
    service = SubmissionOutcomeService(fetch_rows=lambda slug: rows, max_attempts=1, poll_interval_sec=0.0)
    outcome = service.wait_for_outcome(
        slug="demo",
        message="demo",
        submitted_at=datetime.now(UTC),
    )
    assert isinstance(outcome, dict)
    assert outcome.get("rank") == 88
    assert outcome.get("total_teams") == 2700
    assert outcome.get("rank_percentile") == 88 / 2700


def test_submission_outcome_service_handles_prefixed_error_status_as_terminal() -> None:
    rows = [
        {
            "description": "demo",
            "status": "SubmissionStatus.ERROR",
            "publicScore": "",
            "date": "2026-02-16T10:00:00Z",
        }
    ]
    service = SubmissionOutcomeService(fetch_rows=lambda slug: rows, max_attempts=3, poll_interval_sec=0.0)
    outcome = service.wait_for_outcome(
        slug="demo",
        message="demo",
        submitted_at=datetime.now(UTC),
    )
    assert isinstance(outcome, dict)
    assert outcome.get("status") == "error"
    assert outcome.get("score") is None


def test_submission_outcome_service_handles_failed_status_as_terminal() -> None:
    rows = [
        {
            "description": "demo",
            "status": "failed",
            "date": "2026-02-16T10:00:00Z",
        }
    ]
    service = SubmissionOutcomeService(fetch_rows=lambda slug: rows, max_attempts=3, poll_interval_sec=0.0)
    outcome = service.wait_for_outcome(
        slug="demo",
        message="demo",
        submitted_at=datetime.now(UTC),
    )
    assert isinstance(outcome, dict)
    assert outcome.get("status") == "failed"
    assert outcome.get("score") is None


def test_submission_outcome_service_treats_complete_without_score_as_terminal_success() -> None:
    rows = [
        {
            "description": "demo",
            "status": "complete",
            "date": "2026-02-16T10:00:00Z",
        }
    ]
    service = SubmissionOutcomeService(fetch_rows=lambda slug: rows, max_attempts=3, poll_interval_sec=0.0)
    outcome = service.wait_for_outcome(
        slug="demo",
        message="demo",
        submitted_at=datetime.now(UTC),
    )
    assert isinstance(outcome, dict)
    assert outcome.get("status") == "complete"
    assert outcome.get("score") is None


def test_submission_outcome_service_does_not_match_stale_row_with_other_message() -> None:
    submitted_at = datetime(2026, 7, 15, 0, 0, tzinfo=UTC)
    rows = [
        {
            "description": "previous-run",
            "status": "complete",
            "publicScore": "1.0",
            "date": (submitted_at - timedelta(minutes=5)).isoformat(),
        }
    ]
    service = SubmissionOutcomeService(fetch_rows=lambda slug: rows, max_attempts=1, poll_interval_sec=0.0)

    outcome = service.wait_for_outcome(slug="demo", message="new-run", submitted_at=submitted_at)

    assert outcome is None


def test_submission_outcome_service_matches_message_less_code_row_near_request() -> None:
    submitted_at = datetime(2026, 7, 15, 0, 0, tzinfo=UTC)
    rows = [
        {
            "description": "",
            "status": "complete",
            "publicScore": "0.5",
            "date": (submitted_at + timedelta(seconds=3)).isoformat(),
        }
    ]
    service = SubmissionOutcomeService(fetch_rows=lambda slug: rows, max_attempts=1, poll_interval_sec=0.0)

    outcome = service.wait_for_outcome(slug="demo", message="new-code-run", submitted_at=submitted_at)

    assert isinstance(outcome, dict)
    assert outcome["score"] == 0.5


def test_submission_outcome_service_stops_on_unidentifiable_terminal_code_row() -> None:
    calls = {"count": 0}

    def fetch_rows(_slug: str) -> list[dict[str, str]]:
        calls["count"] += 1
        return [{"description": "", "status": "complete", "publicScore": "0.5"}]

    service = SubmissionOutcomeService(fetch_rows=fetch_rows, max_attempts=5, poll_interval_sec=30.0)

    outcome = service.wait_for_outcome(
        slug="demo",
        message="new-code-run",
        submitted_at=datetime(2026, 7, 15, 0, 0, tzinfo=UTC),
    )

    assert outcome is None
    assert calls["count"] == 1


def test_submission_outcome_service_does_not_cross_associate_sequential_messages() -> None:
    submitted_at = datetime(2026, 7, 15, 0, 0, tzinfo=UTC)
    rows = [
        {
            "description": "iteration-1",
            "status": "complete",
            "publicScore": "0.9",
            "date": (submitted_at - timedelta(seconds=30)).isoformat(),
        },
        {
            "description": "iteration-2",
            "status": "error",
            "publicScore": "",
            "date": (submitted_at + timedelta(seconds=1)).isoformat(),
        },
    ]
    service = SubmissionOutcomeService(fetch_rows=lambda slug: rows, max_attempts=1, poll_interval_sec=0.0)

    outcome = service.wait_for_outcome(slug="demo", message="iteration-2", submitted_at=submitted_at)

    assert isinstance(outcome, dict)
    assert outcome["status"] == "error"
    assert outcome["score"] is None


def test_submission_outcome_service_raises_after_consecutive_fetch_errors() -> None:
    calls = {"count": 0}

    def failing_fetch(_slug: str) -> list[dict[str, str]]:
        calls["count"] += 1
        raise RuntimeError("kaggle api unreachable")

    service = SubmissionOutcomeService(
        fetch_rows=failing_fetch,
        max_attempts=None,
        poll_interval_sec=0.0,
        max_fetch_errors=2,
    )

    with pytest.raises(SubmissionOutcomePollingError) as exc_info:
        service.wait_for_outcome(
            slug="demo",
            message="demo",
            submitted_at=datetime.now(UTC),
        )

    assert calls["count"] == 2
    assert exc_info.value.consecutive_errors == 2
