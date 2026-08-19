from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pytest import MonkeyPatch

import kagglebot.event_notifications as event_notifications
from kagglebot.event_notifications import (
    DELIVERED_LIFECYCLE_KEYS,
    DELIVERY_RECEIPTS_FILENAME,
    LEDGER_CURSOR_INITIALIZED_KEY,
    LEDGER_OFFSET_KEY,
    HttpEventSink,
    HttpEventSinkConfig,
    _read_json_object,
    build_autopilot_status_payload,
    dispatch_events_once,
    event_sink_from_env,
)


class RecordingSink(HttpEventSink):
    def __init__(self) -> None:
        super().__init__(None)
        self.events: list[dict[str, object]] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, **kwargs):  # type: ignore[no-untyped-def]
        self.events.append(kwargs)
        return True


class FailingOnceSink(RecordingSink):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def emit(self, **kwargs):  # type: ignore[no-untyped-def]
        self.events.append(kwargs)
        if not self.failed:
            self.failed = True
            return False
        return True


class UrlResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.raw = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> UrlResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.raw


def test_event_sink_requires_at_least_one_matched_route(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        event_notifications.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: UrlResponse({"matched_routes": 0}),
    )
    sink = HttpEventSink(HttpEventSinkConfig(api_url="https://events.invalid/api", api_token="test-token"))

    assert not sink.emit(
        event_type="autopilot.status",
        severity="info",
        dedupe_key="test-event",
        payload={},
    )


def test_event_sink_uses_generic_env_and_omits_optional_account(monkeypatch: MonkeyPatch) -> None:
    requests = []

    def accept(request, **_kwargs):  # type: ignore[no-untyped-def]
        requests.append(request)
        return UrlResponse({"matched_routes": 1, "event": {"id": "server-1", "status": "queued"}})

    monkeypatch.setenv("KAGGLEBOT_EVENT_SINK_URL", "https://events.invalid/api")
    monkeypatch.setenv("KAGGLEBOT_EVENT_SINK_TOKEN", "test-token")
    monkeypatch.setattr(event_notifications.urllib.request, "urlopen", accept)

    receipt = event_sink_from_env().emit(
        event_type="autopilot.status",
        severity="info",
        dedupe_key="test-event",
        payload={},
    )

    body = json.loads(requests[0].data)
    assert "account" not in body
    assert receipt.response_event_id == "server-1"
    assert receipt.response_status == "queued"


def test_lifecycle_cursor_advances_only_after_durable_api_receipt(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    artifacts = tmp_path / "artifacts"
    watch_dir = artifacts / "_watch"
    watch_dir.mkdir(parents=True)
    (watch_dir / "state.json").write_text(json.dumps({"last_status": "failed"}), encoding="utf-8")
    (watch_dir / "event_delivery_state.json").write_text(
        json.dumps({LEDGER_CURSOR_INITIALIZED_KEY: True, LEDGER_OFFSET_KEY: 0}),
        encoding="utf-8",
    )
    ledger_path = watch_dir / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps(
            {
                "ts": "2026-07-17T06:04:57+00:00",
                "event": "failed",
                "slug": "demo",
                "run_id": "run-1",
                "error": "format mismatch",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        event_notifications.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: UrlResponse(
            {"matched_routes": 1, "event_id": "server-event-1", "delivery_status": "queued"}
        ),
    )
    sink = HttpEventSink(HttpEventSinkConfig(api_url="https://events.invalid/api", api_token="test-token"))

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=datetime(2026, 7, 17, 6, 5, 57, tzinfo=UTC),
    )

    receipts = [
        json.loads(line) for line in (watch_dir / DELIVERY_RECEIPTS_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    lifecycle = next(record for record in receipts if record["source"] == "watch_lifecycle")
    assert lifecycle["accepted"] is True
    assert lifecycle["event_id"].startswith("evt-kaggle-autopilot-20260717T060457Z-")
    assert lifecycle["response_event_id"] == "server-event-1"
    assert lifecycle["response_status"] == "queued"
    assert lifecycle["matched_routes"] == 1
    assert lifecycle["ledger_offset"] == 0
    assert lifecycle["ledger_next_offset"] == ledger_path.stat().st_size
    assert _read_json_object(watch_dir / "event_delivery_state.json")[LEDGER_OFFSET_KEY] == ledger_path.stat().st_size


def test_read_json_object_returns_empty_for_missing_invalid_or_non_object_payload(tmp_path: Path) -> None:
    assert _read_json_object(tmp_path / "missing.json") == {}

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert _read_json_object(invalid) == {}

    array_payload = tmp_path / "array.json"
    array_payload.write_text("[]", encoding="utf-8")
    assert _read_json_object(array_payload) == {}


def test_build_autopilot_status_payload_reports_current_running_iteration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    (artifacts / "_watch").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (artifacts / "_watch" / "state.json").write_text(
        json.dumps({"active_slug": "demo", "active_run_id": "run-1", "last_status": "running"}),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "slug": "demo",
                "status": "running",
                "config": {
                    "max_iterations": 12,
                    "target_metric": "accuracy",
                    "score_source": "cv",
                    "submit": True,
                },
            }
        ),
        encoding="utf-8",
    )
    iter3 = run_dir / "iter-3"
    iter4 = run_dir / "iter-4"
    iter3.mkdir()
    iter4.mkdir()
    (iter3 / "iteration_state.json").write_text(
        json.dumps({"iteration_complete": True, "submit_phase_state": "deferred_non_improving"}),
        encoding="utf-8",
    )
    (iter3 / "metrics.json").write_text(
        json.dumps({"metric": "balanced_accuracy", "offline_value": 0.98, "score_source": "cv"}),
        encoding="utf-8",
    )
    (iter4 / "metrics.json").write_text(
        json.dumps({"metric": "balanced_accuracy", "offline_value": 0.97, "score_source": "cv"}),
        encoding="utf-8",
    )

    payload = build_autopilot_status_payload(
        artifacts_dir=artifacts,
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    assert payload["competition"] == "demo"
    assert payload["run_id"] == "run-1"
    assert payload["phase"] == "kernel_running"
    assert payload["current_iteration"] == 4
    assert payload["latest_completed_iteration"] == 3
    assert payload["latest_score"] == 0.97
    assert payload["best_score"] == 0.98
    assert payload["submit_phase_state"] == "pending"


def test_build_autopilot_status_payload_does_not_mix_best_metrics(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    (artifacts / "_watch").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (artifacts / "_watch" / "state.json").write_text(
        json.dumps({"active_slug": "demo", "active_run_id": "run-1", "last_status": "running"}),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "slug": "demo",
                "status": "running",
                "config": {"max_iterations": 3, "target_metric": "accuracy", "target_direction": "maximize"},
            }
        ),
        encoding="utf-8",
    )
    iter1 = run_dir / "iter-1"
    iter2 = run_dir / "iter-2"
    iter1.mkdir()
    iter2.mkdir()
    (iter1 / "metrics.json").write_text(
        json.dumps({"metric": "accuracy", "offline_value": 0.9805555556}),
        encoding="utf-8",
    )
    (iter2 / "metrics.json").write_text(
        json.dumps({"metric": "balanced_accuracy", "offline_value": 0.9800451920}),
        encoding="utf-8",
    )

    payload = build_autopilot_status_payload(
        artifacts_dir=artifacts,
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    assert payload["metric"] == "balanced_accuracy"
    assert payload["latest_score"] == 0.980045192
    assert payload["best_score"] == 0.980045192


def test_build_autopilot_status_payload_includes_submission_scores(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    (artifacts / "_watch").mkdir(parents=True)
    (artifacts / "demo" / "submissions").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (artifacts / "_watch" / "state.json").write_text(
        json.dumps({"active_slug": "demo", "active_run_id": "run-1", "last_status": "running"}),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "slug": "demo",
                "status": "running",
                "config": {"max_iterations": 3, "target_metric": "accuracy", "target_direction": "maximize"},
            }
        ),
        encoding="utf-8",
    )
    for iteration, offline_value in ((1, 0.7), (2, 0.8)):
        iter_dir = run_dir / f"iter-{iteration}"
        iter_dir.mkdir()
        (iter_dir / "metrics.json").write_text(
            json.dumps({"metric": "accuracy", "offline_value": offline_value}),
            encoding="utf-8",
        )
    (artifacts / "demo" / "submissions" / "ledger.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-04-23T00:00:00+00:00",
                        "event": "outcome",
                        "slug": "demo",
                        "run_id": "run-1",
                        "message": "kb run-1 i=1 offline=0.7",
                        "outcome": {"status": "complete", "score": 0.62, "rank": 20, "total_teams": 100},
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-04-23T01:00:00+00:00",
                        "event": "outcome",
                        "slug": "demo",
                        "run_id": "run-1",
                        "message": "kb run-1 i=2 offline=0.8",
                        "outcome": {"status": "complete", "score": 0.64, "rank": 12, "total_teams": 100},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_autopilot_status_payload(
        artifacts_dir=artifacts,
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    assert payload["latest_score"] == 0.8
    assert payload["best_score"] == 0.8
    assert payload["latest_submission_score"] == 0.64
    assert payload["best_submission_score"] == 0.64
    assert payload["submission_score_source"] == "submission_public_score"
    assert payload["latest_submission_iteration"] == 2
    assert payload["best_submission_iteration"] == 2
    assert payload["latest_submission_rank"] == 12
    assert payload["latest_submission_total_teams"] == 100
    assert payload["latest_submission_rank_percentile"] == 0.12
    assert payload["best_submission_rank"] == 12
    assert payload["best_submission_total_teams"] == 100
    assert payload["best_submission_rank_percentile"] == 0.12


def test_build_autopilot_status_payload_recomputes_stale_rank_guard_estimate(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    leaderboard_dir = artifacts / "demo" / "context" / "leaderboard"
    (artifacts / "_watch").mkdir(parents=True)
    leaderboard_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (artifacts / "_watch" / "state.json").write_text(
        json.dumps({"active_slug": "demo", "active_run_id": "run-1", "last_status": "running"}),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "slug": "demo",
                "status": "running",
                "config": {"max_iterations": 3, "target_metric": "rmse", "target_direction": "minimize"},
            }
        ),
        encoding="utf-8",
    )
    iter1 = run_dir / "iter-1"
    iter1.mkdir()
    (iter1 / "metrics.json").write_text(
        json.dumps(
            {
                "metric": "rmse",
                "offline_value": 0.3,
                "submission_score": 0.25,
                "rank_guard": {
                    "estimated_rank": 1,
                    "estimated_total_teams": 1,
                    "rank_estimate_source": "stale_fixture",
                },
            }
        ),
        encoding="utf-8",
    )
    (leaderboard_dir / "demo-publicleaderboard.csv").write_text(
        "Rank,Score\n1,0.100\n2,0.200\n3,0.300\n",
        encoding="utf-8",
    )

    payload = build_autopilot_status_payload(
        artifacts_dir=artifacts,
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    assert "latest_submission_rank" not in payload
    assert payload["latest_submission_estimated_rank"] == 3
    assert payload["latest_submission_estimated_total_teams"] == 3
    assert payload["latest_submission_rank_estimate_source"] == "cached_leaderboard_score_estimate"


def test_build_autopilot_status_payload_prefers_explicit_watch_phase(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    (artifacts / "_watch" / "kaggle_gpu").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    state_path = artifacts / "_watch" / "kaggle_gpu" / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "active_slug": "demo",
                "active_run_id": "run-1",
                "last_status": "running",
                "phase": "gpt_autofix_fixing",
                "phase_detail": "repairing write guard",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "run-1", "slug": "demo", "status": "running", "config": {"max_iterations": 4}}),
        encoding="utf-8",
    )

    payload = build_autopilot_status_payload(
        artifacts_dir=artifacts,
        watch_state_path=state_path,
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    assert payload["compute"] == "kaggle_gpu"
    assert payload["phase"] == "gpt_autofix_fixing"
    assert payload["phase_detail"] == "repairing write guard"
    assert "gpt autofix fixing" in payload["message"]


def test_build_autopilot_status_payload_prefers_active_recovery_over_failed_run_record(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    watch_dir = artifacts / "_watch"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    watch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (watch_dir / "state.json").write_text(
        json.dumps(
            {
                "active_slug": "demo",
                "active_run_id": "run-1",
                "last_status": "running",
                "phase": "gpt_submit_autofix_thinking",
                "phase_detail": "Oracle is analyzing the submit failure",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps({"status": "submit_failed", "config": {"max_iterations": 3}}),
        encoding="utf-8",
    )
    iter_dir = run_dir / "iter-1"
    iter_dir.mkdir()
    (iter_dir / "iteration_state.json").write_text(
        json.dumps({"iteration_complete": True, "submit_phase_state": "submit_failed"}),
        encoding="utf-8",
    )

    payload = build_autopilot_status_payload(
        artifacts_dir=artifacts,
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    assert payload["status"] == "running"
    assert payload["run_record_status"] == "submit_failed"
    assert payload["phase"] == "gpt_submit_autofix_thinking"
    assert event_notifications._event_type_for_snapshot(payload) == "autopilot.status"
    assert event_notifications._severity_for_snapshot(payload) == "info"
    assert "gpt submit autofix thinking" in payload["message"]


def test_event_dispatcher_replaces_transient_failure_with_active_autofix_status(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    watch_dir = artifacts / "_watch"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    watch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    state_path = watch_dir / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "active_slug": "demo",
                "active_run_id": "run-1",
                "last_status": "submit_failed",
                "phase": "submit_failed",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps({"status": "submit_failed", "config": {"max_iterations": 3}}),
        encoding="utf-8",
    )
    (watch_dir / "event_delivery_state.json").write_text(
        json.dumps({LEDGER_CURSOR_INITIALIZED_KEY: True, LEDGER_OFFSET_KEY: 0, "last_run_id": "run-1"}),
        encoding="utf-8",
    )
    sink = RecordingSink()
    first = datetime(2026, 4, 23, 0, 0, tzinfo=UTC)

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=first,
    )

    state_path.write_text(
        json.dumps(
            {
                "active_slug": "demo",
                "active_run_id": "run-1",
                "last_status": "running",
                "phase": "gpt_submit_autofix_thinking",
                "phase_detail": "Oracle is analyzing the submit failure",
            }
        ),
        encoding="utf-8",
    )
    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=first + timedelta(minutes=1),
    )

    assert [event["event_type"] for event in sink.events] == ["autopilot.failed", "autopilot.status"]
    assert [event["severity"] for event in sink.events] == ["error", "info"]
    assert sink.events[-1]["payload"]["status"] == "running"
    assert sink.events[-1]["payload"]["phase"] == "gpt_submit_autofix_thinking"
    assert sink.events[-1]["payload"]["coalesce_key"] == ("kaggle-autopilot:local_gpu:run:run-1")


def test_dispatch_events_once_suppresses_unchanged_idle_snapshot_even_after_heartbeat(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    (artifacts / "_watch").mkdir(parents=True)
    (artifacts / "_watch" / "state.json").write_text(json.dumps({"last_status": "idle"}), encoding="utf-8")
    sink = RecordingSink()
    first = datetime(2026, 4, 23, 0, 0, tzinfo=UTC)

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=first,
    )
    assert not dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=first + timedelta(minutes=5),
    )
    assert not dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=first + timedelta(minutes=31),
    )
    assert len(sink.events) == 1
    assert sink.events[0]["payload"]["coalesce_key"] == "kaggle-autopilot:local_gpu:idle"


def test_coalesce_key_changes_between_idle_and_active_competition(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    watch_dir = artifacts / "_watch"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    watch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    state_path = watch_dir / "state.json"
    state_path.write_text(json.dumps({"last_status": "idle"}), encoding="utf-8")

    idle_payload = build_autopilot_status_payload(
        artifacts_dir=artifacts,
        watch_state_path=state_path,
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    state_path.write_text(
        json.dumps({"active_slug": "demo", "active_run_id": "run-1", "last_status": "running"}),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "run-1", "slug": "demo", "status": "running", "config": {"max_iterations": 3}}),
        encoding="utf-8",
    )
    active_payload = build_autopilot_status_payload(
        artifacts_dir=artifacts,
        watch_state_path=state_path,
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    assert idle_payload["phase"] == "idle"
    assert active_payload["competition"] == "demo"
    assert idle_payload["coalesce_key"] == "kaggle-autopilot:local_gpu:idle"
    assert active_payload["coalesce_key"] == "kaggle-autopilot:local_gpu:run:run-1"
    assert active_payload["lease_expires_at"] == "2026-04-23T01:05:00+00:00"
    for private_key in ("host", "artifact_root", "run_dir"):
        assert private_key not in idle_payload
        assert private_key not in active_payload


def test_installation_id_is_opt_in(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_INSTALLATION_ID", "worker-01")
    payload = build_autopilot_status_payload(
        artifacts_dir=tmp_path / "artifacts",
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    assert payload["coalesce_key"] == "kaggle-autopilot:worker-01:local_gpu:idle"


def test_dispatch_events_once_reemits_active_snapshot_after_heartbeat(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    (artifacts / "_watch").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (artifacts / "_watch" / "state.json").write_text(
        json.dumps({"active_slug": "demo", "active_run_id": "run-1", "last_status": "running"}),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "run-1", "slug": "demo", "status": "running", "config": {"max_iterations": 3}}),
        encoding="utf-8",
    )
    sink = RecordingSink()
    first = datetime(2026, 4, 23, 0, 0, tzinfo=UTC)

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=first,
    )
    assert not dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=first + timedelta(minutes=5),
    )
    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=first + timedelta(minutes=31),
    )
    assert len(sink.events) == 2


def test_dispatch_events_once_marks_new_active_run_as_started(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-2"
    (artifacts / "_watch").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (artifacts / "_watch" / "state.json").write_text(
        json.dumps({"active_slug": "demo", "active_run_id": "run-2", "last_status": "running"}),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "run-2", "slug": "demo", "status": "running", "config": {"max_iterations": 3}}),
        encoding="utf-8",
    )
    sink = RecordingSink()

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=datetime(2026, 4, 23, 0, 0, tzinfo=UTC),
    )

    assert sink.events[0]["event_type"] == "autopilot.started"
    assert sink.events[0]["payload"]["coalesce_key"] == ("kaggle-autopilot:local_gpu:run:run-2")


def test_dispatch_events_once_creates_a_new_message_when_competition_changes(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    watch_dir = artifacts / "_watch"
    watch_dir.mkdir(parents=True)
    state_path = watch_dir / "state.json"
    for slug, run_id in (("first-comp", "run-1"), ("second-comp", "run-2")):
        run_dir = artifacts / slug / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps({"status": "running", "config": {"max_iterations": 3}}),
            encoding="utf-8",
        )
    sink = RecordingSink()
    first = datetime(2026, 4, 23, 0, 0, tzinfo=UTC)

    state_path.write_text(
        json.dumps({"active_slug": "first-comp", "active_run_id": "run-1", "last_status": "running"}),
        encoding="utf-8",
    )
    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=first,
    )

    state_path.write_text(
        json.dumps({"active_slug": "second-comp", "active_run_id": "run-2", "last_status": "running"}),
        encoding="utf-8",
    )
    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=first + timedelta(minutes=5),
    )

    assert [event["event_type"] for event in sink.events] == ["autopilot.started", "autopilot.started"]
    assert [event["payload"]["competition"] for event in sink.events] == ["first-comp", "second-comp"]
    assert [event["payload"]["coalesce_key"] for event in sink.events] == [
        "kaggle-autopilot:local_gpu:run:run-1",
        "kaggle-autopilot:local_gpu:run:run-2",
    ]


def test_dispatch_events_once_does_not_mark_planning_as_iteration_completed(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    (artifacts / "_watch").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (artifacts / "_watch" / "state.json").write_text(
        json.dumps(
            {
                "active_slug": "demo",
                "active_run_id": "run-1",
                "last_status": "running",
                "phase": "gpt_planning",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "run-1", "slug": "demo", "status": "running", "config": {"max_iterations": 3}}),
        encoding="utf-8",
    )
    sink = RecordingSink()

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=datetime(2026, 4, 23, 0, 0, tzinfo=UTC),
    )

    assert sink.events[0]["event_type"] == "autopilot.started"


def test_dispatch_events_once_reports_kaggle_gpu_sidecar_scope(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    main_run_dir = artifacts / "main-demo" / "runs" / "run-main"
    side_run_dir = artifacts / "side-demo" / "runs" / "run-side"
    (artifacts / "_watch" / "kaggle_gpu").mkdir(parents=True)
    main_run_dir.mkdir(parents=True)
    side_run_dir.mkdir(parents=True)
    (artifacts / "_watch" / "state.json").write_text(
        json.dumps({"active_slug": "main-demo", "active_run_id": "run-main", "last_status": "running"}),
        encoding="utf-8",
    )
    (artifacts / "_watch" / "kaggle_gpu" / "state.json").write_text(
        json.dumps({"active_slug": "side-demo", "active_run_id": "run-side", "last_status": "running"}),
        encoding="utf-8",
    )
    for run_dir, max_iterations in ((main_run_dir, 12), (side_run_dir, 4)):
        (run_dir / "run.json").write_text(
            json.dumps({"status": "running", "config": {"max_iterations": max_iterations}}),
            encoding="utf-8",
        )

    sink = RecordingSink()

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=datetime(2026, 4, 23, 0, 0, tzinfo=UTC),
    )

    assert len(sink.events) == 2
    payloads = [event["payload"] for event in sink.events]
    side_payload = next(payload for payload in payloads if payload["competition"] == "side-demo")
    assert side_payload["compute"] == "kaggle_gpu"
    assert side_payload["state_scope"] == "kaggle_gpu"
    assert side_payload["coalesce_key"] == "kaggle-autopilot:kaggle_gpu:run:run-side"


def test_dispatch_events_once_replays_lifecycle_events_from_watch_ledger(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    watch_dir = artifacts / "_watch"
    watch_dir.mkdir(parents=True)
    (watch_dir / "state.json").write_text(json.dumps({"last_status": "finished"}), encoding="utf-8")
    (watch_dir / "event_delivery_state.json").write_text(
        json.dumps({LEDGER_CURSOR_INITIALIZED_KEY: True, LEDGER_OFFSET_KEY: 0}),
        encoding="utf-8",
    )
    ledger_path = watch_dir / "ledger.jsonl"
    ledger_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-04-23T00:00:00+00:00",
                        "event": "started",
                        "slug": "first-comp",
                        "run_id": "run-1",
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-04-23T01:00:00+00:00",
                        "event": "finished",
                        "slug": "first-comp",
                        "run_id": "run-1",
                        "submission_status": "submitted",
                        "submission_url": "https://www.kaggle.com/competitions/first-comp/writeups/demo",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sink = RecordingSink()
    now = datetime(2026, 4, 23, 2, 0, tzinfo=UTC)

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=now,
    )
    assert not dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=now + timedelta(minutes=5),
    )

    assert [event["event_type"] for event in sink.events] == ["autopilot.started", "autopilot.finished"]
    assert sink.events[0]["payload"]["coalesce_key"] == ("kaggle-autopilot:local_gpu:run:run-1")
    assert sink.events[1]["payload"]["coalesce_key"] == ("kaggle-autopilot:local_gpu:run:run-1:finished")
    assert sink.events[1]["payload"]["submission_status"] == "submitted"
    assert sink.events[1]["payload"]["submission_url"].endswith("/writeups/demo")
    delivery_state = _read_json_object(watch_dir / "event_delivery_state.json")
    assert delivery_state[LEDGER_OFFSET_KEY] == ledger_path.stat().st_size
    assert delivery_state[DELIVERED_LIFECYCLE_KEYS] == [
        "kaggle-autopilot:local_gpu:autopilot.started:first-comp:run-1",
        "kaggle-autopilot:local_gpu:autopilot.finished:first-comp:run-1",
    ]


def test_dispatch_events_once_suppresses_duplicate_started_lifecycle(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    watch_dir = artifacts / "_watch"
    watch_dir.mkdir(parents=True)
    (watch_dir / "state.json").write_text(json.dumps({"last_status": "finished"}), encoding="utf-8")
    (watch_dir / "event_delivery_state.json").write_text(
        json.dumps({LEDGER_CURSOR_INITIALIZED_KEY: True, LEDGER_OFFSET_KEY: 0}),
        encoding="utf-8",
    )
    ledger_path = watch_dir / "ledger.jsonl"
    ledger_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "ts": f"2026-04-23T0{hour}:00:00+00:00",
                    "event": "started",
                    "slug": "demo",
                    "run_id": "run-1",
                }
            )
            for hour in (0, 1, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    sink = RecordingSink()

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=datetime(2026, 4, 23, 3, 0, tzinfo=UTC),
    )

    assert [event["event_type"] for event in sink.events] == ["autopilot.started"]
    assert _read_json_object(watch_dir / "event_delivery_state.json")[LEDGER_OFFSET_KEY] == ledger_path.stat().st_size


def test_started_lifecycle_is_immediately_updated_to_current_oracle_phase(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    watch_dir = artifacts / "_watch"
    run_dir = artifacts / "demo" / "runs" / "run-1"
    watch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (watch_dir / "state.json").write_text(
        json.dumps(
            {
                "active_slug": "demo",
                "active_run_id": "run-1",
                "last_status": "running",
                "phase": "oracle_strategy",
                "phase_detail": "Oracle Pro is producing the implementation strategy",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps({"status": "running", "config": {"max_iterations": 3}}),
        encoding="utf-8",
    )
    (watch_dir / "event_delivery_state.json").write_text(
        json.dumps({LEDGER_CURSOR_INITIALIZED_KEY: True, LEDGER_OFFSET_KEY: 0}),
        encoding="utf-8",
    )
    ledger_path = watch_dir / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps(
            {
                "ts": "2026-04-23T00:00:00+00:00",
                "event": "started",
                "slug": "demo",
                "run_id": "run-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sink = RecordingSink()

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=datetime(2026, 4, 23, 0, 1, tzinfo=UTC),
    )

    assert [event["event_type"] for event in sink.events] == ["autopilot.started", "autopilot.status"]
    assert sink.events[-1]["payload"]["phase"] == "oracle_strategy"
    assert sink.events[0]["payload"]["coalesce_key"] == sink.events[-1]["payload"]["coalesce_key"]


def test_historical_failure_replay_does_not_restart_current_run_notification(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    watch_dir = artifacts / "_watch"
    run_dir = artifacts / "current-comp" / "runs" / "current-run"
    watch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (watch_dir / "state.json").write_text(
        json.dumps(
            {
                "active_slug": "current-comp",
                "active_run_id": "current-run",
                "last_status": "running",
                "phase": "preparing_data",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    current_started_key = "kaggle-autopilot:local_gpu:autopilot.started:current-comp:current-run"
    (watch_dir / "event_delivery_state.json").write_text(
        json.dumps(
            {
                LEDGER_CURSOR_INITIALIZED_KEY: True,
                LEDGER_OFFSET_KEY: 0,
                DELIVERED_LIFECYCLE_KEYS: [current_started_key],
                "last_run_id": "current-run",
            }
        ),
        encoding="utf-8",
    )
    (watch_dir / "ledger.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-04-23T00:00:00+00:00",
                "event": "failed",
                "slug": "historical-comp",
                "run_id": "historical-run",
                "reason": "missing_competition_data",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sink = RecordingSink()

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=datetime(2026, 4, 23, 0, 1, tzinfo=UTC),
    )

    assert [event["event_type"] for event in sink.events] == ["autopilot.failed", "autopilot.status"]
    assert sink.events[-1]["payload"]["competition"] == "current-comp"
    assert sink.events[-1]["payload"]["phase"] == "preparing_data"


def test_dispatch_events_once_retries_lifecycle_without_advancing_cursor(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    watch_dir = artifacts / "_watch"
    run_dir = artifacts / "new-comp" / "runs" / "run-2"
    watch_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (watch_dir / "state.json").write_text(
        json.dumps({"active_slug": "new-comp", "active_run_id": "run-2", "last_status": "running"}),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps({"status": "running", "config": {"max_iterations": 3}}),
        encoding="utf-8",
    )
    (watch_dir / "event_delivery_state.json").write_text(
        json.dumps({LEDGER_CURSOR_INITIALIZED_KEY: True, LEDGER_OFFSET_KEY: 0}),
        encoding="utf-8",
    )
    ledger_path = watch_dir / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps(
            {
                "ts": "2026-04-23T00:00:00+00:00",
                "event": "started",
                "slug": "new-comp",
                "run_id": "run-2",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sink = FailingOnceSink()
    now = datetime(2026, 4, 23, 0, 5, tzinfo=UTC)

    assert not dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=now,
    )
    assert _read_json_object(watch_dir / "event_delivery_state.json")[LEDGER_OFFSET_KEY] == 0
    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=now + timedelta(minutes=5),
    )

    assert len(sink.events) == 3
    assert [event["event_type"] for event in sink.events] == [
        "autopilot.started",
        "autopilot.started",
        "autopilot.status",
    ]
    assert sink.events[0]["dedupe_key"] == sink.events[1]["dedupe_key"]
    assert sink.events[-1]["payload"]["coalesce_key"] == ("kaggle-autopilot:local_gpu:run:run-2")
    assert _read_json_object(watch_dir / "event_delivery_state.json")[LEDGER_OFFSET_KEY] == (ledger_path.stat().st_size)


def test_dispatch_events_once_initializes_ledger_cursor_without_replaying_history(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    watch_dir = artifacts / "_watch"
    watch_dir.mkdir(parents=True)
    (watch_dir / "state.json").write_text(json.dumps({"last_status": "idle"}), encoding="utf-8")
    ledger_path = watch_dir / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps(
            {
                "ts": "2026-01-01T00:00:00+00:00",
                "event": "started",
                "slug": "old-comp",
                "run_id": "old-run",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sink = RecordingSink()

    assert dispatch_events_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        sink=sink,
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    assert [event["event_type"] for event in sink.events] == ["autopilot.status"]
    delivery_state = _read_json_object(watch_dir / "event_delivery_state.json")
    assert delivery_state[LEDGER_CURSOR_INITIALIZED_KEY] is True
    assert delivery_state[LEDGER_OFFSET_KEY] == ledger_path.stat().st_size
