from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kagglebot.discord_notifications import (
    DiscordEventNotifier,
    _read_json_object,
    build_autopilot_status_payload,
    run_discord_notifier_once,
)


class RecordingNotifier(DiscordEventNotifier):
    def __init__(self) -> None:
        super().__init__(None)
        self.events: list[dict[str, object]] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, **kwargs):  # type: ignore[no-untyped-def]
        self.events.append(kwargs)
        return True


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
                "config": {"max_iterations": 12, "target_metric": "accuracy", "score_source": "cv"},
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


def test_run_discord_notifier_once_suppresses_unchanged_idle_snapshot_even_after_heartbeat(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    (artifacts / "_watch").mkdir(parents=True)
    (artifacts / "_watch" / "state.json").write_text(json.dumps({"last_status": "idle"}), encoding="utf-8")
    notifier = RecordingNotifier()
    first = datetime(2026, 4, 23, 0, 0, tzinfo=UTC)

    assert run_discord_notifier_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        notifier=notifier,
        now=first,
    )
    assert not run_discord_notifier_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        notifier=notifier,
        now=first + timedelta(minutes=5),
    )
    assert not run_discord_notifier_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        notifier=notifier,
        now=first + timedelta(minutes=31),
    )
    assert len(notifier.events) == 1
    assert notifier.events[0]["payload"]["discord_update_key"] == "kaggle-autopilot:lab_rdp:local_gpu"


def test_discord_update_key_is_stable_between_idle_and_active_competition(tmp_path: Path) -> None:
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
    assert idle_payload["discord_update_key"] == active_payload["discord_update_key"]


def test_run_discord_notifier_once_reemits_active_snapshot_after_heartbeat(tmp_path: Path) -> None:
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
    notifier = RecordingNotifier()
    first = datetime(2026, 4, 23, 0, 0, tzinfo=UTC)

    assert run_discord_notifier_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        notifier=notifier,
        now=first,
    )
    assert not run_discord_notifier_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        notifier=notifier,
        now=first + timedelta(minutes=5),
    )
    assert run_discord_notifier_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        notifier=notifier,
        now=first + timedelta(minutes=31),
    )
    assert len(notifier.events) == 2


def test_run_discord_notifier_once_marks_new_active_run_as_started(tmp_path: Path) -> None:
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
    notifier = RecordingNotifier()

    assert run_discord_notifier_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        notifier=notifier,
        now=datetime(2026, 4, 23, 0, 0, tzinfo=UTC),
    )

    assert notifier.events[0]["event_type"] == "autopilot.started"


def test_run_discord_notifier_once_does_not_mark_planning_as_iteration_completed(tmp_path: Path) -> None:
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
    notifier = RecordingNotifier()

    assert run_discord_notifier_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        notifier=notifier,
        now=datetime(2026, 4, 23, 0, 0, tzinfo=UTC),
    )

    assert notifier.events[0]["event_type"] == "autopilot.started"


def test_run_discord_notifier_once_reports_kaggle_gpu_sidecar_scope(tmp_path: Path) -> None:
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

    notifier = RecordingNotifier()

    assert run_discord_notifier_once(
        artifacts_dir=artifacts,
        heartbeat_sec=1800,
        notifier=notifier,
        now=datetime(2026, 4, 23, 0, 0, tzinfo=UTC),
    )

    assert len(notifier.events) == 2
    payloads = [event["payload"] for event in notifier.events]
    side_payload = next(payload for payload in payloads if payload["competition"] == "side-demo")
    assert side_payload["compute"] == "kaggle_gpu"
    assert side_payload["state_scope"] == "kaggle_gpu"
    assert side_payload["discord_update_key"] == "kaggle-autopilot:lab_rdp:kaggle_gpu"
