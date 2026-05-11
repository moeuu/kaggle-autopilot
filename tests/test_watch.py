from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from kagglebot.cli import app
from kagglebot.exceptions import KaggleCliResourceError, KernelCapacityError
from kagglebot.kaggle_api import EnteredCompetition
from kagglebot.supervisor import (
    WatchConfig,
    WatchLedger,
    _parse_kaggle_gpu_quota_text,
    _read_kaggle_gpu_quota_file,
    run_watch_once,
    select_next_competition,
)


def _competition(
    slug: str,
    *,
    category: str = "Playground",
    reward: str = "",
    submissions_disabled: bool = False,
    team_count: int | None = 100,
    deadline: datetime | None = None,
    new_entrant_deadline: datetime | None = None,
    evaluation_metric: str = "auc",
) -> EnteredCompetition:
    return EnteredCompetition(
        slug=slug,
        title=slug,
        url=f"https://www.kaggle.com/competitions/{slug}",
        category=category,
        reward=reward,
        evaluation_metric=evaluation_metric,
        deadline=deadline,
        enabled_date=None,
        new_entrant_deadline=new_entrant_deadline,
        merger_deadline=None,
        team_count=team_count,
        max_daily_submissions=5,
        is_kernels_submissions_only=False,
        submissions_disabled=submissions_disabled,
        source="test",
    )


def _config(tmp_path: Path, **overrides: object) -> WatchConfig:
    base = WatchConfig(
        workdir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        compute="local_gpu",
        accelerator="gpu",
        strict_accelerator=False,
        kaggle_username=None,
        kernel_name=None,
        internet="on",
        time_budget_min=None,
        seed=None,
        score_source=None,
        holdout_frac=None,
        cv_folds=None,
        max_iterations=5,
        max_total_min=None,
        patience=None,
        min_improvement=None,
        submit_policy="improved",
        verify_cmd="uv run pytest -q",
        auto_eval_spec=False,
        page_limit=5,
        allow_slugs=(),
        block_slugs=(),
        cooldown_hours=24.0,
        dry_run=False,
        force=True,
    )
    return base.__class__(**{**base.__dict__, **overrides})


def test_select_next_competition_filters_disabled_and_blocked(monkeypatch, tmp_path: Path) -> None:
    candidates = [
        _competition("disabled", submissions_disabled=True),
        _competition("blocked"),
        _competition("eligible"),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)

    config = _config(tmp_path, block_slugs=("blocked",))

    selected = select_next_competition(config)

    assert [item.slug for item in selected] == ["eligible"]


def test_select_next_competition_honors_allowlist(monkeypatch, tmp_path: Path) -> None:
    candidates = [_competition("first"), _competition("second")]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)

    selected = select_next_competition(_config(tmp_path, allow_slugs=("second",)))

    assert [item.slug for item in selected] == ["second"]


def test_select_next_competition_skips_recent_failures(monkeypatch, tmp_path: Path) -> None:
    candidates = [_competition("failed-recently"), _competition("fresh")]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(tmp_path)
    WatchLedger(config.ledger_path).append("failed", slug="failed-recently", run_id="run-1")

    selected = select_next_competition(config)

    assert [item.slug for item in selected] == ["fresh"]


def test_select_next_competition_skips_recent_resource_blocks(monkeypatch, tmp_path: Path) -> None:
    candidates = [_competition("resource-heavy"), _competition("fresh")]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(tmp_path)
    WatchLedger(config.ledger_path).append("resource_blocked", slug="resource-heavy", run_id="run-1")

    selected = select_next_competition(config)

    assert [item.slug for item in selected] == ["fresh"]


def test_select_next_competition_manual_next_bypasses_cooldown_and_resource_block(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidates = [_competition("fresh", reward="1000 usd"), _competition("birdclef-2026", reward="50000 usd")]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(tmp_path)
    config.watch_dir.mkdir(parents=True)
    (config.watch_dir / "next_slug.txt").write_text("birdclef-2026\n", encoding="utf-8")
    WatchLedger(config.ledger_path).append("failed", slug="birdclef-2026", run_id="old-run")
    WatchLedger(config.ledger_path).append("resource_blocked", slug="birdclef-2026", run_id="old-run")

    selected = select_next_competition(config)

    assert [item.slug for item in selected] == ["birdclef-2026"]


def test_select_next_competition_prioritizes_never_submitted(monkeypatch, tmp_path: Path) -> None:
    candidates = [_competition("submitted"), _competition("never-submitted")]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(tmp_path)
    ledger_path = config.artifacts_dir / "submitted" / "submissions" / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps({"event": "submit", "slug": "submitted", "ts": "2026-04-22T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )

    selected = select_next_competition(config)

    assert [item.slug for item in selected][:2] == ["never-submitted", "submitted"]


def test_select_next_competition_prioritizes_never_submitted_before_money_prizes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidates = [
        _competition("no-prize-never-submitted", reward=""),
        _competition("prize-submitted", reward="$25,000"),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(tmp_path)
    ledger_path = config.artifacts_dir / "prize-submitted" / "submissions" / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps({"event": "submit", "slug": "prize-submitted", "ts": "2026-04-22T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )

    selected = select_next_competition(config)

    assert [item.slug for item in selected][:2] == ["no-prize-never-submitted", "prize-submitted"]


def test_select_next_competition_parses_kaggle_usd_reward_text(monkeypatch, tmp_path: Path) -> None:
    candidates = [
        _competition("no-prize", reward=""),
        _competition("usd-prize", reward="1,000 Usd"),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)

    selected = select_next_competition(_config(tmp_path))

    assert [item.slug for item in selected][:2] == ["usd-prize", "no-prize"]


def test_select_next_competition_excludes_late_submit_competitions(monkeypatch, tmp_path: Path) -> None:
    now = datetime.now(UTC)
    candidates = [
        _competition("late-prize", reward="$25,000", deadline=now - timedelta(days=1)),
        _competition("open-prize", reward="$10,000", deadline=now + timedelta(days=7)),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)

    selected = select_next_competition(_config(tmp_path))

    assert [item.slug for item in selected] == ["open-prize"]


def test_select_next_competition_prioritizes_medal_prize_and_near_deadline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    candidates = [
        _competition("no-prize-near", reward="", category="Playground", deadline=now + timedelta(days=2)),
        _competition(
            "prize-far",
            reward="$100,000",
            category="Featured",
            deadline=now + timedelta(days=60),
            team_count=500,
        ),
        _competition(
            "medal-prize-near",
            reward="$25,000",
            category="Featured",
            deadline=now + timedelta(days=5),
            team_count=500,
        ),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)

    selected = select_next_competition(_config(tmp_path))

    assert [item.slug for item in selected][:3] == ["medal-prize-near", "prize-far", "no-prize-near"]


def test_select_next_competition_ignores_non_money_rewards(monkeypatch, tmp_path: Path) -> None:
    candidates = [_competition("submitted-swag", reward="Swag"), _competition("never-submitted", reward="")]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(tmp_path)
    ledger_path = config.artifacts_dir / "submitted-swag" / "submissions" / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps({"event": "submit", "slug": "submitted-swag", "ts": "2026-04-22T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )

    selected = select_next_competition(config)

    assert [item.slug for item in selected][:2] == ["never-submitted", "submitted-swag"]


def test_select_next_competition_prioritizes_submitted_competition_with_more_rank_headroom(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidates = [_competition("good-rank"), _competition("poor-rank")]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(tmp_path)
    for slug, rank, total in (("good-rank", 5, 100), ("poor-rank", 80, 100)):
        ledger_path = config.artifacts_dir / slug / "submissions" / "ledger.jsonl"
        ledger_path.parent.mkdir(parents=True)
        ledger_path.write_text(
            "\n".join(
                [
                    json.dumps({"event": "submit", "slug": slug, "ts": "2026-04-22T00:00:00+00:00"}),
                    json.dumps(
                        {
                            "event": "outcome",
                            "slug": slug,
                            "ts": "2026-04-22T00:01:00+00:00",
                            "outcome": {"rank": rank, "total_teams": total},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    selected = select_next_competition(config)

    assert [item.slug for item in selected][:2] == ["poor-rank", "good-rank"]


def test_select_next_competition_enriches_submitted_rank_from_leaderboard_score(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidates = [_competition("good-rank"), _competition("poor-rank")]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)

    def fake_rank_for_score(slug: str, output_dir: Path, *, score: float, direction: str, dry_run: bool) -> dict:
        assert output_dir.name == "context"
        assert direction == "maximize"
        assert dry_run is False
        if slug == "good-rank":
            assert score == 0.95
            return {"rank": 5, "total_teams": 100, "rank_percentile": 0.05}
        assert slug == "poor-rank"
        assert score == 0.8
        return {"rank": 80, "total_teams": 100, "rank_percentile": 0.8}

    monkeypatch.setattr("kagglebot.supervisor.leaderboard_rank_for_score", fake_rank_for_score)
    config = _config(tmp_path)
    for slug, score in (("good-rank", 0.95), ("poor-rank", 0.8)):
        context_dir = config.artifacts_dir / slug / "context"
        context_dir.mkdir(parents=True)
        (context_dir / "evaluation_spec.json").write_text(
            json.dumps({"metric_name": "auc", "direction": "maximize"}),
            encoding="utf-8",
        )
        ledger_path = config.artifacts_dir / slug / "submissions" / "ledger.jsonl"
        ledger_path.parent.mkdir(parents=True)
        ledger_path.write_text(
            "\n".join(
                [
                    json.dumps({"event": "submit", "slug": slug, "ts": "2026-04-22T00:00:00+00:00"}),
                    json.dumps(
                        {
                            "event": "outcome",
                            "slug": slug,
                            "ts": "2026-04-22T00:01:00+00:00",
                            "outcome": {"status": "complete", "score": score},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    selected = select_next_competition(config)

    assert [item.slug for item in selected][:2] == ["poor-rank", "good-rank"]


def test_lightweight_sidecar_selection_skips_active_and_slow_candidates(monkeypatch, tmp_path: Path) -> None:
    now = datetime.now(UTC)
    candidates = [
        _competition("active", reward="$5,000", category="Featured", deadline=now + timedelta(days=10)),
        _competition(
            "slow-image-detection",
            reward="$5,000",
            category="Featured",
            deadline=now + timedelta(days=10),
            evaluation_metric="mAP@50",
        ),
        _competition(
            "fast-stock-prediction",
            reward="$5,000",
            category="Featured",
            deadline=now + timedelta(days=10),
            evaluation_metric="Mean Squared Error",
        ),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(
        tmp_path,
        state_scope="kaggle_gpu",
        lightweight_only=True,
        lightweight_max_training_min=120,
    )
    main_state = config.artifacts_dir / "_watch" / "state.json"
    main_state.parent.mkdir(parents=True)
    run_dir = config.artifacts_dir / "active" / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    main_state.write_text(json.dumps({"active_slug": "active", "active_run_id": "run-1"}), encoding="utf-8")

    selected = select_next_competition(config)

    assert [item.slug for item in selected] == ["fast-stock-prediction"]


def test_lightweight_sidecar_does_not_prefer_getting_started_over_prize_competitions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    candidates = [
        _competition(
            "getting-started-light",
            category="Getting Started",
            reward="",
            deadline=now + timedelta(days=365),
        ),
        _competition(
            "featured-prize-light",
            category="Featured",
            reward="$5,000",
            deadline=now + timedelta(days=10),
            team_count=200,
        ),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    selected = select_next_competition(
        _config(
            tmp_path,
            state_scope="kaggle_gpu",
            lightweight_only=True,
            lightweight_max_training_min=180,
        )
    )

    assert [item.slug for item in selected][:2] == ["featured-prize-light", "getting-started-light"]


def test_run_watch_once_marks_kaggle_gpu_capacity_as_no_capacity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "kagglebot.supervisor.list_entered_competitions",
        lambda **kwargs: [_competition("demo")],
    )
    monkeypatch.setattr("kagglebot.supervisor._prepare_competition", lambda **kwargs: None)

    def fake_run_autopilot(config) -> None:  # noqa: ARG001
        raise KernelCapacityError("limit", ["kaggle", "kernels", "push"], output="maximum batch gpu session count")

    monkeypatch.setattr("kagglebot.supervisor.run_autopilot", fake_run_autopilot)

    result = run_watch_once(_config(tmp_path, compute="kaggle_gpu", state_scope="kaggle_gpu"))

    assert result.status == "no_capacity"
    assert result.slug == "demo"


def test_parse_kaggle_gpu_quota_text_available_of_total() -> None:
    quota = _parse_kaggle_gpu_quota_text("14h 36m available of 30h", source="test")

    assert quota is not None
    assert quota.available_minutes == 876
    assert quota.total_minutes == 1800
    assert quota.used_minutes == 924
    assert quota.source == "test"


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

    assert _read_kaggle_gpu_quota_file(quota_path) is None


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

    quota = _read_kaggle_gpu_quota_file(quota_path)

    assert quota is not None
    assert quota.available_minutes == 1800


def test_run_watch_once_blocks_new_kaggle_gpu_competition_when_quota_low(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KAGGLEBOT_KAGGLE_GPU_QUOTA_TEXT", "14h 36m available of 30h")

    def fail_list_entered_competitions(**kwargs):  # noqa: ANN003, ARG001
        raise AssertionError("quota guard should run before candidate selection")

    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", fail_list_entered_competitions)

    config = _config(
        tmp_path,
        compute="kaggle_gpu",
        state_scope="kaggle_gpu",
        kaggle_gpu_min_available_minutes_for_new_competition=900,
    )

    result = run_watch_once(config)

    assert result.status == "no_capacity"
    assert result.slug is None
    assert result.reason == "kaggle_gpu_quota_low"
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["last_status"] == "no_capacity"
    assert state["phase"] == "kaggle_gpu_quota_low"
    assert state["available_minutes"] == 876
    assert state["threshold_minutes"] == 900


def test_run_watch_once_blocks_new_kaggle_gpu_competition_when_quota_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_list_entered_competitions(**kwargs):  # noqa: ANN003, ARG001
        raise AssertionError("quota guard should run before candidate selection")

    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", fail_list_entered_competitions)

    config = _config(
        tmp_path,
        compute="kaggle_gpu",
        state_scope="kaggle_gpu",
        kaggle_gpu_min_available_minutes_for_new_competition=900,
    )

    result = run_watch_once(config)

    assert result.status == "no_capacity"
    assert result.slug is None
    assert result.reason == "kaggle_gpu_quota_unavailable"


def test_run_watch_once_resumes_active_kaggle_gpu_run_when_quota_low(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_KAGGLE_GPU_QUOTA_TEXT", "14h 36m available of 30h")
    config = _config(
        tmp_path,
        compute="kaggle_gpu",
        state_scope="kaggle_gpu",
        kaggle_gpu_min_available_minutes_for_new_competition=900,
    )
    run_id = "20260422T000000Z-abcd1234"
    run_dir = config.artifacts_dir / "demo" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(
        json.dumps(
            {
                "active_slug": "demo",
                "active_run_id": run_id,
                "started_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    def fail_list_entered_competitions(**kwargs):  # noqa: ANN003, ARG001
        raise AssertionError("resuming active runs should not select new candidates")

    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", fail_list_entered_competitions)
    monkeypatch.setattr("kagglebot.supervisor._prepare_competition", lambda **kwargs: None)
    captured: dict[str, object] = {}

    def fake_run_autopilot(config) -> None:
        captured["slug"] = config.slug
        captured["run_id"] = config.run_id

    monkeypatch.setattr("kagglebot.supervisor.run_autopilot", fake_run_autopilot)

    result = run_watch_once(config)

    assert result.status == "finished"
    assert result.slug == "demo"
    assert captured == {"slug": "demo", "run_id": None}


def test_run_watch_once_dry_run_does_not_call_autopilot(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "kagglebot.supervisor.list_entered_competitions",
        lambda **kwargs: [_competition("demo")],
    )
    calls = {"autopilot": 0}
    monkeypatch.setattr("kagglebot.supervisor.run_autopilot", lambda config: calls.update(autopilot=1))

    result = run_watch_once(_config(tmp_path, dry_run=True))

    assert result.status == "dry_run"
    assert result.slug == "demo"
    assert calls["autopilot"] == 0


def test_run_watch_once_passes_improved_submit_policy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "kagglebot.supervisor.list_entered_competitions",
        lambda **kwargs: [_competition("demo")],
    )
    monkeypatch.setattr("kagglebot.supervisor._prepare_competition", lambda **kwargs: None)
    captured: dict[str, object] = {}

    def fake_run_autopilot(config) -> None:
        captured["submit_policy"] = config.submit_policy
        captured["submit"] = config.submit
        captured["force_submit"] = config.force_submit

    monkeypatch.setattr("kagglebot.supervisor.run_autopilot", fake_run_autopilot)

    result = run_watch_once(_config(tmp_path))

    assert result.status == "finished"
    assert captured == {"submit_policy": "improved", "submit": True, "force_submit": False}


def test_run_watch_once_caps_iterations_to_plan_max(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "kagglebot.supervisor.list_entered_competitions",
        lambda **kwargs: [_competition("demo")],
    )

    def fake_prepare(**kwargs) -> None:
        paths = kwargs["paths"]
        paths.base_dir.mkdir(parents=True, exist_ok=True)
        paths.plan_path.write_text(json.dumps({"max_iterations": 3}), encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_autopilot(config) -> None:
        captured["max_iterations"] = config.max_iterations

    monkeypatch.setattr("kagglebot.supervisor._prepare_competition", fake_prepare)
    monkeypatch.setattr("kagglebot.supervisor.run_autopilot", fake_run_autopilot)

    result = run_watch_once(_config(tmp_path, max_iterations=5))

    assert result.status == "finished"
    assert captured == {"max_iterations": 3}


def test_run_watch_once_resumes_active_run(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_id = "20260422T000000Z-abcd1234"
    run_dir = config.artifacts_dir / "demo" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(
        json.dumps(
            {
                "active_slug": "demo",
                "active_run_id": run_id,
                "started_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("kagglebot.supervisor._prepare_competition", lambda **kwargs: None)
    captured: dict[str, object] = {}

    def fake_run_autopilot(config) -> None:
        captured["run_id"] = config.run_id

    monkeypatch.setattr("kagglebot.supervisor.run_autopilot", fake_run_autopilot)

    result = run_watch_once(config)

    assert result.status == "finished"
    assert result.slug == "demo"
    assert captured["run_id"] is None


def test_run_watch_once_failure_is_recorded_and_next_cycle_selects_new_competition(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "kagglebot.supervisor.list_entered_competitions",
        lambda **kwargs: [_competition("first"), _competition("second")],
    )
    monkeypatch.setattr("kagglebot.supervisor._prepare_competition", lambda **kwargs: None)
    attempts: list[str] = []

    def fake_run_autopilot(config) -> None:
        attempts.append(config.slug)
        if config.slug == "first":
            raise RuntimeError("boom")

    monkeypatch.setattr("kagglebot.supervisor.run_autopilot", fake_run_autopilot)
    config = _config(tmp_path)

    first = run_watch_once(config)
    second = run_watch_once(config)

    assert first.status == "failed"
    assert first.slug == "first"
    assert second.status == "finished"
    assert second.slug == "second"
    assert attempts == ["first", "second"]


def test_run_watch_once_consumes_manual_next_slug(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "kagglebot.supervisor.list_entered_competitions",
        lambda **kwargs: [_competition("first"), _competition("birdclef-2026")],
    )
    monkeypatch.setattr("kagglebot.supervisor._prepare_competition", lambda **kwargs: None)
    monkeypatch.setattr("kagglebot.supervisor.run_autopilot", lambda config: None)
    config = _config(tmp_path)
    config.watch_dir.mkdir(parents=True)
    next_slug_path = config.watch_dir / "next_slug.txt"
    next_slug_path.write_text("birdclef-2026\n", encoding="utf-8")

    result = run_watch_once(config)

    assert result.status == "finished"
    assert result.slug == "birdclef-2026"
    assert not next_slug_path.exists()
    records = WatchLedger(config.ledger_path).records()
    assert any(
        record.get("event") == "selected"
        and record.get("slug") == "birdclef-2026"
        and record.get("reason") == "manual_next_slug"
        for record in records
    )


def test_run_watch_once_resource_error_records_resource_block(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "kagglebot.supervisor.list_entered_competitions",
        lambda **kwargs: [_competition("resource-heavy"), _competition("fresh")],
    )

    def fake_prepare_competition(**kwargs) -> None:  # noqa: ANN003
        raise KaggleCliResourceError("resource guard", ["kaggle"], exit_code=-9)

    monkeypatch.setattr("kagglebot.supervisor._prepare_competition", fake_prepare_competition)
    config = _config(tmp_path)

    result = run_watch_once(config)
    selected_after = select_next_competition(config)

    records = WatchLedger(config.ledger_path).records()
    assert result.status == "skipped"
    assert result.reason == "kaggle_cli_resource_limit"
    assert any(record.get("event") == "resource_blocked" for record in records)
    assert [item.slug for item in selected_after] == ["fresh"]


def test_watch_cli_requires_force_for_enabled_submissions(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "watch",
            "--once",
        ],
    )

    assert result.exit_code == 2
    assert "without" in result.output
    assert "--force" in result.output


def test_watch_cli_dry_run_once(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_watch_once(config):
        captured["submit_policy"] = config.submit_policy
        captured["compute"] = config.compute
        captured["max_iterations"] = config.max_iterations
        return type("Result", (), {"status": "dry_run", "slug": "demo", "run_id": "run-1"})()

    monkeypatch.setattr("kagglebot.cli.run_watch_once", fake_run_watch_once)

    result = CliRunner().invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--dry-run",
            "watch",
            "--once",
        ],
    )

    assert result.exit_code == 0
    assert captured == {"submit_policy": "improved", "compute": "local_gpu", "max_iterations": 5}


def test_watch_kaggle_gpu_sidecar_cli_builds_lightweight_config(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_watch_once(config):
        captured["compute"] = config.compute
        captured["state_scope"] = config.state_scope
        captured["lightweight_only"] = config.lightweight_only
        captured["lightweight_max_data_bytes"] = config.lightweight_max_data_bytes
        captured["lightweight_max_training_min"] = config.lightweight_max_training_min
        captured["time_budget_min"] = config.time_budget_min
        captured["max_iterations"] = config.max_iterations
        captured["max_total_min"] = config.max_total_min
        captured["kaggle_gpu_min_available_minutes_for_new_competition"] = (
            config.kaggle_gpu_min_available_minutes_for_new_competition
        )
        return type("Result", (), {"status": "dry_run", "slug": "demo", "run_id": "run-1"})()

    monkeypatch.setattr("kagglebot.cli.run_watch_once", fake_run_watch_once)

    result = CliRunner().invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "watch-kaggle-gpu-sidecar",
            "--once",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "compute": "kaggle_gpu",
        "state_scope": "kaggle_gpu",
        "lightweight_only": True,
        "lightweight_max_data_bytes": None,
        "lightweight_max_training_min": 600,
        "time_budget_min": 600,
        "max_iterations": 3,
        "max_total_min": 1800,
        "kaggle_gpu_min_available_minutes_for_new_competition": 900,
    }


def test_watch_kaggle_gpu_sidecar_cli_allows_disabling_new_comp_quota_guard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_watch_once(config):
        captured["kaggle_gpu_min_available_minutes_for_new_competition"] = (
            config.kaggle_gpu_min_available_minutes_for_new_competition
        )
        return type("Result", (), {"status": "dry_run", "slug": "demo", "run_id": "run-1"})()

    monkeypatch.setattr("kagglebot.cli.run_watch_once", fake_run_watch_once)

    result = CliRunner().invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "watch-kaggle-gpu-sidecar",
            "--once",
            "--min-gpu-quota-hours-for-new-comp",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert captured == {"kaggle_gpu_min_available_minutes_for_new_competition": None}
