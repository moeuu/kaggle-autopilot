from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

import kagglebot.supervisor as supervisor
from kagglebot.cli import app
from kagglebot.exceptions import KaggleCliResourceError, KernelCapacityError, MissingCompetitionDataError
from kagglebot.kaggle_api import EnteredCompetition
from kagglebot.paths import CompetitionPaths
from kagglebot.supervisor import (
    WatchConfig,
    WatchLedger,
    _build_autopilot_config,
    _estimate_training_minutes,
    _plan_max_iterations,
    _reconcile_active_writeup_submission,
    run_watch_once,
    select_next_competition,
)

pytestmark = pytest.mark.slow


def _competition(
    slug: str,
    *,
    title: str | None = None,
    category: str = "Playground",
    reward: str = "",
    submissions_disabled: bool = False,
    team_count: int | None = 100,
    deadline: datetime | None = None,
    new_entrant_deadline: datetime | None = None,
    evaluation_metric: str = "auc",
    awards_points: bool | None = None,
) -> EnteredCompetition:
    return EnteredCompetition(
        slug=slug,
        title=title or slug,
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
        awards_points=(
            awards_points
            if awards_points is not None
            else category.strip().lower() in {"featured", "research", "community"} and bool(reward)
        ),
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


def test_published_self_improvement_restarts_watch_process() -> None:
    calls: list[tuple[str, list[str]]] = []

    restarted = supervisor._restart_after_published_self_improvement(
        {
            "status": "written",
            "codex_improvement": {
                "status": "completed",
                "publish": {"status": "pushed", "commit": "abc123"},
            },
        },
        dry_run=False,
        execv_func=lambda executable, argv: calls.append((executable, argv)),
    )

    assert restarted is True
    assert len(calls) == 1
    assert calls[0][1][0] == calls[0][0]


def test_unpublished_self_improvement_does_not_restart_watch_process() -> None:
    restarted = supervisor._restart_after_published_self_improvement(
        {
            "status": "written",
            "codex_improvement": {
                "status": "completed",
                "publish": {"status": "disabled"},
            },
        },
        dry_run=False,
        execv_func=lambda *_args: (_ for _ in ()).throw(AssertionError("must not restart")),
    )

    assert restarted is False


def test_repository_head_change_restarts_watch_even_without_publish_result(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(supervisor, "_repository_head", lambda _root: "new-head")

    restarted = supervisor._restart_after_repository_head_change(  # noqa: SLF001
        repository_root=tmp_path,
        loaded_head="old-head",
        dry_run=False,
        execv_func=lambda executable, argv: calls.append((executable, argv)),
    )

    assert restarted is True
    assert len(calls) == 1


def test_repository_head_change_does_not_restart_dry_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(supervisor, "_repository_head", lambda _root: "new-head")

    restarted = supervisor._restart_after_repository_head_change(  # noqa: SLF001
        repository_root=tmp_path,
        loaded_head="old-head",
        dry_run=True,
        execv_func=lambda *_args: (_ for _ in ()).throw(AssertionError("must not restart")),
    )

    assert restarted is False


def test_published_self_improvement_schedules_one_preflight_retry(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = {
        "status": "written",
        "codex_improvement": {
            "status": "completed",
            "publish": {"status": "pushed", "commit": "abc123"},
        },
        "latest_watch_incident": {
            "slug": "demo",
            "run_id": "run-preflight",
            "run_directory_exists": False,
            "phase": "preparing_data",
            "fingerprint": "failure-123",
            "cause_tags": ["orchestration_preflight_failure"],
        },
    }

    scheduled = supervisor._schedule_preflight_retry_after_published_self_improvement(  # noqa: SLF001
        config,
        result,
    )

    assert scheduled is True
    state = supervisor.load_watch_state(config.state_path)
    assert state["active_slug"] == "demo"
    assert state["active_run_id"] == "run-preflight"
    assert state["reason"] == "verified_self_improvement_preflight_retry"
    ledger = WatchLedger(config.ledger_path).records()
    assert ledger[-1]["event"] == "self_improvement_retry_scheduled"
    assert ledger[-1]["incident_fingerprint"] == "failure-123"


def test_published_self_improvement_schedules_fresh_run_for_leaderboard_anomaly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(supervisor, "new_run_id", lambda: "run-repaired")
    result = {
        "status": "written",
        "codex_improvement": {
            "status": "completed",
            "publish": {"status": "pushed", "commit": "abc123"},
        },
        "latest_leaderboard_anomaly": {
            "slug": "demo",
            "run_id": "run-bottom",
            "fingerprint": "bottom-signals-123",
            "severity": "critical",
        },
    }

    scheduled = supervisor._schedule_preflight_retry_after_published_self_improvement(  # noqa: SLF001
        config,
        result,
    )

    assert scheduled is True
    state = supervisor.load_watch_state(config.state_path)
    assert state["active_slug"] == "demo"
    assert state["active_run_id"] == "run-repaired"
    assert state["original_run_id"] == "run-bottom"
    assert state["reason"] == "verified_self_improvement_leaderboard_anomaly_retry"
    ledger = WatchLedger(config.ledger_path).records()
    assert ledger[-1]["reason"] == "verified_leaderboard_anomaly_repair"
    assert ledger[-1]["anomaly_fingerprint"] == "bottom-signals-123"


def test_watch_failure_incident_preserves_preflight_traceback(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    monkeypatch.setenv("KAGGLE_KEY", "supersecretvalue")
    supervisor.write_watch_state(
        config.state_path,
        {
            "active_slug": "demo",
            "active_run_id": "run-preflight",
            "last_status": "running",
            "phase": "preparing_data",
        },
    )
    try:
        raise ValueError("All arrays must be of the same length KAGGLE_KEY=supersecretvalue")
    except ValueError as exc:
        incident_path = supervisor._write_watch_failure_incident(  # noqa: SLF001
            config=config,
            slug="demo",
            run_id="run-preflight",
            reason="ValueError",
            error=exc,
        )

    payload = json.loads(incident_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "preparing_data"
    assert payload["run_directory_exists"] is False
    assert "ValueError: All arrays must be of the same length" in payload["traceback"]
    assert "supersecretvalue" not in json.dumps(payload)
    assert "KAGGLE_KEY=<redacted>" in payload["traceback"]
    assert payload["repo_root"] == str(tmp_path)


def test_watch_self_improvement_uses_resolved_repository_root(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Path] = {}

    def fake_run_self_improvement_cycle(config):
        captured["workdir"] = config.knowledge_paths.workdir
        return {"status": "skipped_not_due"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(supervisor, "run_self_improvement_cycle", fake_run_self_improvement_cycle)

    supervisor.run_watch_self_improvement(_config(tmp_path), force=True)

    assert captured["workdir"] == Path(supervisor.__file__).resolve().parents[2]


def test_watch_optional_int_rejects_bool_and_fractional_values() -> None:
    assert supervisor._optional_int(True) is None  # noqa: SLF001
    assert supervisor._optional_int("3.5") is None  # noqa: SLF001
    assert supervisor._optional_int(3.0) == 3  # noqa: SLF001
    assert supervisor._optional_int("4") == 4  # noqa: SLF001


def test_watch_optional_float_rejects_bool_and_non_finite_values() -> None:
    assert supervisor._optional_float(True) is None  # noqa: SLF001
    assert supervisor._optional_float("nan") is None  # noqa: SLF001
    assert supervisor._optional_float("inf") is None  # noqa: SLF001
    assert supervisor._optional_float("0.25") == 0.25  # noqa: SLF001


def test_prepare_competition_reuses_eval_spec_despite_global_force(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path, auto_eval_spec=True, force=True)
    candidate = _competition("demo")
    paths = supervisor.CompetitionPaths(slug="demo", artifacts_dir=config.artifacts_dir)
    knowledge_paths = supervisor.KnowledgePaths(workdir=tmp_path)
    advisor_kwargs: dict[str, object] = {}
    phases: list[str] = []
    phase_contexts: list[tuple[str, str]] = []

    class FakeAdvisor:
        def __init__(self, **kwargs) -> None:
            advisor_kwargs.update(kwargs)

        def ensure_spec(self):
            return {"metric_name": "auc"}, "frozen"

    monkeypatch.delenv("KAGGLEBOT_REFRESH_EVALUATION_SPEC", raising=False)

    def fake_bootstrap(**kwargs) -> None:  # noqa: ANN003
        bootstrap_paths = kwargs["paths"]
        bootstrap_paths.context_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_paths.data_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_paths.dataset_profile_path.write_text(
            json.dumps({"status": "ready", "task": "classification", "modality": "tabular"}),
            encoding="utf-8",
        )
        (bootstrap_paths.data_dir / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    monkeypatch.setattr(supervisor, "bootstrap_competition", fake_bootstrap)
    monkeypatch.setattr(supervisor, "EvaluationAdvisor", FakeAdvisor)
    monkeypatch.setattr(
        supervisor,
        "update_watch_phase",
        lambda config, run_id, phase, **kwargs: (
            phase_contexts.append((config.slug, config.compute)),
            phases.append(phase),
        ),
    )

    supervisor._prepare_competition(
        config=config,
        candidate=candidate,
        paths=paths,
        knowledge_paths=knowledge_paths,
        run_id="run-1",
    )

    assert advisor_kwargs["force"] is False
    assert phases == ["preparing_data", "oracle_evaluation_advisor"]
    assert phase_contexts == [("demo", "local_gpu"), ("demo", "local_gpu")]


def test_prepare_competition_resume_reuses_existing_data_without_kaggle_download(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, auto_eval_spec=True, force=True)
    candidate = _competition("demo")
    paths = supervisor.CompetitionPaths(
        slug="demo",
        artifacts_dir=config.artifacts_dir,
        repo_root=config.workdir,
    )
    paths.context_dir.mkdir(parents=True)
    paths.data_dir.mkdir(parents=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"status": "ready", "task": "classification", "modality": "tabular"}),
        encoding="utf-8",
    )
    (paths.data_dir / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    knowledge_paths = supervisor.KnowledgePaths(workdir=tmp_path)
    phases: list[tuple[str, str]] = []

    monkeypatch.setattr(
        supervisor,
        "bootstrap_competition",
        lambda **kwargs: pytest.fail("resume must not re-download existing competition data"),
    )

    class FakeAdvisor:
        def __init__(self, **kwargs) -> None:
            assert kwargs["paths"] is paths

        def ensure_spec(self):
            return {"deliverable_mode": "leaderboard"}, "frozen"

    monkeypatch.setattr(supervisor, "EvaluationAdvisor", FakeAdvisor)
    monkeypatch.setattr(
        supervisor,
        "update_watch_phase",
        lambda _context, _run_id, phase, **kwargs: phases.append((phase, kwargs.get("detail", ""))),
    )

    supervisor._prepare_competition(
        config=config,
        candidate=candidate,
        paths=paths,
        knowledge_paths=knowledge_paths,
        run_id="run-1",
        resume=True,
    )

    assert phases == [
        ("preparing_data", "reusing existing competition data and context"),
        ("oracle_evaluation_advisor", "resolving evaluation specification"),
    ]


def test_prepare_competition_accepts_writeup_without_bundled_training_data(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidate = _competition("writeup-demo")
    paths = supervisor.CompetitionPaths(slug=candidate.slug, artifacts_dir=config.artifacts_dir)

    def fake_bootstrap(**kwargs) -> None:  # noqa: ANN003
        bootstrap_paths = kwargs["paths"]
        bootstrap_paths.context_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_paths.data_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_paths.rules_md_path.write_text(
            "The final submission is a Kaggle writeup.\n"
            "Teams must submit a writeup describing the artifact.\n"
            "The writeup is the primary competition deliverable.\n",
            encoding="utf-8",
        )
        bootstrap_paths.dataset_profile_path.write_text(
            json.dumps({"status": "non_tabular_data", "task": "text", "modality": "text"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(supervisor, "bootstrap_competition", fake_bootstrap)

    supervisor._prepare_competition(
        config=config,
        candidate=candidate,
        paths=paths,
        knowledge_paths=supervisor.KnowledgePaths(workdir=tmp_path),
        run_id="run-1",
    )

    assert paths.dataset_profile_path.is_file()


def test_watch_training_completion_rejects_diagnostic_only_run(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_dir = paths.run_dir("run-1")
    iter_dir = run_dir / "iter-1"
    iter_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"status": "validated_unscored_artifact"}),
        encoding="utf-8",
    )
    (iter_dir / "iteration_state.json").write_text(
        json.dumps({"iteration_complete": True, "trained": False}),
        encoding="utf-8",
    )

    issue = supervisor._watch_training_completion_issue(paths=paths, run_id="run-1")

    assert issue == "run status is validated_unscored_artifact"


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


def test_watch_autopilot_config_defaults_to_exhaustive_top1(tmp_path: Path) -> None:
    config = _config(tmp_path, time_budget_min=1200)
    paths = config.artifacts_dir / "demo"
    from kagglebot.paths import CompetitionPaths, KnowledgePaths

    autopilot_config = _build_autopilot_config(
        config=config,
        candidate=_competition("demo"),
        paths=CompetitionPaths(slug="demo", artifacts_dir=config.artifacts_dir),
        knowledge_paths=KnowledgePaths(workdir=tmp_path),
        run_id="run-1",
    )

    assert paths.name == "demo"
    assert autopilot_config.campaign_mode == "top1"
    assert autopilot_config.top1_exhaustive is True
    assert autopilot_config.method_scout == "refresh"
    assert autopilot_config.research_scout == "refresh"
    assert autopilot_config.portfolio_execution == "budgeted"
    assert autopilot_config.validation_lab == "force"
    assert autopilot_config.candidate_budget_min == 1200
    assert autopilot_config.max_candidates_per_iteration == 3


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


def test_select_next_competition_skips_submitted_writeup(monkeypatch, tmp_path: Path) -> None:
    candidates = [_competition("submitted-writeup", reward="$25,000"), _competition("fresh")]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(tmp_path)
    config.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    config.ledger_path.write_text(
        json.dumps(
            {
                "event": "finished",
                "slug": "submitted-writeup",
                "run_id": "run-1",
                "submission_status": "submitted",
                "submission_url": "https://www.kaggle.com/competitions/submitted-writeup/writeups/demo",
                "writeup_title": "Demo",
                "ts": "2026-04-22T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    selected = select_next_competition(config)

    assert [item.slug for item in selected] == ["fresh"]
    skipped = [
        record for record in WatchLedger(config.ledger_path).records() if record.get("event") == "candidate_skipped"
    ]
    assert skipped[-1]["slug"] == "submitted-writeup"
    assert skipped[-1]["reason"] == "writeup_already_submitted"


def test_select_next_competition_uses_submission_history_before_prize_score(
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


def test_select_next_competition_orders_prize_then_medal_then_standard_within_submission_tier(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidates = [
        _competition("standard", reward="", awards_points=False),
        _competition("medal", reward="", awards_points=True),
        _competition("prize", reward="$5,000", awards_points=False),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(tmp_path)
    for candidate in candidates:
        ledger_path = config.artifacts_dir / candidate.slug / "submissions" / "ledger.jsonl"
        ledger_path.parent.mkdir(parents=True)
        ledger_path.write_text(
            json.dumps({"event": "submit", "slug": candidate.slug, "ts": "2026-04-22T00:00:00+00:00"}) + "\n",
            encoding="utf-8",
        )

    selected = select_next_competition(config)

    assert [item.slug for item in selected][:3] == ["prize", "medal", "standard"]


def test_select_next_competition_prioritizes_prize_and_medal_combinations_within_unsubmitted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    candidates = [
        _competition(
            "standard-high-score",
            reward="",
            awards_points=False,
            category="Featured",
            deadline=now + timedelta(hours=1),
            team_count=5000,
        ),
        _competition("medal-only", reward="", awards_points=True),
        _competition("prize-only", reward="$1,000", awards_points=False),
        _competition("prize-and-medal", reward="$1,000", awards_points=True),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)

    selected = select_next_competition(_config(tmp_path))

    assert [item.slug for item in selected] == [
        "prize-and-medal",
        "prize-only",
        "medal-only",
        "standard-high-score",
    ]


def test_select_next_competition_uses_score_before_autopilot_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidates = [
        _competition("prize-started", reward="$100,000", category="Featured"),
        _competition("never-started", reward="", category="Playground"),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(tmp_path)
    run_dir = config.artifacts_dir / "prize-started" / "runs" / "20260422T000000Z-started"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "20260422T000000Z-started", "status": "completed"}),
        encoding="utf-8",
    )

    selected = select_next_competition(config)

    assert [item.slug for item in selected][:2] == ["prize-started", "never-started"]


def test_select_next_competition_prioritizes_urgent_started_candidate_over_far_unstarted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    candidates = [
        _competition(
            "urgent-started",
            reward="$50,000",
            category="Research",
            deadline=now + timedelta(days=1),
        ),
        _competition("far-unstarted", deadline=now + timedelta(days=90)),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)
    config = _config(tmp_path)
    run_dir = config.artifacts_dir / "urgent-started" / "runs" / "20260422T000000Z-started"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "20260422T000000Z-started", "status": "completed"}),
        encoding="utf-8",
    )

    selected = select_next_competition(config)

    assert [item.slug for item in selected][:2] == ["urgent-started", "far-unstarted"]


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


def test_select_next_competition_allows_entered_after_new_entrant_deadline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    candidates = [
        _competition(
            "already-entered",
            deadline=now + timedelta(days=14),
            new_entrant_deadline=now - timedelta(days=1),
        ),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)

    selected = select_next_competition(_config(tmp_path))

    assert [item.slug for item in selected] == ["already-entered"]


def test_select_next_competition_keeps_entered_competition_with_unknown_metric(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidates = [
        _competition(
            "custom-leaderboard-task",
            title="Custom leaderboard task",
            evaluation_metric="Competition custom score",
        ),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)

    selected = select_next_competition(_config(tmp_path))

    assert [item.slug for item in selected] == ["custom-leaderboard-task"]


def test_select_next_competition_keeps_known_task_when_metric_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidates = [
        _competition(
            "triagegeist",
            title="Triagegeist",
            category="Community",
            evaluation_metric="",
        ),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)

    selected = select_next_competition(_config(tmp_path))

    assert [item.slug for item in selected] == ["triagegeist"]


def test_select_next_competition_keeps_unfamiliar_entered_competition_types(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    candidates = [
        _competition(
            "arc-prize-2026-arc-agi-2",
            title="ARC Prize 2026 - ARC-AGI-2",
            category="Featured",
            evaluation_metric="Abstraction and Reasoning Challenge",
            deadline=now + timedelta(days=120),
        ),
        _competition(
            "orbit-wars",
            title="Orbit Wars",
            category="Featured",
            evaluation_metric="orbit_wars",
            deadline=now + timedelta(days=30),
        ),
        _competition(
            "pierce-the-veil",
            title="Pierce the VEIL: Hack It and Crack It Simulation",
            category="Community",
            evaluation_metric="",
            deadline=now + timedelta(days=7),
        ),
    ]
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", lambda **kwargs: candidates)

    selected = select_next_competition(_config(tmp_path))

    assert {item.slug for item in selected} == {item.slug for item in candidates}


def test_estimate_training_minutes_is_finite_for_complex_competitions() -> None:
    candidate = _competition(
        "nvidia-nemotron-model-reasoning-challenge",
        title="NVIDIA Nemotron Model Reasoning Challenge",
        category="Featured",
        evaluation_metric="NVIDIA Nemotron Metric",
    )

    assert _estimate_training_minutes(candidate) == 360


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


def test_select_next_competition_dry_run_enriches_submitted_rank_from_leaderboard_score(
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
    config = _config(tmp_path, dry_run=True)
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


def test_run_watch_once_recovers_self_improvement_before_competition_work(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    lock_handle = object()

    monkeypatch.setattr("kagglebot.supervisor.has_interrupted_self_improvement", lambda config: True)
    monkeypatch.setattr(
        "kagglebot.supervisor.recover_interrupted_self_improvement",
        lambda config: events.append("recover") or {"status": "completed"},
    )
    monkeypatch.setattr("kagglebot.supervisor._try_acquire_watch_resource_lock", lambda config, ledger: lock_handle)
    monkeypatch.setattr(
        "kagglebot.supervisor._run_watch_once_unlocked",
        lambda config, ledger: events.append("watch")
        or supervisor.WatchCycleResult(status="no_candidates", slug=None, run_id=None),
    )
    monkeypatch.setattr("kagglebot.supervisor._release_watch_resource_lock", lambda handle: None)

    result = run_watch_once(_config(tmp_path))

    assert result.status == "no_candidates"
    assert events == ["recover", "watch"]


def test_run_watch_once_skips_when_watch_resource_is_locked(monkeypatch, tmp_path: Path) -> None:
    def fake_flock(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise BlockingIOError()

    def fail_list_entered_competitions(**kwargs) -> list[EnteredCompetition]:  # noqa: ANN003
        raise AssertionError("locked watcher must not select another competition")

    monkeypatch.setattr(supervisor.fcntl, "flock", fake_flock)
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", fail_list_entered_competitions)

    config = _config(tmp_path, compute="local_gpu", hardware_profile="rtx3060")
    result = run_watch_once(config)

    assert result.status == "locked"
    assert result.reason == "watch_resource_locked"
    records = WatchLedger(config.ledger_path).records()
    assert any(record.get("event") == "locked" for record in records)


def test_watch_env_hour_parsers_fallback_for_invalid_and_non_finite(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_RESOURCE_BLOCK_TTL_HOURS", "nan")
    assert supervisor._resource_block_ttl_hours() == 168.0  # noqa: SLF001

    monkeypatch.setenv("KAGGLEBOT_RESOURCE_BLOCK_TTL_HOURS", "-1")
    assert supervisor._resource_block_ttl_hours() == 0.0  # noqa: SLF001


def test_plan_max_iterations_ignores_invalid_or_non_object_payload(tmp_path: Path) -> None:
    assert _plan_max_iterations(tmp_path / "missing.json") is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert _plan_max_iterations(invalid) is None

    array_payload = tmp_path / "array.json"
    array_payload.write_text("[]", encoding="utf-8")
    assert _plan_max_iterations(array_payload) is None


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
    monkeypatch.setattr("kagglebot.supervisor._watch_training_completion_issue", lambda **kwargs: None)

    result = run_watch_once(config)

    assert result.status == "finished"
    assert result.slug == "demo"
    assert captured == {"slug": "demo", "run_id": None}
    records = WatchLedger(config.ledger_path).records()
    assert any(record.get("event") == "resumed" for record in records)
    assert not any(record.get("event") == "started" for record in records)


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
        captured["repo_root"] = config.paths.repo_root

    monkeypatch.setattr("kagglebot.supervisor.run_autopilot", fake_run_autopilot)

    result = run_watch_once(_config(tmp_path))

    assert result.status == "finished"
    assert captured == {
        "submit_policy": "improved",
        "submit": True,
        "force_submit": True,
        "repo_root": tmp_path,
    }


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
    monkeypatch.setattr("kagglebot.supervisor._watch_training_completion_issue", lambda **kwargs: None)

    result = run_watch_once(config)

    assert result.status == "finished"
    assert result.slug == "demo"
    assert captured["run_id"] is None
    records = WatchLedger(config.ledger_path).records()
    assert any(record.get("event") == "resumed" for record in records)
    assert not any(record.get("event") == "started" for record in records)


def test_run_watch_once_restarts_interrupted_preflight_with_same_run_id(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_id = "20260423T010203Z-preflight"
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(
        json.dumps(
            {
                "active_slug": "demo",
                "active_run_id": run_id,
                "last_status": "running",
                "phase": "oracle_evaluation_advisor",
                "started_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    def fail_list_entered_competitions(**kwargs):  # noqa: ANN003, ARG001
        raise AssertionError("an interrupted preflight must restart the same competition")

    captured: dict[str, object] = {}
    monkeypatch.setattr("kagglebot.supervisor.list_entered_competitions", fail_list_entered_competitions)
    monkeypatch.setattr(
        "kagglebot.supervisor._prepare_competition",
        lambda **kwargs: captured.update(prepared_run_id=kwargs["run_id"]),
    )
    monkeypatch.setattr(
        "kagglebot.supervisor.run_autopilot",
        lambda config: captured.update(config_run_id=config.run_id),
    )

    result = run_watch_once(config)

    assert result.status == "finished"
    assert result.run_id == run_id
    assert captured == {"prepared_run_id": run_id, "config_run_id": run_id}


def test_run_watch_once_clears_stale_active_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "kagglebot.supervisor.list_entered_competitions",
        lambda **kwargs: [_competition("fresh")],
    )
    monkeypatch.setattr("kagglebot.supervisor._prepare_competition", lambda **kwargs: None)
    config = _config(tmp_path)
    stale_run_id = "20260422T000000Z-abcd1234"
    run_dir = config.artifacts_dir / "demo" / "runs" / stale_run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(
        json.dumps(
            {
                "active_slug": "demo",
                "active_run_id": stale_run_id,
                "last_status": "running",
                "updated_at": (datetime.now(UTC) - timedelta(hours=48)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_autopilot(config) -> None:
        captured["slug"] = config.slug
        captured["run_id"] = config.run_id

    monkeypatch.setattr("kagglebot.supervisor.run_autopilot", fake_run_autopilot)

    result = run_watch_once(config)

    assert result.status == "finished"
    assert result.slug == "fresh"
    assert captured["slug"] == "fresh"
    assert captured["run_id"] is not None
    records = WatchLedger(config.ledger_path).records()
    assert any(record.get("event") == "stale_active_cleared" for record in records)


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


def test_run_watch_once_reports_missing_data_as_failed_and_selects_next_candidate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "kagglebot.supervisor.list_entered_competitions",
        lambda **kwargs: [_competition("blocked-data"), _competition("fresh")],
    )
    monkeypatch.setattr("kagglebot.supervisor._prepare_competition", lambda **kwargs: None)
    attempts: list[str] = []

    def fake_run_autopilot(config) -> None:
        attempts.append(config.slug)
        if config.slug == "blocked-data":
            paths = CompetitionPaths(slug=config.slug, artifacts_dir=config.paths.artifacts_dir)
            paths.context_dir.mkdir(parents=True, exist_ok=True)
            paths.data_dir.mkdir(parents=True, exist_ok=True)
            paths.dataset_profile_path.write_text(
                json.dumps({"status": "missing_required_files"}),
                encoding="utf-8",
            )
            run_dir = paths.run_dir(config.run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "failure_kind": "blocked_on_data",
                        "retryable": True,
                    }
                ),
                encoding="utf-8",
            )
            raise MissingCompetitionDataError("data unavailable")

    monkeypatch.setattr("kagglebot.supervisor.run_autopilot", fake_run_autopilot)
    config = _config(tmp_path)

    failed = run_watch_once(config)

    assert failed.status == "failed"
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["active_slug"] == "blocked-data"
    assert state["active_run_id"] == failed.run_id
    assert state["last_status"] == "failed"
    assert state["phase"] == "blocked_on_data"
    records = WatchLedger(config.ledger_path).records()
    failure = next(record for record in records if record.get("event") == "failed")
    assert failure["reason"] == "missing_competition_data"
    assert failure["failure_kind"] == "blocked_on_data"
    assert failure["retryable"] is True
    assert not any(record.get("event") == "finished" for record in records)

    resumed = run_watch_once(config)

    assert resumed.status == "finished"
    assert resumed.slug == "fresh"
    assert attempts == ["blocked-data", "fresh"]


def test_run_watch_once_resumes_retryable_data_failure_after_training_archive_is_staged(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    slug = "blocked-data"
    run_id = "20260422T000000Z-data1234"
    paths = CompetitionPaths(slug=slug, artifacts_dir=config.artifacts_dir, repo_root=config.workdir)
    paths.context_dir.mkdir(parents=True)
    paths.data_dir.mkdir(parents=True)
    paths.run_dir(run_id).mkdir(parents=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"status": "missing_required_files"}),
        encoding="utf-8",
    )
    (paths.data_dir / "HAR.zip").write_bytes(b"staged training archive")
    (paths.run_dir(run_id) / "run.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "failure_kind": "blocked_on_data",
                "retryable": True,
            }
        ),
        encoding="utf-8",
    )
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(
        json.dumps(
            {
                "active_slug": slug,
                "active_run_id": run_id,
                "last_status": "failed",
                "phase": "blocked_on_data",
                "started_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kagglebot.supervisor.list_entered_competitions",
        lambda **kwargs: pytest.fail("staged data must resume the active run"),
    )
    monkeypatch.setattr("kagglebot.supervisor._prepare_competition", lambda **kwargs: None)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "kagglebot.supervisor.run_autopilot",
        lambda autopilot_config: captured.update(run_id=autopilot_config.run_id),
    )
    monkeypatch.setattr("kagglebot.supervisor._watch_training_completion_issue", lambda **kwargs: None)

    result = run_watch_once(config)

    assert result.status == "finished"
    assert result.slug == slug
    assert result.run_id == run_id
    assert captured == {"run_id": None}


def test_reconcile_active_writeup_submission_marks_run_terminal(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    slug = "writeup-demo"
    run_id = "run-1"
    run_dir = config.artifacts_dir / slug / "runs" / run_id
    report_path = run_dir / "iter-1" / "output" / "launch-demo" / "writeup.md"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("# Demo Writeup\n\nEvidence-backed body.\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "slug": slug,
                "status": "running",
                "config": {"deliverable_mode": "writeup", "submit": True},
            }
        ),
        encoding="utf-8",
    )
    config.state_path.parent.mkdir(parents=True, exist_ok=True)
    config.state_path.write_text(
        json.dumps(
            {
                "active_slug": slug,
                "active_run_id": run_id,
                "last_status": "running",
                "phase": "local_kernel_running",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kagglebot.supervisor.find_submitted_writeup",
        lambda **kwargs: {
            "status": "submitted",
            "reason": "kaggle-submitted-confirmation-observed",
            "url": "https://www.kaggle.com/competitions/writeup-demo/writeups/demo",
        },
    )
    ledger = WatchLedger(config.ledger_path)

    reconciled = _reconcile_active_writeup_submission(
        config=config,
        ledger=ledger,
        slug=slug,
        run_id=run_id,
    )

    assert reconciled is True
    run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_payload["status"] == "submitted"
    assert run_payload["writeup_bundle"]["submission"]["url"].endswith("/writeups/demo")
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["active_slug"] is None
    assert state["active_run_id"] is None
    finished = [record for record in ledger.records() if record.get("event") == "finished"]
    assert finished[-1]["reason"] == "reconciled_writeup_submission"
    assert finished[-1]["submission_status"] == "submitted"


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
        captured["time_budget_min"] = config.time_budget_min
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
    assert captured == {
        "submit_policy": "improved",
        "compute": "local_gpu",
        "max_iterations": 5,
        "time_budget_min": 1440,
    }


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
