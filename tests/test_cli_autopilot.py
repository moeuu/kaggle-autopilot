from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from kagglebot.cli import app
from kagglebot.exceptions import SubmitAbortedError
from kagglebot.paths import CompetitionPaths
from kagglebot.submission_service import SubmissionConfig


def test_autopilot_uses_preferred_artifacts_dir_by_default(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, Path] = {}

    def fake_bootstrap(**kwargs):
        captured["artifacts_dir"] = kwargs["paths"].artifacts_dir
        captured["repo_root"] = kwargs["paths"].repo_root

    monkeypatch.setattr("kagglebot.cli._preferred_artifacts_dir", lambda: Path("/data/kaggle-autopilot-artifacts"))
    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", fake_bootstrap)
    monkeypatch.setattr("kagglebot.cli.run_autopilot", lambda config: None)  # noqa: ARG005

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert captured["artifacts_dir"] == Path("/data/kaggle-autopilot-artifacts")
    assert captured["repo_root"] == tmp_path


def test_autopilot_requires_force_before_external_side_effects(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    calls = {"bootstrap": 0, "autopilot": 0}

    monkeypatch.setattr(
        "kagglebot.cli.bootstrap_competition",
        lambda **kwargs: calls.__setitem__("bootstrap", calls["bootstrap"] + 1),
    )
    monkeypatch.setattr(
        "kagglebot.cli.run_autopilot",
        lambda config: calls.__setitem__("autopilot", calls["autopilot"] + 1),
    )

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 2
    assert "without --force" in result.stderr
    assert calls == {"bootstrap": 0, "autopilot": 0}


def test_autopilot_submit_aborted_exits_clean(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)

    def fake_run_autopilot(config):  # noqa: ARG001
        raise SubmitAbortedError("Local submission validation failed; Kaggle CLI submit is skipped.")

    monkeypatch.setattr("kagglebot.cli.run_autopilot", fake_run_autopilot)

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == SubmitAbortedError.exit_code
    assert "submit aborted" in result.stdout.lower()
    assert "Local submission validation failed" in result.stdout
    assert "Traceback" not in result.stdout


def test_submit_cli_force_submit_allows_side_effect_guard(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    submission = tmp_path / "submission.csv"
    submission.write_text("id,target\n1,0\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeSubmissionService:
        def __init__(self, config: SubmissionConfig) -> None:
            captured["force_submit"] = config.force_submit

        def submit(self, **kwargs: object) -> None:
            captured["submit_kwargs"] = kwargs

    monkeypatch.setattr("kagglebot.cli.check_rules_accepted", lambda *args, **kwargs: True)
    monkeypatch.setattr("kagglebot.cli.SubmissionService", FakeSubmissionService)

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "submit",
            "demo",
            "-f",
            str(submission),
            "-m",
            "manual repair submit",
            "--force-submit",
        ],
    )

    assert result.exit_code == 0
    assert captured["force_submit"] is True
    assert captured["submit_kwargs"]


def test_autopilot_cli_defaults_to_top1_campaign_mode(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.cli.run_autopilot", lambda config: captured.update(campaign_mode=config.campaign_mode)
    )

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert captured["campaign_mode"] == "top1"


def test_autopilot_cli_accepts_baseline_campaign_mode(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.cli.run_autopilot", lambda config: captured.update(campaign_mode=config.campaign_mode)
    )

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--campaign-mode",
            "baseline",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert captured["campaign_mode"] == "baseline"


def test_autopilot_cli_uses_shared_campaign_mode_aliases(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.cli.run_autopilot", lambda config: captured.update(campaign_mode=config.campaign_mode)
    )

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--campaign-mode",
            "top-1",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert captured["campaign_mode"] == "top1"


def test_autopilot_cli_accepts_method_scout_options(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.cli.run_autopilot",
        lambda config: captured.update(
            method_scout=config.method_scout,
            method_scout_max_sources=config.method_scout_max_sources,
        ),
    )

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--method-scout",
            "refresh",
            "--method-scout-max-sources",
            "7",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert captured["method_scout"] == "refresh"
    assert captured["method_scout_max_sources"] == 7


def test_autopilot_cli_accepts_portfolio_execution_mode(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.cli.run_autopilot",
        lambda config: captured.update(portfolio_execution=config.portfolio_execution),
    )

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--portfolio-execution",
            "parallel",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert captured["portfolio_execution"] == "parallel"


def test_autopilot_cli_uses_shared_portfolio_execution_aliases(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.cli.run_autopilot",
        lambda config: captured.update(portfolio_execution=config.portfolio_execution),
    )

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--portfolio-execution",
            "budget",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert captured["portfolio_execution"] == "budgeted"


def test_autopilot_cli_top1_exhaustive_applies_safe_defaults(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.cli.run_autopilot",
        lambda config: captured.update(
            campaign_mode=config.campaign_mode,
            method_scout=config.method_scout,
            research_scout=config.research_scout,
            portfolio_execution=config.portfolio_execution,
            validation_lab=config.validation_lab,
            candidate_budget_min=config.candidate_budget_min,
            max_candidates_per_iteration=config.max_candidates_per_iteration,
            top1_exhaustive=config.top1_exhaustive,
            top1_submit_policy=config.top1_submit_policy,
        ),
    )

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--top1-exhaustive",
            "--top1-submit-policy",
            "final_lock",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert captured["campaign_mode"] == "top1"
    assert captured["method_scout"] == "refresh"
    assert captured["research_scout"] == "refresh"
    assert captured["portfolio_execution"] == "budgeted"
    assert captured["validation_lab"] == "force"
    assert captured["candidate_budget_min"] == 60
    assert captured["max_candidates_per_iteration"] == 3
    assert captured["top1_exhaustive"] is True
    assert captured["top1_submit_policy"] == "final_lock"


def test_autopilot_resume_run_id_reuses_existing_run(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    slug = "playground-series-s6e2"
    run_id = "20260216T000000Z-abcd1234"
    paths = CompetitionPaths(slug=slug, artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.run_dir(run_id).mkdir(parents=True, exist_ok=True)

    bootstrap_calls = {"count": 0}
    captured: dict[str, object] = {}

    def fake_bootstrap(**kwargs):  # noqa: ARG001
        bootstrap_calls["count"] += 1

    def fake_run_autopilot(config):
        captured["run_id"] = config.run_id
        captured["resume_id"] = os.environ.get("KAGGLEBOT_RESUME_RUN_ID")
        captured["resume_slug"] = os.environ.get("KAGGLEBOT_RESUME_SLUG")

    monkeypatch.delenv("KAGGLEBOT_RESUME_RUN_ID", raising=False)
    monkeypatch.delenv("KAGGLEBOT_RESUME_SLUG", raising=False)
    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", fake_bootstrap)
    monkeypatch.setattr("kagglebot.cli.run_autopilot", fake_run_autopilot)

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            slug,
            "--compute",
            "local_gpu",
            "--resume-run-id",
            run_id,
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert bootstrap_calls["count"] == 0
    assert captured["run_id"] is None
    assert captured["resume_id"] == run_id
    assert captured["resume_slug"] == slug
    assert "requested run" in result.stdout.lower()
    assert "skipping bootstrap" in result.stdout.lower()


def test_autopilot_resume_latest_selects_most_recent_run(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    slug = "playground-series-s6e2"
    paths = CompetitionPaths(slug=slug, artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    older = paths.run_dir("20260216T000000Z-old11111")
    newer = paths.run_dir("20260216T000100Z-new22222")
    older.mkdir(parents=True, exist_ok=True)
    newer.mkdir(parents=True, exist_ok=True)
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    captured: dict[str, object] = {}
    monkeypatch.delenv("KAGGLEBOT_RESUME_RUN_ID", raising=False)
    monkeypatch.delenv("KAGGLEBOT_RESUME_SLUG", raising=False)
    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.cli.run_autopilot",
        lambda config: captured.update(
            run_id=config.run_id,
            resume_id=os.environ.get("KAGGLEBOT_RESUME_RUN_ID"),
            resume_slug=os.environ.get("KAGGLEBOT_RESUME_SLUG"),
        ),
    )

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            slug,
            "--compute",
            "local_gpu",
            "--resume-latest",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert captured["run_id"] is None
    assert captured["resume_id"] == newer.name
    assert captured["resume_slug"] == slug


def test_autopilot_resume_run_id_requires_existing_run(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    calls = {"bootstrap": 0, "run_autopilot": 0}
    monkeypatch.delenv("KAGGLEBOT_RESUME_RUN_ID", raising=False)
    monkeypatch.delenv("KAGGLEBOT_RESUME_SLUG", raising=False)
    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: calls.update(bootstrap=1))
    monkeypatch.setattr("kagglebot.cli.run_autopilot", lambda config: calls.update(run_autopilot=1))

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--resume-run-id",
            "missing-run-id",
        ],
    )

    assert result.exit_code == 2
    assert calls["bootstrap"] == 0
    assert calls["run_autopilot"] == 0


def test_autopilot_resume_run_id_accepts_unique_prefix(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    slug = "playground-series-s6e2"
    run_id = "20260216T000000Z-abcd1234"
    prefix = "20260216T000000Z-abcd123"
    paths = CompetitionPaths(slug=slug, artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.run_dir(run_id).mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}
    monkeypatch.delenv("KAGGLEBOT_RESUME_RUN_ID", raising=False)
    monkeypatch.delenv("KAGGLEBOT_RESUME_SLUG", raising=False)
    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)
    monkeypatch.setattr(
        "kagglebot.cli.run_autopilot",
        lambda config: captured.update(
            run_id=config.run_id,
            resume_id=os.environ.get("KAGGLEBOT_RESUME_RUN_ID"),
            resume_slug=os.environ.get("KAGGLEBOT_RESUME_SLUG"),
        ),
    )

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            slug,
            "--compute",
            "local_gpu",
            "--resume-run-id",
            prefix,
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert captured["run_id"] is None
    assert captured["resume_id"] == run_id
    assert captured["resume_slug"] == slug


def test_autopilot_resume_run_id_rejects_ambiguous_prefix(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    slug = "playground-series-s6e2"
    paths = CompetitionPaths(slug=slug, artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.run_dir("20260216T000000Z-abcd1111").mkdir(parents=True, exist_ok=True)
    paths.run_dir("20260216T000000Z-abcd2222").mkdir(parents=True, exist_ok=True)

    calls = {"bootstrap": 0, "run_autopilot": 0}
    monkeypatch.delenv("KAGGLEBOT_RESUME_RUN_ID", raising=False)
    monkeypatch.delenv("KAGGLEBOT_RESUME_SLUG", raising=False)
    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: calls.update(bootstrap=1))
    monkeypatch.setattr("kagglebot.cli.run_autopilot", lambda config: calls.update(run_autopilot=1))

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            slug,
            "--compute",
            "local_gpu",
            "--resume-run-id",
            "20260216T000000Z-abcd",
        ],
    )

    assert result.exit_code == 2
    assert calls["bootstrap"] == 0
    assert calls["run_autopilot"] == 0


def test_autopilot_auto_eval_spec_enabled_by_default(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    calls = {"ensure_spec": 0, "run_autopilot": 0, "advisor_force": None}

    class _FakeAdvisor:
        def __init__(self, **kwargs):
            self.spec_path = Path("/tmp/evaluation_spec.json")
            calls["advisor_force"] = kwargs["force"]

        def ensure_spec(self):
            calls["ensure_spec"] += 1
            return {"metric_name": "rmse", "split_strategy": "kfold"}, "advisor"

    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)
    monkeypatch.setattr("kagglebot.cli.EvaluationAdvisor", _FakeAdvisor)
    monkeypatch.setattr("kagglebot.cli.run_autopilot", lambda config: calls.update(run_autopilot=1))  # noqa: ARG005

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
        ],
    )

    assert result.exit_code == 0
    assert calls["ensure_spec"] == 1
    assert calls["run_autopilot"] == 1
    assert calls["advisor_force"] is False
    assert "evaluation advisor" in result.stdout.lower()


def test_autopilot_no_auto_eval_spec_skips_advisor(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    calls = {"ensure_spec": 0, "run_autopilot": 0}

    class _FakeAdvisor:
        def __init__(self, **kwargs):  # noqa: ARG002
            self.spec_path = Path("/tmp/evaluation_spec.json")

        def ensure_spec(self):
            calls["ensure_spec"] += 1
            return {"metric_name": "rmse", "split_strategy": "kfold"}, "advisor"

    monkeypatch.setattr("kagglebot.cli.bootstrap_competition", lambda **kwargs: None)
    monkeypatch.setattr("kagglebot.cli.EvaluationAdvisor", _FakeAdvisor)
    monkeypatch.setattr("kagglebot.cli.run_autopilot", lambda config: calls.update(run_autopilot=1))  # noqa: ARG005

    result = runner.invoke(
        app,
        [
            "--workdir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--force",
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
            "--no-auto-eval-spec",
        ],
    )

    assert result.exit_code == 0
    assert calls["ensure_spec"] == 0
    assert calls["run_autopilot"] == 1
    assert "evaluation advisor" not in result.stdout.lower()
