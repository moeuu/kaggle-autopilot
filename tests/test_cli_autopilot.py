from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from kagglebot.cli import app
from kagglebot.exceptions import SubmitAbortedError
from kagglebot.paths import CompetitionPaths


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
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
        ],
    )

    assert result.exit_code == SubmitAbortedError.exit_code
    assert "submit aborted" in result.stdout.lower()
    assert "Local submission validation failed" in result.stdout
    assert "Traceback" not in result.stdout


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
            "autopilot",
            slug,
            "--compute",
            "local_gpu",
            "--resume-run-id",
            run_id,
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
            "autopilot",
            slug,
            "--compute",
            "local_gpu",
            "--resume-latest",
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
            "autopilot",
            slug,
            "--compute",
            "local_gpu",
            "--resume-run-id",
            prefix,
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
            "autopilot",
            "playground-series-s6e2",
            "--compute",
            "local_gpu",
        ],
    )

    assert result.exit_code == 0
    assert calls["ensure_spec"] == 1
    assert calls["run_autopilot"] == 1
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
