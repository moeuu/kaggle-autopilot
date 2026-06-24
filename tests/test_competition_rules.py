from __future__ import annotations

from pathlib import Path

from kagglebot.competition_rules import (
    CompetitionRuleConstraints,
    extract_submission_limit_per_day,
    load_competition_rule_constraints,
    runtime_limit_for_compute,
)
from kagglebot.paths import CompetitionPaths


def test_extract_submission_limit_per_day_uses_strictest_positive_limit() -> None:
    text = "maximum of five submissions per day, but only 2 submissions within 24 hours for final scoring"

    assert extract_submission_limit_per_day(text) == 2


def test_load_competition_rule_constraints_reads_rules_files(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True)
    paths.rules_md_path.write_text(
        """
        Code submissions to this competition must be made through Notebooks.
        Your Notebook cannot use internet access in this competition.
        GPU Notebook <= 9 hours.
        CPU Notebook <= 12 hours.
        You may make three submissions per day.
        """,
        encoding="utf-8",
    )

    constraints = load_competition_rule_constraints(paths)

    assert constraints.notebook_submissions_only is True
    assert constraints.internet_must_be_off is True
    assert constraints.submission_limit_detected is True
    assert constraints.submission_limit_per_day == 3
    assert constraints.gpu_runtime_limit_min == 540
    assert constraints.cpu_runtime_limit_min == 720


def test_runtime_limit_for_compute_prefers_gpu_limit_for_kaggle_gpu() -> None:
    constraints = CompetitionRuleConstraints(cpu_runtime_limit_min=720, gpu_runtime_limit_min=540)

    assert runtime_limit_for_compute(constraints=constraints, compute="kaggle_gpu") == 540
    assert runtime_limit_for_compute(constraints=constraints, compute="kaggle_tpu") == 540
    assert runtime_limit_for_compute(constraints=constraints, compute="local_gpu") is None
