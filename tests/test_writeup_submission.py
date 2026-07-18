from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from kagglebot.kaggle_api import EnteredCompetition
from kagglebot.writeup_submission import (
    _KAGGLE_WRITEUP_CDP_SCRIPT,
    WriteupSubmissionRequest,
    submit_validated_writeup,
)


class _Adapter:
    def __init__(self, status: str = "submitted") -> None:
        self.status = status
        self.calls = 0

    def submit(self, *, slug: str, title: str, body: str) -> dict[str, object]:
        self.calls += 1
        assert slug == "demo"
        assert title == "Demo solution"
        assert "evidence" in body
        return {"status": self.status, "reason": "test"}


def _competition(slug: str = "demo") -> EnteredCompetition:
    return EnteredCompetition(
        slug=slug,
        title=slug,
        url=f"https://www.kaggle.com/competitions/{slug}",
        category="Community",
        reward="",
        evaluation_metric="",
        deadline=None,
        enabled_date=None,
        new_entrant_deadline=None,
        merger_deadline=None,
        team_count=1,
        max_daily_submissions=1,
        is_kernels_submissions_only=False,
        submissions_disabled=False,
        awards_points=True,
        source="test",
    )


def _request(tmp_path: Path, *, force: bool = True) -> WriteupSubmissionRequest:
    report_path = tmp_path / "report.md"
    body = "# Demo solution\n\nThis evidence-backed writeup contains enough validated content for submission.\n"
    report_path.write_text(body, encoding="utf-8")
    metadata = {
        "status": "ready_for_submit",
        "report_path": str(report_path),
        "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "validation": {"valid": True},
    }
    return WriteupSubmissionRequest(
        slug="demo",
        metadata=metadata,
        attempts_path=tmp_path / "attempts.jsonl",
        force=force,
        dry_run=False,
    )


def test_writeup_submission_requires_force(tmp_path: Path) -> None:
    adapter = _Adapter()
    result = submit_validated_writeup(_request(tmp_path, force=False), adapter=adapter)

    assert result["status"] == "blocked_force_required"
    assert adapter.calls == 0


def test_writeup_submission_checks_entered_rules_and_deduplicates(tmp_path: Path) -> None:
    request = _request(tmp_path)
    adapter = _Adapter()

    result = submit_validated_writeup(
        request,
        adapter=adapter,
        entered_loader=lambda **kwargs: [_competition()],
        rules_checker=lambda slug, **kwargs: True,
    )
    duplicate = submit_validated_writeup(
        request,
        adapter=adapter,
        entered_loader=lambda **kwargs: [_competition()],
        rules_checker=lambda slug, **kwargs: True,
    )

    assert result["status"] == "submitted"
    assert duplicate["status"] == "blocked_duplicate"
    assert adapter.calls == 1


def test_writeup_submission_does_not_retry_ambiguous_attempt(tmp_path: Path) -> None:
    request = _request(tmp_path)
    adapter = _Adapter(status="ambiguous")

    first = submit_validated_writeup(
        request,
        adapter=adapter,
        entered_loader=lambda **kwargs: [_competition()],
        rules_checker=lambda slug, **kwargs: True,
    )
    second = submit_validated_writeup(
        request,
        adapter=adapter,
        entered_loader=lambda **kwargs: [_competition()],
        rules_checker=lambda slug, **kwargs: True,
    )

    assert first["status"] == "ambiguous"
    assert second["status"] == "blocked_duplicate"
    assert adapter.calls == 1


def test_writeup_submission_blocks_not_entered_and_rules(tmp_path: Path) -> None:
    request = _request(tmp_path)
    adapter = _Adapter()
    not_entered = submit_validated_writeup(
        request,
        adapter=adapter,
        entered_loader=lambda **kwargs: [_competition("other")],
    )
    rules_blocked = submit_validated_writeup(
        replace(request, attempts_path=tmp_path / "rules-attempts.jsonl"),
        adapter=adapter,
        entered_loader=lambda **kwargs: [_competition()],
        rules_checker=lambda slug, **kwargs: False,
    )

    assert not_entered["status"] == "blocked_not_entered"
    assert rules_blocked["status"] == "blocked_rules_not_accepted"
    assert adapter.calls == 0


def test_writeup_submission_requires_published_notebook_when_contract_requires_it(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.metadata["status"] = "ready_for_notebook_publish"
    request.metadata["notebook"] = {"required": True, "status": "publish_required"}
    adapter = _Adapter()

    result = submit_validated_writeup(request, adapter=adapter)

    assert result["status"] == "blocked_notebook_required"
    assert adapter.calls == 0


def test_writeup_submission_blocks_changed_required_artifact(tmp_path: Path) -> None:
    request = _request(tmp_path)
    artifact = tmp_path / "features.csv"
    artifact.write_text("a\n1\n", encoding="utf-8")
    request.metadata["required_artifacts"] = [
        {
            "name": "features.csv",
            "path": str(artifact),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    ]
    artifact.write_text("a\n2\n", encoding="utf-8")
    adapter = _Adapter()

    result = submit_validated_writeup(request, adapter=adapter)

    assert result["status"] == "blocked_required_artifact"
    assert "changed" in str(result["reason"])
    assert adapter.calls == 0


def test_writeup_cdp_script_is_valid_javascript() -> None:
    node = shutil.which("node")
    if node is None:
        return
    result = subprocess.run(  # noqa: S603
        [node, "-e", "new Function(process.argv[1])", _KAGGLE_WRITEUP_CDP_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
