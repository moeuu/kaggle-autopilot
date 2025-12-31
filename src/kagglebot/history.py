from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from kagglebot.hashing import sha256_file
from kagglebot.paths import CompetitionPaths, repo_root


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    metadata_path: Path


@dataclass(frozen=True)
class RunLedger:
    base_dir: Path

    @staticmethod
    def for_slug(slug: str, root: Path | None = None) -> RunLedger:
        base_root = root if root is not None else repo_root()
        paths = CompetitionPaths(slug=slug, repo_root=base_root)
        paths.runs_dir.mkdir(parents=True, exist_ok=True)
        return RunLedger(base_dir=paths.runs_dir)

    def start_run(
        self,
        *,
        slug: str,
        dry_run: bool,
        force: bool,
        submission_path: str | None,
        sample_path: str | None,
        message: str | None,
        argv: list[str] | None,
    ) -> RunRecord:
        run_id = _new_run_id()
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata_path = run_dir / "metadata.json"

        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "slug": slug,
            "created_at": datetime.now(UTC).isoformat(),
            "dry_run": dry_run,
            "force": force,
            "submission_path": submission_path,
            "sample_path": sample_path,
            "message": message,
            "argv": argv,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return RunRecord(run_id=run_id, run_dir=run_dir, metadata_path=metadata_path)


@dataclass(frozen=True)
class SubmissionLedger:
    ledger_path: Path

    @staticmethod
    def for_slug(slug: str, root: Path | None = None) -> SubmissionLedger:
        base_root = root if root is not None else repo_root()
        paths = CompetitionPaths(slug=slug, repo_root=base_root)
        paths.submissions_dir.mkdir(parents=True, exist_ok=True)
        return SubmissionLedger(ledger_path=paths.submission_ledger)

    def is_duplicate(self, submission_path: str) -> bool:
        new_hash = sha256_file(submission_path)
        if not self.ledger_path.exists():
            return False
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("sha256") == new_hash:
                return True
        return False

    def record(self, submission_path: str, message: str, run_id: str | None) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now(UTC).isoformat(),
            "sha256": sha256_file(submission_path),
            "submission_path": submission_path,
            "message": message,
            "run_id": run_id,
        }
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
