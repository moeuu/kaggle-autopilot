from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.hashing import sha256_file
from kagglebot.paths import CompetitionPaths, repo_root


@dataclass(frozen=True)
class SubmissionHistory:
    history_file: Path

    @staticmethod
    def for_slug(slug: str) -> SubmissionHistory:
        paths = CompetitionPaths(slug=slug, repo_root=repo_root())
        paths.submissions_dir.mkdir(parents=True, exist_ok=True)
        hf = paths.submissions_dir / "history.jsonl"
        return SubmissionHistory(history_file=hf)

    def is_duplicate(self, submission_path: str) -> bool:
        new_hash = sha256_file(submission_path)
        if not self.history_file.exists():
            return False
        for line in self.history_file.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("sha256") == new_hash:
                return True
        return False

    def record(self, submission_path: str, message: str) -> None:
        rec = {
            "ts": datetime.now(UTC).isoformat(),
            "sha256": sha256_file(submission_path),
            "submission_path": submission_path,
            "message": message,
        }
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
