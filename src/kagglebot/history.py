from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from kagglebot.hashing import sha256_file


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def submission_fingerprint(slug: str, message: str, submission_path: Path) -> str:
    file_hash = sha256_file(str(submission_path))
    combined = f"{slug}\n{message}\n{file_hash}".encode()
    return sha256_file_bytes(combined)


def sha256_file_bytes(payload: bytes) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class SubmissionLedger:
    ledger_path: Path

    def is_duplicate(self, *, slug: str, message: str, submission_path: Path) -> bool:
        fingerprint = submission_fingerprint(slug, message, submission_path)
        if not self.ledger_path.exists():
            return False
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("fingerprint") == fingerprint:
                return True
        return False

    def record(self, *, slug: str, message: str, submission_path: Path, run_id: str | None) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "slug": slug,
            "message": message,
            "submission_path": str(submission_path),
            "sha256": sha256_file(str(submission_path)),
            "fingerprint": submission_fingerprint(slug, message, submission_path),
            "run_id": run_id,
        }
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def last_submission_time(self) -> datetime | None:
        if not self.ledger_path.exists():
            return None
        last_ts = None
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ts_str = rec.get("ts")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if last_ts is None or ts > last_ts:
                last_ts = ts
        return last_ts

    def recent_submission_count(self, *, hours: float) -> int:
        if not self.ledger_path.exists():
            return 0
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=hours)
        count = 0
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ts_str = rec.get("ts")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts >= cutoff:
                count += 1
        return count
