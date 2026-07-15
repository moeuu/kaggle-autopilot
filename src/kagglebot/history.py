from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from kagglebot.datetime_utils import parse_iso_datetime_utc
from kagglebot.hashing import sha256_path
from kagglebot.json_utils import append_jsonl_record, load_jsonl_records


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def submission_fingerprint(slug: str, message: str, submission_path: Path) -> str:
    file_hash = sha256_path(submission_path)
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

    @staticmethod
    def _is_submit_event(record: dict[str, object]) -> bool:
        event = record.get("event")
        return event in (None, "submit")

    def _iter_records(self) -> list[dict[str, object]]:
        return load_jsonl_records(self.ledger_path)

    @staticmethod
    def _record_timestamp(record: dict[str, object]) -> datetime | None:
        return parse_iso_datetime_utc(record.get("ts"))

    def is_duplicate(
        self,
        *,
        slug: str,
        message: str,
        submission_path: Path,
        submission_identity: str | None = None,
    ) -> bool:
        normalized_identity = str(submission_identity or "").strip()
        if normalized_identity:
            return any(
                self._is_submit_event(rec)
                and rec.get("slug") == slug
                and rec.get("submission_identity") == normalized_identity
                for rec in self._iter_records()
            )
        fingerprint = submission_fingerprint(slug, message, submission_path)
        submission_sha = sha256_path(submission_path)
        for rec in self._iter_records():
            if rec.get("fingerprint") == fingerprint:
                return True
            if not self._is_submit_event(rec):
                continue
            if rec.get("slug") == slug and rec.get("sha256") == submission_sha:
                return True
        return False

    def record(
        self,
        *,
        slug: str,
        message: str,
        submission_path: Path,
        run_id: str | None,
        iteration: int | None = None,
        metrics_path: Path | None = None,
        offline_score: float | None = None,
        score_source: str | None = None,
        pipeline_name: str | None = None,
        submission_kind: str | None = None,
        out_of_band: bool = False,
        source_run_id: str | None = None,
        source_iteration: int | None = None,
        submission_identity: str | None = None,
        submission_ref: str | None = None,
    ) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        submission_sha = sha256_path(submission_path)
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": "submit",
            "slug": slug,
            "message": message,
            "submission_path": str(submission_path),
            "sha256": submission_sha,
            "fingerprint": submission_fingerprint(slug, message, submission_path),
            "run_id": run_id,
            "iteration": iteration,
            "metrics_path": str(metrics_path) if metrics_path is not None else None,
            "offline_score": offline_score,
            "score_source": score_source,
            "pipeline_name": pipeline_name,
            "submission_kind": submission_kind,
            "out_of_band": out_of_band,
            "source_run_id": source_run_id,
            "source_iteration": source_iteration,
            "submission_identity": str(submission_identity or "").strip() or None,
            "submission_ref": str(submission_ref or "").strip() or None,
        }
        append_jsonl_record(self.ledger_path, record)

    def record_outcome(
        self,
        *,
        slug: str,
        message: str,
        submission_path: Path,
        run_id: str | None,
        outcome: dict[str, object],
        submission_kind: str | None = None,
        out_of_band: bool = False,
        source_run_id: str | None = None,
        source_iteration: int | None = None,
    ) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": "outcome",
            "slug": slug,
            "message": message,
            "submission_path": str(submission_path),
            "sha256": sha256_path(submission_path),
            "run_id": run_id,
            "outcome": outcome,
            "submission_kind": submission_kind,
            "out_of_band": out_of_band,
            "source_run_id": source_run_id,
            "source_iteration": source_iteration,
        }
        append_jsonl_record(self.ledger_path, record)

    def last_submission_time(self) -> datetime | None:
        last_ts = None
        for rec in self._iter_records():
            if not self._is_submit_event(rec):
                continue
            ts = self._record_timestamp(rec)
            if ts is None:
                continue
            if last_ts is None or ts > last_ts:
                last_ts = ts
        return last_ts

    def recent_submission_count(self, *, hours: float) -> int:
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=hours)
        count = 0
        for rec in self._iter_records():
            if not self._is_submit_event(rec):
                continue
            ts = self._record_timestamp(rec)
            if ts is None:
                continue
            if ts >= cutoff:
                count += 1
        return count
