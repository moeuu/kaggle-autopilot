from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from kagglebot.history import SubmissionLedger
from kagglebot.submission.guard import run_kaggle_submit
from kagglebot.submission.validate import validate_submission
from kagglebot.validation import ensure_not_duplicate_submission, ensure_submission_rate_limit


@dataclass(frozen=True)
class SubmissionConfig:
    slug: str
    data_dir: Path
    sample_submission_path: Path
    submission_ledger_path: Path
    dry_run: bool = False
    force_submit: bool = False
    bypass_rate_limit: bool = False


@dataclass(frozen=True)
class SubmissionResult:
    message: str
    submission_path: Path
    exit_code: int
    stdout: str
    stderr: str


class SubmissionService:
    def __init__(self, config: SubmissionConfig):
        self._config = config

    def submit(self, *, submission_path: Path, message: str, run_id: str | None) -> SubmissionResult:
        prepared_path = self.validate_and_prepare_submission(submission_path)
        return self.submit_prepared(prepared_path=prepared_path, message=message, run_id=run_id)

    def validate_and_prepare_submission(self, submission_path: Path) -> Path:
        sample_path = self._resolve_sample_submission()
        validate_submission(str(submission_path), str(sample_path))
        return self._prepare_submission_path(sample_path, submission_path)

    def submit_prepared(self, *, prepared_path: Path, message: str, run_id: str | None) -> SubmissionResult:
        ledger = SubmissionLedger(self._config.submission_ledger_path)
        if not self._config.bypass_rate_limit:
            ensure_submission_rate_limit(ledger)
        if not self._config.force_submit:
            ensure_not_duplicate_submission(
                ledger,
                slug=self._config.slug,
                message=message,
                submission_path=str(prepared_path),
            )

        command_result = run_kaggle_submit(
            slug=self._config.slug,
            submission_file=prepared_path,
            message=message,
            dry_run=self._config.dry_run,
        )
        ledger.record(
            slug=self._config.slug,
            message=message,
            submission_path=prepared_path,
            run_id=run_id,
        )
        return SubmissionResult(
            message=message,
            submission_path=prepared_path,
            exit_code=command_result.returncode,
            stdout=command_result.stdout,
            stderr=command_result.stderr,
        )

    def _resolve_sample_submission(self) -> Path:
        sample_path = self._config.sample_submission_path
        if sample_path.exists():
            if self._has_data_rows(sample_path):
                return sample_path
        synthesized = self._config.data_dir / ".kagglebot_cache" / "sample_submission_synth.csv"
        if synthesized.exists() and self._has_data_rows(synthesized):
            return synthesized

        from kagglebot.solver.io import ensure_sample_submission, find_competition_files

        discovered: Path | None = None
        try:
            _, _, discovered = find_competition_files(self._config.data_dir)
        except FileNotFoundError:
            pass
        ensured = ensure_sample_submission(self._config.data_dir)
        for candidate in (discovered, ensured, synthesized):
            if candidate is None:
                continue
            if candidate.exists() and self._has_data_rows(candidate):
                return candidate
        return sample_path

    def _prepare_submission_path(self, sample_path: Path, submission_path: Path) -> Path:
        if not sample_path.exists() or not submission_path.exists():
            return submission_path
        sample_delim = self._sniff_delimiter(sample_path)
        submission_delim = self._sniff_delimiter(submission_path)
        if sample_delim == "\t" and submission_delim == "\t" and submission_path.suffix.lower() != ".tsv":
            tsv_path = submission_path.with_suffix(".tsv")
            if tsv_path != submission_path:
                shutil.copy2(submission_path, tsv_path)
            return tsv_path
        return submission_path

    @staticmethod
    def _sniff_delimiter(path: Path, default: str = ",") -> str:
        candidates: list[str] = []
        for sep in (default, "\t", ","):
            if sep and sep not in candidates:
                candidates.append(sep)
        counts = {sep: 0 for sep in candidates}
        lines_seen = 0
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    lines_seen += 1
                    for sep in candidates:
                        counts[sep] += line.count(sep)
                    if lines_seen >= 100:
                        break
        except OSError:
            return default
        if lines_seen == 0:
            return default
        best = max(candidates, key=lambda sep: counts[sep])
        if counts[best] == 0:
            return default
        if counts.get(default, 0) >= counts[best]:
            return default
        return best

    @staticmethod
    def _has_data_rows(path: Path) -> bool:
        non_empty = 0
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    non_empty += 1
                    if non_empty >= 2:
                        return True
        except OSError:
            return True
        return False
