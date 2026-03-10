from __future__ import annotations

import csv
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.history import SubmissionLedger
from kagglebot.submission.guard import run_kaggle_submit
from kagglebot.submission.validate import infer_required_id_suffix, validate_submission
from kagglebot.submission_artifacts import (
    ARTIFACT_CLASS_BUNDLE,
    ARTIFACT_CLASS_MULTI_FILE_ZIP,
    ARTIFACT_CLASS_SINGLE_FILE,
    ARTIFACT_CLASS_TABULAR,
    ARTIFACT_CLASS_UNKNOWN,
    SUBMISSION_MANIFEST_FILENAME,
    find_submission_manifest,
    normalize_artifact_class,
    resolve_manifest_references,
)
from kagglebot.submission_format import (
    SubmissionFormatHint,
    extract_submission_section,
    load_submission_format_hint,
    parse_submission_format,
)
from kagglebot.validation import ensure_not_duplicate_submission, ensure_submission_rate_limit

_KAGGLE_SUBMISSION_SOFT_MAX_BYTES = 10_000_000
_KAGGLE_SUBMISSION_MESSAGE_MAX_CHARS = 100
_KAGGLE_SUBMISSION_COMPACT_FLOAT_FORMAT = "%.10g"
_TABULAR_SUBMISSION_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"}
_ZIP_SUBMISSION_SUFFIX = ".zip"
_SAMPLE_STAGE_PATTERN = re.compile(r"(?:stage|phase|round)[_-]?(\d+)", re.IGNORECASE)
_ID_LIKE_COLUMN_NAMES = {"id", "rowid", "row_id", "identifier"}
_PREDICTION_LIKE_COLUMN_NAMES = {
    "prediction",
    "pred",
    "target",
    "label",
    "category",
    "class",
    "value",
    "y",
    "result",
    "output",
}


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

    @staticmethod
    def _normalize_submission_message(message: str) -> str:
        normalized = " ".join(str(message or "").split()).strip()
        if not normalized:
            raise SubmissionValidationError("submission message is empty")
        if len(normalized) <= _KAGGLE_SUBMISSION_MESSAGE_MAX_CHARS:
            return normalized
        ellipsis = "..."
        keep = max(_KAGGLE_SUBMISSION_MESSAGE_MAX_CHARS - len(ellipsis), 0)
        if keep <= 0:
            return normalized[:_KAGGLE_SUBMISSION_MESSAGE_MAX_CHARS]
        return normalized[:keep].rstrip() + ellipsis

    def validate_and_prepare_submission(self, submission_path: Path) -> Path:
        self._validate_submission_input_exists(submission_path)
        format_hint = self._resolve_submission_format_hint()
        artifact_class = self._resolve_submission_artifact_class(
            format_hint=format_hint, submission_path=submission_path
        )
        manifest_path = self._resolve_submission_manifest_path(submission_path)
        if manifest_path is not None:
            artifact_class, manifest_submission_path, staging_dir, members = resolve_manifest_references(manifest_path)
            if artifact_class in {ARTIFACT_CLASS_BUNDLE, ARTIFACT_CLASS_MULTI_FILE_ZIP}:
                prepared_bundle = self._prepare_bundle_submission(
                    base_path=manifest_path,
                    submission_path=manifest_submission_path,
                    staging_dir=staging_dir,
                    members=members,
                )
                return self._enforce_expected_submission_format(
                    submission_path=prepared_bundle,
                    format_hint=format_hint,
                )
            if manifest_submission_path is not None:
                submission_path = manifest_submission_path
                artifact_class = self._resolve_submission_artifact_class(
                    format_hint=format_hint,
                    submission_path=submission_path,
                    manifest_artifact_class=artifact_class,
                )
        if submission_path.is_dir():
            if artifact_class not in {ARTIFACT_CLASS_BUNDLE, ARTIFACT_CLASS_MULTI_FILE_ZIP}:
                raise SubmissionValidationError(
                    f"submission path is a directory but artifact class is not bundle/multi_file_zip: {submission_path}"
                )
            prepared_bundle = self._prepare_bundle_submission(
                base_path=submission_path,
                submission_path=None,
                staging_dir=submission_path,
                members=[],
            )
            return self._enforce_expected_submission_format(
                submission_path=prepared_bundle,
                format_hint=format_hint,
            )
        if self._is_zip_submission(submission_path):
            self._validate_zip_submission(submission_path)
            return self._enforce_expected_submission_format(
                submission_path=submission_path,
                format_hint=format_hint,
            )
        if artifact_class == ARTIFACT_CLASS_SINGLE_FILE or not self._is_tabular_submission(submission_path):
            self._validate_non_tabular_submission_file(submission_path)
            return self._enforce_expected_submission_format(
                submission_path=submission_path,
                format_hint=format_hint,
            )

        sample_path = self._resolve_sample_submission_for_submission(submission_path=submission_path)
        try:
            validate_submission(str(submission_path), str(sample_path), data_dir=self._config.data_dir)
            prepared = self._prepare_submission_path(sample_path, submission_path)
            compacted = self._maybe_compact_submission_csv(sample_path, prepared)
            if compacted != prepared:
                validate_submission(str(compacted), str(sample_path), data_dir=self._config.data_dir)
            return self._finalize_prepared_tabular_submission(
                sample_path=sample_path,
                submission_path=compacted,
                format_hint=format_hint,
            )
        except SubmissionValidationError as exc:
            original = exc
            message = str(exc)
            autofixable = any(
                marker in message
                for marker in (
                    "columns mismatch",
                    "row count mismatch",
                    "id column missing",
                    "id values appear to require",
                    "missing a header row",
                    "header does not resemble",
                )
            )
            if not autofixable:
                raise
            autofixed = self._attempt_autofix_submission(sample_path=sample_path, submission_path=submission_path)
            if autofixed == submission_path:
                raise
            try:
                validate_submission(str(autofixed), str(sample_path), data_dir=self._config.data_dir)
            except SubmissionValidationError as exc2:
                raise SubmissionValidationError(
                    f"{original}\n\nAutofix wrote: {autofixed}\nBut validation still failed:\n{exc2}"
                ) from exc2
            prepared = self._prepare_submission_path(sample_path, autofixed)
            compacted = self._maybe_compact_submission_csv(sample_path, prepared)
            if compacted != prepared:
                validate_submission(str(compacted), str(sample_path), data_dir=self._config.data_dir)
            return self._finalize_prepared_tabular_submission(
                sample_path=sample_path,
                submission_path=compacted,
                format_hint=format_hint,
            )

    def _resolve_sample_submission_for_submission(self, *, submission_path: Path) -> Path:
        """Resolve a sample file, preferring one that already validates the current submission."""
        primary_sample = self._resolve_sample_submission()
        candidates = [primary_sample, *self._find_all_usable_sample_submissions_in_data_dir()]
        deduped: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(candidate)
        desired_stage = self._resolve_desired_submission_stage()
        primary_stage = self._sample_stage_score(primary_sample)
        primary_has_rows = primary_sample.exists() and self._has_data_rows(primary_sample)
        for candidate in deduped:
            candidate_stage = self._sample_stage_score(candidate)
            if desired_stage is None and primary_has_rows and primary_stage == 0 and candidate_stage > 0:
                # If the configured sample submission has data rows but lacks an explicit stage
                # marker in its filename, treat it as the canonical template and avoid switching
                # to staged samples implicitly. Multi-stage competitions often ship both stage
                # templates, but Kaggle will only accept one at a time.
                continue
            if desired_stage is None and primary_stage > 0 and candidate_stage > 0 and candidate_stage > primary_stage:
                # Avoid jumping to a later stage when no explicit override is set.
                continue
            try:
                validate_submission(str(submission_path), str(candidate), data_dir=self._config.data_dir)
            except SubmissionValidationError:
                continue
            return candidate
        return primary_sample

    @staticmethod
    def _validate_submission_input_exists(path: Path) -> None:
        if not path.exists():
            raise SubmissionValidationError(f"submission file not found: {path}")

    @staticmethod
    def _is_zip_submission(path: Path) -> bool:
        return path.suffix.lower() == _ZIP_SUBMISSION_SUFFIX

    @staticmethod
    def _is_tabular_submission(path: Path) -> bool:
        return path.suffix.lower() in _TABULAR_SUBMISSION_SUFFIXES

    @staticmethod
    def _validate_non_tabular_submission_file(path: Path) -> None:
        if not path.is_file():
            raise SubmissionValidationError(f"submission path is not a file: {path}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise SubmissionValidationError(f"unable to read submission file metadata: {path}") from exc
        if size <= 0:
            raise SubmissionValidationError(f"submission file is empty: {path}")

    @staticmethod
    def _validate_zip_submission(path: Path) -> None:
        try:
            with zipfile.ZipFile(path, "r") as archive:
                members = [info for info in archive.infolist() if not info.is_dir()]
        except (OSError, zipfile.BadZipFile) as exc:
            raise SubmissionValidationError(f"submission zip is invalid: {path}") from exc
        if not members:
            raise SubmissionValidationError(f"submission zip has no files: {path}")

    def _finalize_prepared_tabular_submission(
        self,
        *,
        sample_path: Path,
        submission_path: Path,
        format_hint: SubmissionFormatHint | None,
    ) -> Path:
        """Apply format constraints and re-validate the final prepared submission."""
        finalized = self._enforce_expected_submission_format(
            submission_path=submission_path,
            format_hint=format_hint,
        )
        if self._is_zip_submission(finalized):
            self._validate_zip_submission(finalized)
            return finalized
        if self._is_tabular_submission(finalized):
            validate_submission(str(finalized), str(sample_path), data_dir=self._config.data_dir)
            return finalized
        self._validate_non_tabular_submission_file(finalized)
        return finalized

    def _resolve_submission_format_hint(self) -> SubmissionFormatHint | None:
        """Load submission format hints from context files."""
        for context_dir in self._candidate_context_dirs():
            format_hint = load_submission_format_hint(context_dir / "submission_format.md")
            if format_hint is not None and self._hint_has_any_signal(format_hint):
                return format_hint
            for name in ("overview.md", "data.md", "rules.md", "discussion.md"):
                path = context_dir / name
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                section = extract_submission_section(text) or ""
                if not section.strip():
                    continue
                hint = parse_submission_format(section)
                if self._hint_has_any_signal(hint):
                    return hint
        return None

    @staticmethod
    def _hint_has_any_signal(hint: SubmissionFormatHint) -> bool:
        """Return whether a parsed hint has usable signals."""
        return bool(hint.columns or hint.delimiter or hint.expected_suffixes or hint.artifact_class)

    def _candidate_context_dirs(self) -> list[Path]:
        """Discover possible context directories that may contain submission hints."""
        candidates: list[Path] = []
        seen: set[Path] = set()

        def add(path: Path) -> None:
            resolved = path.resolve()
            if resolved in seen:
                return
            seen.add(resolved)
            candidates.append(resolved)

        sample_parent = self._config.sample_submission_path.parent
        data_dir = self._config.data_dir
        add(sample_parent)
        add(data_dir)
        for root in [sample_parent, *sample_parent.parents]:
            add(root / "context")
        for root in [data_dir, *data_dir.parents]:
            add(root / "context")
        return candidates

    def _enforce_expected_submission_format(
        self,
        *,
        submission_path: Path,
        format_hint: SubmissionFormatHint | None,
    ) -> Path:
        """Coerce submission artifact into the expected file format when inferable."""
        expected_suffixes = self._expected_submission_suffixes(format_hint)
        if not expected_suffixes:
            return submission_path
        current_suffix = submission_path.suffix.lower()
        if current_suffix in expected_suffixes:
            return submission_path
        preferred_suffix = expected_suffixes[0]
        converted = self._convert_submission_to_suffix(
            submission_path=submission_path,
            target_suffix=preferred_suffix,
            format_hint=format_hint,
        )
        if converted is not None:
            return converted
        expected_display = ", ".join(expected_suffixes)
        actual_display = current_suffix or "<no extension>"
        raise SubmissionValidationError(
            "submission file format mismatch:\n"
            f"  expected one of: {expected_display}\n"
            f"  actual:          {actual_display}\n"
            f"  file:            {submission_path}"
        )

    def _resolve_submission_artifact_class(
        self,
        *,
        format_hint: SubmissionFormatHint | None,
        submission_path: Path,
        manifest_artifact_class: str | None = None,
    ) -> str:
        manifest_class = normalize_artifact_class(manifest_artifact_class, default="")
        if manifest_class:
            return manifest_class
        hint_class = normalize_artifact_class(getattr(format_hint, "artifact_class", None), default="")
        if hint_class:
            return hint_class
        if submission_path.is_dir():
            return ARTIFACT_CLASS_BUNDLE
        if self._is_zip_submission(submission_path):
            return ARTIFACT_CLASS_MULTI_FILE_ZIP
        if self._is_tabular_submission(submission_path):
            return ARTIFACT_CLASS_TABULAR
        if submission_path.name == SUBMISSION_MANIFEST_FILENAME:
            return ARTIFACT_CLASS_UNKNOWN
        return ARTIFACT_CLASS_SINGLE_FILE

    @staticmethod
    def _resolve_submission_manifest_path(submission_path: Path) -> Path | None:
        if submission_path.is_file() and submission_path.name == SUBMISSION_MANIFEST_FILENAME:
            return submission_path
        if submission_path.is_dir():
            return find_submission_manifest(submission_path)
        parent_manifest = find_submission_manifest(submission_path.parent)
        return parent_manifest

    @staticmethod
    def _expected_submission_suffixes(format_hint: SubmissionFormatHint | None) -> list[str]:
        """Extract normalized expected suffixes from a parsed submission format hint."""
        if format_hint is None or not format_hint.expected_suffixes:
            return []
        suffixes: list[str] = []
        for suffix in format_hint.expected_suffixes:
            normalized = str(suffix or "").strip().lower()
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            if normalized not in suffixes:
                suffixes.append(normalized)
        return suffixes

    def _convert_submission_to_suffix(
        self,
        *,
        submission_path: Path,
        target_suffix: str,
        format_hint: SubmissionFormatHint | None,
    ) -> Path | None:
        """Convert submission artifact to target suffix when safe and deterministic."""
        if target_suffix == ".zip":
            return self._build_submission_zip(submission_path)
        if target_suffix not in _TABULAR_SUBMISSION_SUFFIXES:
            return None
        if not self._is_tabular_submission(submission_path):
            return None
        try:
            frame = self._read_tabular_submission(submission_path)
        except Exception:
            return None
        destination = submission_path.with_suffix(target_suffix)
        try:
            self._write_tabular_submission(
                frame=frame,
                destination=destination,
                target_suffix=target_suffix,
                format_hint=format_hint,
            )
        except Exception:
            return None
        return destination

    @staticmethod
    def _build_submission_zip(
        submission_path: Path,
        *,
        members: list[Path] | None = None,
        staging_dir: Path | None = None,
    ) -> Path:
        """Create a zip archive from a single file or a staged directory."""
        if submission_path.is_dir():
            source_dir = submission_path
            archive_members = members or [path for path in source_dir.rglob("*") if path.is_file()]
            destination = source_dir.parent / f"{source_dir.name}.zip"
        else:
            source_dir = staging_dir
            archive_members = members or [submission_path]
            destination = submission_path.with_suffix(".zip")
        if not archive_members:
            raise SubmissionValidationError(f"submission bundle has no files to archive: {submission_path}")
        with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member in archive_members:
                if not member.exists() or not member.is_file():
                    raise SubmissionValidationError(f"submission bundle member missing: {member}")
                if source_dir is not None:
                    try:
                        arcname = member.relative_to(source_dir).as_posix()
                    except ValueError:
                        arcname = member.name
                else:
                    arcname = member.name
                archive.write(member, arcname=arcname)
        return destination

    def _prepare_bundle_submission(
        self,
        *,
        base_path: Path,
        submission_path: Path | None,
        staging_dir: Path | None,
        members: list[Path],
    ) -> Path:
        if submission_path is not None and submission_path.exists() and submission_path.is_file():
            if self._is_zip_submission(submission_path):
                self._validate_zip_submission(submission_path)
                return submission_path
            if not members and staging_dir is None:
                return self._build_submission_zip(submission_path)
        source_dir = staging_dir
        if source_dir is None and submission_path is not None and submission_path.is_dir():
            source_dir = submission_path
        if source_dir is None and base_path.is_dir():
            source_dir = base_path
        if source_dir is None:
            raise SubmissionValidationError("bundle submission requires a staging_dir or archive file")
        archive_members = members or [path for path in source_dir.rglob("*") if path.is_file()]
        prepared = self._build_submission_zip(source_dir, members=archive_members)
        self._validate_zip_submission(prepared)
        return prepared

    def _read_tabular_submission(self, path: Path):
        """Read a tabular submission using a suffix-aware parser."""
        try:
            import pandas as pd
        except Exception as exc:  # noqa: BLE001
            raise SubmissionValidationError("pandas is required to convert submission format") from exc
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".json", ".jsonl"}:
            try:
                return pd.read_json(path, lines=(suffix == ".jsonl"))
            except ValueError:
                return pd.read_json(path)
        if suffix in {".tsv", ".txt"}:
            sep = "\t"
            if suffix == ".txt":
                sep = self._sniff_delimiter(path, default="\t")
            return pd.read_csv(path, sep=sep)
        return pd.read_csv(path)

    def _write_tabular_submission(
        self,
        *,
        frame,
        destination: Path,
        target_suffix: str,
        format_hint: SubmissionFormatHint | None,
    ) -> None:
        """Write a tabular submission frame in the requested target format."""
        if target_suffix == ".parquet":
            frame.to_parquet(destination, index=False)
            return
        if target_suffix == ".jsonl":
            frame.to_json(destination, orient="records", lines=True)
            return
        if target_suffix == ".json":
            frame.to_json(destination, orient="records")
            return
        if target_suffix == ".tsv":
            frame.to_csv(destination, index=False, sep="\t")
            return
        if target_suffix == ".txt":
            sep = "\t"
            if format_hint is not None and format_hint.delimiter in {",", "\t"}:
                sep = format_hint.delimiter
            frame.to_csv(destination, index=False, sep=sep)
            return
        frame.to_csv(destination, index=False, sep=",")

    def submit_prepared(self, *, prepared_path: Path, message: str, run_id: str | None) -> SubmissionResult:
        message = self._normalize_submission_message(message)
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
        if not self._config.dry_run:
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
        discovered_text_sample = self._find_usable_sample_submission_in_data_dir()
        if discovered_text_sample is not None:
            return discovered_text_sample
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

    def _maybe_compact_submission_csv(self, sample_path: Path, submission_path: Path) -> Path:
        if not sample_path.exists() or not submission_path.exists():
            return submission_path
        if submission_path.suffix.lower() != ".csv":
            return submission_path
        try:
            size_bytes = submission_path.stat().st_size
        except OSError:
            return submission_path
        if size_bytes <= _KAGGLE_SUBMISSION_SOFT_MAX_BYTES:
            return submission_path

        sample_delim = self._sniff_delimiter(sample_path)
        submission_delim = self._sniff_delimiter(submission_path, default=sample_delim)
        if submission_delim != ",":
            return submission_path

        try:
            import pandas as pd
        except Exception:
            return submission_path

        try:
            frame = pd.read_csv(submission_path, sep=submission_delim)
        except Exception:
            return submission_path

        compact_path = submission_path.with_name(f"{submission_path.stem}.compact{submission_path.suffix}")
        try:
            frame.to_csv(
                compact_path,
                index=False,
                sep=submission_delim,
                float_format=_KAGGLE_SUBMISSION_COMPACT_FLOAT_FORMAT,
            )
        except Exception:
            return submission_path

        try:
            compact_size = compact_path.stat().st_size
        except OSError:
            return submission_path
        if compact_size >= size_bytes:
            return submission_path
        return compact_path

    def _attempt_autofix_submission(self, *, sample_path: Path, submission_path: Path) -> Path:
        """Best-effort local fixups to make a submission match sample_submission.csv.

        This is intentionally conservative:
        - Only runs after strict validation fails.
        - Produces a new file next to the original submission.
        - Uses sample_submission rows/columns as the template when present.
        """
        if not sample_path.exists() or not submission_path.exists():
            return submission_path

        try:
            import pandas as pd
        except Exception:
            return submission_path

        sample_delim = self._sniff_delimiter(sample_path)
        expected = pd.read_csv(sample_path, sep=sample_delim)
        if expected.columns.empty:
            return submission_path

        expected_columns = list(expected.columns)
        sample_has_data_rows = self._has_data_rows(sample_path)
        if not sample_has_data_rows and self._is_synthesized_sample_submission(sample_path):
            format_hint = self._resolve_submission_format_hint()
            if (
                format_hint is not None
                and format_hint.columns
                and list(format_hint.columns) != expected_columns
                and self._sample_columns_look_placeholder(expected_columns)
            ):
                expected_columns = list(format_hint.columns)

        source_submission_path = submission_path
        submission_delim = self._sniff_delimiter(source_submission_path, default=sample_delim)
        columns_only_autofix = self._attempt_autofix_submission_columns_only(
            expected_columns=expected_columns,
            sample_delim=sample_delim,
            submission_path=source_submission_path,
            submission_delim=submission_delim,
        )
        if columns_only_autofix is not None:
            source_submission_path = columns_only_autofix
            submission_delim = sample_delim

        id_suffix_autofix = self._attempt_autofix_submission_id_suffix(
            sample_path=sample_path,
            expected_columns=expected_columns,
            sample_delim=sample_delim,
            submission_path=source_submission_path,
            submission_delim=submission_delim,
        )
        if id_suffix_autofix is not None:
            source_submission_path = id_suffix_autofix
            submission_delim = sample_delim

        if not sample_has_data_rows:
            return source_submission_path

        pred_cols = []
        for col in expected_columns[1:]:
            if col in expected.columns and self._sample_column_is_numeric(expected[col]):
                pred_cols.append(col)
        if not pred_cols and len(expected_columns) >= 2:
            pred_cols = expected_columns[1:]
        key_cols = [c for c in expected_columns if c not in pred_cols] or [expected_columns[0]]

        header, col_index = self._sniff_header_and_column_index(
            submission_path=source_submission_path, delim=submission_delim, expected_columns=expected_columns
        )
        if col_index is None:
            return submission_path

        key_positions = [col_index[c] for c in key_cols if c in col_index]
        pred_positions = [col_index[c] for c in pred_cols if c in col_index]
        if not key_positions or not pred_positions:
            return submission_path

        # Aggregate predictions by key (mean for duplicates).
        sums: dict[tuple[str, ...], list[float]] = {}
        counts: dict[tuple[str, ...], int] = {}
        try:
            with source_submission_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.reader(handle, delimiter=submission_delim)
                first = True
                for row in reader:
                    if not row or all(not cell.strip() for cell in row):
                        continue
                    if first and header:
                        first = False
                        continue
                    first = False
                    if len(row) < max(key_positions + pred_positions) + 1:
                        continue
                    key = tuple(str(row[pos]).strip() for pos in key_positions)
                    if any(not part for part in key):
                        continue
                    values: list[float] = []
                    ok = True
                    for pos in pred_positions:
                        try:
                            values.append(float(str(row[pos]).strip()))
                        except ValueError:
                            ok = False
                            break
                    if not ok:
                        continue
                    if key not in sums:
                        sums[key] = values
                        counts[key] = 1
                    else:
                        sums[key] = [a + b for a, b in zip(sums[key], values, strict=False)]
                        counts[key] += 1
        except OSError:
            return submission_path

        if not sums:
            return submission_path

        if not sample_has_data_rows:
            return submission_path
        prepared = self._expand_autofix_template_if_placeholder(
            sample=expected,
            key_cols=key_cols,
            pred_cols=pred_cols,
        )
        if prepared is None:
            prepared = expected.copy()

        # Fill prediction columns by matching keys against the sample template.
        key_df = prepared[key_cols].astype(str)
        for pred_idx, pred_col in enumerate(pred_cols):
            if pred_col not in prepared.columns:
                continue
            filled = []
            default_series = prepared[pred_col]
            for row_idx in range(len(prepared)):
                key = tuple(key_df.iloc[row_idx, k].strip() for k in range(len(key_cols)))
                if key in sums:
                    filled.append(sums[key][pred_idx] / counts[key])
                else:
                    filled.append(default_series.iloc[row_idx])
            prepared[pred_col] = pd.to_numeric(pd.Series(filled), errors="coerce").fillna(default_series)

        prepared = prepared[expected_columns]
        prepared_path = submission_path.with_name(
            f"{submission_path.stem}.autofixed{'.tsv' if sample_delim == chr(9) else '.csv'}"
        )
        prepared.to_csv(prepared_path, index=False, sep=sample_delim)
        return prepared_path

    @classmethod
    def _sample_columns_look_placeholder(cls, columns: list[str]) -> bool:
        if len(columns) != 2:
            return False
        normalized = [cls._normalize_column_name(c) for c in columns]
        id_like = normalized[0] in {cls._normalize_column_name(v) for v in _ID_LIKE_COLUMN_NAMES}
        prediction_like = normalized[1] in {cls._normalize_column_name(v) for v in _PREDICTION_LIKE_COLUMN_NAMES} | {
            "target",
            "label",
            "value",
            "y",
        }
        return bool(id_like and prediction_like)

    @staticmethod
    def _is_synthesized_sample_submission(path: Path) -> bool:
        name = path.name.lower()
        if "sample_submission_synth" in name:
            return True
        if ".kagglebot_cache" in {part.lower() for part in path.parts}:
            return True
        return False

    def _attempt_autofix_submission_columns_only(
        self,
        *,
        expected_columns: list[str],
        sample_delim: str,
        submission_path: Path,
        submission_delim: str,
    ) -> Path | None:
        """Try to rewrite the submission with expected columns/ordering.

        This handles common failures where the submission is structurally correct but uses
        different column names/casing (e.g., `Id,Category` vs `id,prediction`), including
        header-only sample submissions.
        """
        if not expected_columns:
            return None

        try:
            import pandas as pd
        except Exception:
            return None

        try:
            frame = pd.read_csv(submission_path, sep=submission_delim)
        except Exception:
            return None

        actual_columns = list(frame.columns)
        if len(actual_columns) != len(expected_columns):
            return None

        mapping = self._resolve_column_mapping(expected_columns=expected_columns, actual_columns=actual_columns)
        if mapping is None:
            return None

        if expected_columns == actual_columns and all(mapping.get(col) == col for col in actual_columns):
            return None

        renamed = frame.rename(columns=mapping, errors="raise")
        try:
            renamed = renamed[expected_columns]
        except Exception:
            return None

        prepared_path = submission_path.with_name(
            f"{submission_path.stem}.autofixed{'.tsv' if sample_delim == chr(9) else '.csv'}"
        )
        try:
            renamed.to_csv(prepared_path, index=False, sep=sample_delim)
        except Exception:
            return None
        return prepared_path

    def _attempt_autofix_submission_id_suffix(
        self,
        *,
        sample_path: Path,
        expected_columns: list[str],
        sample_delim: str,
        submission_path: Path,
        submission_delim: str,
    ) -> Path | None:
        """Append a required filename suffix to id values when inferred confidently."""
        if not expected_columns:
            return None

        try:
            import pandas as pd
        except Exception:
            return None

        try:
            frame = pd.read_csv(submission_path, sep=submission_delim)
        except Exception:
            return None
        if frame.empty:
            return None

        id_col = expected_columns[0]
        if id_col not in frame.columns:
            return None

        raw_ids = [str(value).strip() for value in frame[id_col].tolist()]
        required_suffix = infer_required_id_suffix(
            sample_csv=sample_path,
            data_dir=self._config.data_dir,
            submission_ids=raw_ids,
        )
        if not required_suffix:
            return None

        rewritten = (
            frame[id_col]
            .astype(str)
            .map(
                lambda raw: (
                    raw.strip() if (not raw.strip()) or Path(raw.strip()).suffix else f"{raw.strip()}{required_suffix}"
                )
            )
        )
        current = frame[id_col].astype(str).map(str.strip)
        if rewritten.equals(current):
            return None

        frame[id_col] = rewritten
        if all(col in frame.columns for col in expected_columns):
            frame = frame[expected_columns]

        prepared_path = submission_path.with_name(
            f"{submission_path.stem}.autofixed{'.tsv' if sample_delim == chr(9) else '.csv'}"
        )
        try:
            frame.to_csv(prepared_path, index=False, sep=sample_delim)
        except Exception:
            return None
        return prepared_path

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())

    @classmethod
    def _semantic_group_for_column(cls, normalized: str) -> str:
        if normalized in {cls._normalize_column_name(v) for v in _ID_LIKE_COLUMN_NAMES}:
            return "id"
        if normalized in {cls._normalize_column_name(v) for v in _PREDICTION_LIKE_COLUMN_NAMES}:
            return "prediction"
        return normalized

    @classmethod
    def _resolve_column_mapping(
        cls, *, expected_columns: list[str], actual_columns: list[str]
    ) -> dict[str, str] | None:
        expected_norm = [cls._normalize_column_name(c) for c in expected_columns]
        actual_norm = [cls._normalize_column_name(c) for c in actual_columns]

        # Fast-paths: exact or case/format-only differences.
        if expected_columns == actual_columns:
            return {c: c for c in actual_columns}
        if expected_norm == actual_norm:
            return {actual: expected for actual, expected in zip(actual_columns, expected_columns, strict=False)}

        expected_groups = [cls._semantic_group_for_column(n) for n in expected_norm]
        actual_groups = [cls._semantic_group_for_column(n) for n in actual_norm]

        if len(expected_groups) != len(actual_groups):
            return None

        remaining_actual = set(range(len(actual_columns)))
        mapping: dict[str, str] = {}

        # 1) Prefer exact normalized matches.
        for exp_idx, exp_norm in enumerate(expected_norm):
            matches = [i for i in remaining_actual if actual_norm[i] == exp_norm]
            if len(matches) == 1:
                act_idx = matches[0]
                remaining_actual.remove(act_idx)
                mapping[actual_columns[act_idx]] = expected_columns[exp_idx]

        # 2) Then match by semantic group (id/prediction), but only when unambiguous.
        for exp_idx, exp_group in enumerate(expected_groups):
            expected_name = expected_columns[exp_idx]
            if expected_name in mapping.values():
                continue
            matches = [i for i in remaining_actual if actual_groups[i] == exp_group]
            if len(matches) != 1:
                return None
            act_idx = matches[0]
            remaining_actual.remove(act_idx)
            mapping[actual_columns[act_idx]] = expected_name

        if len(mapping) != len(expected_columns):
            return None
        return mapping

    def _expand_autofix_template_if_placeholder(
        self,
        *,
        sample,
        key_cols: list[str],
        pred_cols: list[str],
    ):
        """Return a test-sized template when sample_submission rows look truncated.

        Some competitions ship a tiny example `sample_submission.csv` while Kaggle expects
        one row per test id. If we can confidently detect that pattern, expand the template
        to all test ids and use safe numeric defaults for missing predictions.
        """
        try:
            import pandas as pd
        except Exception:
            return None

        if len(key_cols) != 1:
            return None
        id_col = key_cols[0]
        if id_col not in getattr(sample, "columns", ()):
            return None
        if len(sample) > 10:
            return None
        if sample[id_col].duplicated().any():
            return None

        try:
            from kagglebot.solver.io import find_competition_files
        except Exception:
            return None
        try:
            train_path, test_path, _ = find_competition_files(self._config.data_dir)
        except Exception:  # noqa: BLE001
            return None
        if not test_path.exists():
            return None

        try:
            test_delim = self._sniff_delimiter(test_path)
            test = pd.read_csv(test_path, sep=test_delim, usecols=[id_col], dtype={id_col: str})
        except Exception:  # noqa: BLE001
            return None
        if id_col not in test.columns:
            return None

        test_ids = test[id_col].astype(str).tolist()
        sample_ids = sample[id_col].astype(str).tolist()
        if len(test_ids) <= max(len(sample_ids) * 3, len(sample_ids) + 10):
            return None
        if sample_ids and test_ids[: len(sample_ids)] != sample_ids:
            return None

        defaults: dict[str, float] = {col: 0.0 for col in pred_cols}
        if train_path.exists() and pred_cols:
            try:
                train_delim = self._sniff_delimiter(train_path)
                train = pd.read_csv(train_path, sep=train_delim, usecols=[c for c in pred_cols if c != id_col])
            except Exception:  # noqa: BLE001
                train = None
            if train is not None:
                for col in pred_cols:
                    if col not in train.columns:
                        continue
                    series = pd.to_numeric(train[col], errors="coerce").dropna()
                    if not series.empty:
                        defaults[col] = float(series.mean())

        template = pd.DataFrame({id_col: test_ids})
        for col in pred_cols:
            template[col] = defaults.get(col, 0.0)
        # Match sample column order (and preserve any additional key columns if present).
        for col in list(sample.columns):
            if col not in template.columns:
                template[col] = ""
        return template[list(sample.columns)]

    @staticmethod
    def _sniff_header_and_column_index(
        *, submission_path: Path, delim: str, expected_columns: list[str]
    ) -> tuple[bool, dict[str, int] | None]:
        try:
            with submission_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    header_fields = next(csv.reader([line], delimiter=delim))
                    normalized = [f.strip() for f in header_fields]
                    if normalized == expected_columns:
                        return True, {name: i for i, name in enumerate(normalized)}
                    if len(normalized) == len(expected_columns) and set(normalized) == set(expected_columns):
                        return True, {name: i for i, name in enumerate(normalized)}
                    # Headerless: assume the file is in expected column order.
                    return False, {name: i for i, name in enumerate(expected_columns)}
        except OSError:
            return False, None
        return False, None

    @staticmethod
    def _sample_column_is_numeric(sample_col) -> bool:  # type: ignore[no-untyped-def]
        try:
            import pandas as pd
        except Exception:
            return False
        if sample_col is None or getattr(sample_col, "empty", True):
            return False
        if pd.api.types.is_numeric_dtype(sample_col):
            return True
        coerced = pd.to_numeric(sample_col, errors="coerce")
        return coerced.notna().all()

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

    def _find_usable_sample_submission_in_data_dir(self) -> Path | None:
        candidates = self._find_all_usable_sample_submissions_in_data_dir()
        if not candidates:
            return None
        return candidates[0]

    def _find_all_usable_sample_submissions_in_data_dir(self) -> list[Path]:
        """List usable sample-submission CSV candidates in ranked preference order."""
        data_dir = self._config.data_dir
        if not data_dir.exists():
            return []
        candidates: list[Path] = []
        for path in data_dir.rglob("*.csv"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if ("sample" not in name) or ("submission" not in name):
                continue
            if not self._has_data_rows(path):
                continue
            if not self._csv_has_two_or_more_columns(path):
                continue
            candidates.append(path)
        return sorted(candidates, key=self._sample_candidate_key, reverse=True)

    @staticmethod
    def _sample_candidate_key(path: Path) -> tuple[int, int, int, int, str]:
        """Return ranking key for sample-submission CSV candidates."""
        name_score = SubmissionService._sample_name_score(path)
        stage_score = SubmissionService._sample_stage_score(path)
        desired_stage = SubmissionService._resolve_desired_submission_stage()
        stage_match = 1 if (desired_stage is not None and stage_score == desired_stage) else 0
        stage_preference = 0 if stage_score == 0 else -stage_score
        desired_distance = 0
        if desired_stage is not None:
            desired_distance = -abs(stage_score - desired_stage) if stage_score else -10_000
        return (name_score, stage_match, desired_distance, stage_preference, path.name.lower())

    @staticmethod
    def _sample_name_score(path: Path) -> int:
        """Score how clearly a filename indicates a sample-submission CSV."""
        name = path.name.lower()
        compact = name.replace("_", "")
        if "sample_submission" in name or "samplesubmission" in compact:
            return 3
        if "sample" in name and "submission" in name:
            return 2
        if "submission" in name:
            return 1
        return 0

    @staticmethod
    def _sample_stage_score(path: Path) -> int:
        """Extract stage/phase/round number from filename for ranking."""
        matches = _SAMPLE_STAGE_PATTERN.findall(path.name.lower())
        if not matches:
            return 0
        return max(int(value) for value in matches)

    @staticmethod
    def _resolve_desired_submission_stage() -> int | None:
        raw = (
            os.environ.get("KAGGLEBOT_SUBMISSION_STAGE") or os.environ.get("KAGGLEBOT_SAMPLE_SUBMISSION_STAGE") or ""
        ).strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        if value <= 0:
            return None
        return value

    @staticmethod
    def _csv_has_two_or_more_columns(path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = next(csv.reader([line], delimiter=","))
                    return len(row) >= 2
        except OSError:
            return False
        return False
