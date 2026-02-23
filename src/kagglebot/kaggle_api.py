from __future__ import annotations

import csv
import logging
import math
import os
import re
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.competition import parse_competition_slug
from kagglebot.exceptions import KaggleCliError, KaggleNetworkError, KernelCapacityError, RulesNotAcceptedError
from kagglebot.exec_utils import run_command
from kagglebot.submission.guard import run_kaggle_submit
from kagglebot.validators import safe_extract_zip

_KERNEL_URL_RE = re.compile(
    r"https?://(?:www\.)?kaggle\.com/(?:code/)?(?P<user>[A-Za-z0-9_-]+)/(?P<slug>[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
_KERNEL_ID_RE = re.compile(r"^(?P<user>[A-Za-z0-9_-]+)/(?P<slug>[A-Za-z0-9_.-]+)$")
_NEXT_PAGE_TOKEN_PREFIX = "next page token ="
_FILES_PAGE_SIZE = 200
_DEFAULT_SPLIT_THRESHOLD_BYTES = 8 * 1024**3
_DEFAULT_DOWNLOAD_ATTEMPTS = 8
_DEFAULT_RETRY_BACKOFF_SEC = 2.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CompetitionFile:
    name: str
    size_bytes: int


DownloadProgressCallback = Callable[[int, int, str | None], None]


def _normalize_kernel_title(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _get_row_value(row: dict[str, str], key: str) -> str | None:
    target = key.lower()
    for k, v in row.items():
        if k and k.strip().lower() == target:
            return v
    return None


def download_competition(
    slug: str,
    dest_dir: Path,
    *,
    force: bool,
    quiet: bool,
    dry_run: bool = False,
    progress_callback: DownloadProgressCallback | None = None,
) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = _list_competition_files_with_sizes(slug, dry_run=dry_run)
    total_size = sum(file.size_bytes for file in files)
    threshold = _split_download_threshold_bytes()
    total_files = len(files)
    completed_files = _count_downloaded_competition_files(dest_dir, files)
    if files and total_size >= threshold:
        return _download_competition_by_file(
            slug,
            dest_dir,
            files,
            force=force,
            quiet=quiet,
            dry_run=dry_run,
            progress_callback=progress_callback,
            completed_files=completed_files,
        )

    _emit_download_progress(progress_callback, completed_files=completed_files, total_files=total_files, file_name=None)

    args = _competition_download_args(slug=slug, dest_dir=dest_dir, force=force, quiet=quiet, file_name=None)
    try:
        output = _run_kaggle(args, slug=slug, dry_run=dry_run)
        completed_after = _count_downloaded_competition_files(dest_dir, files)
        _emit_download_progress(
            progress_callback,
            completed_files=completed_after,
            total_files=total_files,
            file_name=None,
        )
        return output
    except KaggleCliError as exc:
        if files and _is_retryable_download_error(exc):
            completed_after = _count_downloaded_competition_files(dest_dir, files)
            _emit_download_progress(
                progress_callback,
                completed_files=completed_after,
                total_files=total_files,
                file_name=None,
            )
            return _download_competition_by_file(
                slug,
                dest_dir,
                files,
                force=force,
                quiet=quiet,
                dry_run=dry_run,
                progress_callback=progress_callback,
                completed_files=completed_after,
            )
        raise


def _list_competition_files_with_sizes(slug: str, *, dry_run: bool) -> list[_CompetitionFile]:
    if dry_run:
        return []

    files: list[_CompetitionFile] = []
    seen_names: set[str] = set()
    seen_tokens: set[str] = set()
    page_token: str | None = None

    while True:
        args = [
            "kaggle",
            "competitions",
            "files",
            slug,
            "--csv",
            "--page-size",
            str(_FILES_PAGE_SIZE),
        ]
        if page_token:
            args += ["--page-token", page_token]

        output = _run_kaggle_with_retry(args, slug=slug, dry_run=False)
        page_files, next_page_token = _parse_competition_files_csv(output)

        for file in page_files:
            if file.name in seen_names:
                continue
            seen_names.add(file.name)
            files.append(file)

        if not next_page_token:
            return files
        if next_page_token in seen_tokens:
            return files
        seen_tokens.add(next_page_token)
        page_token = next_page_token


def _parse_competition_files_csv(output: str) -> tuple[list[_CompetitionFile], str | None]:
    csv_lines: list[str] = []
    next_page_token: str | None = None

    for raw_line in output.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith(_NEXT_PAGE_TOKEN_PREFIX):
            _prefix, _eq, remainder = stripped.partition("=")
            token = remainder.strip()
            next_page_token = token or None
            continue
        csv_lines.append(raw_line)

    if not csv_lines:
        return [], next_page_token

    files: list[_CompetitionFile] = []
    for row in csv.DictReader(csv_lines):
        if not row:
            continue
        name = str(row.get("name") or row.get("Name") or "").strip()
        if not name:
            continue
        size_value = row.get("size") or row.get("sizeBytes") or row.get("Size") or row.get("sizebytes")
        size_bytes = _parse_int(size_value)
        files.append(_CompetitionFile(name=name, size_bytes=size_bytes))
    return files, next_page_token


def _parse_int(value: str | None) -> int:
    if value is None:
        return 0
    cleaned = str(value).strip().replace(",", "")
    if not cleaned:
        return 0
    try:
        return max(0, int(cleaned))
    except ValueError:
        return 0


def _split_download_threshold_bytes() -> int:
    value = _read_int_env("KAGGLEBOT_DOWNLOAD_SPLIT_THRESHOLD_BYTES", _DEFAULT_SPLIT_THRESHOLD_BYTES)
    return max(1, value)


def _download_attempts() -> int | None:
    value = _read_int_env("KAGGLEBOT_DOWNLOAD_RETRY_ATTEMPTS", _DEFAULT_DOWNLOAD_ATTEMPTS)
    if value <= 0:
        return None
    return value


def _download_retry_backoff_sec() -> float:
    raw = os.getenv("KAGGLEBOT_DOWNLOAD_RETRY_BACKOFF_SEC")
    if raw is None:
        return _DEFAULT_RETRY_BACKOFF_SEC
    try:
        value = float(raw.strip())
    except ValueError:
        return _DEFAULT_RETRY_BACKOFF_SEC
    if value < 0:
        return _DEFAULT_RETRY_BACKOFF_SEC
    return value


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _build_basename_counts(files: list[_CompetitionFile]) -> dict[str, int]:
    basename_counts: dict[str, int] = {}
    for item in files:
        basename = Path(item.name).name
        basename_counts[basename] = basename_counts.get(basename, 0) + 1
    return basename_counts


def _count_downloaded_competition_files(dest_dir: Path, files: list[_CompetitionFile]) -> int:
    if not files:
        return 0
    basename_counts = _build_basename_counts(files)
    return sum(
        1 for item in files if _is_competition_file_already_downloaded(dest_dir, item, basename_counts=basename_counts)
    )


def _emit_download_progress(
    callback: DownloadProgressCallback | None,
    *,
    completed_files: int,
    total_files: int,
    file_name: str | None,
) -> None:
    if callback is None or total_files <= 0:
        return
    done = min(max(completed_files, 0), total_files)
    callback(done, total_files, file_name)


def _competition_download_args(
    *,
    slug: str,
    dest_dir: Path,
    force: bool,
    quiet: bool,
    file_name: str | None,
) -> list[str]:
    args = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        slug,
    ]
    if file_name:
        args += ["-f", file_name]
    args += ["-p", str(dest_dir)]
    if force:
        args.append("--force")
    if quiet:
        args.append("--quiet")
    return args


def _download_competition_by_file(
    slug: str,
    dest_dir: Path,
    files: list[_CompetitionFile],
    *,
    force: bool,
    quiet: bool,
    dry_run: bool,
    progress_callback: DownloadProgressCallback | None = None,
    completed_files: int | None = None,
) -> str:
    outputs: list[str] = []
    max_attempts = _download_attempts()
    base_backoff = _download_retry_backoff_sec()
    basename_counts = _build_basename_counts(files)
    total_files = len(files)
    if completed_files is None:
        completed_files = _count_downloaded_competition_files(dest_dir, files)
    _emit_download_progress(
        progress_callback,
        completed_files=completed_files,
        total_files=total_files,
        file_name=None,
    )

    for file in sorted(files, key=lambda item: item.name):
        if _is_competition_file_already_downloaded(dest_dir, file, basename_counts=basename_counts):
            continue

        args = _competition_download_args(slug=slug, dest_dir=dest_dir, force=force, quiet=quiet, file_name=file.name)

        attempt = 1
        while True:
            try:
                output = _run_kaggle(args, slug=slug, dry_run=dry_run)
                if output:
                    outputs.append(output)
                completed_files += 1
                _emit_download_progress(
                    progress_callback,
                    completed_files=completed_files,
                    total_files=total_files,
                    file_name=file.name,
                )
                break
            except KaggleCliError as exc:
                if _is_rate_limited_download_error(exc):
                    logger.warning(
                        "rate limited while downloading %s in split mode; "
                        "falling back to single-shot competition download",
                        file.name,
                    )
                    fallback_output = _run_kaggle_with_retry(
                        _competition_download_args(
                            slug=slug,
                            dest_dir=dest_dir,
                            force=force,
                            quiet=quiet,
                            file_name=None,
                        ),
                        slug=slug,
                        dry_run=dry_run,
                    )
                    if fallback_output:
                        outputs.append(fallback_output)
                    _emit_download_progress(
                        progress_callback,
                        completed_files=total_files,
                        total_files=total_files,
                        file_name=None,
                    )
                    return "\n".join(outputs).strip()
                if _should_stop_retrying(attempt=attempt, max_attempts=max_attempts, error=exc):
                    raise
                sleep_sec = base_backoff * (2 ** (attempt - 1))
                _log_download_retry(exc=exc, attempt=attempt, max_attempts=max_attempts, sleep_sec=sleep_sec)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                attempt += 1
    return "\n".join(outputs).strip()


def _is_rate_limited_download_error(exc: KaggleCliError) -> bool:
    text = (exc.output or "").lower()
    return "too many requests" in text or "429" in text or "rate limit" in text or "ratelimit" in text


def _log_download_retry(*, exc: KaggleCliError, attempt: int, max_attempts: int | None, sleep_sec: float) -> None:
    budget = "unbounded" if max_attempts is None else str(max_attempts)
    detail = _summarize_error_output(exc.output)
    logger.warning(
        "download retry %s/%s after Kaggle CLI error (sleep %.1fs): %s",
        attempt,
        budget,
        sleep_sec,
        detail,
    )


def _summarize_error_output(output: str) -> str:
    text = (output or "").strip().replace("\n", " ")
    if not text:
        return "<no stderr/stdout>"
    return text[:200] + "..." if len(text) > 200 else text


def _is_retryable_download_error(exc: KaggleCliError) -> bool:
    if isinstance(exc, KaggleNetworkError):
        return True
    if exc.exit_code in {130, 137, 143}:
        return True
    text = (exc.output or "").lower()
    retry_tokens = (
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "try again",
    )
    return any(token in text for token in retry_tokens)


def _run_kaggle_with_retry(args: list[str], *, slug: str, dry_run: bool) -> str:
    max_attempts = _download_attempts()
    base_backoff = _download_retry_backoff_sec()

    attempt = 1
    while True:
        try:
            return _run_kaggle(args, slug=slug, dry_run=dry_run)
        except KaggleCliError as exc:
            if _should_stop_retrying(attempt=attempt, max_attempts=max_attempts, error=exc):
                raise
            sleep_sec = base_backoff * (2 ** (attempt - 1))
            _log_download_retry(exc=exc, attempt=attempt, max_attempts=max_attempts, sleep_sec=sleep_sec)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            attempt += 1


def _should_stop_retrying(*, attempt: int, max_attempts: int | None, error: KaggleCliError) -> bool:
    if not _is_retryable_download_error(error):
        return True
    if max_attempts is None:
        return False
    return attempt >= max_attempts


def _is_competition_file_already_downloaded(
    dest_dir: Path,
    file: _CompetitionFile,
    *,
    basename_counts: dict[str, int],
) -> bool:
    direct_path = dest_dir / file.name
    if _path_looks_downloaded(direct_path, file_size=file.size_bytes):
        return True

    basename = Path(file.name).name
    if basename_counts.get(basename, 0) != 1:
        return False
    basename_path = dest_dir / basename
    return _path_looks_downloaded(basename_path, file_size=file.size_bytes)


def _path_looks_downloaded(path: Path, *, file_size: int) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if file_size <= 0:
        return True
    try:
        return path.stat().st_size == file_size
    except OSError:
        return False


def submit_competition(slug: str, submission_file: Path, message: str, *, dry_run: bool = False) -> str:
    result = run_kaggle_submit(
        slug=slug,
        submission_file=submission_file,
        message=message,
        dry_run=dry_run,
    )
    return result.output


def list_competition_submissions(slug: str, *, dry_run: bool = False) -> list[dict[str, str]]:
    output = _run_kaggle(
        ["kaggle", "competitions", "submissions", "-c", slug, "--csv"],
        slug=slug,
        dry_run=dry_run,
    )
    if dry_run:
        return []
    rows: list[dict[str, str]] = []
    for row in csv.DictReader(output.splitlines()):
        if not row:
            continue
        normalized = {str(key): "" if value is None else str(value) for key, value in row.items() if key}
        if normalized:
            rows.append(normalized)
    return rows


def check_rules_accepted(slug: str, *, dry_run: bool = False) -> bool:
    if dry_run:
        return True

    # `kaggle competitions list` is not a reliable signal for rules acceptance: the CSV
    # schema varies and often omits acceptance columns. Prefer the smallest command that
    # requires accepted rules / entry: listing competition files.
    try:
        competitions_files(slug, dry_run=False)
    except RulesNotAcceptedError:
        return False
    except KaggleCliError as exc:
        # Preserve historical behavior: unknown competitions returned `False` rather than
        # crashing the CLI, while other failures (auth/network) should surface.
        text = (exc.output or "").lower()
        if "not found" in text or "404" in text or "no competition found" in text:
            return False
        raise

    # Some competitions expose file metadata publicly even when the account cannot access
    # competition resources in kernels. Cross-check with the submissions endpoint, which
    # requires actual competition participation.
    try:
        list_competition_submissions(slug, dry_run=False)
    except RulesNotAcceptedError:
        return False
    except KaggleCliError as exc:
        text = (exc.output or "").lower()
        if "not found" in text or "404" in text or "no competition found" in text:
            return False
        if "forbidden" in text or "permission" in text or "not allowed" in text:
            return False
        raise
    return True


def kernels_init(kernel_dir: Path, *, dry_run: bool = False) -> str:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    return _run_kaggle(["kaggle", "kernels", "init", "-p", str(kernel_dir)], slug=None, dry_run=dry_run)


def kernels_push(kernel_dir: Path, *, slug: str | None = None, dry_run: bool = False) -> str:
    return _run_kaggle(["kaggle", "kernels", "push", "-p", str(kernel_dir)], slug, dry_run=dry_run)


def kernels_status(kernel_id: str, *, slug: str | None = None, dry_run: bool = False) -> str:
    return _run_kaggle(["kaggle", "kernels", "status", kernel_id], slug, dry_run=dry_run)


def kernels_output(
    kernel_id: str,
    output_dir: Path,
    *,
    slug: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    quiet: bool = False,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    args = ["kaggle", "kernels", "output", kernel_id, "-p", str(output_dir)]
    if force:
        args.append("--force")
    if quiet:
        args.append("--quiet")
    return _run_kaggle(args, slug, dry_run=dry_run)


def kernels_pull(
    kernel_id: str,
    output_dir: Path,
    *,
    slug: str | None = None,
    dry_run: bool = False,
    metadata: bool = True,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    args = ["kaggle", "kernels", "pull", kernel_id, "-p", str(output_dir)]
    if metadata:
        args.append("-m")
    return _run_kaggle(args, slug, dry_run=dry_run)


def kernels_list(
    *, mine: bool = False, user: str | None = None, sort_by: str = "dateCreated", dry_run: bool = False
) -> str:
    args = ["kaggle", "kernels", "list"]
    if mine:
        args.append("-m")
    elif user:
        args += ["-u", user]
    if sort_by:
        args += ["--sort-by", sort_by]
    args.append("--csv")
    return _run_kaggle(args, slug=None, dry_run=dry_run)


def list_competition_kernels(
    slug: str,
    *,
    page: int = 1,
    page_size: int = 200,
    sort_by: str = "scoreDescending",
    kernel_type: str = "notebook",
    dry_run: bool = False,
) -> list[dict[str, str]]:
    args = [
        "kaggle",
        "kernels",
        "list",
        "--competition",
        slug,
        "--page",
        str(page),
        "--page-size",
        str(page_size),
        "--sort-by",
        sort_by,
        "--kernel-type",
        kernel_type,
        "--csv",
    ]
    output = _run_kaggle(args, slug=slug, dry_run=dry_run)
    if dry_run:
        return []
    rows: list[dict[str, str]] = []
    for row in csv.DictReader(output.splitlines()):
        if not row:
            continue
        normalized = {str(key): "" if value is None else str(value) for key, value in row.items() if key}
        if normalized:
            rows.append(normalized)
    return rows


def kernel_exists(kernel_id: str, *, dry_run: bool = False) -> bool:
    output = kernels_list(mine=True, sort_by="dateCreated", dry_run=dry_run)
    if dry_run:
        return True
    target = kernel_id.strip().lower()
    rows = [row for row in csv.DictReader(output.splitlines()) if row]
    for row in rows:
        for key in ("ref", "url", "link"):
            ref = _normalize_kernel_ref(row.get(key))
            if ref and ref.lower() == target:
                return True
        raw_ref = row.get("ref")
        if raw_ref and target in str(raw_ref).lower():
            return True
    return False


def kernel_id_by_title(title: str, *, dry_run: bool = False) -> str | None:
    output = kernels_list(mine=True, sort_by="dateCreated", dry_run=dry_run)
    if dry_run:
        return None
    rows = [row for row in csv.DictReader(output.splitlines()) if row]
    target = _normalize_kernel_title(title)
    for row in rows:
        row_title = _get_row_value(row, "title")
        if not row_title:
            continue
        if _normalize_kernel_title(row_title) != target:
            continue
        ref_value = row.get("ref") or row.get("url") or row.get("link")
        normalized = _normalize_kernel_ref(ref_value)
        if normalized:
            return normalized
        if ref_value:
            return str(ref_value)
    return None


def _normalize_kernel_ref(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _KERNEL_URL_RE.search(text)
    if match:
        return f"{match.group('user')}/{match.group('slug')}"
    match = _KERNEL_ID_RE.match(text)
    if match:
        return f"{match.group('user')}/{match.group('slug')}"
    return None


def competitions_files(slug: str, *, dry_run: bool = False) -> str:
    return _run_kaggle(["kaggle", "competitions", "files", "-c", slug], slug, dry_run=dry_run)


def leaderboard_top1(
    slug: str,
    output_dir: Path,
    *,
    dry_run: bool = False,
    metric_hint: str | None = None,
) -> dict[str, object]:
    rows, csv_path, load_error = _load_leaderboard_rows(slug=slug, output_dir=output_dir, dry_run=dry_run)
    if load_error is not None:
        warning = _warn_invalid_leaderboard_csv(csv_path, load_error)
        return {
            "score": None,
            "timestamp": int(datetime.now(UTC).timestamp()),
            "source": "kaggle competitions leaderboard --download",
            "scope": "public",
            "error": load_error,
            "warning": warning,
        }

    best_score = _extract_top_rank_score(rows)
    if best_score is None:
        reason = "No finite numeric score values found in leaderboard CSV."
        warning = _warn_invalid_leaderboard_csv(csv_path, reason)
        return {
            "score": None,
            "timestamp": int(datetime.now(UTC).timestamp()),
            "source": "kaggle competitions leaderboard --download",
            "scope": "public",
            "error": reason,
            "warning": warning,
        }
    if _requires_positive_score(metric_hint) and best_score <= 0.0:
        reason = (
            f"Top leaderboard score {best_score} is not valid for metric '{metric_hint}' (expected a positive value)."
        )
        warning = _warn_invalid_leaderboard_csv(csv_path, reason)
        return {
            "score": None,
            "timestamp": int(datetime.now(UTC).timestamp()),
            "source": "kaggle competitions leaderboard --download",
            "scope": "public",
            "error": reason,
            "warning": warning,
        }
    return {
        "score": best_score,
        "timestamp": int(datetime.now(UTC).timestamp()),
        "source": "kaggle competitions leaderboard --download",
        "scope": "public",
    }


def leaderboard_rank_for_score(
    slug: str,
    output_dir: Path,
    *,
    score: float,
    direction: str,
    dry_run: bool = False,
) -> dict[str, object]:
    rows, _csv_path, load_error = _load_leaderboard_rows(slug=slug, output_dir=output_dir, dry_run=dry_run)
    if load_error is not None:
        return {
            "rank": None,
            "total_teams": None,
            "rank_percentile": None,
            "source": "kaggle competitions leaderboard --download",
            "scope": "public",
            "error": load_error,
        }

    scores: list[float] = []
    for row in rows:
        parsed = _extract_score_or_none(row)
        if parsed is not None:
            scores.append(parsed)
    if not scores:
        return {
            "rank": None,
            "total_teams": None,
            "rank_percentile": None,
            "source": "kaggle competitions leaderboard --download",
            "scope": "public",
            "error": "Unable to parse score column from leaderboard CSV.",
        }

    if direction == "minimize":
        better = sum(1 for value in scores if value < score)
    else:
        better = sum(1 for value in scores if value > score)
    rank = better + 1
    total_teams = len(scores)
    rank_percentile = (rank / total_teams) if total_teams > 0 else None
    return {
        "rank": rank,
        "total_teams": total_teams,
        "rank_percentile": rank_percentile,
        "source": "kaggle competitions leaderboard --download",
        "scope": "public",
    }


def _load_leaderboard_rows(
    *,
    slug: str,
    output_dir: Path,
    dry_run: bool,
) -> tuple[list[dict[str, str]], Path | None, str | None]:
    leaderboard_dir = output_dir / "leaderboard"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        _run_kaggle(
            [
                "kaggle",
                "competitions",
                "leaderboard",
                "--download",
                "-c",
                slug,
                "-p",
                str(leaderboard_dir),
            ],
            slug=slug,
            dry_run=dry_run,
        )

    csv_path = _find_leaderboard_csv(leaderboard_dir, slug)
    if csv_path is None:
        zip_paths = list(leaderboard_dir.glob("*.zip"))
        for zip_path in zip_paths:
            safe_extract_zip(zip_path, leaderboard_dir)
        csv_path = _find_leaderboard_csv(leaderboard_dir, slug)
    if dry_run:
        return [], csv_path, None
    if csv_path is None or not csv_path.exists():
        return [], csv_path, "No leaderboard CSV file was found after download/extract."
    if csv_path.stat().st_size == 0:
        return [], csv_path, "Leaderboard CSV is empty."

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return [], csv_path, "Leaderboard CSV is missing a header row."
        rows = [row for row in reader if row]
    if not rows:
        return [], csv_path, "Leaderboard CSV has a header but no score rows."
    return rows, csv_path, None


def _find_leaderboard_csv(output_dir: Path, slug: str) -> Path | None:
    csvs = list(output_dir.glob("*.csv"))
    if not csvs:
        return None
    slug_lower = slug.lower()
    preferred = [path for path in csvs if "leaderboard" in path.name.lower()]
    if not preferred:
        preferred = [path for path in csvs if slug_lower in path.name.lower()]
    candidates = preferred or csvs
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_score(row: dict[str, str]) -> float:
    value = _extract_score_or_none(row)
    if value is not None:
        return value
    raise ValueError("Unable to parse a numeric score from leaderboard CSV.")


def _extract_score_or_none(row: dict[str, str]) -> float | None:
    preferred_keys = ("Score", "PublicScore", "Public Score", "PrivateScore", "Private Score")
    for key in preferred_keys:
        if key in row:
            value = _parse_score_value(row.get(key))
            if value is not None:
                return value
    for key, value in row.items():
        if "score" not in key.lower():
            continue
        parsed = _parse_score_value(value)
        if parsed is not None:
            return parsed
    return None


def _parse_score_value(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = str(value).strip().replace(",", "")
    if not cleaned:
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _extract_top_rank_score(rows: list[dict[str, str]]) -> float | None:
    candidates: list[tuple[int, int, float]] = []
    for idx, row in enumerate(rows):
        score = _extract_score_or_none(row)
        if score is None:
            continue
        rank = _extract_rank_or_none(row)
        normalized_rank = rank if rank is not None else idx + 1
        candidates.append((normalized_rank, idx, score))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _extract_rank_or_none(row: dict[str, str]) -> int | None:
    for key in ("Rank", "rank", "PublicRank", "Public Rank"):
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None:
            continue
        try:
            rank = int(float(str(raw).strip()))
        except ValueError:
            continue
        if rank > 0:
            return rank
    return None


def _requires_positive_score(metric_hint: str | None) -> bool:
    if not metric_hint:
        return False
    normalized = metric_hint.strip().lower()
    return "rmse" in normalized or "rmsle" in normalized


def _warn_invalid_leaderboard_csv(csv_path: Path | None, reason: str) -> str:
    location = str(csv_path) if csv_path is not None else "<missing>"
    message = f"Ignoring leaderboard CSV ({location}): {reason}"
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return message


def _parse_rules_accepted(row: dict[str, str]) -> bool | None:
    normalized = {key.strip().lower(): value for key, value in row.items() if key}
    raw = normalized.get("hasacceptedrules") or normalized.get("userhasentered")
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    return None


def _matches_slug(ref: str | None, slug: str) -> bool:
    if not ref:
        return False
    ref_value = ref.strip()
    if ref_value == slug:
        return True
    try:
        return parse_competition_slug(ref_value) == slug
    except ValueError:
        return False


def _run_kaggle(args: list[str], slug: str | None, *, dry_run: bool) -> str:
    try:
        result = run_command(args, dry_run=dry_run)
    except FileNotFoundError as exc:
        raise KaggleCliError("Kaggle CLI not found. Install `kaggle` and ensure it is on PATH.", args) from exc

    output = result.output
    if result.returncode != 0:
        if _is_kernel_capacity_limit(output):
            raise KernelCapacityError(
                "Kaggle GPU session limit reached; free running GPU sessions and retry.",
                args,
                exit_code=result.returncode,
                output=output,
            )
        if _is_network_error(output):
            raise KaggleNetworkError(
                "Kaggle CLI failed due to a network or DNS error. Check connectivity to www.kaggle.com and retry.",
                args,
                exit_code=result.returncode,
                output=output,
            )
        if slug and _is_rules_not_accepted(output):
            raise RulesNotAcceptedError("Competition rules not accepted.")
        raise KaggleCliError(
            f"Kaggle CLI failed with exit code {result.returncode}.",
            args,
            exit_code=result.returncode,
            output=output,
        )
    return output


def _is_rules_not_accepted(output: str) -> bool:
    text = output.lower()
    if "rules" in text and ("accept" in text or "accepted" in text):
        return True
    if "competition rules" in text and "not" in text:
        return True
    if "forbidden" in text and "competition" in text:
        return True
    return False


def _is_network_error(output: str) -> bool:
    text = output.lower()
    tokens = (
        "nameresolutionerror",
        "failed to resolve",
        "temporary failure in name resolution",
        "nodename nor servname provided",
        "no address associated with hostname",
        "getaddrinfo failed",
        "connectionerror",
        "newconnectionerror",
        "max retries exceeded",
        "connection refused",
        "network is unreachable",
        "no route to host",
        "failed to establish a new connection",
    )
    return any(token in text for token in tokens)


def _is_kernel_capacity_limit(output: str) -> bool:
    text = output.lower()
    return "maximum batch gpu session count" in text
