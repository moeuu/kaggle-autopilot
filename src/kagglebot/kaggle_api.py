from __future__ import annotations

import csv
import re
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


def download_competition(slug: str, dest_dir: Path, *, force: bool, quiet: bool, dry_run: bool = False) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        slug,
        "-p",
        str(dest_dir),
    ]
    if force:
        args.append("--force")
    if quiet:
        args.append("--quiet")
    return _run_kaggle(args, slug=slug, dry_run=dry_run)


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
    args = ["kaggle", "competitions", "list", "--search", slug, "--csv"]
    output = _run_kaggle(args, slug=None, dry_run=dry_run)
    if dry_run:
        return True
    rows = [row for row in csv.DictReader(output.splitlines()) if _matches_slug(row.get("ref"), slug)]
    if not rows:
        return False
    accepted = _parse_rules_accepted(rows[0])
    if accepted is not None:
        return accepted
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


def leaderboard_top1(slug: str, output_dir: Path, *, dry_run: bool = False) -> dict[str, object]:
    rows = _load_leaderboard_rows(slug=slug, output_dir=output_dir, dry_run=dry_run)
    if not rows:
        return {
            "score": None,
            "timestamp": int(datetime.now(UTC).timestamp()),
            "source": "kaggle competitions leaderboard --download",
            "scope": "public",
        }
    try:
        score = _extract_score(rows[0])
    except ValueError as exc:
        return {
            "score": None,
            "timestamp": int(datetime.now(UTC).timestamp()),
            "source": "kaggle competitions leaderboard --download",
            "scope": "public",
            "error": str(exc),
        }
    return {
        "score": score,
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
    rows = _load_leaderboard_rows(slug=slug, output_dir=output_dir, dry_run=dry_run)
    if not rows:
        return {
            "rank": None,
            "total_teams": None,
            "rank_percentile": None,
            "source": "kaggle competitions leaderboard --download",
            "scope": "public",
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


def _load_leaderboard_rows(*, slug: str, output_dir: Path, dry_run: bool) -> list[dict[str, str]]:
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
    if dry_run or csv_path is None or not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row]


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
        return float(cleaned)
    except ValueError:
        return None


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
