from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.exceptions import KaggleCliError, RulesNotAcceptedError
from kagglebot.exec_utils import run_command


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
    args = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        slug,
        "-f",
        str(submission_file),
        "-m",
        message,
    ]
    return _run_kaggle(args, slug=slug, dry_run=dry_run)


def check_rules_accepted(slug: str, *, dry_run: bool = False) -> bool:
    args = ["kaggle", "competitions", "list", "--csv"]
    output = _run_kaggle(args, slug=None, dry_run=dry_run)
    if dry_run:
        return True
    rows = list(csv.DictReader(output.splitlines()))
    for row in rows:
        ref = (row.get("ref") or "").strip()
        if ref == slug:
            return True
    return False


def kernels_init(kernel_dir: Path, *, dry_run: bool = False) -> str:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    return _run_kaggle(["kaggle", "kernels", "init", "-p", str(kernel_dir)], slug=None, dry_run=dry_run)


def kernels_push(kernel_dir: Path, *, slug: str | None = None, dry_run: bool = False) -> str:
    return _run_kaggle(["kaggle", "kernels", "push", "-p", str(kernel_dir)], slug, dry_run=dry_run)


def kernels_status(kernel_id: str, *, slug: str | None = None, dry_run: bool = False) -> str:
    return _run_kaggle(["kaggle", "kernels", "status", kernel_id], slug, dry_run=dry_run)


def kernels_output(kernel_id: str, output_dir: Path, *, slug: str | None = None, dry_run: bool = False) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    return _run_kaggle(["kaggle", "kernels", "output", kernel_id, "-p", str(output_dir)], slug, dry_run=dry_run)


def competitions_files(slug: str, *, dry_run: bool = False) -> str:
    return _run_kaggle(["kaggle", "competitions", "files", "-c", slug], slug, dry_run=dry_run)


def leaderboard_top1(slug: str, output_dir: Path, *, dry_run: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"{slug}.csv"
    csv_path = output_dir / csv_name
    if not dry_run:
        _run_kaggle(
            [
                "kaggle",
                "competitions",
                "leaderboard",
                slug,
                "--download",
                "--path",
                str(output_dir),
            ],
            slug=slug,
            dry_run=dry_run,
        )
    if dry_run or not csv_path.exists():
        return {
            "score": None,
            "timestamp": int(datetime.now(UTC).timestamp()),
            "source": "kaggle competitions leaderboard --download",
            "scope": "public",
        }
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        try:
            first = next(reader)
        except StopIteration:
            raise ValueError("Leaderboard CSV is empty.")
    score = _extract_score(first)
    return {
        "score": score,
        "timestamp": int(datetime.now(UTC).timestamp()),
        "source": "kaggle competitions leaderboard --download",
        "scope": "public",
    }


def _extract_score(row: dict[str, str]) -> float:
    if "Score" in row:
        return float(str(row["Score"]).replace(",", ""))
    for key, value in row.items():
        try:
            return float(str(value).replace(",", ""))
        except ValueError:
            continue
    raise ValueError("Unable to parse a numeric score from leaderboard CSV.")


def _run_kaggle(args: list[str], slug: str | None, *, dry_run: bool) -> str:
    try:
        result = run_command(args, dry_run=dry_run)
    except FileNotFoundError as exc:
        raise KaggleCliError("Kaggle CLI not found. Install `kaggle` and ensure it is on PATH.", args) from exc

    output = result.output
    if result.returncode != 0:
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
