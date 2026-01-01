from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from kagglebot.exceptions import KaggleCliError, RulesNotAcceptedError


def download_competition(slug: str, dest_dir: Path, *, force: bool, quiet: bool) -> str:
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
    return _run_kaggle(args, slug)


def submit_competition(slug: str, submission_file: Path, message: str) -> str:
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
    return _run_kaggle(args, slug)


def check_rules_accepted(slug: str) -> bool:
    args = ["kaggle", "competitions", "list", "--csv"]
    output = _run_kaggle(args, slug=None)
    rows = list(csv.DictReader(output.splitlines()))
    for row in rows:
        ref = (row.get("ref") or "").strip()
        if ref == slug:
            return True
    return False


def kernels_init(kernel_dir: Path) -> str:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    return _run_kaggle(["kaggle", "kernels", "init", "-p", str(kernel_dir)], slug=None)


def kernels_push(kernel_dir: Path, *, slug: str | None = None) -> str:
    return _run_kaggle(["kaggle", "kernels", "push", "-p", str(kernel_dir)], slug)


def kernels_status(kernel_id: str, *, slug: str | None = None) -> str:
    return _run_kaggle(["kaggle", "kernels", "status", kernel_id], slug)


def kernels_output(kernel_id: str, output_dir: Path, *, slug: str | None = None) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    return _run_kaggle(["kaggle", "kernels", "output", kernel_id, "-p", str(output_dir)], slug)


def competitions_files(slug: str) -> str:
    return _run_kaggle(["kaggle", "competitions", "files", "-c", slug], slug)


def _run_kaggle(args: list[str], slug: str | None) -> str:
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise KaggleCliError("Kaggle CLI not found. Install `kaggle` and ensure it is on PATH.", "") from exc

    output = "".join([completed.stdout or "", completed.stderr or ""]).strip()
    if completed.returncode != 0:
        if slug and _is_rules_not_accepted(output):
            raise RulesNotAcceptedError("Competition rules not accepted.")
        raise KaggleCliError(f"Kaggle CLI failed with exit code {completed.returncode}.", output)
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
