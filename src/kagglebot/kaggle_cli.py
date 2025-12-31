from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KaggleCliError(RuntimeError):
    message: str
    output: str

    def __str__(self) -> str:
        return f"{self.message}\n{self.output}".strip()


@dataclass(frozen=True)
class RulesNotAcceptedError(KaggleCliError):
    slug: str


def download_competition(slug: str, dest_dir: Path, *, overwrite: bool = False) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        slug,
        "-p",
        str(dest_dir),
        "--unzip",
    ]
    if overwrite:
        args.append("--force")
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


def _run_kaggle(args: list[str], slug: str) -> str:
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise KaggleCliError(
            "Kaggle CLI not found. Install the kaggle package and ensure `kaggle` is on PATH.",
            "",
        ) from exc

    output = "".join([completed.stdout or "", completed.stderr or ""]).strip()
    if completed.returncode != 0:
        if _is_rules_not_accepted(output):
            raise RulesNotAcceptedError(
                message="Competition rules not accepted.",
                output=output,
                slug=slug,
            )
        raise KaggleCliError(
            message=f"Kaggle CLI failed with exit code {completed.returncode}.",
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
