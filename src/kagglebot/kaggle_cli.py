from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CmdResult:
    code: int
    stdout: str
    stderr: str


def run_kaggle(args: Sequence[str]) -> CmdResult:
    """
    Run `kaggle ...` command. We keep a thin wrapper so we can standardize error handling.
    """
    proc = subprocess.run(
        ["kaggle", *args],
        text=True,
        capture_output=True,
    )
    return CmdResult(code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def kaggle_submit(slug: str, submission_file: str, message: str) -> None:
    res = run_kaggle(["competitions", "submit", "-c", slug, "-f", submission_file, "-m", message])
    if res.code != 0:
        raise RuntimeError(f"kaggle submit failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
