from __future__ import annotations

import zipfile

from rich import print

from kagglebot.kaggle_cli import run_kaggle
from kagglebot.paths import CompetitionPaths, repo_root


def _looks_like_rules_not_accepted(stderr: str, stdout: str) -> bool:
    text = (stderr + "\n" + stdout).lower()
    # Kaggle messages may change, so match loosely.
    needles = [
        "accept the rules",
        "must accept",
        "competition rules",
        "not permitted",
        "permission denied",
    ]
    return any(n in text for n in needles)


def bootstrap_competition(slug: str, force: bool = False) -> None:
    paths = CompetitionPaths(slug=slug, repo_root=repo_root())
    paths.data_raw.mkdir(parents=True, exist_ok=True)

    # 1) Rules acceptance check (safe: run CLI and guide on failure).
    check = run_kaggle(["competitions", "files", "-c", slug])
    if check.code != 0 and _looks_like_rules_not_accepted(check.stderr, check.stdout):
        rules_url = f"https://www.kaggle.com/competitions/{slug}/rules"
        print(
            "[red]Rules not accepted (or not joined) for this competition.[/red]\n"
            "Please open the Rules page in your browser, click Join, and accept the rules.\n"
            f"Rules URL: {rules_url}"
        )
        raise SystemExit(2)
    elif check.code != 0:
        raise RuntimeError(f"kaggle competitions files failed:\n{check.stderr}\n{check.stdout}")

    # 2) Download zip to data/raw
    # Kaggle CLI will place a zip in the target directory.
    download_args = ["competitions", "download", "-c", slug, "-p", str(paths.data_raw)]
    if force:
        download_args.append("--force")

    dl = run_kaggle(download_args)
    if dl.code != 0:
        raise RuntimeError(f"kaggle download failed:\nSTDOUT:\n{dl.stdout}\nSTDERR:\n{dl.stderr}")

    # 3) Unzip all zips in data/raw
    zips = sorted(paths.data_raw.glob("*.zip"))
    if not zips:
        print("[yellow]No zip files found after download.[/yellow]")
        return

    for z in zips:
        print(f"Unzipping: {z.name}")
        with zipfile.ZipFile(z, "r") as zipf:
            zipf.extractall(paths.data_raw)
