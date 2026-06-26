from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from kagglebot.json_utils import load_json_object

KAGGLE_CREDENTIALS_ERROR = (
    "Kaggle API credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY or point KAGGLE_CONFIG_DIR "
    "to a directory (or kaggle.json file) containing username/key."
)


def kaggle_json_candidates(*, config_dir_env: str | None = None, home: Path | None = None) -> list[Path]:
    """Return Kaggle config candidates in lookup order."""
    candidates: list[Path] = []
    resolved_home = home or Path.home()
    if config_dir_env:
        config_path = Path(config_dir_env).expanduser()
        if config_path.suffix.lower() == ".json":
            candidates.append(config_path)
        else:
            candidates.extend([config_path / "kaggle.json", config_path / "kaggle" / "kaggle.json"])
    else:
        candidates.append(Path("~/.kaggle/kaggle.json").expanduser())

    candidates.extend(
        [
            resolved_home / ".kaggle" / "kaggle.json",
            resolved_home / ".config" / "kaggle" / "kaggle.json",
        ]
    )
    return list(dict.fromkeys(candidates))


def resolve_kaggle_username(explicit: str | None) -> str:
    """Resolve a Kaggle username from explicit input, environment, or kaggle.json."""
    if explicit:
        return explicit
    env_user = os.getenv("KAGGLE_USERNAME")
    if env_user:
        return env_user

    for kaggle_json in kaggle_json_candidates(config_dir_env=os.getenv("KAGGLE_CONFIG_DIR")):
        data = load_json_object(kaggle_json)
        if data is None:
            continue
        username = data.get("username")
        if username:
            return str(username)
    raise ValueError(
        "Kaggle username is required for kaggle_* compute modes. "
        "Set --kaggle-username, KAGGLE_USERNAME, or point KAGGLE_CONFIG_DIR "
        "to a directory (or kaggle.json file) containing a username."
    )


def resolve_kaggle_api_credentials(*, config_candidates: Iterable[Path] | None = None) -> tuple[str, str]:
    """Resolve Kaggle API username/key from environment or kaggle.json candidates."""
    username = os.getenv("KAGGLE_USERNAME")
    api_key = os.getenv("KAGGLE_KEY")
    if username and api_key:
        return username, api_key

    candidates = config_candidates
    if candidates is None:
        candidates = kaggle_json_candidates(config_dir_env=os.getenv("KAGGLE_CONFIG_DIR"))
    for path in candidates:
        payload = load_json_object(path)
        if payload is None:
            continue
        username = str(payload.get("username") or "").strip()
        api_key = str(payload.get("key") or "").strip()
        if username and api_key:
            return username, api_key
    raise ValueError(KAGGLE_CREDENTIALS_ERROR)
