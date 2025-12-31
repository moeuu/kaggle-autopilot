from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.paths import CompetitionPaths, repo_root


def _default_config(paths: CompetitionPaths) -> dict[str, object]:
    return {
        "schema_version": 1,
        "slug": paths.slug,
        "created_at": datetime.now(UTC).isoformat(),
        "paths": {
            "data_raw": str(paths.data_raw),
            "artifacts": str(paths.artifacts),
            "models_dir": str(paths.models_dir),
            "submissions_dir": str(paths.submissions_dir),
            "runs_dir": str(paths.runs_dir),
        },
    }


def bootstrap_competition(slug: str, force: bool = False, root: Path | None = None) -> Path:
    """
    Prepare local workspace directories and write a config file.
    Does not join competitions or perform any network actions.
    """
    base_root = root if root is not None else repo_root()
    paths = CompetitionPaths(slug=slug, repo_root=base_root)

    paths.data_raw.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.submissions_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)

    config_path = paths.config_file
    if config_path.exists() and not force:
        return config_path

    config_path.write_text(
        json.dumps(_default_config(paths), indent=2),
        encoding="utf-8",
    )
    return config_path
