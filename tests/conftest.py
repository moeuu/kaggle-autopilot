from __future__ import annotations

import os
from pathlib import Path

from kagglebot.verify_artifacts import mirror_verify_artifacts


def pytest_sessionstart(session) -> None:  # noqa: ANN001
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = os.environ.get("KAGGLEBOT_TEST_ARTIFACTS_DIR", "/data/morita/kaggle-autopilot-artifacts")
    external_artifacts_dir = Path(artifacts_dir)
    if not external_artifacts_dir.exists():
        return
    mirror_verify_artifacts(external_artifacts_dir, repo_root=repo_root)
