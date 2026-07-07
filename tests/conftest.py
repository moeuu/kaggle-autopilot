from __future__ import annotations

import os
from pathlib import Path

from kagglebot.verify_artifacts import mirror_verify_artifacts


def pytest_collection_modifyitems(items) -> None:  # noqa: ANN001
    slow_kernel_runner_prefixes = (
        "test_generated_kernel_",
        "test_rendered_kernel_",
    )
    for item in items:
        if item.path.name == "test_kernel_submit_wrapper.py" and item.name.startswith(
            "test_rendered_submission_kernel_"
        ):
            item.add_marker("slow")
        if item.path.name == "test_kaggle_notebook_runner.py" and item.name.startswith(slow_kernel_runner_prefixes):
            item.add_marker("slow")


def pytest_sessionstart(session) -> None:  # noqa: ANN001
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = os.environ.get("KAGGLEBOT_TEST_ARTIFACTS_DIR")
    if not artifacts_dir:
        return
    external_artifacts_dir = Path(artifacts_dir)
    if not external_artifacts_dir.exists():
        return
    mirror_verify_artifacts(external_artifacts_dir, repo_root=repo_root)
