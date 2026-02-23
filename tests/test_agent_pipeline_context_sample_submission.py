from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kagglebot.orchestrator.agent_pipeline import _ensure_context_materials, _resolve_blocked_modules_for_runtime
from kagglebot.paths import CompetitionPaths


def test_ensure_context_materials_refreshes_header_only_sample_submission(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")

    (paths.context_dir / "submission_format.md").write_text(
        "## Submission File\n\n```csv\nfilename,right_place,prediction_string\n0.jpg,0,-\n```\n",
        encoding="utf-8",
    )

    images_test_dir = paths.data_dir / "images" / "test"
    images_test_dir.mkdir(parents=True, exist_ok=True)
    (images_test_dir / "0.jpg").write_bytes(b"")
    (images_test_dir / "1.jpg").write_bytes(b"")

    paths.sample_submission_path.write_text("id,target\n", encoding="utf-8")

    _ensure_context_materials(paths)

    sample = pd.read_csv(paths.sample_submission_path)
    assert sample.columns.tolist() == ["filename", "right_place", "prediction_string"]
    assert len(sample) == 2


def test_ensure_context_materials_creates_community_placeholders(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")

    _ensure_context_materials(paths)

    assert "/code" in paths.code_md_path.read_text(encoding="utf-8")
    assert "/models" in paths.models_md_path.read_text(encoding="utf-8")
    assert "/discussions" in paths.discussion_md_path.read_text(encoding="utf-8")
    assert paths.code_notebooks_index_path.exists()
    assert paths.discussion_threads_index_path.exists()
    assert paths.code_notebooks_dir.exists()
    assert paths.discussion_threads_dir.exists()


def test_ensure_context_materials_uses_prediction_header_for_placeholder_sample(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = CompetitionPaths(slug="demo", artifacts_dir=artifacts_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")

    _ensure_context_materials(paths)

    assert paths.sample_submission_path.read_text(encoding="utf-8").strip() == "id,prediction"


def test_resolve_blocked_modules_for_runtime_adds_missing_xgboost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "blocked_modules.json").write_text('["custom_pkg"]\n', encoding="utf-8")

    monkeypatch.setattr(
        "kagglebot.orchestrator.agent_pipeline.importlib.util.find_spec",
        lambda name: None if name == "xgboost" else object(),
    )
    resolved = _resolve_blocked_modules_for_runtime(context_dir, compute="local_gpu")
    assert resolved == ["custom_pkg", "xgboost"]
