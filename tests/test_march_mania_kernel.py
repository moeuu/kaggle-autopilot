from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.competition_artifact


def _load_kernel_module():
    kernel_path = Path("artifacts/march-machine-learning-mania-2026/kernel/kernel.py")
    spec = importlib.util.spec_from_file_location("march_mania_kernel", kernel_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_march_mania_kernel_prefers_stage2_sample_over_stale_canonical(tmp_path: Path) -> None:
    mod = _load_kernel_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "sample_submission.csv").write_text("ID,Pred\n2022_1_2,0.5\n", encoding="utf-8")
    (data_dir / "SampleSubmissionStage1.csv").write_text("ID,Pred\n2022_1_2,0.5\n2022_1_3,0.5\n", encoding="utf-8")
    stage2 = data_dir / "SampleSubmissionStage2.csv"
    stage2.write_text("ID,Pred\n2026_1_2,0.5\n2026_1_3,0.5\n", encoding="utf-8")

    assert mod._resolve_sample_submission(data_dir) == stage2


def test_march_mania_kernel_select_training_labels_excludes_target_season() -> None:
    mod = _load_kernel_module()
    labels = pd.DataFrame(
        {
            "League": ["M", "M", "M", "W"],
            "Season": [2024, 2025, 2026, 2025],
            "TeamID1": [1, 1, 1, 3001],
            "TeamID2": [2, 2, 2, 3002],
            "y": [1, 0, 1, 1],
        }
    )

    selected = mod.select_training_labels(labels, league="M", min_train_season=2003, target_season=2026)

    assert selected["Season"].tolist() == [2024, 2025]
    assert int(selected["Season"].max()) == 2025
