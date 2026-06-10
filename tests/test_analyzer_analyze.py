from __future__ import annotations

import json

import pandas as pd

from kagglebot.analyzer.analyze import analyze_competition
from kagglebot.paths import CompetitionPaths


def test_analyze_competition_uses_current_competition_paths(tmp_path):
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "feature": [0.1, 0.2, 0.3, 0.4],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(paths.data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [5, 6], "feature": [0.5, 0.6]}).to_csv(paths.data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "label": [0.0, 0.0]}).to_csv(
        paths.data_dir / "sample_submission.csv",
        index=False,
    )

    result = analyze_competition(
        slug="demo",
        paths=paths,
        time_budget_minutes=30,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    assert result.analysis_path == paths.analysis_path
    assert result.analysis_path.exists()
    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["slug"] == "demo"
    assert payload["schema"]["train_path"] == str(paths.data_raw / "train.csv")
