"""Tests for competition analyzer."""

from __future__ import annotations

import pandas as pd

from kagglebot.analyzer import analyze_competition
from kagglebot.bootstrap import bootstrap_competition
from kagglebot.paths import CompetitionPaths


def test_analyze_competition_classification(tmp_path) -> None:
    slug = "demo"
    bootstrap_competition(slug=slug, root=tmp_path)
    paths = CompetitionPaths(slug=slug, repo_root=tmp_path)

    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "feature_num": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "feature_cat": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "id": [7, 8],
            "feature_num": [4.0, 5.0],
            "feature_cat": ["a", "b"],
        }
    )
    sample = pd.DataFrame({"id": [7, 8], "target": [0.5, 0.5]})

    paths.data_raw.mkdir(parents=True, exist_ok=True)
    train.to_csv(paths.data_raw / "train.csv", index=False)
    test.to_csv(paths.data_raw / "test.csv", index=False)
    sample.to_csv(paths.data_raw / "sample_submission.csv", index=False)

    analysis = analyze_competition(
        slug=slug,
        paths=paths,
        time_budget_minutes=1,
        cv_folds=3,
        models=None,
        use_stacking=False,
    )

    metadata = analysis.metadata
    assert metadata.task == "classification"
    assert metadata.metric == "accuracy"
    assert metadata.prediction_kind == "probability"
    assert metadata.schema.id_column == "id"
    assert metadata.schema.target_columns == ["target"]
    assert analysis.analysis_path.exists()
