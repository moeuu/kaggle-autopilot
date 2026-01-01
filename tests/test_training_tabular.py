"""Tests for tabular training and prediction."""

from __future__ import annotations

import pandas as pd

from kagglebot.analyzer import analyze_competition
from kagglebot.bootstrap import bootstrap_competition
from kagglebot.paths import CompetitionPaths
from kagglebot.training import predict_tabular, train_tabular
from kagglebot.validation import validate_submission


def test_train_and_predict_tabular(tmp_path) -> None:
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
            "id": [7, 8, 9],
            "feature_num": [4.0, 5.0, 6.0],
            "feature_cat": ["a", "b", "a"],
        }
    )
    sample = pd.DataFrame({"id": [7, 8, 9], "target": [0.5, 0.5, 0.5]})

    paths.data_raw.mkdir(parents=True, exist_ok=True)
    train.to_csv(paths.data_raw / "train.csv", index=False)
    test.to_csv(paths.data_raw / "test.csv", index=False)
    sample.to_csv(paths.data_raw / "sample_submission.csv", index=False)

    analysis = analyze_competition(
        slug=slug,
        paths=paths,
        time_budget_minutes=1,
        cv_folds=2,
        models=["logreg"],
        use_stacking=False,
    )

    result = train_tabular(
        analysis.metadata,
        paths=paths,
        time_budget_minutes=1,
        model_names=["logreg"],
        cv_folds=2,
    )

    assert result.model_path.exists()
    assert result.model_info_path.exists()
    assert result.report_path.exists()

    submission_path = predict_tabular(analysis.metadata, paths=paths)
    validate_submission(str(paths.data_raw / "sample_submission.csv"), str(submission_path))
