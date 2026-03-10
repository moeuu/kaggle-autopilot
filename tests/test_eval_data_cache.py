from __future__ import annotations

from pathlib import Path

import pandas as pd

from kagglebot.autopilot import AutopilotConfig, _ensure_eval_data_cache
from kagglebot.paths import CompetitionPaths, KnowledgePaths


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_ensure_eval_data_cache_drift_selection_avoids_strict_getitem(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path)

    _write_csv(
        paths.data_dir / "train.csv",
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "oare_id": [10, 11, 12, 13],
                "feature": [0.1, 0.2, 0.3, 0.4],
                "label": [0, 1, 0, 1],
            }
        ),
    )
    _write_csv(
        paths.data_dir / "test.csv",
        pd.DataFrame(
            {
                "id": [5, 6],
                "oare_id": [14, 15],
                "feature": [0.5, 0.6],
            }
        ),
    )
    _write_csv(paths.data_dir / "sample_submission.csv", pd.DataFrame({"id": [5, 6], "label": [0, 0]}))

    config = AutopilotConfig(
        run_id="run-1",
        slug="demo",
        competition_url=None,
        paths=paths,
        knowledge_paths=knowledge_paths,
        agent="codex",
        compute="local_cpu",
        accelerator="cpu",
        strict_accelerator=False,
        kaggle_username=None,
        kernel_name=None,
        internet=None,
        time_budget_min=None,
        seed=42,
        score_source="cv",
        holdout_frac=0.2,
        cv_folds=2,
        target_metric=None,
        target_score=None,
        target_direction=None,
        max_iterations=1,
        max_total_min=5,
        patience=1,
        min_improvement=0.0,
        submit=False,
        force_submit=False,
        message=None,
        verify_cmd=":",
        dry_run=True,
    )

    original_getitem = pd.DataFrame.__getitem__

    def flaky_getitem(self, key):  # noqa: ANN001
        if isinstance(key, list) and "oare_id" in key:
            raise KeyError("['oare_id'] not in index")
        return original_getitem(self, key)

    monkeypatch.setattr(pd.DataFrame, "__getitem__", flaky_getitem)

    cache = _ensure_eval_data_cache(
        config=config,
        cv_folds=2,
        split_strategy="kfold",
        seed=42,
        eval_seeds=[42],
        eval_repeats=1,
        score_source="cv",
        eval_data_cache=None,
    )

    assert cache["drift_train_x"] is not None
    assert cache["drift_test_x"] is not None
    assert list(cache["drift_train_x"].columns) == ["oare_id", "feature"]
    assert list(cache["drift_test_x"].columns) == ["oare_id", "feature"]


def test_ensure_eval_data_cache_preserves_requested_timeseries_split_on_fallback(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    knowledge_paths = KnowledgePaths(workdir=tmp_path)
    config = AutopilotConfig(
        run_id="run-1",
        slug="demo",
        competition_url=None,
        paths=paths,
        knowledge_paths=knowledge_paths,
        agent="codex",
        compute="local_cpu",
        accelerator="cpu",
        strict_accelerator=False,
        kaggle_username=None,
        kernel_name=None,
        internet=None,
        time_budget_min=None,
        seed=42,
        score_source="cv",
        holdout_frac=0.2,
        cv_folds=5,
        target_metric=None,
        target_score=None,
        target_direction=None,
        max_iterations=1,
        max_total_min=5,
        patience=1,
        min_improvement=0.0,
        submit=False,
        force_submit=False,
        message=None,
        verify_cmd=":",
        dry_run=True,
    )

    monkeypatch.setattr(
        "kagglebot.autopilot.load_competition_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    cache = _ensure_eval_data_cache(
        config=config,
        cv_folds=5,
        split_strategy="timeseries_split",
        seed=42,
        eval_seeds=[42],
        eval_repeats=1,
        score_source="cv",
        eval_data_cache=None,
    )

    assert cache["split_strategy"] == "timeseries_split"
    assert cache["n_splits"] == 5
