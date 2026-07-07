from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kagglebot.analyzer.schema import SchemaFrames, infer_schema


def test_infer_schema_target_inference_error_mentions_actual_files(tmp_path: Path) -> None:
    train_path = tmp_path / "train.tsv"
    test_path = tmp_path / "test.tsv"
    sample_path = tmp_path / "sample_submission.tsv"
    frames = SchemaFrames(
        train=pd.DataFrame({"id": [1], "feature": [0.1]}),
        test=pd.DataFrame({"id": [2], "feature": [0.2]}),
        sample=pd.DataFrame({"prediction": [0.0]}),
    )

    with pytest.raises(ValueError) as exc_info:
        infer_schema(frames=frames, train_path=train_path, test_path=test_path, sample_path=sample_path)

    message = str(exc_info.value)
    assert "train='train.tsv'" in message
    assert "test='test.tsv'" in message
    assert "sample='sample_submission.tsv'" in message


def test_infer_schema_no_feature_error_mentions_actual_train_and_test_files(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    sample_path = tmp_path / "sample_submission.jsonl"
    frames = SchemaFrames(
        train=pd.DataFrame({"target": [0.0]}),
        test=pd.DataFrame(index=[0]),
        sample=pd.DataFrame({"target": [0.0]}),
    )

    with pytest.raises(ValueError) as exc_info:
        infer_schema(frames=frames, train_path=train_path, test_path=test_path, sample_path=sample_path)

    message = str(exc_info.value)
    assert "train='train.jsonl'" in message
    assert "test='test.jsonl'" in message
