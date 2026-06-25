from __future__ import annotations

import numpy as np
import pandas as pd

from kagglebot.submission_templates import build_submission_template_for_test, is_tiny_public_sample_for_test


def test_build_submission_template_for_test_expands_tiny_public_sample() -> None:
    sample = pd.DataFrame({"id": [1, 2, 3], "target": [0.2, 0.4, 0.6]})
    test = pd.DataFrame({"id": [10, 11, 12, 13], "feature": [1, 2, 3, 4]})

    template = build_submission_template_for_test(sample_submission=sample, test_df=test, id_col="id")

    assert list(template.columns) == ["id", "target"]
    assert template["id"].tolist() == [10, 11, 12, 13]
    assert np.allclose(template["target"], [0.4, 0.4, 0.4, 0.4])


def test_build_submission_template_for_test_keeps_regular_sample_copy() -> None:
    sample = pd.DataFrame({"id": [1, 2, 3], "target": [0.2, 0.4, 0.6]})
    test = pd.DataFrame({"id": [1, 2, 3], "feature": [1, 2, 3]})

    template = build_submission_template_for_test(sample_submission=sample, test_df=test, id_col="id")
    template.loc[0, "target"] = 0.9

    assert sample.loc[0, "target"] == 0.2
    assert template["id"].tolist() == [1, 2, 3]


def test_is_tiny_public_sample_for_test_rejects_duplicate_ids() -> None:
    sample = pd.DataFrame({"id": [1, 1, 2], "target": [0.0, 0.0, 0.0]})
    test = pd.DataFrame({"id": [10, 11, 12, 13]})

    assert not is_tiny_public_sample_for_test(sample_submission=sample, test_df=test, id_col="id")
