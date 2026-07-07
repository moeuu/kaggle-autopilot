from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def build_submission_template_for_test(
    *,
    sample_submission: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str | None,
    target_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Return a submission template, expanding tiny public samples to test ids."""
    if id_col is None and is_tiny_public_sample_without_id(sample_submission=sample_submission, test_df=test_df):
        expanded = pd.DataFrame(index=range(len(test_df)))
        for col in sample_submission.columns:
            expanded[col] = sample_column_default(sample_submission, col)
        for col in target_cols:
            if col not in expanded.columns:
                expanded[col] = sample_column_default(sample_submission, col)
        return expanded[list(sample_submission.columns)]
    if (
        id_col
        and id_col in sample_submission.columns
        and id_col in test_df.columns
        and len(sample_submission) != len(test_df)
        and is_tiny_public_sample_for_test(sample_submission=sample_submission, test_df=test_df, id_col=id_col)
    ):
        expanded = pd.DataFrame({id_col: test_df[id_col].to_numpy()})
        for col in sample_submission.columns:
            if col == id_col:
                continue
            expanded[col] = sample_column_default(sample_submission, col)
        for col in target_cols:
            if col not in expanded.columns:
                expanded[col] = sample_column_default(sample_submission, col)
        return expanded[list(sample_submission.columns)]
    return sample_submission.copy()


def is_tiny_public_sample_without_id(*, sample_submission: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Detect placeholder samples for target-only submissions."""
    if len(sample_submission) <= 0 or len(sample_submission) > 10:
        return False
    return len(test_df) > len(sample_submission)


def is_tiny_public_sample_for_test(
    *,
    sample_submission: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
) -> bool:
    """Detect notebook/code competitions where sample_submission is only a public placeholder."""
    if len(sample_submission) <= 0 or len(sample_submission) > 10:
        return False
    if len(test_df) <= len(sample_submission):
        return False
    if sample_submission[id_col].duplicated().any() or test_df[id_col].duplicated().any():
        return False
    return True


def sample_column_default(sample_submission: pd.DataFrame, column: str):
    if column not in sample_submission.columns or sample_submission.empty:
        return 0.0
    non_null = sample_submission[column].dropna()
    if non_null.empty:
        return 0.0
    numeric = pd.to_numeric(non_null, errors="coerce").dropna()
    if not numeric.empty:
        return float(numeric.mean())
    return non_null.iloc[0]
