import sys
from pathlib import Path

import pandas as pd


def validate_titanic_submission(submission_path: Path) -> None:
    """
    Validate Titanic submission format.

    Requirements:
    - Must have exactly 2 columns: PassengerId, Survived
    - Must have exactly 418 rows
    - Survived values must be 0 or 1
    """
    df = pd.read_csv(submission_path)

    required = ["PassengerId", "Survived"]
    missing = [c for c in required if c not in df.columns]
    extra = [c for c in df.columns if c not in required]

    assert not missing, f"missing columns: {missing}"
    assert not extra, f"extra columns: {extra}"
    assert df.shape[0] == 418, f"expected 418 rows, got {df.shape[0]}"
    assert set(df["Survived"].unique()).issubset({0, 1}), "Survived must be 0/1"

    print(f"OK: {submission_path}")
    print(f"  Rows: {df.shape[0]}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Survived distribution: {df['Survived'].value_counts().to_dict()}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/titanic/submissions/submission.csv")
    validate_titanic_submission(p)
