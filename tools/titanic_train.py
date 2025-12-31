from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


def main(data_dir: Path, out_csv: Path) -> None:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    y = train["Survived"].astype(int)

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = train[features].copy()
    X_test = test[features].copy()

    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    cat_cols = ["Sex", "Embarked"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=2000)),
    ])

    model.fit(X, y)
    preds = model.predict(X_test).astype(int)

    sub = pd.DataFrame({
        "PassengerId": test["PassengerId"].astype(int),
        "Survived": preds,
    })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"wrote: {out_csv} shape={sub.shape} columns={list(sub.columns)}")


if __name__ == "__main__":
    main(Path("data/titanic/raw"), Path("data/titanic/submissions/submission.csv"))
