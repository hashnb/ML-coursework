"""Check training-fold preparation with a majority-class reference."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import FEATURES, clean_data, load_data, make_split
from src.preprocessing import make_cv, make_preprocessor


def main():
    train, _ = make_split(clean_data(load_data()))
    X, y = train[FEATURES], train["Churn"]
    reference = Pipeline([
        ("preprocessing", make_preprocessor()),
        ("classifier", DummyClassifier(strategy="most_frequent")),
    ])
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    folds = list(make_cv().split(X, y))
    results = cross_validate(reference, X, y, cv=folds, scoring=scoring,
                             return_estimator=True, error_score="raise")
    table = pd.DataFrame({
        "fold": range(1, 6),
        "training_rows": [len(fit) for fit, _ in folds],
        "validation_rows": [len(valid) for _, valid in folds],
        "encoded_features": [len(model["preprocessing"].get_feature_names_out())
                             for model in results["estimator"]],
        **{metric: results[f"test_{metric}"] for metric in scoring},
    })
    path = ROOT / "outputs/tables/majority_reference.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")
    print(table.to_string(index=False))
    print("\nFive-fold training CV (mean and sample standard deviation):")
    print(table[list(scoring)].agg(["mean", "std"]).round(6).to_string())
    print("\nSaved outputs/tables/majority_reference.csv. Final test set not evaluated.")


if __name__ == "__main__":
    main()
