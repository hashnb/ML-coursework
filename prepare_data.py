"""Clean types, separate predictors from the target, and reserve a final test set."""

import hashlib
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from audit_data import NUMERIC_COLUMNS, read_original, validate_schema
from download_data import EXPECTED_SHA256, ROOT

TEST_SIZE = 0.20
RANDOM_STATE = 42
NUMERIC_FEATURES = NUMERIC_COLUMNS.copy()
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_data(raw):
    cleaned = validate_schema(raw).copy()
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned["Churn"] = cleaned["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    # Keep missing totals as NaN. Any learned imputation belongs inside model CV.
    return cleaned


def make_split(cleaned):
    return train_test_split(
        cleaned, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=cleaned["Churn"],
    )


def main():
    cleaned = clean_data(read_original())
    train, test = make_split(cleaned)
    if set(train["customerID"]) & set(test["customerID"]):
        raise ValueError("A customer appears in both partitions.")
    if set(train.index) & set(test.index) or len(train) + len(test) != len(cleaned):
        raise ValueError("Split overlap or missing records.")
    summary = {
        "source_sha256": EXPECTED_SHA256,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "method": "stratified random holdout by Churn",
        "target_encoding": {"No": 0, "Yes": 1},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_predictors": ["customerID", "Churn"],
        "imputation_scaling_encoding_fitted": False,
        "partitions": {},
    }
    for name, partition in [("train", train), ("test", test)]:
        data_path = ROOT / "data" / "processed" / f"{name}.csv"
        manifest_path = ROOT / "data" / "splits" / f"{name}_rows.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        partition[FEATURES + ["Churn"]].to_csv(data_path, index=False, lineterminator="\n")
        manifest = partition[["customerID"]].copy()
        manifest.insert(0, "original_row_index", partition.index)
        manifest.to_csv(manifest_path, index=False, lineterminator="\n")
        summary["partitions"][name] = {
            "rows": len(partition),
            "class_counts": {
                str(label): int(partition["Churn"].eq(label).sum()) for label in [0, 1]
            },
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
    (ROOT / "results" / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved split: {len(train):,} training rows; {len(test):,} reserved test rows.")
    return summary


if __name__ == "__main__":
    main()
