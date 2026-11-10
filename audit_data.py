"""Inspect the original dataset and save aggregate quality checks."""

import json
import pandas as pd
from download_data import RAW_PATH, ROOT, verify_bytes

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORY_VALUES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
    "Churn": ["Yes", "No"],
}
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


def read_original():
    verify_bytes(RAW_PATH.read_bytes())
    return pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False)


def validate_schema(frame):
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("Unexpected column names or order.")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    stripped = frame.apply(lambda column: column.str.strip())
    if stripped["customerID"].eq("").any() or stripped["customerID"].duplicated().any():
        raise ValueError("Missing or duplicate customer IDs need review before splitting.")
    for column, allowed in CATEGORY_VALUES.items():
        unknown = set(stripped[column]) - set(allowed)
        if unknown:
            raise ValueError(f"Unexpected values in {column}: {sorted(unknown)}")
    for column in NUMERIC_COLUMNS:
        values = stripped[column]
        parsed = pd.to_numeric(values, errors="coerce")
        invalid = parsed.isna() & ~((column == "TotalCharges") & values.eq(""))
        if invalid.any():
            raise ValueError(f"Unexpected non-numeric or missing values in {column}.")
        if (parsed.dropna() < 0).any() or (parsed.dropna() == float("inf")).any():
            raise ValueError(f"Invalid numeric range in {column}.")
        if column == "tenure" and (parsed.dropna() % 1 != 0).any():
            raise ValueError("Tenure must contain whole months.")
    return stripped


def audit(frame):
    checked = validate_schema(frame)
    blanks = checked.eq("").sum()
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_customer_ids": int(checked["customerID"].duplicated().sum()),
        "duplicate_predictor_rows_excluding_id": int(
            checked.drop(columns=["customerID", "Churn"]).duplicated().sum()
        ),
        "blank_counts": {column: int(value) for column, value in blanks.items()},
        "churn_counts": {label: int(checked["Churn"].eq(label).sum()) for label in ["No", "Yes"]},
        "blank_total_charges_with_zero_tenure": int(
            (checked["TotalCharges"].eq("") & checked["tenure"].eq("0")).sum()
        ),
        "column_types_on_read": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "categories": {column: sorted(checked[column].unique().tolist()) for column in CATEGORY_VALUES},
    }


def main():
    result = audit(read_original())
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / "data_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Audit passed: {result['rows']:,} rows, {result['columns']} columns.")
    return result


if __name__ == "__main__":
    main()
