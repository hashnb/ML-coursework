"""Load the original CSV, clean its types and reproduce the reserved split."""

import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
SOURCE_SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    """Download once if needed; reject a changed source before reading it."""
    if RAW_PATH.exists():
        contents = RAW_PATH.read_bytes()
    else:
        request = Request(DOWNLOAD_URL, headers={"User-Agent": "ML-coursework"})
        with urlopen(request, timeout=60) as response:
            with ZipFile(BytesIO(response.read())) as archive:
                contents = archive.read(RAW_PATH.name)
    if hashlib.sha256(contents).hexdigest() != SOURCE_SHA256:
        raise ValueError("Dataset hash changed. Review the source before continuing.")
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_bytes(contents)
    return pd.read_csv(BytesIO(contents), dtype=str, keep_default_na=False)


def clean_data(raw):
    """Keep blank TotalCharges missing; preserve valid service categories."""
    if set(raw.columns) != set(FEATURES + ["customerID", "Churn"]):
        raise ValueError("Unexpected dataset columns.")
    data = raw.copy().apply(lambda column: column.str.strip())
    required = data.drop(columns="TotalCharges")
    if required.isna().any().any() or required.eq("").any().any():
        raise ValueError("Unexpected missing value outside TotalCharges.")
    if data["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs need review before splitting.")
    if not data["Churn"].isin(["No", "Yes"]).all():
        raise ValueError("Churn must contain No or Yes.")
    for column in NUMERIC_FEATURES:
        values = data[column].replace("", np.nan)
        data[column] = pd.to_numeric(values, errors="raise")
        observed = data[column].dropna()
        if not np.isfinite(observed).all() or observed.lt(0).any():
            raise ValueError(f"Invalid numeric values in {column}.")
    if data["tenure"].mod(1).ne(0).any():
        raise ValueError("Tenure must contain whole months.")
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    return data


def make_split(data):
    """Use the original 80/20 stratified split, without fitting transforms."""
    return train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["Churn"],
    )
