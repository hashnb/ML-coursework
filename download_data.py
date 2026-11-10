"""Download the coursework dataset without changing the supplied CSV."""

import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parent
DATASET_PAGE = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
EXPECTED_SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
RAW_PATH = ROOT / "data" / "raw" / FILENAME


def verify_bytes(data):
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Dataset version differs: expected {EXPECTED_SHA256}, got {digest}")
    return digest


def download():
    if RAW_PATH.exists():
        verify_bytes(RAW_PATH.read_bytes())
        print("Existing original matches the recorded SHA-256.")
        return RAW_PATH
    with urlopen(DOWNLOAD_URL, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_bytes = archive.read(FILENAME)
    verify_bytes(csv_bytes)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_bytes(csv_bytes)
    print(f"Downloaded verified original: {RAW_PATH.name}")
    return RAW_PATH


if __name__ == "__main__":
    download()
