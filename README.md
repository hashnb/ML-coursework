# Telco customer churn coursework

CM2604 Coursework Part 1. This first increment downloads and audits the specified
Telco Customer Churn CSV. Models and model scores have not been produced.

## Run the dataset audit

Use Python 3.12. From this folder:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe audit_data.py
~~~

The downloader verifies the original CSV against a recorded SHA-256. It stops if
the dataset has changed. The audit reports schema, blank values, duplicates and
target counts in results/data_audit.json.

## Dataset and publication

Source: [Telco Customer Churn, blastchar on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
The downloaded metadata identifies version 1, last updated 23 February 2018,
and credits IBM sample data. The download is dated 10 November 2026.

Kaggle labels the data license "Data files © Original Authors". No broader
redistribution permission is assumed. Original and prepared customer-level data
remain local and are excluded from Git. Run the downloader to obtain the same
bytes from Kaggle.

The supplied university specification and personal study notes stay outside this
repository. This repository contains coursework code and aggregate analysis.
