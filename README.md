# Telco customer churn coursework

CM2604 Coursework Part 1. The work currently covers dataset inspection,
preparation and training-only EDA. Models and model scores have not been produced.

Read [the findings and plots](results/eda/EDA.md), then open
01_data_and_eda.ipynb for a short walkthrough of the executed workflow.

## Run the dataset audit

Use Python 3.12. From this folder:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe audit_data.py
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe explore_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_notebook.py
~~~

The downloader verifies the original CSV against a recorded SHA-256. It stops if
the dataset has changed. The audit reports schema, blank values, duplicates and
target counts in results/data_audit.json.

## Preparation decisions

The split reserves 20% for the final test and uses 80% for development. A fixed
seed of 42 makes it repeatable; stratification keeps the churn proportions close
in both partitions. These are implementation choices, not prescribed coursework
settings. Training data will supply the EDA and later cross-validation.

Cleaning trims surrounding spaces, converts tenure and charge fields to numbers,
and maps Churn from No/Yes to 0/1. The 11 blank total-charge values remain missing.
Customer IDs stay in local split manifests and are excluded from predictors.
SeniorCitizen stays categorical despite its numeric-looking values. Structural
labels such as "No internet service" remain valid categories.

No imputer, scaler or encoder has been fitted. Later model pipelines must fit
learned transformations inside training folds. The reserved test must not guide
feature selection, hyperparameters or thresholds. The split summary records the
seed, feature groups, counts and checksums. Prepared rows and split manifests
remain local.

Different customer IDs with the same predictor values are retained as distinct
records. Matching profiles alone do not establish duplicate customers.

## Outputs and explanation

The EDA uses 5,634 training records. It saves six aggregate CSV tables and four
PNG figures covering churn balance, numeric distributions, category churn
rates with group sizes, and correlations. The findings explain the observed
associations and the reasons for the preparation choices. There are no test
feature summaries or model performance claims.

The notebook calls the same small Python files used by the command-line
workflow and displays their aggregate outputs. Keep these files together.
run_notebook.py selects the current Python interpreter explicitly and executes
the notebook with a fresh kernel. In an editor, choose the environment where
the requirements were installed.

The checked local environment is recorded in results/environment.json. On the
author's Windows computer, the bundled Python 3.12 runtime and a local
.venv312 environment were used after application control rejected a newly
installed Python 3.14 pandas build. Security settings were not changed.

## Next milestone

Build preprocessing pipelines and the three required baseline classifiers.
Imputation, encoding and scaling must be fitted within training folds.
Clarify the brief's phrase "both model experimental settings" before finalizing
the experiment comparison. Keep the final test reserved until selection is fixed.

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
