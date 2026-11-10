# Dataset record

The original CSV was downloaded directly from the [dataset specified in the brief](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
on 10 November 2026. Kaggle metadata reports version 1 and a last update of
23 February 2018. It credits IBM sample data and describes Churn as customers
who left within the last month. The file contains one row per customer.

Original filename: WA_Fn-UseC_-Telco-Customer-Churn.csv

SHA-256: 88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a

Downloaded license label: "Data files © Original Authors". Dataset files and
prepared customer records are excluded from the public repository. The exact
metadata response and CSV are preserved locally. The download script verifies
the CSV hash before use.

## Column dictionary

The meanings below follow the source description and column names. Exact observed
category labels are recorded in results/data_audit.json. Charges have no verified
currency unit in the retrieved metadata.

| Column | Meaning | Intended use |
| --- | --- | --- |
| customerID | Customer identifier | Retain for split checks; exclude from predictors |
| gender | Recorded gender category | Categorical predictor; review fairness implications |
| SeniorCitizen | Senior-citizen indicator recorded as 0 or 1 | Categorical predictor |
| Partner | Recorded partner indicator | Categorical predictor |
| Dependents | Recorded dependents indicator | Categorical predictor |
| tenure | Customer tenure in months | Numeric predictor |
| PhoneService | Phone service subscription | Categorical predictor |
| MultipleLines | Multiple phone lines or no phone service | Categorical predictor |
| InternetService | Internet service category | Categorical predictor |
| OnlineSecurity | Online security subscription or no internet service | Categorical predictor |
| OnlineBackup | Online backup subscription or no internet service | Categorical predictor |
| DeviceProtection | Device protection subscription or no internet service | Categorical predictor |
| TechSupport | Technical support subscription or no internet service | Categorical predictor |
| StreamingTV | TV streaming subscription or no internet service | Categorical predictor |
| StreamingMovies | Movie streaming subscription or no internet service | Categorical predictor |
| Contract | Contract duration category | Categorical predictor |
| PaperlessBilling | Paperless billing indicator | Categorical predictor |
| PaymentMethod | Recorded payment method | Categorical predictor |
| MonthlyCharges | Monthly charge amount | Numeric predictor |
| TotalCharges | Accumulated charges; contains blank strings | Numeric predictor with missing values |
| Churn | Left within the last month, Yes or No | Target; never a predictor |

## Scope and limitations

The audit describes the full file's structure and data quality. Feature-to-target
analysis will use the training partition only. The source does not establish a
prospective observation date for each field, so later modelling must state the
assumption that predictors would be available when a prediction is made. This
snapshot cannot demonstrate future performance or the effect of a retention offer.
