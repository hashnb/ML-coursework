# Telco customer churn

CM2604 Coursework Part 1. The project explores the training data and prepares
it for modelling. All three classifiers have baseline and tuned configurations.
Final-test evaluation and the report are later phases.

## Run

On the existing development machine, run these commands from the project folder:

~~~powershell
.\.venv312\Scripts\python.exe scripts/run_exploration.py
.\.venv312\Scripts\python.exe scripts/run_preprocessing.py
.\.venv312\Scripts\python.exe scripts/run_notebook.py
.\.venv312\Scripts\python.exe -m unittest discover -s tests -v
~~~

The runner executes [the notebook](notebooks/01_data_exploration.ipynb) in a
fresh kernel and saves its outputs, figures and CSV tables. The notebook also
opens in a Python notebook editor using the same environment.

For a fresh clone, use Python 3.12 to create an environment with
python -m venv .venv312, then install dependencies with
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
before running the commands above. These setup commands assume python points
to Python 3.12.

## Data

The [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
credits IBM sample data. The loader downloads the CSV on its first run and
checks its SHA-256 against version 1:

~~~text
88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a
~~~

If the automatic download fails, download and extract the dataset from Kaggle
and put WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/, then rerun. Customer
records stay local and are excluded from Git. The retrieved license label is
"Data files © Original Authors".

There are 7,043 customers and 21 columns. Churn is the target (No = 0, Yes = 1).
Customer IDs are excluded from the 19 predictors. Eleven blank TotalCharges
values remain missing. Valid categories such as "No internet service" are kept.
The 80/20 stratified split uses seed 42: 5,634 training rows and 1,409 reserved
test rows. Feature analysis and development cross-validation use training data
only. The exploration notebook keeps missing totals unchanged. The separate
preprocessing pipeline learns its transformations within each training fold.

## Preprocessing

make_preprocessor() in src/preprocessing.py creates an unfitted transformer:
numeric values receive median imputation and standard scaling; categorical
values receive one-hot encoding. SeniorCitizen stays categorical. Service-absence
categories are retained, and unseen validation categories map to all-zero
indicators for that field. Customer IDs and the target are excluded.

Use make_preprocessor(scale_numeric=False) for random forest, which does not
need numeric scaling. KNN and logistic regression will use the scaled version.
Put the transformer and classifier in the same Pipeline before cross-validation
so validation data cannot affect the medians, scales or learned categories.

make_cv() creates five stratified folds with shuffle=True and random_state=42.
The preprocessing runner uses these folds on the 5,634 training customers and
records a DummyClassifier(strategy="most_frequent") reference in
outputs/tables/majority_reference.csv. Churn is positive class 1. Precision,
recall and F1 use zero_division=0 because this reference predicts no churners.
The final test set is reserved for later evaluation.

The reference's mean validation accuracy is 73.4647% (sample standard deviation
0.0105 percentage points). Churn precision, recall and F1 are all 0; ROC-AUC
is 0.5. It predicts that every customer stays. Both preprocessing variants
produce 46 numeric/encoded features on the current training data.

## Files

The comparison notebook also runs the small tuning grids: 6 KNN, 4 logistic
regression and 8 random forest candidates, each on five training folds.
Selection uses mean churn F1. Tuned F1 is 0.6003 for KNN, 0.5965 for logistic
regression and 0.5814 for random forest. The selected KNN uses 25 neighbours
and uniform votes. Its lead is small compared with fold variation; these
selection-CV scores do not establish final-test performance.

All selected settings and the chosen model are fixed in
outputs/frozen_configuration.json before evaluating the test set. The standard
classifier.predict decision rule is unchanged. Exact score ties retain the
first grid candidate, then the first model in KNN/logistic/forest order.
Candidate results and fold F1 values are in outputs/tables/tuning_candidates.csv;
the selected configurations and baseline changes are in tuned_comparison.csv.
The baseline-settings hash in the frozen file uses UTF-8 text with LF newlines.
The user confirmed baseline versus tuned as the working interpretation of both
experimental settings; the lecturer's intended meaning remains unclarified.

The [model-comparison notebook](notebooks/02_model_comparison.ipynb) compares
KNN, logistic regression and random forest on the same five training folds.
Each complete pipeline is fitted within its fold. Full settings are saved in
outputs/baseline_settings.json; per-fold scores and mean/sample-deviation
summaries are in outputs/tables/baseline_folds.csv and baseline_comparison.csv.

Logistic regression has the highest mean baseline churn F1 (0.5930), compared
with KNN (0.5463) and random forest (0.5385). Mean churn F1 is the primary score
for the planned tuning phase, with the other metrics retained to explain
tradeoffs. These are validation results; the final test remains reserved.
The notebook explains false alarms, missed churners and limitations.

- notebooks/01_data_exploration.ipynb: inspection, cleaning decisions and EDA.
- notebooks/02_model_comparison.ipynb: baseline comparison and explanation.
- src/data.py: loading, type cleaning and the reproducible split.
- src/preprocessing.py: numeric/categorical preparation and shared CV folds.
- src/models.py and src/train.py: baselines, small tuning grids and shared-fold scoring.
- scripts/run_exploration.py: fresh-kernel execution.
- scripts/run_preprocessing.py: training-fold majority-class reference.
- scripts/run_notebook.py: execute the comparison notebook, or a supplied notebook path.
- outputs/: figures and aggregate tables regenerated by the runners.
- tests/test_data.py: data quality and original split checks.
- tests/test_preprocessing.py: missing values, new categories, scaling and fold isolation.
- tests/test_models.py: pipeline roles and independent prediction/metric checks.
- tests/test_tuning.py: baseline grid coverage and independent selected-candidate fold scores.

The notebook records findings and limitations. These sample data cannot establish
why a customer leaves or guarantee performance on another company's customers.
Gender and the senior-citizen indicator need a fairness review before deployment.

## Assistance

OpenAI Codex assisted with planning, code, checks and explanation drafts. The
student should review the work, be able to explain each step, and disclose
assistance according to university requirements.
