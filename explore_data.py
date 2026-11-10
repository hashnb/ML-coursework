"""Create training-only EDA tables, figures and evidence for the report."""

import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

from download_data import ROOT
from prepare_data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

RESULTS = ROOT / "results" / "eda"
FIGURES = RESULTS / "figures"
COLORS = ["#4477AA", "#CC6677"]


def load_training():
    path = ROOT / "data" / "processed" / "train.csv"
    summary = json.loads((ROOT / "results" / "split_summary.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != summary["partitions"]["train"]["data_sha256"]:
        raise ValueError("Training data differs from the recorded split. Run preparation again.")
    frame = pd.read_csv(path, dtype={column: str for column in CATEGORICAL_FEATURES})
    if list(frame.columns) != FEATURES + ["Churn"]:
        raise ValueError("Unexpected training columns.")
    return frame


def category_rates(frame, column):
    table = frame.groupby(column, observed=True)["Churn"].agg(customers="size", churned="sum")
    table["churn_rate"] = table["churned"] / table["customers"]
    return table.sort_index()


def save_figure(figure, name):
    figure.savefig(FIGURES / name, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    frame = load_training()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    counts = frame["Churn"].value_counts().reindex([0, 1], fill_value=0)
    balance = pd.DataFrame({"customers": counts, "proportion": counts / len(frame)})
    balance.index.name = "Churn"
    balance.to_csv(RESULTS / "class_balance.csv")
    frame[NUMERIC_FEATURES].describe().T.to_csv(RESULTS / "numeric_summary.csv")
    frame.groupby("Churn")[NUMERIC_FEATURES].median().to_csv(RESULTS / "numeric_medians_by_churn.csv")
    missing = pd.DataFrame({"missing": frame.isna().sum(), "proportion": frame.isna().mean()})
    missing.to_csv(RESULTS / "missing_values.csv")
    groups = pd.concat(
        [category_rates(frame, column).assign(feature=column).rename_axis("category").reset_index()
         for column in CATEGORICAL_FEATURES],
        ignore_index=True,
    )[["feature", "category", "customers", "churned", "churn_rate"]]
    groups.to_csv(RESULTS / "categorical_churn_rates.csv", index=False)
    correlation = frame[NUMERIC_FEATURES].corr()
    correlation.to_csv(RESULTS / "numeric_correlations.csv")

    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(["Stayed (0)", "Churned (1)"], counts, color=COLORS)
    axis.bar_label(bars, labels=[f"{count:,}\n{count / len(frame):.1%}" for count in counts], padding=4)
    axis.set(title=f"Churn distribution: training data (n={len(frame):,})", ylabel="Customers")
    axis.set_ylim(0, max(counts) * 1.25)
    save_figure(figure, "churn_distribution.png")

    figure, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    labels = {"tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges (source units)",
              "TotalCharges": "Total charges (source units)"}
    for column, top, bottom in zip(NUMERIC_FEATURES, axes[0], axes[1]):
        top.hist(frame[column].dropna(), bins=24, color=COLORS[0], edgecolor="white")
        top.set(xlabel=labels[column], ylabel="Customers")
        bottom.boxplot(
            [frame.loc[frame["Churn"].eq(label), column].dropna() for label in [0, 1]],
            tick_labels=["Stayed", "Churned"], showfliers=True,
        )
        bottom.set(ylabel=labels[column])
    figure.suptitle("Numeric distributions and churn groups: training data")
    save_figure(figure, "numeric_distributions.png")

    selected = ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport", "SeniorCitizen"]
    figure, axes = plt.subplots(3, 2, figsize=(15, 11), layout="constrained")
    overall_rate = float(frame["Churn"].mean())
    for column, axis in zip(selected, axes.flat):
        table = category_rates(frame, column)
        bars = axis.barh(table.index.astype(str), table["churn_rate"], color=COLORS[0])
        axis.bar_label(
            bars,
            labels=[f"{row.churn_rate:.1%} (n={int(row.customers):,})" for row in table.itertuples()],
            padding=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )
        axis.axvline(overall_rate, color=COLORS[1], linestyle="--", label="Overall training rate")
        axis.set(title=column, xlabel="Observed churn rate", xlim=(0, 0.7))
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.invert_yaxis()
    figure.suptitle(
        "Churn rates by category: training data\n"
        f"Labels include group size. Dashed line: overall training churn ({overall_rate:.1%})."
    )
    save_figure(figure, "categorical_churn_rates.png")

    figure, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    heatmap = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(3), NUMERIC_FEATURES, rotation=20, ha="right")
    axis.set_yticks(range(3), NUMERIC_FEATURES)
    for row in range(3):
        for column in range(3):
            value = correlation.iloc[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                      color="white" if abs(value) > 0.75 else "black")
    axis.set_title("Numeric correlations (training)")
    figure.colorbar(heatmap, ax=axis, label="Correlation")
    save_figure(figure, "numeric_correlations.png")

    contract = category_rates(frame, "Contract")
    medians = frame.groupby("Churn")[NUMERIC_FEATURES].median()
    rate = lambda label: contract.loc[label, "churn_rate"]
    report = f"""# Dataset inspection and EDA findings

Analysis date: 10 November 2026. These are measured descriptive results and AI interpretations, not model evaluation scores. Feature-to-target analysis uses only the {len(frame):,} training records. The 1,409 reserved test records have not been used for EDA.

## Measured findings

- Training contains {int(counts[1]):,} churned customers ({overall_rate:.2%}) and {int(counts[0]):,} who stayed. The majority-class training proportion is {counts[0] / len(frame):.2%}. Accuracy alone could conceal missed churners.
- Month-to-month customers have an observed churn rate of {rate('Month-to-month'):.2%}, compared with {rate('One year'):.2%} for one-year and {rate('Two year'):.2%} for two-year contracts. Group sizes and numerators appear in categorical_churn_rates.csv.
- Median tenure is {medians.loc[1, 'tenure']:.0f} months among churned customers and {medians.loc[0, 'tenure']:.0f} months among those who stayed.
- Median monthly charges are {medians.loc[1, 'MonthlyCharges']:.2f} among churned customers and {medians.loc[0, 'MonthlyCharges']:.2f} among those who stayed, in the source's charge units.
- There are {int(frame['TotalCharges'].isna().sum())} missing total-charge values in training. Histograms, boxplots and numeric summaries omit missing values for that field; group denominators remain the full category counts.
- Tenure and TotalCharges have Pearson correlation {correlation.loc['tenure', 'TotalCharges']:.3f} in training, calculated on observed pairs.

## Interpretation and preparation decisions

Contract and tenure show associations with churn in this training sample. These comparisons do not establish that a contract change or retention offer would prevent churn. Monthly and accumulated charges measure different aspects of the account, and accumulated charges are related to tenure. Retain all three numeric predictors initially and assess the models with training cross-validation.

Keep the original data unchanged. Retain distinct customer IDs even where profiles match. Exclude the identifier from predictors and encode the target separately. Preserve valid service-absence categories. Keep SeniorCitizen categorical. Numeric scaling is planned for KNN and logistic regression; it must be fitted inside their training pipelines.

The full-file audit found 11 blank total-charge strings, all on records with zero tenure. This pattern does not prove the correct replacement value. Retain them as missing and plan median imputation within training folds. Extreme values in the displayed distributions are retained because no verified data error has been established.

## Ethics and limits

Publish aggregate tables and code; keep customer-level records local. Gender and the senior-citizen indicator need scrutiny when evaluating subgroup errors later. Observed group churn rates alone do not establish model fairness. Consider whether these fields are appropriate for the intended use and explain that decision before deployment.

Kaggle describes the target as customers who left within the last month. This is a static sample without verified observation dates or prospective follow-up. Predictor availability at decision time remains an assumption. Later test performance will describe this held-out sample, not prove performance on future customers. The source metadata does not establish a representative deployment population.

## Evidence and next step

Tables and four figures in this folder are generated by explore_data.py. The source checksum, split seed and partition hashes are recorded in results/split_summary.json. The data dictionary and provenance are in data/DATASET.md.

Next: construct leakage-safe model pipelines, fit all three baseline classifiers, and agree how the brief's undefined "both model experimental settings" should be interpreted. No models, performance scores or post-deployment monitoring have been implemented in this milestone.

## Figures

![Training class counts and proportions](figures/churn_distribution.png)

![Numeric distributions and churn-group comparisons in training](figures/numeric_distributions.png)

![Training category churn rates with group sizes](figures/categorical_churn_rates.png)

![Pearson correlations between the three numeric training features](figures/numeric_correlations.png)

References: [specified Kaggle dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn); [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
"""
    (RESULTS / "EDA.md").write_text(report, encoding="utf-8")
    print(f"EDA complete: {len(frame):,} training rows, six tables and four figures.")
    return frame


if __name__ == "__main__":
    main()
