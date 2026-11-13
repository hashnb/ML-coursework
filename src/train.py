"""Compare baseline pipelines using the same training-only validation folds."""

import json
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate

from src.data import FEATURES, ROOT, clean_data, load_data, make_split
from src.models import make_models
from src.preprocessing import make_cv

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def compare_models(X, y):
    """Return per-fold metrics and unweighted fold means/sample deviations."""
    folds = list(make_cv().split(X, y))
    records = []
    for name, pipeline in make_models().items():
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            results = cross_validate(pipeline, X, y, cv=folds, scoring=SCORING,
                                     error_score="raise")
        for fold, (fit, valid) in enumerate(folds):
            records.append({"model": name, "fold": fold + 1,
                            "training_rows": len(fit), "validation_rows": len(valid),
                            **{metric: results[f"test_{metric}"][fold] for metric in SCORING}})
    scores = pd.DataFrame(records)
    summary = scores.groupby("model", sort=False)[list(SCORING)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return scores, summary.reset_index()


def run_comparison():
    train, _ = make_split(clean_data(load_data()))
    scores, summary = compare_models(train[FEATURES], train["Churn"])
    tables = ROOT / "outputs/tables"
    tables.mkdir(parents=True, exist_ok=True)
    scores.to_csv(tables / "baseline_folds.csv", index=False, lineterminator="\n")
    summary.to_csv(tables / "baseline_comparison.csv", index=False, lineterminator="\n")
    settings = {
        name: {"scale_numeric": name != "Random forest",
               "classifier": model["classifier"].get_params(deep=False)}
        for name, model in make_models().items()
    }
    (ROOT / "outputs/baseline_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return scores, summary


if __name__ == "__main__":
    _, summary = run_comparison()
    print(summary.round(4).to_string(index=False))
