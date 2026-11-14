"""Compare baseline pipelines using the same training-only validation folds."""

import json
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_validate

from src.data import FEATURES, ROOT, clean_data, load_data, make_split
from src.models import make_models, make_search_grids
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


def tune_models(X, y, grids=None):
    """Search complete pipelines; tied scores retain the first grid candidate."""
    grids = make_search_grids() if grids is None else grids
    folds = list(make_cv().split(X, y))
    candidates, selected = [], []
    for name, grid in grids.items():
        print(f"Tuning {name}...", flush=True)
        search = GridSearchCV(make_models()[name], grid, scoring=SCORING,
                              refit=False, cv=folds, n_jobs=1, error_score="raise")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            search.fit(X, y)
        result = search.cv_results_
        best = int(result["rank_test_f1"].argmin())
        for index, params in enumerate(result["params"]):
            row = {"model": name, "candidate": index + 1,
                   "parameters": json.dumps(params, sort_keys=True),
                   "selected": index == best}
            for metric in SCORING:
                row[f"{metric}_mean"] = float(result[f"mean_test_{metric}"][index])
                # GridSearchCV reports population SD; use sample SD as in Phase 3.
                values = [float(result[f"split{fold}_test_{metric}"][index]) for fold in range(5)]
                row[f"{metric}_std"] = float(pd.Series(values).std(ddof=1))
                if metric == "f1":
                    row.update({f"fold_{fold+1}_f1": value for fold, value in enumerate(values)})
            candidates.append(row)
            if index == best:
                selected.append(row.copy())
    return pd.DataFrame(candidates), pd.DataFrame(selected)


def run_tuning():
    import hashlib
    import sklearn
    from src.data import SOURCE_SHA256

    train, holdout = make_split(clean_data(load_data()))
    candidates, selected = tune_models(train[FEATURES], train["Churn"])
    tables = ROOT / "outputs/tables"
    baseline = pd.read_csv(tables / "baseline_comparison.csv").set_index("model")
    selected["baseline_f1"] = selected["model"].map(baseline["f1_mean"])
    selected["f1_change"] = selected["f1_mean"] - selected["baseline_f1"]
    candidates.to_csv(tables / "tuning_candidates.csv", index=False, lineterminator="\n")
    selected.to_csv(tables / "tuned_comparison.csv", index=False, lineterminator="\n")
    winner = selected.loc[selected["f1_mean"].idxmax()]
    configurations = {}
    for _, row in selected.iterrows():
        model = make_models()[row["model"]].set_params(**json.loads(row["parameters"]))
        configurations[row["model"]] = {
            "parameters": json.loads(row["parameters"]),
            "classifier": model["classifier"].get_params(deep=False),
            "scale_numeric": row["model"] != "Random forest",
        }
    manifest_hashes = {}
    for name, partition in [("train", train), ("test", holdout)]:
        manifest = partition[["customerID"]].copy()
        manifest.insert(0, "original_row_index", partition.index)
        manifest_hashes[name] = hashlib.sha256(
            manifest.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    frozen = {
        "selected_model": winner["model"], "selection_metric": "mean training-CV churn F1",
        "tie_rule": "first grid candidate; first model in KNN, logistic regression, random forest order",
        "prediction_method": "classifier.predict (standard decision rule, unchanged)",
        "cv": {"n_splits": 5, "shuffle": True, "random_state": 42},
        "split": {"test_size": 0.2, "random_state": 42, "stratify": "Churn",
                  "manifest_sha256": manifest_hashes},
        "source_sha256": SOURCE_SHA256, "sklearn_version": sklearn.__version__,
        "baseline_settings_sha256": hashlib.sha256((ROOT / "outputs/baseline_settings.json").read_text(encoding="utf-8").encode()).hexdigest(),
        "features": FEATURES, "positive_class": 1, "final_test_evaluated": False,
        "experimental_settings": "baseline versus tuned; user-approved interpretation; lecturer clarification pending",
        "configurations": configurations,
    }
    (ROOT / "outputs/frozen_configuration.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return candidates, selected, frozen


if __name__ == "__main__":
    _, summary = run_comparison()
    print(summary.round(4).to_string(index=False))
