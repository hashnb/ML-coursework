"""Baseline pipelines and small, explicit hyperparameter grids."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import make_preprocessor


def make_models():
    classifiers = {
        "Majority reference": DummyClassifier(strategy="most_frequent"),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="uniform",
                                    algorithm="brute", metric="minkowski", p=2, n_jobs=1),
        "Logistic regression": LogisticRegression(C=1.0, solver="lbfgs",
                                                   max_iter=1000, tol=1e-4,
                                                   class_weight=None, random_state=42),
        "Random forest": RandomForestClassifier(n_estimators=100, criterion="gini",
                                                max_depth=None, min_samples_split=2,
                                                min_samples_leaf=1, max_features="sqrt",
                                                bootstrap=True, class_weight=None,
                                                random_state=42, n_jobs=1),
    }
    return {
        name: Pipeline([("preprocessing", make_preprocessor(name != "Random forest")),
                        ("classifier", classifier)])
        for name, classifier in classifiers.items()
    }


def make_search_grids():
    """Include each original baseline so tuning can retain its settings."""
    return {
        "KNN": {"classifier__n_neighbors": [5, 15, 25],
                "classifier__weights": ["uniform", "distance"]},
        "Logistic regression": {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        "Random forest": {"classifier__n_estimators": [100, 200],
                          "classifier__max_depth": [None, 8],
                          "classifier__min_samples_leaf": [1, 5]},
    }
