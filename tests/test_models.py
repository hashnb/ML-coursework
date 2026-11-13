"""Check baseline roles and independently reproduce fold metric calculations."""

import unittest

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data import FEATURES, clean_data, load_data, make_split
from src.models import make_models
from src.preprocessing import make_cv
from src.train import compare_models


class ModelTests(unittest.TestCase):
    def test_pipeline_roles_and_fresh_instances(self):
        models = make_models()
        self.assertEqual(set(models), {"Majority reference", "KNN", "Logistic regression", "Random forest"})
        for name, model in models.items():
            self.assertFalse(hasattr(model["preprocessing"], "transformers_"))
            numeric = model["preprocessing"].transformers[0][1]
            self.assertEqual("scaler" in numeric.named_steps, name != "Random forest")
            self.assertIsNot(model, make_models()[name])

    def test_scores_match_independent_predictions(self):
        train, _ = make_split(clean_data(load_data()))
        sample = train.iloc[:250]
        X, y = sample[FEATURES], sample.Churn
        scores, summary = compare_models(X, y)
        self.assertEqual(len(scores), 20)
        self.assertEqual(len(summary), 4)
        for name, model in make_models().items():
            for fold, (fit, valid) in enumerate(make_cv().split(X, y), 1):
                fitted = clone(model).fit(X.iloc[fit], y.iloc[fit])
                prediction = fitted.predict(X.iloc[valid])
                positive = list(fitted.classes_).index(1)
                probability = fitted.predict_proba(X.iloc[valid])[:, positive]
                truth = y.iloc[valid]
                expected = [accuracy_score(truth, prediction),
                            precision_score(truth, prediction, zero_division=0),
                            recall_score(truth, prediction, zero_division=0),
                            f1_score(truth, prediction, zero_division=0),
                            roc_auc_score(truth, probability)]
                row = scores.loc[(scores.model == name) & (scores.fold == fold)].iloc[0]
                np.testing.assert_allclose(row[["accuracy", "precision", "recall", "f1", "roc_auc"]].astype(float), expected)
            rows = scores.loc[scores.model == name]
            result = summary.loc[summary.model == name].iloc[0]
            self.assertAlmostEqual(result.f1_mean, rows.f1.mean())
            self.assertAlmostEqual(result.f1_std, rows.f1.std(ddof=1))


if __name__ == "__main__":
    unittest.main()
