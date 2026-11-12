"""Exercise missing values, new categories and isolation of validation folds."""

import unittest

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.data import FEATURES, NUMERIC_FEATURES, clean_data, load_data, make_split
from src.preprocessing import make_cv, make_preprocessor


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train, _ = make_split(clean_data(load_data()))
        cls.X, cls.y = train[FEATURES], train["Churn"]

    def test_missing_totals_and_unseen_categories(self):
        fit = self.X.iloc[:100].copy()
        validation = self.X.iloc[100:102].copy()
        validation.loc[:, "TotalCharges"] = np.nan
        validation.loc[:, "Contract"] = "Unseen contract"
        for scale in [True, False]:
            with self.subTest(scale=scale):
                prep = make_preprocessor(scale)
                training_values = prep.fit_transform(fit)
                medians = prep.named_transformers_["numeric"]["imputer"].statistics_.copy()
                validation_values = prep.transform(validation)
                self.assertTrue(np.isfinite(validation_values).all())
                self.assertEqual(training_values.shape[1], validation_values.shape[1])
                encoder = prep.named_transformers_["categorical"]
                self.assertNotIn("Unseen contract", encoder.categories_[13])
                np.testing.assert_array_equal(medians, prep.named_transformers_["numeric"]["imputer"].statistics_)
                np.testing.assert_allclose(medians, fit[NUMERIC_FEATURES].median())

    def test_numeric_scaling_and_predictor_roles(self):
        scaled = make_preprocessor()
        unscaled = make_preprocessor(False)
        a = scaled.fit_transform(self.X)
        b = unscaled.fit_transform(self.X)
        np.testing.assert_allclose(a[:, :3].mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(a[:, :3].std(axis=0), 1, atol=1e-10)
        expected = self.X[NUMERIC_FEATURES].fillna(self.X[NUMERIC_FEATURES].median())
        np.testing.assert_allclose(b[:, :3], expected)
        self.assertEqual(a.shape, b.shape)
        names = scaled.get_feature_names_out()
        self.assertTrue(any("SeniorCitizen_0" in name for name in names))
        self.assertTrue(any("No internet service" in name for name in names))
        self.assertFalse(any("customerID" in name or "Churn" in name for name in names))

    def test_cv_learns_only_from_each_fit_partition(self):
        model = Pipeline([("preprocessing", make_preprocessor()),
                          ("classifier", DummyClassifier(strategy="most_frequent"))])
        folds = list(make_cv().split(self.X, self.y))
        outcome = cross_validate(model, self.X, self.y, cv=folds,
                                 return_estimator=True, error_score="raise")
        self.assertFalse(hasattr(model["preprocessing"], "transformers_"))
        for (fit, valid), fitted in zip(folds, outcome["estimator"]):
            self.assertFalse(set(fit) & set(valid))
            numeric = fitted["preprocessing"].named_transformers_["numeric"]
            expected = self.X.iloc[fit][NUMERIC_FEATURES]
            np.testing.assert_allclose(numeric["imputer"].statistics_, expected.median())
            np.testing.assert_allclose(numeric["scaler"].mean_, expected.fillna(expected.median()).mean())
            self.assertTrue(np.isfinite(fitted["preprocessing"].transform(self.X.iloc[valid])).all())

    def test_folds_repeat_and_cover_training_once(self):
        first = list(make_cv().split(self.X, self.y))
        second = list(make_cv().split(self.X, self.y))
        for (a, b), (c, d) in zip(first, second):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)
            self.assertEqual(self.y.iloc[b].sum(), 299)
        validation = np.concatenate([valid for _, valid in first])
        np.testing.assert_array_equal(np.sort(validation), np.arange(len(self.X)))


if __name__ == "__main__":
    unittest.main()
