"""Check grid coverage, selected candidates and fold-score aggregation."""

import json
import unittest

import numpy as np
from sklearn.model_selection import cross_val_score

from src.data import FEATURES, clean_data, load_data, make_split
from src.models import make_models, make_search_grids
from src.preprocessing import make_cv
from src.train import SCORING, tune_models


class TuningTests(unittest.TestCase):
    def test_grids_include_baselines(self):
        for name, grid in make_search_grids().items():
            baseline = make_models()[name].get_params()
            for parameter, choices in grid.items():
                self.assertIn(baseline[parameter], choices)

    def test_selection_and_independent_fold_scores(self):
        train, _ = make_split(clean_data(load_data()))
        sample = train.iloc[:200]
        X, y = sample[FEATURES], sample.Churn
        grids = {"KNN": {"classifier__n_neighbors": [5, 15]},
                 "Logistic regression": {"classifier__C": [0.1, 1]},
                 "Random forest": {"classifier__n_estimators": [10],
                                   "classifier__max_depth": [4, 8]}}
        candidates, selected = tune_models(X, y, grids)
        self.assertEqual((len(candidates), len(selected)), (6, 3))
        for _, row in selected.iterrows():
            rows = candidates.loc[candidates.model == row.model]
            self.assertEqual(row.candidate, rows.loc[rows.f1_mean.idxmax()].candidate)
            self.assertEqual(rows.selected.sum(), 1)
            model = make_models()[row.model].set_params(**json.loads(row.parameters))
            scores = cross_val_score(model, X, y, cv=make_cv(), scoring=SCORING["f1"], error_score="raise")
            np.testing.assert_allclose(scores, row[[f"fold_{i}_f1" for i in range(1, 6)]].astype(float))
            self.assertAlmostEqual(row.f1_mean, scores.mean())
            self.assertAlmostEqual(row.f1_std, scores.std(ddof=1))


if __name__ == "__main__":
    unittest.main()
