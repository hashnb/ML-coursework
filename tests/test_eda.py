"""Reconcile EDA aggregates against the original CSV and saved training manifest."""

import csv
from collections import Counter
import unittest

from download_data import RAW_PATH, ROOT
from explore_data import category_rates, load_training
from prepare_data import CATEGORICAL_FEATURES


class EDAChecks(unittest.TestCase):
    def test_contract_rates_match_original_training_customers(self):
        with (ROOT / "data/splits/train_rows.csv").open(newline="") as stream:
            ids = {row["customerID"] for row in csv.DictReader(stream)}
        totals, churned = Counter(), Counter()
        with RAW_PATH.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row["customerID"] in ids:
                    totals[row["Contract"]] += 1
                    churned[row["Contract"]] += row["Churn"] == "Yes"
        rates = category_rates(load_training(), "Contract")
        for category in totals:
            self.assertEqual(rates.loc[category, "customers"], totals[category])
            self.assertEqual(rates.loc[category, "churned"], churned[category])
            self.assertAlmostEqual(
                rates.loc[category, "churn_rate"], churned[category] / totals[category],
            )

    def test_each_category_summary_covers_training_only(self):
        frame = load_training()
        self.assertEqual(len(frame), 5634)
        for column in CATEGORICAL_FEATURES:
            table = category_rates(frame, column)
            self.assertEqual(table["customers"].sum(), len(frame))
            self.assertEqual(table["churned"].sum(), 1495)


if __name__ == "__main__":
    unittest.main()
