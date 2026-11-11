"""Check cleaning decisions and preserve the original holdout membership."""

import hashlib
import unittest

import pandas as pd

from src.data import FEATURES, ROOT, clean_data, load_data, make_split


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_data()
        cls.data = clean_data(cls.raw)
        cls.train, cls.test = make_split(cls.data)

    def test_source_and_cleaning(self):
        self.assertEqual(self.raw.shape, (7043, 21))
        self.assertEqual(self.data["TotalCharges"].isna().sum(), 11)
        self.assertEqual(self.data["Churn"].value_counts().to_dict(), {0: 5174, 1: 1869})
        self.assertEqual(self.data["OnlineSecurity"].eq("No internet service").sum(), 1526)
        self.assertEqual(len(FEATURES), 19)
        self.assertNotIn("customerID", FEATURES)
        self.assertNotIn("Churn", FEATURES)
        pd.testing.assert_frame_equal(load_data(), self.raw)
        self.assertTrue(self.raw["TotalCharges"].str.strip().eq("").any())

    def test_reserved_split(self):
        self.assertEqual((len(self.train), len(self.test)), (5634, 1409))
        self.assertFalse(set(self.train.customerID) & set(self.test.customerID))
        self.assertEqual(set(self.train.index) | set(self.test.index), set(self.data.index))
        self.assertEqual(self.train.Churn.value_counts().to_dict(), {0: 4139, 1: 1495})
        self.assertEqual(self.test.Churn.value_counts().to_dict(), {0: 1035, 1: 374})
        self.assertEqual(self.train.TotalCharges.isna().sum(), 8)
        expected = {
            "train": "b40bab5915da4f073e517c1552b56a724076ba055b4c01c466ef3bd74d576629",
            "test": "1778b191a462b1df1b25710b8410b7baec8fc8de1fd9814bd7d39c55eca69dd0",
        }
        for name, partition in [("train", self.train), ("test", self.test)]:
            manifest = partition[["customerID"]].copy()
            manifest.insert(0, "original_row_index", partition.index)
            contents = manifest.to_csv(index=False, lineterminator="\n").encode()
            self.assertEqual(hashlib.sha256(contents).hexdigest(), expected[name])
            saved = ROOT / "data/splits" / f"{name}_rows.csv"
            if saved.exists():
                self.assertEqual(saved.read_bytes(), contents)

    def test_invalid_values_are_rejected(self):
        for column, value in [("Churn", "Unknown"), ("tenure", "bad"),
                              ("MonthlyCharges", ""), ("TotalCharges", "bad")]:
            with self.subTest(column=column):
                invalid = self.raw.copy()
                invalid.loc[0, column] = value
                with self.assertRaises(ValueError):
                    clean_data(invalid)

    def test_duplicate_customers_are_rejected(self):
        invalid = self.raw.copy()
        invalid.loc[1, "customerID"] = invalid.loc[0, "customerID"]
        with self.assertRaisesRegex(ValueError, "Duplicate customer"):
            clean_data(invalid)


if __name__ == "__main__":
    unittest.main()
