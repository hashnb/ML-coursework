"""Check data failures that could silently damage later evaluation."""

import unittest

from audit_data import audit, read_original, validate_schema
from download_data import verify_bytes
from prepare_data import FEATURES, clean_data, make_split


class DataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = read_original()

    def test_original_quality_counts(self):
        result = audit(self.original)
        self.assertEqual((result["rows"], result["columns"]), (7043, 21))
        self.assertEqual(result["blank_counts"]["TotalCharges"], 11)
        self.assertEqual(result["churn_counts"], {"No": 5174, "Yes": 1869})

    def test_changed_original_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Dataset version differs"):
            verify_bytes(b"changed data")

    def test_duplicate_customer_is_rejected(self):
        frame = self.original.copy()
        frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
        with self.assertRaisesRegex(ValueError, "duplicate customer"):
            validate_schema(frame)

    def test_unknown_target_is_rejected(self):
        frame = self.original.copy()
        frame.loc[0, "Churn"] = "Maybe"
        with self.assertRaisesRegex(ValueError, "Unexpected values in Churn"):
            validate_schema(frame)

    def test_unexpected_numeric_text_is_rejected(self):
        frame = self.original.copy()
        frame.loc[0, "TotalCharges"] = "not known"
        with self.assertRaisesRegex(ValueError, "Unexpected non-numeric"):
            validate_schema(frame)

    def test_cleaning_preserves_missing_totals_and_rows(self):
        cleaned = clean_data(self.original)
        self.assertEqual(len(cleaned), len(self.original))
        self.assertEqual(int(cleaned["TotalCharges"].isna().sum()), 11)
        self.assertEqual(set(cleaned["Churn"]), {0, 1})
        self.assertNotIn("customerID", FEATURES)
        self.assertNotIn("Churn", FEATURES)
        self.assertEqual(len(FEATURES), 19)
        self.assertEqual(int(self.original["TotalCharges"].eq(" ").sum()), 11)

    def test_split_is_disjoint_complete_and_repeatable(self):
        cleaned = clean_data(self.original)
        train, test = make_split(cleaned)
        train_again, test_again = make_split(cleaned)
        self.assertEqual((len(train), len(test)), (5634, 1409))
        self.assertFalse(set(train["customerID"]) & set(test["customerID"]))
        self.assertEqual(set(train.index) | set(test.index), set(cleaned.index))
        self.assertEqual(train.index.tolist(), train_again.index.tolist())
        self.assertEqual(test.index.tolist(), test_again.index.tolist())
        for partition in [train, test]:
            self.assertLess(abs(partition["Churn"].mean() - cleaned["Churn"].mean()), 0.001)

    def test_missing_monthly_charge_is_rejected(self):
        frame = self.original.copy()
        frame.loc[0, "MonthlyCharges"] = ""
        with self.assertRaisesRegex(ValueError, "Unexpected non-numeric"):
            clean_data(frame)


if __name__ == "__main__":
    unittest.main()
