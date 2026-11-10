"""Check data failures that could silently damage later evaluation."""

import unittest

from audit_data import audit, read_original, validate_schema
from download_data import verify_bytes


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


if __name__ == "__main__":
    unittest.main()
