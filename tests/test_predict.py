"""
Unit tests for the prediction pipeline.
Compatible with standard unittest and pytest.
"""

import unittest
from src.predict import predict


class TestPredict(unittest.TestCase):

    def test_predict_empty(self):
        """Verify empty or whitespace strings return legitimate safely."""
        res = predict("")
        self.assertFalse(res["is_scam"])
        self.assertEqual(res["label"], "LEGITIMATE")

    def test_predict_scam_sample(self):
        """Verify high-risk scam message classification."""
        sample = "URGENT: Your SBI bank account is blocked. Update KYC at http://bit.ly/sbi-verify immediately."
        res = predict(sample)
        self.assertTrue(res["is_scam"])
        self.assertEqual(res["label"], "SCAM")
        self.assertGreaterEqual(res["scam_probability"], 0.5)
        self.assertIn("red_flags", res)

    def test_predict_legitimate_sample(self):
        """Verify normal interpersonal message classification."""
        sample = "Hey, are we still meeting for lunch at 1 PM today?"
        res = predict(sample)
        self.assertFalse(res["is_scam"])
        self.assertEqual(res["label"], "LEGITIMATE")
        self.assertGreaterEqual(res["legit_probability"], 0.5)

    def test_predict_schema(self):
        """Verify all expected keys exist in the returned dictionary."""
        res = predict("Test message")
        expected_keys = [
            "text",
            "label",
            "is_scam",
            "confidence_score",
            "scam_probability",
            "legit_probability",
            "red_flags",
            "top_suspicious_tokens",
            "highlighted_html",
        ]
        for key in expected_keys:
            self.assertIn(key, res)


if __name__ == "__main__":
    unittest.main()
