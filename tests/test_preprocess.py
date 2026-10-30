"""
Unit tests for text preprocessing and entity normalization.
Compatible with standard unittest and pytest.
"""

import unittest
from src.preprocess import clean_text, extract_text_statistics


class TestPreprocess(unittest.TestCase):

    def test_clean_text_url_replacement(self):
        """Verify that varied URLs and shorteners are correctly replaced with <URL>."""
        sample = "Click http://bit.ly/claim-prize or visit www.fake-bank-login.com for info"
        cleaned = clean_text(sample, replace_urls=True)
        self.assertIn("<URL>", cleaned)
        self.assertNotIn("http://bit.ly", cleaned)
        self.assertNotIn("www.fake-bank-login.com", cleaned)

    def test_clean_text_phone_replacement(self):
        """Verify that phone numbers are replaced with <PHONE>."""
        sample = "Call our executive at +91 9876543210 or 9123456780 immediately."
        cleaned = clean_text(sample, replace_phones=True)
        self.assertIn("<PHONE>", cleaned)
        self.assertNotIn("9876543210", cleaned)

    def test_clean_text_html_unescaping(self):
        """Verify HTML entities like &amp; &lt; &gt; are unescaped properly."""
        sample = "Win cash &amp; prizes &gt; Rs 1000!"
        cleaned = clean_text(sample)
        self.assertIn("&", cleaned)
        self.assertNotIn("&amp;", cleaned)

    def test_clean_text_empty_input(self):
        """Verify empty or whitespace strings return empty string without errors."""
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   \n\t  "), "")
        self.assertEqual(clean_text(None), "")

    def test_extract_text_statistics(self):
        """Verify structural statistics calculations."""
        text = "URGENT! Call 9876543210 at http://scam.cc now? Win Rs 5000!"
        stats = extract_text_statistics(text)

        self.assertEqual(stats["has_url"], 1)
        self.assertEqual(stats["has_phone"], 1)
        self.assertEqual(stats["has_amount"], 1)
        self.assertGreaterEqual(stats["exclamation_count"], 1)
        self.assertEqual(stats["question_count"], 1)
        self.assertGreater(stats["digit_count"], 0)
        self.assertGreater(stats["caps_count"], 0)
        self.assertTrue(0.0 < stats["caps_ratio"] <= 1.0)
        self.assertTrue(0.0 < stats["digit_ratio"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
