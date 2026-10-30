"""
Text Preprocessing and Entity Normalization Module.
Replaces dynamic entities (URLs, phone numbers, amounts) with standardized tokens
to improve machine learning model generalization across unseen messages.
"""

import re
import html
from typing import Dict, Any


# --- Regular Expressions for Entity Replacement ---
# Matches web links (http, https, www) and typical shortener domains
URL_REGEX: re.Pattern = re.compile(
    r"(?:https?:\/\/|www\.)[^\s]+|(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|cc|gov|in|ly|gy|gl|to|me|io|xyz|top|site|online|app)(?:\/[^\s]*)?",
    re.IGNORECASE,
)

# Matches Indian and international phone numbers (e.g., +91 9876543210, 98765-43210, (555) 123-4567, 10-digit numbers)
PHONE_REGEX: re.Pattern = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}|\b\d{10}\b",
    re.IGNORECASE,
)

# Matches monetary amounts in INR (Rs, INR, ₹), USD ($), EUR (€), GBP (£) with commas and decimals, or shorthand like 50k
AMOUNT_REGEX: re.Pattern = re.compile(
    r"(?:(?:Rs\.?|INR|₹|\$|€|£)\s*\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:Rs|INR|lakh|crore|usd|eur|gbp|k\b))",
    re.IGNORECASE,
)

# Matches excessive whitespaces
WHITESPACE_REGEX: re.Pattern = re.compile(r"\s+")


def clean_text(
    text: str,
    replace_urls: bool = True,
    replace_phones: bool = True,
    replace_amounts: bool = False,
    to_lower: bool = True,
) -> str:
    """
    Standardize raw message text by unescaping HTML, normalizing whitespace,
    and optionally replacing URLs and phone numbers with meta-tokens.

    Args:
        text (str): Raw input message.
        replace_urls (bool): Whether to replace links with '<URL>'.
        replace_phones (bool): Whether to replace phone numbers with '<PHONE>'.
        replace_amounts (bool): Whether to replace money figures with '<AMOUNT>'.
        to_lower (bool): Whether to convert text to lowercase.

    Returns:
        str: Normalized, preprocessed text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unescape HTML entities (e.g., &amp; -> &, &lt; -> <)
    cleaned = html.unescape(text)

    # 2. Replace URLs first (to prevent phone/amount regexes from corrupting URLs with numbers)
    if replace_urls:
        cleaned = URL_REGEX.sub(" <URL> ", cleaned)

    # 3. Replace Phone Numbers
    if replace_phones:
        cleaned = PHONE_REGEX.sub(" <PHONE> ", cleaned)

    # 4. Replace Currency and Amounts (optional)
    if replace_amounts:
        cleaned = AMOUNT_REGEX.sub(" <AMOUNT> ", cleaned)

    # 5. Lowercase if requested (keep special tokens uppercase like <URL>)
    if to_lower:
        cleaned = cleaned.lower()
        cleaned = cleaned.replace("<url>", "<URL>").replace("<phone>", "<PHONE>").replace("<amount>", "<AMOUNT>")

    # 6. Normalize excessive spaces
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()

    return cleaned


def extract_text_statistics(text: str) -> Dict[str, Any]:
    """
    Extract raw structural statistics from the original message prior to normalization.
    These statistics serve as features for machine learning models.

    Args:
        text (str): Raw input message.

    Returns:
        Dict[str, Any]: Dictionary containing counts and character ratios.
    """
    if not isinstance(text, str) or not text:
        return {
            "char_length": 0,
            "word_count": 0,
            "digit_count": 0,
            "digit_ratio": 0.0,
            "caps_count": 0,
            "caps_ratio": 0.0,
            "exclamation_count": 0,
            "question_count": 0,
            "has_url": 0,
            "has_phone": 0,
            "has_amount": 0,
        }

    total_chars = len(text)
    words = text.split()
    word_count = len(words)

    digit_count = sum(1 for c in text if c.isdigit())
    caps_count = sum(1 for c in text if c.isupper())
    exclamation_count = text.count("!")
    question_count = text.count("?")

    has_url = 1 if URL_REGEX.search(text) else 0
    has_phone = 1 if PHONE_REGEX.search(text) else 0
    has_amount = 1 if AMOUNT_REGEX.search(text) else 0

    return {
        "char_length": total_chars,
        "word_count": word_count,
        "digit_count": digit_count,
        "digit_ratio": (digit_count / total_chars) if total_chars > 0 else 0.0,
        "caps_count": caps_count,
        "caps_ratio": (caps_count / total_chars) if total_chars > 0 else 0.0,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "has_url": has_url,
        "has_phone": has_phone,
        "has_amount": has_amount,
    }
