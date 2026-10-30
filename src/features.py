"""
Feature Engineering Module.
Implements custom domain-specific feature extractors and combines them
with TF-IDF n-gram vectorization inside a scikit-learn compatible pipeline.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    URGENCY_KEYWORDS,
    FINANCIAL_SECURITY_KEYWORDS,
    REWARD_LOTTERY_KEYWORDS,
    SHORTENER_DOMAINS,
    TRUSTED_OFFICIAL_DOMAINS,
)
from src.preprocess import (
    clean_text,
    extract_text_statistics,
    URL_REGEX,
)


class HandcraftedFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract domain-informed behavioral and stylistic security features from raw text messages.
    Focuses specifically on fraud indicators (urgency threats, credential harvesting,
    unrealistic lures, link masking, and deceptive combinations) while avoiding
    generic text statistics that penalize legitimate transactional messages.
    """

    def __init__(self):
        self.feature_names_: List[str] = [
            "urgency_score",
            "financial_security_score",
            "reward_lottery_score",
            "is_shortened_url",
            "has_suspicious_combo",
            "caps_ratio",
            "exclamation_count",
        ]

    def fit(self, X, y=None):
        """Fit method (stateless extractor, returns self)."""
        return self

    def _extract_single_message_features(self, text: str) -> List[float]:
        """Compute targeted numeric handcrafted security signals for a single message string."""
        if not isinstance(text, str):
            text = ""

        lower_text = text.lower()

        # 1. Keyword Group Scores
        urgency_score = sum(1 for kw in URGENCY_KEYWORDS if kw in lower_text)
        financial_security_score = sum(1 for kw in FINANCIAL_SECURITY_KEYWORDS if kw in lower_text)
        reward_lottery_score = sum(1 for kw in REWARD_LOTTERY_KEYWORDS if kw in lower_text)

        # 2. Structural & Statistical Signals
        stats = extract_text_statistics(text)

        # 3. Check for URL shorteners specifically (exclude verified official domains like gpay.app.goo.gl)
        is_shortened = 0
        if stats["has_url"]:
            is_trusted = any(dom in lower_text for dom in TRUSTED_OFFICIAL_DOMAINS)
            if not is_trusted:
                for shortener in SHORTENER_DOMAINS:
                    if shortener in lower_text:
                        is_shortened = 1
                        break

        # 4. Dangerous Combinations (e.g. URL + urgency, urgency + credential/PAN demand, or job lure + upfront fee demand)
        has_suspicious_combo = 0
        if stats["has_url"] and (urgency_score > 0 or financial_security_score > 0 or reward_lottery_score > 0):
            has_suspicious_combo = 1
        elif is_shortened:
            has_suspicious_combo = 1
        elif stats["has_phone"] and urgency_score > 0:
            has_suspicious_combo = 1
        elif reward_lottery_score > 0 and financial_security_score > 0:
            has_suspicious_combo = 1
        elif urgency_score > 0 and financial_security_score > 0:
            has_suspicious_combo = 1

        return [
            float(urgency_score),
            float(financial_security_score),
            float(reward_lottery_score),
            float(is_shortened),
            float(has_suspicious_combo),
            float(stats["caps_ratio"]),
            float(stats["exclamation_count"]),
        ]

    def transform(self, X) -> np.ndarray:
        """
        Transform an iterable of raw text messages into a 2D numpy array of handcrafted features.

        Args:
            X: Iterable of raw message strings.

        Returns:
            np.ndarray: Matrix of shape (n_samples, n_features).
        """
        features_list = [self._extract_single_message_features(text) for text in X]
        return np.array(features_list, dtype=np.float64)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return the list of engineered feature names."""
        return self.feature_names_


class TextSelector(BaseEstimator, TransformerMixin):
    """Transformer to extract and clean text for TF-IDF."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> List[str]:
        return [clean_text(str(x)) for x in X]


def build_tfidf_pipeline(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
) -> TfidfVectorizer:
    """
    Build standalone TF-IDF Vectorizer with sublinear term frequency scaling and English stopwords.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        stop_words="english",
        strip_accents="unicode",
        lowercase=True,
    )


def build_hybrid_feature_pipeline(
    max_tfidf_features: int = 5000,
    ngram_range: tuple = (1, 2),
) -> FeatureUnion:
    """
    Construct a unified feature pipeline combining:
    1. Sparse TF-IDF word & bi-gram representations.
    2. Dense handcrafted domain security features (scaled).

    Returns:
        FeatureUnion: scikit-learn FeatureUnion pipeline.
    """
    tfidf_branch = Pipeline([
        ("cleaner", TextSelector()),
        ("tfidf", build_tfidf_pipeline(max_features=max_tfidf_features, ngram_range=ngram_range)),
    ])

    handcrafted_branch = Pipeline([
        ("extractor", HandcraftedFeatureExtractor()),
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False preserves sparsity compatibility
    ])

    union = FeatureUnion([
        ("tfidf_features", tfidf_branch),
        ("handcrafted_features", handcrafted_branch),
    ])

    return union
