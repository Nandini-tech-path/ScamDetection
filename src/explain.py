"""
Model Explainability and Suspicious Signal Interpretation Module.
Extracts top contributing vocabulary tokens and detects domain-specific red flags
to provide clear, human-understandable explanations for non-technical users.
"""

from typing import Dict, List, Any, Tuple
import re
import html
import numpy as np
from src.config import (
    URGENCY_KEYWORDS,
    FINANCIAL_SECURITY_KEYWORDS,
    REWARD_LOTTERY_KEYWORDS,
    SHORTENER_DOMAINS,
    TRUSTED_OFFICIAL_DOMAINS,
)
from src.preprocess import URL_REGEX, PHONE_REGEX, AMOUNT_REGEX


def extract_red_flags(text: str) -> List[Dict[str, str]]:
    """
    Identify heuristic domain red flags present in the message.

    Args:
        text (str): Raw input message.

    Returns:
        List[Dict[str, str]]: List of detected red flag dictionaries with title, description, and severity.
    """
    flags = []
    lower_text = text.lower()

    # 1. Check for Urgency / Threat Triggers
    found_urgency = [kw for kw in URGENCY_KEYWORDS if kw in lower_text]
    if found_urgency:
        flags.append({
            "category": "Urgency & Coercion",
            "title": "Artificial Urgency Detected",
            "description": f"Message uses high-pressure words ({', '.join(found_urgency[:3])}) to force hasty actions.",
            "severity": "HIGH",
        })

    # 2. Check for Financial / Credential Demands, Identity Harvesting, or Upfront Fee Traps
    found_fin = [kw for kw in FINANCIAL_SECURITY_KEYWORDS if kw in lower_text]
    if found_fin:
        is_fee_demand = any("fee" in kw or "deposit" in kw for kw in found_fin)
        is_identity_demand = any("pan" in kw or "aadhaar" in kw or "information" in kw for kw in found_fin)
        flags.append({
            "category": (
                "Advance Fee / Upfront Trap"
                if is_fee_demand
                else ("Identity & PAN Harvesting" if is_identity_demand else "Credential Harvesting")
            ),
            "title": (
                "Upfront Fee / Registration Demand"
                if is_fee_demand
                else ("Urgent Identity / PAN Verification Request" if is_identity_demand else "Requests Sensitive Information")
            ),
            "description": (
                f"Demands payment or upfront fees ({', '.join(found_fin[:3])}). Legitimate employers and lottery organizations never ask candidates to pay fees to join."
                if is_fee_demand
                else f"Demands sensitive personal/identity assets ({', '.join(found_fin[:3])}). Financial institutions and tax authorities do not solicit unverified PAN/identity updates via SMS."
            ),
            "severity": "CRITICAL",
        })

    # 3. Check for Prize / Unrealistic Rewards & Work from home job offers
    found_rewards = [kw for kw in REWARD_LOTTERY_KEYWORDS if kw in lower_text]
    if found_rewards:
        is_job_lure = any("job" in kw or "work" in kw or "selected" in kw for kw in found_rewards)
        flags.append({
            "category": "Unsolicited Job / Bait",
            "title": "Unverified Job / Work-from-Home Offer" if is_job_lure else "Lure / Reward Promise",
            "description": f"Promises high-paying jobs, work-from-home salary, or instant money ({', '.join(found_rewards[:3])}) without standard interview processes.",
            "severity": "HIGH" if is_job_lure else "MEDIUM",
        })

    # 4. Check for Suspicious or Shortened Links (excluding verified official services)
    urls = URL_REGEX.findall(text)
    if urls:
        is_trusted = any(dom in lower_text for dom in TRUSTED_OFFICIAL_DOMAINS)
        is_shortened = (not is_trusted) and any(short in lower_text for short in SHORTENER_DOMAINS)
        has_threat_context = bool(found_urgency or found_fin or found_rewards)
        if is_shortened or (has_threat_context and not is_trusted):
            flags.append({
                "category": "Suspicious Link",
                "title": "Masked / Shortened Link" if is_shortened else "External Link with Urgency/Lure",
                "description": "Contains an external web link combined with pressure tactics or domain masking. Fraudsters often use these to hide phishing sites.",
                "severity": "CRITICAL" if is_shortened else "HIGH",
            })

    # 5. Check for Phone Callback / Support numbers with threat
    phones = PHONE_REGEX.findall(text)
    if phones and ("call" in lower_text or "care" in lower_text or "officer" in lower_text) and (found_urgency or found_fin):
        flags.append({
            "category": "Fake Support Callback",
            "title": "Urgent Callback Solicitation",
            "description": "Urges you to dial an unverified phone number under panic or threat instead of official verified customer service channels.",
            "severity": "HIGH",
        })

    return flags


def get_token_explanations(
    text: str,
    pipeline: Any,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Extract highest contributing vocabulary tokens for the prediction using model coefficients.

    Args:
        text (str): Input text message.
        pipeline (Any): Trained sklearn Pipeline.
        top_n (int): Number of top influential tokens to return.

    Returns:
        List[Tuple[str, float]]: List of (token, score) pairs.
    """
    # Check if classifier is linear and has coef_
    classifier = None
    vectorizer = None

    if hasattr(pipeline, "named_steps"):
        for step_name in ["classifier", "clf", "model"]:
            if step_name in pipeline.named_steps:
                classifier = pipeline.named_steps[step_name]
                break

        # Check for vectorizer in pipeline or features union
        if "tfidf" in pipeline.named_steps:
            vectorizer = pipeline.named_steps["tfidf"]
        elif "features" in pipeline.named_steps:
            union = pipeline.named_steps["features"]
            if hasattr(union, "transformer_list"):
                for name, trans in union.transformer_list:
                    if hasattr(trans, "named_steps") and "tfidf" in trans.named_steps:
                        vectorizer = trans.named_steps["tfidf"]
                        break

    # Determine feature weights from Linear model (coef_) or Naive Bayes (feature_log_prob_)
    has_weights = False
    tfidf_weights = None

    if classifier is not None and vectorizer is not None:
        if hasattr(classifier, "coef_"):
            feature_names = vectorizer.get_feature_names_out()
            n_tfidf = len(feature_names)
            tfidf_weights = classifier.coef_[0][:n_tfidf]
            has_weights = True
        elif hasattr(classifier, "feature_log_prob_"):
            # Naive Bayes log-ratio of P(word|scam) vs P(word|legit)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_weights = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
            has_weights = True

    if not has_weights or tfidf_weights is None:
        # Fallback to keyword matching scoring
        lower_words = re.findall(r"\b\w+\b", text.lower())
        scored_tokens = []
        for w in lower_words:
            score = 0.0
            if any(w in kw for kw in URGENCY_KEYWORDS):
                score += 2.0
            if any(w in kw for kw in FINANCIAL_SECURITY_KEYWORDS):
                score += 3.0
            if any(w in kw for kw in REWARD_LOTTERY_KEYWORDS):
                score += 2.5
            if score > 0:
                scored_tokens.append((w, score))
        scored_tokens = sorted(list(set(scored_tokens)), key=lambda x: x[1], reverse=True)
        return scored_tokens[:top_n]

    # Tokenize input message using vectorizer analyzer
    analyzer = vectorizer.build_analyzer()
    tokens_in_text = set(analyzer(text))

    # Common English stopwords to ignore in suspicious token highlights
    benign_stopwords = {
        "the", "a", "an", "is", "in", "it", "to", "for", "of", "and", "or", "on", "at", 
        "by", "with", "from", "as", "be", "this", "that", "are", "was", "were", "been",
        "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "she", "her",
        "they", "them", "their", "so", "if", "but", "do", "did", "have", "has", "had"
    }

    token_scores = []
    for token in tokens_in_text:
        if token.lower() in benign_stopwords:
            continue
        if token in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[token]
            weight = tfidf_weights[idx]
            # Require positive log-odds weight
            if weight > 0.15:
                token_scores.append((token, float(weight)))

    # Fallback to domain keywords if no TF-IDF token met the threshold
    if not token_scores:
        lower_text = text.lower()
        for kw_list in [FINANCIAL_SECURITY_KEYWORDS, REWARD_LOTTERY_KEYWORDS, URGENCY_KEYWORDS]:
            for kw in kw_list:
                if kw in lower_text:
                    token_scores.append((kw, 2.5))

    # Sort descending by influence weight
    token_scores.sort(key=lambda x: x[1], reverse=True)
    return token_scores[:top_n]


def generate_highlighted_html(text: str, suspicious_tokens: List[str]) -> str:
    """
    Generate styled HTML text with suspicious words highlighted in luminous amber/red badge spans.

    Args:
        text (str): Raw input message.
        suspicious_tokens (List[str]): List of suspicious words to highlight.

    Returns:
        str: HTML formatted string.
    """
    safe_text = html.escape(text)

    if not suspicious_tokens:
        return safe_text

    # Sort by length descending to prevent sub-token collisions during replacement
    tokens = sorted(set(suspicious_tokens), key=len, reverse=True)

    for token in tokens:
        if not token.strip():
            continue
        # Use regex boundary matching
        pattern = re.compile(rf"\b({re.escape(token)})\b", re.IGNORECASE)
        replacement = r'<mark style="background: rgba(239, 68, 68, 0.25); color: #ef4444; font-weight: 600; padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(239, 68, 68, 0.4);">\1</mark>'
        safe_text = pattern.sub(replacement, safe_text)

    return safe_text
