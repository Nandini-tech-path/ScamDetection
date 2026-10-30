"""
Production Prediction Pipeline Module.
Provides inference API to classify raw messages as SCAM or LEGITIMATE,
estimate confidence probabilities, and package explainability signals.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import joblib

from src.config import BEST_PIPELINE_PATH, INVERSE_LABEL_MAP
from src.explain import (
    extract_red_flags,
    get_token_explanations,
    generate_highlighted_html,
)

# Global cached model instance to avoid disk re-reads per inference
_CACHED_PIPELINE = None


def load_inference_pipeline(model_path: Path = BEST_PIPELINE_PATH):
    """
    Load and cache the trained model pipeline from disk.

    Args:
        model_path (Path): Path to serialized joblib pipeline.

    Returns:
        Trained sklearn Pipeline.
    """
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact '{model_path}' not found. Please train the model first by running `python -m src.train`."
            )
        _CACHED_PIPELINE = joblib.load(model_path)
    return _CACHED_PIPELINE


def predict(text: str, model_path: Path = BEST_PIPELINE_PATH) -> Dict[str, Any]:
    """
    Classify a text message and generate rich explainability metadata.

    Args:
        text (str): Raw input text message (SMS, WhatsApp, Email).
        model_path (Path, optional): Path to model file.

    Returns:
        Dict[str, Any]: Prediction payload containing:
            - text: Original message
            - label: "SCAM" or "LEGITIMATE"
            - is_scam: bool
            - confidence_score: float (0.0 to 100.0)
            - scam_probability: float (0.0 to 1.0)
            - legit_probability: float (0.0 to 1.0)
            - red_flags: List of detected security heuristics
            - top_suspicious_tokens: List of (token, score) tuples
            - highlighted_html: Formatted HTML with annotated tokens
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "text": "",
            "label": "LEGITIMATE",
            "is_scam": False,
            "confidence_score": 100.0,
            "scam_probability": 0.0,
            "legit_probability": 1.0,
            "red_flags": [],
            "top_suspicious_tokens": [],
            "highlighted_html": "",
        }

    pipeline = load_inference_pipeline(model_path)

    # Scikit-learn pipeline expects an iterable of strings
    input_data = [text]

    # Predict class & probabilities
    pred_int = int(pipeline.predict(input_data)[0])

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_data)[0]
        legit_prob = float(probabilities[0])
        scam_prob = float(probabilities[1])
    else:
        legit_prob = 0.0 if pred_int == 1 else 1.0
        scam_prob = 1.0 if pred_int == 1 else 0.0

    # Extract explainability signals & red flags
    red_flags = extract_red_flags(text)
    token_explanations = get_token_explanations(text, pipeline, top_n=5)

    has_critical_flag = any(f.get("severity") == "CRITICAL" for f in red_flags)
    has_high_flag = any(f.get("severity") == "HIGH" for f in red_flags)

    # Dual-layer safety: ML probability with heuristic threat corroboration
    if scam_prob >= 0.50:
        is_scam = True
    elif (has_critical_flag or len(red_flags) >= 2) and scam_prob >= 0.30:
        is_scam = True
        scam_prob = max(scam_prob, 0.75)
        legit_prob = 1.0 - scam_prob
    else:
        is_scam = False

    label_str = "SCAM" if is_scam else "LEGITIMATE"
    confidence = (scam_prob if is_scam else legit_prob) * 100.0

    # Only highlight tokens if scam or if red flags detected
    if is_scam or red_flags:
        suspicious_words = [t[0] for t in token_explanations]
    else:
        suspicious_words = []

    highlighted_html = generate_highlighted_html(text, suspicious_words)

    return {
        "text": text,
        "label": label_str,
        "is_scam": is_scam,
        "confidence_score": round(confidence, 2),
        "scam_probability": round(scam_prob, 4),
        "legit_probability": round(legit_prob, 4),
        "red_flags": red_flags,
        "top_suspicious_tokens": token_explanations,
        "highlighted_html": highlighted_html,
    }


if __name__ == "__main__":
    sample = "URGENT: Your SBI account is blocked. Update KYC at http://bit.ly/sbi-kyc now."
    res = predict(sample)
    print(res)
