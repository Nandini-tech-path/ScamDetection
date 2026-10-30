"""
Central configuration module for the Scam Message Detection System.
Defines file paths, random seeds, regex patterns, and keyword vocabularies.
"""

from pathlib import Path
from typing import Dict, List, Set

# --- Project Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Ensure essential directories exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# --- Data File Names ---
UCI_DATASET_FILENAME: str = "SMSSpamCollection"
CUSTOM_DATASET_FILENAME: str = "custom_scams.csv"
TRAIN_FILE: Path = DATA_PROCESSED_DIR / "train.csv"
VAL_FILE: Path = DATA_PROCESSED_DIR / "val.csv"
TEST_FILE: Path = DATA_PROCESSED_DIR / "test.csv"

# --- Model Artifact Paths ---
BEST_PIPELINE_PATH: Path = MODELS_DIR / "best_pipeline.joblib"
METRICS_REPORT_PATH: Path = MODELS_DIR / "model_comparison_metrics.json"

# --- Reproducibility & Splitting Constants ---
RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# Label Mapping: 0 -> Legitimate, 1 -> Scam
LABEL_MAP: Dict[str, int] = {
    "ham": 0,
    "legitimate": 0,
    "legit": 0,
    "spam": 1,
    "scam": 1,
    "fraud": 1,
}
INVERSE_LABEL_MAP: Dict[int, str] = {
    0: "LEGITIMATE",
    1: "SCAM",
}

# --- Trusted Official Domains (never treated as phishing shorteners) ---
TRUSTED_OFFICIAL_DOMAINS: Set[str] = {
    "app.goo.gl",
    "gpay.app.goo.gl",
    "maps.app.goo.gl",
    "google.com",
    "jio.com",
    "airtel.in",
    "myvi.in",
    "amazon.in",
    "amazon.com",
    "flipkart.com",
    "swiggy.com",
    "zomato.com",
    "uber.com",
    "irctc.co.in",
    "goindigo.in",
    "paytm.com",
    "phonepe.com",
    "youtube.com",
    "youtu.be",
    "netflix.com",
}

# --- Known Masked Shortener & Suspicious Domains ---
SHORTENER_DOMAINS: Set[str] = {
    "bit.ly",
    "tinyurl.com",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "rebrand.ly",
    "cutt.ly",
    "rb.gy",
    "wa.me",
    "t.me",
}

# --- Suspicious Keyword Lists for Handcrafted Signals ---
# Coercive, panic-inducing urgency & threat terms
URGENCY_KEYWORDS: List[str] = [
    "urgent",
    "immediately",
    "action required",
    "important notice",
    "urgent notice",
    "mandatory update",
    "compliance alert",
    "immediate verification",
    "require immediate",
    "failure to verify",
    "account restriction",
    "account restrictions",
    "suspended",
    "blocked",
    "terminated",
    "deactivated",
    "within 24 hours",
    "expires today",
    "final notice",
    "disconnected tonight",
    "power supply will be disconnected",
    "digital arrest",
    "police action",
    "court warrant",
    "restricted access",
    "may result in",
    "service interruption",
]

# Sensitive credential and security harvesting demands & advance fee / identity fraud
FINANCIAL_SECURITY_KEYWORDS: List[str] = [
    "send your otp",
    "share your otp",
    "send otp to claim",
    "share otp to avoid",
    "enter otp to claim",
    "enter pin",
    "upi pin",
    "cvv",
    "pan details",
    "pan card",
    "aadhaar details",
    "submit your information",
    "submit details",
    "submit your pan",
    "verify your information",
    "update kyc",
    "pending kyc",
    "verify pan",
    "link aadhaar",
    "netbanking password",
    "enter password",
    "login details",
    "biometric verification",
    "no documentation required",
    "registration fee",
    "joining fee",
    "processing fee",
    "training fee",
    "refundable fee",
    "dispatch fee",
    "customs duty fee",
    "pay fee to complete",
]

# Unrealistic reward, lottery, and employment/job scam lures
REWARD_LOTTERY_KEYWORDS: List[str] = [
    "lottery",
    "lucky draw",
    "winner",
    "won 25,00,000",
    "won rs",
    "claim prize",
    "cash prize",
    "free iphone",
    "spin the lucky wheel",
    "work from home",
    "work-from-home",
    "selected for a job",
    "selected for work",
    "part-time job",
    "earn daily",
    "earn rs",
    "liking youtube videos",
    "guaranteed return",
    "crypto vip",
    "0% interest",
    "pre-approved loan",
]


