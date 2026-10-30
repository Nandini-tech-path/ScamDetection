# 🛡️ AI-Powered Scam Message Detection System (AI PBL)

An intelligent, end-to-end, reproducible machine learning system designed to detect and explain fraudulent and phishing messages across SMS, WhatsApp, and Email. Built as a Project-Based Learning (PBL) system for academic evaluation and real-world deployment demonstrations.

---

## 📌 Problem Statement
With the explosion of digital payments (UPI, NetBanking) and mobile communication, cybercriminals increasingly leverage social engineering tactics—such as fake KYC suspensions, urgent electricity bill disconnection notices, unsolicited job offers, lottery winnings, and fraudulent OTP demands.

This project implements an interpretable AI system that:
1. Classifies messages into **SCAM** or **LEGITIMATE**.
2. Outputs calibrated confidence and threat probabilities.
3. Automatically identifies domain-specific **Red Flags** (Urgency, Credential Theft, Unverified Links).
4. Explains individual predictions by highlighting high-risk keywords using model log-odds weights.

---

## 🛠️ Technology Stack
- **Language**: Python 3.10+
- **Machine Learning & NLP**: `scikit-learn`, `pandas`, `numpy`, `nltk`, `joblib`
- **Visualization**: `matplotlib`, `seaborn`
- **Web Interface**: `Streamlit`
- **Testing & Reproducibility**: `pytest`
- **Deep Learning / Transformer Extension**: `HuggingFace Transformers`, `PyTorch` (Google Colab ready)

---

## 📁 Project Directory Structure

```text
ScamDetection/
├── data/
│   ├── raw/
│   │   ├── SMSSpamCollection           # UCI SMS Spam Collection dataset
│   │   └── custom_scams.csv            # 20 synthetic realistic Indian/global fraud templates
│   └── processed/
│       ├── train.csv                   # Stratified training split (70%)
│       ├── val.csv                     # Stratified validation split (15%)
│       └── test.csv                    # Stratified holdout test split (15%)
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis & visual insights
│   ├── 02_baseline.ipynb               # Baseline modeling & feature experiments
│   └── 03_transformer_colab.ipynb      # DistilBERT transformer fine-tuning for Google Colab
├── src/
│   ├── __init__.py
│   ├── config.py                       # Paths, random seed (42), keyword vocabularies
│   ├── data_loader.py                  # Download UCI data, merge custom CSVs, stratified split
│   ├── preprocess.py                   # Regex token replacement (<URL>, <PHONE>, <AMOUNT>)
│   ├── features.py                     # TF-IDF + Handcrafted security signal pipeline
│   ├── train.py                        # Model trainer and selection engine
│   ├── evaluate.py                     # Metrics (Recall, Precision, F1, ROC-AUC, FPR)
│   ├── predict.py                      # Production inference pipeline
│   └── explain.py                      # Explainability (Model weights & red-flag badges)
├── models/
│   ├── best_pipeline.joblib            # Serialized best model pipeline
│   └── model_comparison_metrics.json   # Full benchmark evaluation metrics
├── docs/
│   └── decisions.md                    # Plain-English viva/presentation decision log
├── tests/
│   ├── test_preprocess.py              # Tests for regex normalizers and cleaning
│   └── test_predict.py                 # Tests for predictions, schema, and edge cases
├── app.py                              # Streamlit web dashboard
├── requirements.txt                    # Pinned project dependencies
└── README.md                           # Documentation & project guide
```

---

## 🚀 Quick Setup & Installation

### 1. Clone the repository and enter directory
```bash
cd ScamDetection
```

### 2. Create and activate virtual environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Project

### Step 1: Ingest Data & Generate Stratified Splits
Automatically checks for the UCI dataset, downloads it if missing, merges custom templates, and creates stratified train/val/test splits:
```bash
python -m src.data_loader
```

### Step 2: Train & Benchmark ML Models
Trains Logistic Regression, Naive Bayes, Random Forest, and Hybrid Pipelines; exports the best model to `models/best_pipeline.joblib`:
```bash
python -m src.train
```

### Step 3: Launch Streamlit Web App
Launch the interactive web application:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Step 4: Run Automated Tests
Execute the unit test suite:
```bash
pytest tests/ -v
```

---

## 📊 Benchmark Results (Holdout Test Split)

| Model Pipeline | Scam Recall | Scam Precision | Scam F1 | ROC-AUC | False Positives | False Negatives (Missed) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid (TF-IDF + Handcrafted + Logistic Reg)** | **95.54%** | **94.69%** | **95.11%** | **0.9934** | 6 | 5 |
| Baseline (TF-IDF + Logistic Reg) | 93.75% | 96.33% | 95.02% | 0.9912 | 4 | 7 |
| Baseline (TF-IDF + Naive Bayes) | 88.39% | 98.02% | 92.96% | 0.9845 | 2 | 13 |
| Hybrid (TF-IDF + Handcrafted + Random Forest) | 86.61% | 98.98% | 92.38% | 0.9880 | 1 | 15 |

> **Key Takeaway**: The **Hybrid Logistic Regression** model achieved the highest **Scam Recall (95.54%)**, successfully catching the most scam messages while maintaining a sub-1% False Positive Rate.

---

## 🔍 Model Explainability
1. **Token Importance**: Uses log-odds coefficients to attribute risk scores to individual vocabulary terms (e.g., `kyc`, `suspended`, `otp`, `immediately`).
2. **Rule-Based Red Flags**: Heuristic detection of high-risk patterns such as shortened links (`bit.ly`, `tinyurl`), credential demands, and financial bait.
3. **HTML Visual Highlighter**: Luminous mark badges highlight suspicious tokens directly within the message context in the Streamlit UI.

---

## ⚠️ Limitations & Ethical Considerations
- **Evolving Scams (Concept Drift)**: Attackers continuously adapt language and obfuscate keywords (e.g., `0TP`, `K-Y-C`, `cl!ck`). Periodic retraining is required.
- **False Positives**: Overly aggressive filters may flag legitimate transactional messages (e.g., genuine bank OTPs containing currency amounts).
- **Privacy**: Message classification in production should be performed client-side or on privacy-compliant servers to prevent logging private conversations.
- **Educational Scope**: Designed as an educational project and decision-support tool, not an absolute guarantee against targeted zero-day social engineering attacks.

---

## 🔮 Future Work
- **Multilingual Support**: Fine-tuning multilingual models (`xlm-roberta-base`, IndicBERT) on Hinglish, Tamil-English, and regional phishing variants.
- **Multimodal Fraud Detection**: Integrating OCR to scan and classify phishing screenshots and QR code payment scams.
- **Active Learning**: Allowing end users to report false classifications to continuously enrich the training set.
