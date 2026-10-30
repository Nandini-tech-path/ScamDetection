# Architectural & Design Decisions (PBL Presentation Reference)

This document provides concise, plain-English rationales for every design choice in the AI-Powered Scam Message Detection System. Use these points during viva/presentation.

---

### 1. Data Ingestion & Splitting Strategy
- **Stratified Split Before Fitting (`train_test_split(..., stratify=y)`):**
  *Rationale:* Scam datasets exhibit severe class imbalance (~87% legitimate, ~13% scam). Stratified splitting ensures that train (70%), validation (15%), and test (15%) partitions all maintain identical class ratios, preventing rare scam patterns from being missing in test evaluations.
- **Strict Leakage Prevention:**
  *Rationale:* Vectorizers (TF-IDF), scalers, and vocabulary lists are fitted *exclusively* on the training split and applied to validation/test sets without refitting. This guarantees zero look-ahead bias and mimics real-world unseen production inputs.

---

### 2. Preprocessing & Entity Generalization
- **Regex Replacement (`<URL>`, `<PHONE>`, `<AMOUNT>`):**
  *Rationale:* Scammers constantly generate new randomized phone numbers, phishing URLs, and monetary figures. Replacing them with generalized tokens allows the ML model to learn that *the presence of a link or amount* is a suspicious structural pattern, rather than memorizing individual digits or URLs.

---

### 3. Handling Class Imbalance
- **Cost-Sensitive Learning (`class_weight='balanced'`):**
  *Rationale:* Rather than synthetically synthesizing text (which often introduces unnatural phrasing or noisy artifacts), we penalize mistakes on minority scam instances proportional to their inverse frequency. This boosts scam recall without distorting text distributions.

---

### 4. Dual Feature Engineering (TF-IDF + Domain Signals)
- **Combining Word N-Grams with Handcrafted Indicators:**
  *Rationale:* While TF-IDF captures lexical cues (e.g., specific terms like *lottery*, *refund*), handcrafted features capture behavioral and structural signals (e.g., urgency words, OTP demands, ALL-CAPS ratios, digit density). Combining both gives robust performance across diverse scam formats.

---

### 5. Primary Evaluation Metric
- **Prioritizing Recall on the Scam Class:**
  *Rationale:* In security and fraud detection, a False Negative (missing an actual scam message and allowing a victim to get defrauded) is vastly more harmful than a False Positive (flagging a legitimate message for review). Therefore, we maximize Scam Recall while keeping Precision acceptable.

---

### 6. Interpretability & Explainability
- **Linear Feature Coefficients & Rule-Based Highlights:**
  *Rationale:* Complex black-box models are difficult for end users and examiners to trust. Highlighting specific contributing tokens and red flags directly maps predictions back to understandable reasons (e.g., *"Detected urgent deadline + request for OTP"*).

---

### 7. Transformer Strategy (Colab Fine-Tuning)
- **Separation of Local ML vs Cloud DistilBERT:**
  *Rationale:* Traditional ML models (Logistic Regression, Naive Bayes, Random Forest) train in seconds and run instantly on lightweight hardware, making them ideal for rapid local deployment. The heavier transformer (DistilBERT) is formatted as an independent Google Colab notebook to leverage free cloud GPUs without bogging down the local machine.
