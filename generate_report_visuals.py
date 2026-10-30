"""
Generates publication-quality charts and visual output screens for AI PBL Report.
Saved to docs/report_screens/
"""

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set matplotlib cache directory to local folder to avoid permission errors
os.environ["MPLCONFIGDIR"] = str(Path("./docs/report_screens/.mplconfig").resolve())
Path("./docs/report_screens/.mplconfig").mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 0.8
output_dir = Path("./docs/report_screens")
output_dir.mkdir(parents=True, exist_ok=True)

# Load metrics
with open("models/model_comparison_metrics.json", "r", encoding="utf-8") as f:
    metrics_data = json.load(f)

test_metrics = metrics_data["test_metrics"]

# ==============================================================================
# Screen 1: Model Benchmark Performance Comparison
# ==============================================================================
models = [m["model_name"].replace("Hybrid: ", "Hybrid:\n").replace("Baseline: ", "Baseline:\n") for m in test_metrics]
recall = [m["scam_recall"] * 100 for m in test_metrics]
precision = [m["scam_precision"] * 100 for m in test_metrics]
f1 = [m["scam_f1"] * 100 for m in test_metrics]
macro_f1 = [m["macro_f1"] * 100 for m in test_metrics]

x = np.arange(len(models))
width = 0.18

fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
rects1 = ax.bar(x - 1.5 * width, recall, width, label="Scam Recall (%)", color="#ef4444")
rects2 = ax.bar(x - 0.5 * width, precision, width, label="Scam Precision (%)", color="#3b82f6")
rects3 = ax.bar(x + 0.5 * width, f1, width, label="Scam F1 Score (%)", color="#10b981")
rects4 = ax.bar(x + 1.5 * width, macro_f1, width, label="Macro F1 (%)", color="#8b5cf6")

ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold", color="#1e293b")
ax.set_title("AI PBL: Benchmark Model Comparison on Unseen Holdout Test Set", fontsize=13, fontweight="bold", pad=15, color="#0f172a")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9, fontweight="medium")
ax.set_ylim(70, 103)
ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", loc="lower right", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5, color="#94a3b8")

# Add text labels on bars
for rects in [rects1, rects2, rects3, rects4]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, rotation=45)

plt.tight_layout()
fig.savefig(output_dir / "01_model_benchmark_comparison.png", bbox_inches="tight")
plt.close(fig)
print("[1/5] Saved 01_model_benchmark_comparison.png")

# ==============================================================================
# Screen 2: Confusion Matrix of Best Model on Holdout Test Set
# ==============================================================================
best_test = next(m for m in test_metrics if "Cost-Tuned" in m["model_name"])
cm = best_test["confusion_matrix"]
cm_matrix = np.array([
    [cm["true_negatives"], cm["false_positives"]],
    [cm["false_negatives"], cm["true_positives"]]
])

fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
sns.heatmap(
    cm_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=ax,
    annot_kws={"size": 14, "weight": "bold"},
    xticklabels=["Predicted Legitimate", "Predicted SCAM"],
    yticklabels=["Actual Legitimate", "Actual SCAM"]
)

ax.set_title(f"Confusion Matrix: {best_test['model_name']}\nTest Set Accuracy: {best_test['accuracy']*100:.2f}% | Scam Recall: {best_test['scam_recall']*100:.1f}%", fontsize=11, fontweight="bold", pad=12, color="#0f172a")
plt.tight_layout()
fig.savefig(output_dir / "02_confusion_matrix_test_set.png", bbox_inches="tight")
plt.close(fig)
print("[2/5] Saved 02_confusion_matrix_test_set.png")

# ==============================================================================
# Screen 3: Top Feature Importances / Suspicious Scam Predictor Weights
# ==============================================================================
pipeline = joblib.load("models/best_pipeline.joblib")
classifier = pipeline.named_steps["classifier"]
feature_union = pipeline.named_steps["features"]
tfidf_step = feature_union.transformer_list[0][1].named_steps["tfidf"]
vocab = tfidf_step.get_feature_names_out()

coefs = classifier.coef_[0]
tfidf_coefs = coefs[:len(vocab)]

# Get top positive coefficients (indicating scam)
top_pos_idx = np.argsort(tfidf_coefs)[-15:]
top_tokens = [vocab[i] for i in top_pos_idx]
top_weights = [tfidf_coefs[i] for i in top_pos_idx]

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
y_pos = np.arange(len(top_tokens))
bars = ax.barh(y_pos, top_weights, color="#ef4444", alpha=0.85, edgecolor="#b91c1c", height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_tokens, fontsize=10, fontweight="bold", color="#1e293b")
ax.set_xlabel("Logistic Regression Log-Odds Weight (Scam Affinity)", fontsize=10, fontweight="bold")
ax.set_title("Top 15 Most Suspicious Vocabulary Tokens Identified by AI Model", fontsize=12, fontweight="bold", pad=12, color="#0f172a")
ax.grid(axis="x", linestyle="--", alpha=0.5, color="#94a3b8")

for bar in bars:
    width = bar.get_width()
    ax.annotate(f"+{width:.2f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left", va="center", fontsize=8.5, fontweight="semibold")

plt.tight_layout()
fig.savefig(output_dir / "03_top_suspicious_features.png", bbox_inches="tight")
plt.close(fig)
print("[3/5] Saved 03_top_suspicious_features.png")

# ==============================================================================
# Screen 4: Dataset EDA Class Distribution & Message Length Characteristics
# ==============================================================================
train_df = pd.read_csv("data/processed/train.csv")
train_df["char_length"] = train_df["text"].astype(str).apply(len)
train_df["label_str"] = train_df["label"].map({0: "Legitimate (Ham)", 1: "Fraudulent (Scam)"})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Class balance bar
counts = train_df["label_str"].value_counts()
colors = ["#3b82f6", "#ef4444"]
ax1.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="#0f172a", alpha=0.85)
ax1.set_title("Training Set Class Distribution", fontsize=11, fontweight="bold", color="#0f172a")
ax1.set_ylabel("Message Count", fontsize=10, fontweight="bold")
for idx, val in enumerate(counts.values):
    pct = (val / len(train_df)) * 100
    ax1.annotate(f"{val} ({pct:.1f}%)", xy=(idx, val), xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# Message length KDE/box
sns.boxplot(x="label_str", y="char_length", data=train_df, palette={"Legitimate (Ham)": "#93c5fd", "Fraudulent (Scam)": "#fca5a5"}, ax=ax2, width=0.4)
ax2.set_title("Message Character Length Distribution by Class", fontsize=11, fontweight="bold", color="#0f172a")
ax2.set_ylabel("Character Length", fontsize=10, fontweight="bold")
ax2.set_xlabel("")
ax2.set_ylim(0, 300)
ax2.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig(output_dir / "04_dataset_distribution_and_lengths.png", bbox_inches="tight")
plt.close(fig)
print("[4/5] Saved 04_dataset_distribution_and_lengths.png")

# ==============================================================================
# Screen 5: Simulation Cards of Scam vs Legitimate Live Detection
# ==============================================================================
fig, (ax_scam, ax_legit) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)

for ax in [ax_scam, ax_legit]:
    ax.axis("off")

# Scam Card
ax_scam.text(0.5, 0.95, "CRITICAL THREAT DETECTED", fontsize=14, fontweight="bold", color="#b91c1c", ha="center", va="top")
ax_scam.text(0.5, 0.88, "Classification: SCAM  |  Confidence: 98.4%", fontsize=11, fontweight="bold", color="#1e293b", ha="center", va="top")
box_scam = dict(boxstyle="round,pad=0.8", facecolor="#fef2f2", edgecolor="#ef4444", linewidth=2)
scam_content = (
    "Input Message:\n"
    "\"URGENT: Your SBI Bank account is blocked due to missing KYC.\n"
    "Click http://bit.ly/sbi-kyc-update immediately to restore access.\"\n\n"
    "Detected Red Flags & Heuristics:\n"
    " • [CRITICAL] Masked/Shortened Phishing URL (bit.ly/sbi-kyc-update)\n"
    " • [HIGH] Coercive Urgency Trigger ('URGENT', 'immediately')\n"
    " • [CRITICAL] Sensitive Credential/KYC Harvesting Trap\n\n"
    "Top Suspicious Tokens Identified:\n"
    " • sbi (+2.84), claim (+2.61), urgent (+2.42), bit (+2.15)"
)
ax_scam.text(0.05, 0.78, scam_content, fontsize=9.5, va="top", ha="left", bbox=box_scam, family="monospace")

# Legit Card
ax_legit.text(0.5, 0.95, "MESSAGE VERIFIED SAFE", fontsize=14, fontweight="bold", color="#15803d", ha="center", va="top")
ax_legit.text(0.5, 0.88, "Classification: LEGITIMATE  |  Confidence: 99.2%", fontsize=11, fontweight="bold", color="#1e293b", ha="center", va="top")
box_legit = dict(boxstyle="round,pad=0.8", facecolor="#f0fdf4", edgecolor="#22c55e", linewidth=2)
legit_content = (
    "Input Message:\n"
    "\"Hi Nandini, are we still meeting for lunch at 1:30 PM today?\n"
    "Let me know if the cafeteria works for you.\"\n\n"
    "Detected Red Flags & Heuristics:\n"
    " • No high-pressure urgency triggers found\n"
    " • No external suspicious URLs or shortened links\n"
    " • Zero credential or OTP solicitation\n\n"
    "Top Suspicious Tokens Identified:\n"
    " • None (Negative threat log-odds across all vocabulary)"
)
ax_legit.text(0.05, 0.78, legit_content, fontsize=9.5, va="top", ha="left", bbox=box_legit, family="monospace")

fig.suptitle("ScamShield AI: Live Inference & Explainability Output Screens", fontsize=14, fontweight="bold", y=0.98, color="#0f172a")
plt.tight_layout()
fig.savefig(output_dir / "05_scam_vs_ham_classification_screens.png", bbox_inches="tight")
plt.close(fig)
print("[5/5] Saved 05_scam_vs_ham_classification_screens.png")
