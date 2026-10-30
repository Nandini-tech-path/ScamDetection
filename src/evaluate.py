"""
Model Evaluation and Error Analysis Module.
Calculates key metrics (Scam Recall, Precision, F1, ROC-AUC, False Positive Rate),
generates formatted summary tables, and conducts qualitative error analysis.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics focused on fraud detection priorities.

    Args:
        y_true (np.ndarray): True binary labels (0 = Legit, 1 = Scam).
        y_pred (np.ndarray): Predicted binary labels.
        y_proba (np.ndarray): Predicted probability of Scam class (1).
        model_name (str): Identifier for reporting.

    Returns:
        Dict[str, Any]: Structured dictionary of evaluation metrics.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Safety against division by zero
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = 0.0

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "scam_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "scam_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "scam_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "legit_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "legit_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(auc),
        "false_positive_rate": float(fpr),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    return metrics


def format_metrics_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Format a list of metric dictionaries into a clean comparison DataFrame.
    """
    rows = []
    for m in results:
        rows.append({
            "Model": m["model_name"],
            "Scam Recall": f"{m['scam_recall'] * 100:.2f}%",
            "Scam Precision": f"{m['scam_precision'] * 100:.2f}%",
            "Scam F1": f"{m['scam_f1'] * 100:.2f}%",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "False Positives": m["confusion_matrix"]["false_positives"],
            "False Negatives (Missed)": m["confusion_matrix"]["false_negatives"],
            "Macro F1": f"{m['macro_f1'] * 100:.2f}%",
        })
    return pd.DataFrame(rows)


def analyze_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    text_column: str = "text",
    max_samples: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract False Positives and False Negatives for qualitative error analysis.

    Args:
        df (pd.DataFrame): Evaluation DataFrame.
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_proba (np.ndarray): Predicted scam probabilities.
        text_column (str): Name of original message column.
        max_samples (int): Max examples to return per error category.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Categorized error cases with message text.
    """
    df_eval = df.copy()
    df_eval["true_label"] = y_true
    df_eval["pred_label"] = y_pred
    df_eval["scam_prob"] = y_proba

    # False Positives: Actual Legit (0) predicted as Scam (1)
    fps = df_eval[(df_eval["true_label"] == 0) & (df_eval["pred_label"] == 1)].head(max_samples)
    
    # False Negatives: Actual Scam (1) predicted as Legit (0)
    fns = df_eval[(df_eval["true_label"] == 1) & (df_eval["pred_label"] == 0)].head(max_samples)

    return {
        "false_positives": fps[[text_column, "scam_prob"]].to_dict(orient="records"),
        "false_negatives": fns[[text_column, "scam_prob"]].to_dict(orient="records"),
    }
