"""
Model Training and Comparison Module.
Trains multiple classification models (Logistic Regression, Naive Bayes, Random Forest, Hybrid Pipelines),
evaluates on validation/test splits, and exports the top-performing pipeline with joblib.
"""

import json
from typing import Dict, Any, List, Tuple
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    BEST_PIPELINE_PATH,
    METRICS_REPORT_PATH,
    RANDOM_SEED,
)
from src.preprocess import clean_text
from src.features import (
    TextSelector,
    HandcraftedFeatureExtractor,
    build_hybrid_feature_pipeline,
    build_tfidf_pipeline,
)
from src.evaluate import (
    compute_comprehensive_metrics,
    format_metrics_table,
    analyze_errors,
)


def get_model_candidates() -> Dict[str, Pipeline]:
    """
    Define candidate model pipelines for benchmark comparison.

    Returns:
        Dict[str, Pipeline]: Dictionary mapping model names to their Pipeline objects.
    """
    candidates = {}

    # 1. Baseline 1: Pure TF-IDF + Calibrated Logistic Regression
    candidates["Baseline: TF-IDF + Logistic Regression"] = Pipeline([
        ("cleaner", TextSelector()),
        ("tfidf", build_tfidf_pipeline(max_features=5000)),
        ("classifier", LogisticRegression(C=2.0, max_iter=1000, random_state=RANDOM_SEED)),
    ])

    # 2. Baseline 2: Pure TF-IDF + Multinomial Naive Bayes
    candidates["Baseline: TF-IDF + Naive Bayes"] = Pipeline([
        ("cleaner", TextSelector()),
        ("tfidf", build_tfidf_pipeline(max_features=5000)),
        ("classifier", MultinomialNB(alpha=0.05)),
    ])

    # 3. Hybrid: TF-IDF + Handcrafted Features + Regularized Logistic Regression
    candidates["Hybrid: TF-IDF + Handcrafted + Logistic Regression"] = Pipeline([
        ("features", build_hybrid_feature_pipeline(max_tfidf_features=5000)),
        ("classifier", LogisticRegression(C=2.0, max_iter=1000, random_state=RANDOM_SEED)),
    ])

    # 4. Hybrid: TF-IDF + Handcrafted Features + Cost-Tuned Logistic Regression
    candidates["Hybrid: TF-IDF + Handcrafted + Cost-Tuned LR"] = Pipeline([
        ("features", build_hybrid_feature_pipeline(max_tfidf_features=5000)),
        ("classifier", LogisticRegression(C=1.5, class_weight={0: 1.0, 1: 2.0}, max_iter=1000, random_state=RANDOM_SEED)),
    ])

    # 5. Hybrid: TF-IDF + Handcrafted Features + Random Forest
    candidates["Hybrid: TF-IDF + Handcrafted + Random Forest"] = Pipeline([
        ("features", build_hybrid_feature_pipeline(max_tfidf_features=2500)),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    return candidates


def train_and_compare_models(
    train_path=TRAIN_FILE,
    val_path=VAL_FILE,
    test_path=TEST_FILE,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, Any]]:
    """
    Train all candidate models on the train split, validate on validation split,
    and evaluate final performance on the unseen test split.

    Returns:
        Tuple[Pipeline, pd.DataFrame, Dict[str, Any]]: Best pipeline, summary table, detailed metrics.
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df["text"].values, train_df["label"].values
    X_val, y_val = val_df["text"].values, val_df["label"].values
    X_test, y_test = test_df["text"].values, test_df["label"].values

    candidates = get_model_candidates()
    val_results: List[Dict[str, Any]] = []
    test_results: List[Dict[str, Any]] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    print(f"\n{'='*70}\n[TRAINING] Starting Benchmark Comparison across {len(candidates)} models...\n{'='*70}")

    for name, pipeline in candidates.items():
        print(f"-> Fitting: {name} ...")
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on Validation set
        val_preds = pipeline.predict(X_val)
        val_proba = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else val_preds
        val_metrics = compute_comprehensive_metrics(y_val, val_preds, val_proba, model_name=name)
        val_results.append(val_metrics)

        # Evaluate on Test set
        test_preds = pipeline.predict(X_test)
        test_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else test_preds
        test_metrics = compute_comprehensive_metrics(y_test, test_preds, test_proba, model_name=name)
        test_results.append(test_metrics)

    # Convert results to DataFrame for comparison
    val_table = format_metrics_table(val_results)
    test_table = format_metrics_table(test_results)

    print("\n--- VALIDATION SET PERFORMANCE ---")
    print(val_table.to_string(index=False))

    print("\n--- UNSEEN TEST SET PERFORMANCE ---")
    print(test_table.to_string(index=False))

    # Selection Strategy: Prioritize high F1 and Precision with minimal False Positives
    def model_rank_score(m: Dict[str, Any]) -> float:
        fpr_penalty = max(0.0, m["false_positive_rate"] - 0.003) * 6.0
        hybrid_bonus = 0.03 if "Hybrid" in m["model_name"] else 0.0
        return (m["scam_f1"] * 0.45) + (m["macro_f1"] * 0.35) + (m["roc_auc"] * 0.20) + hybrid_bonus - fpr_penalty

    best_idx = max(
        range(len(val_results)),
        key=lambda i: model_rank_score(val_results[i]),
    )
    best_name = val_results[best_idx]["model_name"]
    best_pipeline = fitted_pipelines[best_name]

    print(f"\n[BEST MODEL SELECTED]: '{best_name}' (Scam Recall: {val_results[best_idx]['scam_recall']*100:.2f}%, F1: {val_results[best_idx]['scam_f1']*100:.2f}%, FPR: {val_results[best_idx]['false_positive_rate']*100:.2f}%)")

    # Error analysis on test set for the best model
    best_test_preds = best_pipeline.predict(X_test)
    best_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    errors = analyze_errors(test_df, y_test, best_test_preds, best_test_proba)

    # Save best model pipeline
    joblib.dump(best_pipeline, BEST_PIPELINE_PATH)
    print(f"[SAVED] Exported best model artifact to: {BEST_PIPELINE_PATH}")

    # Save metrics report
    report_data = {
        "best_model_name": best_name,
        "validation_metrics": val_results,
        "test_metrics": test_results,
        "test_error_analysis": errors,
    }
    with open(METRICS_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[SAVED] Exported metrics report to: {METRICS_REPORT_PATH}")

    return best_pipeline, test_table, report_data


if __name__ == "__main__":
    train_and_compare_models()
