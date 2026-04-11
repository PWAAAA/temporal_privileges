"""
evaluate.py — Model creation, prediction, metric scoring, and feature importance.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from xgboost import XGBClassifier


def make_model(model_type="rf"):
    """Return a fresh classifier instance."""
    if model_type == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42,
                                      n_jobs=-1, class_weight="balanced",
                                      max_features=0.5)
    elif model_type == "xgb":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=1, random_state=42,
                             n_jobs=-1, eval_metric="logloss")
    elif model_type == "lr":
        return LogisticRegression(max_iter=2000, random_state=42,
                                  class_weight="balanced", solver="saga")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def evaluate(X_train, y_train, X_test, y_test, model_label,
             model_type="rf", show_importances=False, top_n=15):
    """Train, predict, print metrics, and optionally show top features.

    Returns a dict of metric scores.
    """
    model = make_model(model_type)
    if model_type == "xgb":
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        model.set_params(scale_pos_weight=neg / pos)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    f1_weighted  = f1_score(y_test, predictions, average="weighted")
    f1_macro     = f1_score(y_test, predictions, average="macro")
    balanced_acc = balanced_accuracy_score(y_test, predictions)

    print(f"\n-- {model_label} {'-' * (45 - len(model_label))}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}  "
          f"Balanced: {balanced_acc:.4f}  "
          f"Precision: {precision_score(y_test, predictions, average='binary'):.4f}  "
          f"Recall: {recall_score(y_test, predictions, average='binary'):.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}  macro: {f1_macro:.4f}")
    print(classification_report(y_test, predictions))

    if show_importances:
        top = get_top_features(model, X_train.columns, top_n=top_n)
        print(f"Top {top_n} features:\n", top.to_string())

    return {
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "bal_acc": balanced_acc,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="binary"),
        "recall": recall_score(y_test, predictions, average="binary"),
    }


def get_top_features(fitted_model, feature_names, top_n=15):
    """Extract top-N features by importance from a fitted model.

    Works with tree-based models (feature_importances_) and linear models (coef_).
    Returns a DataFrame sorted by importance descending.
    """
    if hasattr(fitted_model, "feature_importances_"):
        importances = fitted_model.feature_importances_
    elif hasattr(fitted_model, "coef_"):
        importances = np.abs(fitted_model.coef_).ravel()
    else:
        raise ValueError("Model has no feature_importances_ or coef_ attribute")

    imp_series = pd.Series(importances, index=feature_names)
    top = imp_series.nlargest(top_n).reset_index()
    top.columns = ["feature", "importance"]
    return top
