# model/evaluate.py

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    average_precision_score,
)
import matplotlib.pyplot as plt

# Load everything we saved 
model = joblib.load("model/baseline_logreg.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Recreate the same preprocessing used in training 
X_test = pd.get_dummies(X_test, columns=["type"], drop_first=True)
# Ensure column order/presence matches training exactly
X_test = X_test.reindex(columns=feature_columns, fill_value=0)

# Handle inf/NaN — same guard as in train.py 
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(-1)

X_test_scaled = scaler.transform(X_test)

# Get predicted probabilities (not just 0/1) 
# We want probabilities so we can choose our own threshold later,
# instead of blindly accepting sklearn's default 0.5 cutoff.
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# Precision-Recall curve 
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

avg_precision = average_precision_score(y_test, y_probs)
print(f"Average Precision (PR-AUC): {avg_precision:.4f}")

# Plot PR curve 
plt.figure(figsize=(7, 5))
plt.plot(recalls, precisions, label=f"PR curve (AP={avg_precision:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Baseline Logistic Regression")
plt.legend()
plt.grid(True)
plt.savefig("docs/pr_curve_baseline.png")
print("\nSaved PR curve to docs/pr_curve_baseline.png")

# Try a few thresholds and print precision/recall/confusion matrix at each 
print("\n--- Threshold comparison ---")
for t in [0.3, 0.5, 0.7, 0.9]:
    y_pred = (y_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\nThreshold = {t}")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

# Default 0.5 threshold classification report (for reference) 
print("\n--- Full report at threshold=0.5 ---")
y_pred_default = (y_probs >= 0.5).astype(int)
print(classification_report(y_test, y_pred_default, digits=4))
