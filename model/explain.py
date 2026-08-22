# model/explain.py

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# Load model and data 
model = xgb.XGBClassifier()
model.load_model("model/xgboost_model.json")

feature_columns = joblib.load("model/xgb_feature_columns.pkl")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

X_test = pd.get_dummies(X_test, columns=["type"], drop_first=True)
X_test = X_test.reindex(columns=feature_columns, fill_value=0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(-1)

# SHAP explainer (TreeExplainer is fast, built specifically for tree models) 
explainer = shap.TreeExplainer(model)

# SHAP on the full test set can be slow — use a sample for the summary plot
sample_idx = X_test.sample(n=5000, random_state=42).index
X_sample = X_test.loc[sample_idx]

shap_values = explainer.shap_values(X_sample)

# Global summary plot: which features matter most, and in which direction 
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("docs/shap_summary.png")
print("Saved SHAP summary plot to docs/shap_summary.png")

# Pick one actual fraud case correctly caught, and explain it
fraud_indices = y_test[y_test == 1].index
example_idx = fraud_indices[0]  # first fraud case in test set

example_row = X_test.loc[[example_idx]]
example_shap = explainer.shap_values(example_row)

print(f"\n--- Explaining prediction for test row index {example_idx} (actual fraud) ---")
print("Feature values:")
print(example_row.T)
print("\nSHAP contributions (positive = pushes toward fraud, negative = pushes toward legit):")
for feat, val in zip(feature_columns, example_shap[0]):
    print(f"  {feat}: {val:+.4f}")


dmatrix = xgb.DMatrix(example_row)
prob = model.get_booster().predict(dmatrix)[0]
print(f"\nModel predicted probability: {prob:.4f}")
print(f"Actual label: {y_test.loc[example_idx]}")