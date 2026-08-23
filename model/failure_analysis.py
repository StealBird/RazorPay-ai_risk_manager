# model/failure_analysis.py

import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb

# Load model and data 
model = xgb.XGBClassifier()
model.load_model("model/xgboost_model.json")
feature_columns = joblib.load("model/xgb_feature_columns.pkl")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

X_test_original = X_test.copy()  # keep original for readable printing
X_test = pd.get_dummies(X_test, columns=["type"], drop_first=True)
X_test = X_test.reindex(columns=feature_columns, fill_value=0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(-1)

# Get predictions 
dmatrix = xgb.DMatrix(X_test)
y_probs = model.get_booster().predict(dmatrix)
y_pred = (y_probs >= 0.5).astype(int)

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred,
    "probability": y_probs,
}, index=X_test.index)

# Find a FALSE POSITIVE: predicted fraud, actually legit 
false_positives = results[(results["actual"] == 0) & (results["predicted"] == 1)]
# Sort by probability descending — pick one the model was MOST confident about (most interesting case)
fp_example_idx = false_positives.sort_values("probability", ascending=False).index[0]

# Find a FALSE NEGATIVE: predicted legit, actually fraud 
false_negatives = results[(results["actual"] == 1) & (results["predicted"] == 0)]
# Sort by probability descending — pick the one CLOSEST to the threshold (near-miss, most interesting)
fn_example_idx = false_negatives.sort_values("probability", ascending=False).index[0]

explainer = shap.TreeExplainer(model)

def explain_case(idx, label):
    print(f"\n{'='*60}")
    print(f"{label} — test row index {idx}")
    print(f"{'='*60}")
    print("\nOriginal feature values:")
    print(X_test_original.loc[idx])
    print(f"\nActual label: {results.loc[idx, 'actual']}")
    print(f"Predicted label: {results.loc[idx, 'predicted']}")
    print(f"Predicted probability: {results.loc[idx, 'probability']:.4f}")

    row = X_test.loc[[idx]]
    shap_vals = explainer.shap_values(row)
    print("\nSHAP contributions:")
    for feat, val in sorted(zip(feature_columns, shap_vals[0]), key=lambda x: -abs(x[1])):
        print(f"  {feat}: {val:+.4f}")

explain_case(fp_example_idx, "FALSE POSITIVE (flagged fraud, actually legit)")
explain_case(fn_example_idx, "FALSE NEGATIVE (missed fraud, model said legit)")