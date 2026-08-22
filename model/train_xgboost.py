# model/train_xgboost.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib

# Load processed data 
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Encode categorical column 
X_train = pd.get_dummies(X_train, columns=["type"], drop_first=True)
X_test = pd.get_dummies(X_test, columns=["type"], drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Handle inf/NaN 
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(-1)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(-1)

print("Features used:", list(X_train.columns))
print("Number of features:", X_train.shape[1])

# Handle class imbalance 
# scale_pos_weight = ratio of negative to positive class — tells XGBoost
# to weight the rare fraud class more heavily, similar to class_weight='balanced'
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")

# Train XGBoost 
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",  # optimize for PR-AUC directly, matches our evaluation goal
    random_state=42,
    n_jobs=-1,  # use all CPU cores
)

model.fit(X_train, y_train)
print("\nXGBoost model trained.")

# Save
joblib.dump(model, "model/xgboost_model.pkl")
joblib.dump(list(X_train.columns), "model/xgb_feature_columns.pkl")
model.save_model("model/xgboost_model.json")
print("Saved XGBoost model to model/")