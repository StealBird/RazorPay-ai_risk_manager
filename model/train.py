# model/train.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load processed data 
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Encode categorical column 
X_train = pd.get_dummies(X_train, columns=["type"], drop_first=True)
X_test = pd.get_dummies(X_test, columns=["type"], drop_first=True)

# Ensure test set has same columns as train (safety net)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Features used:", list(X_train.columns))
print("Number of features:", X_train.shape[1])

# Handle any remaining NaNs (e.g., from first transactions with no history)
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(-1)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(-1)

# Scale numeric features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train 
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("\nModel trained.")

# Save 
joblib.dump(model, "model/baseline_logreg.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(list(X_train.columns), "model/feature_columns.pkl")

print("Saved model, scaler, and feature columns to model/")