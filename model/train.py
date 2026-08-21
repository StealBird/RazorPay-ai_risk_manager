# model/train.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Encode categorical column 
# 'type' only has 2 values now (TRANSFER, CASH_OUT) — one-hot encode it
X_train = pd.get_dummies(X_train, columns=["type"], drop_first=True)
X_test = pd.get_dummies(X_test, columns=["type"], drop_first=True)

print("Features used:", list(X_train.columns)) 

# Scale numeric features
# Logistic regression is sensitive to feature scale — 'amount' and 'step'
# are on very different ranges, so we standardize them.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train baseline model 
# class_weight='balanced' tells the model to pay more attention to the rare
# fraud class, instead of just predicting "not fraud" every time (which would
# already be 99.7% "accurate" and completely useless).
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("\nModel trained.")

# Save model + scaler + feature list (need these for consistent inference later) 
joblib.dump(model, "model/baseline_logreg.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(list(X_train.columns), "model/feature_columns.pkl")

print("Saved model, scaler, and feature columns to model/")
