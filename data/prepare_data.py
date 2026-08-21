# data/prepare_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

# Load 
RAW_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"
df = pd.read_csv(RAW_PATH)

print("Original shape:", df.shape)

# Filter to TRANSFER and CASH_OUT only (Train only on useful cases)
# Fraud in this dataset ONLY occurs in these two types (confirmed via EDA).
# Keeping other types would pad metrics with trivially-easy negatives.

df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)

print("Filtered shape (TRANSFER + CASH_OUT only):", df.shape)
print("\nFraud rate after filtering (%):", df["isFraud"].mean() * 100)
print("\nFraud count after filtering:")
print(df["isFraud"].value_counts())


# Drop leaky columns 
# These reflect POST-fraud state (balances get cancelled/reset once fraud is detected),
# so using them would be leakage — the model would cheat using information that
# wouldn't legitimately exist at prediction time.
LEAKY_COLS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

# Also drop nameOrig/nameDest (just transaction IDs, no predictive value)
# and isFlaggedFraud (business's own naive rule — we're building something new, not copying it)
DROP_COLS = LEAKY_COLS + ["nameOrig", "nameDest", "isFlaggedFraud"]

df = df.drop(columns=DROP_COLS)
print("\nColumns remaining:", list(df.columns))

# ---- Train/test split (stratified — keeps fraud ratio consistent in both sets) ----
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, "  Fraud rate:", y_train.mean() * 100)
print("Test shape:", X_test.shape, "  Fraud rate:", y_test.mean() * 100)

# Save processed data
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("\nSaved processed train/test splits to data/processed/")

