# data/prepare_data.py
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
# Load 
RAW_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"
df = pd.read_csv(RAW_PATH)
print("Original shape:", df.shape)

# Filter to TRANSFER and CASH_OUT only 
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
print("Filtered shape:", df.shape)

# Sort by customer and step — REQUIRED for correct historical feature computation 
# Every "history" feature below must only look BACKWARD in time relative to
# the current row, or we leak future information into past predictions.
df = df.sort_values(["nameOrig", "step"]).reset_index(drop=True)

# =====================================================================
# FEATURE ENGINEERING
# =====================================================================

# 1. Customer historical stats (expanding window, shifted to exclude current row) 
grouped = df.groupby("nameOrig")["amount"]

df["cust_txn_count_so_far"] = grouped.cumcount()  #txns before this one (0 = first ever)
# Cumulative sum and count per customer, shifted to exclude current row
cum_sum = grouped.cumsum() - df["amount"]  # sum of all PRIOR transactions
cum_count = grouped.cumcount()  # count of PRIOR transactions (0-indexed, so this = count before current)

df["cust_hist_avg_amount"] = (cum_sum / cum_count.replace(0, pd.NA))

df["cust_hist_max_amount"] = grouped.cummax().shift(1)

# Ratio features — guard against division by zero / NaN for first-ever transactions
df["amount_vs_hist_median"] = df["amount"] / df["cust_hist_avg_amount"]
df["amount_vs_hist_max"] = df["amount"] / df["cust_hist_max_amount"]
df["is_first_transaction"] = (df["cust_txn_count_so_far"] == 0).astype(int)

# Replace inf/-inf (from division by zero) AND NaN (from 0/0) with a sensible default
df["amount_vs_hist_median"] = df["amount_vs_hist_median"].replace([float("inf"), float("-inf")], pd.NA)
df["amount_vs_hist_max"] = df["amount_vs_hist_max"].replace([float("inf"), float("-inf")], pd.NA)

df["amount_vs_hist_median"] = df["amount_vs_hist_median"].fillna(1.0)
df["amount_vs_hist_max"] = df["amount_vs_hist_max"].fillna(1.0)

# 2. Time since customer's last transaction (in step units) 
df["prev_step"] = df.groupby("nameOrig")["step"].shift(1)
df["steps_since_last_txn"] = (df["step"] - df["prev_step"]).fillna(-1)  # -1 = no prior txn

# 3. Recipient-side feature: how many times has this destination received money before? 
df = df.sort_values(["nameDest", "step"]).reset_index(drop=True)
df["recipient_received_count_so_far"] = df.groupby("nameDest").cumcount()

# 4. Transfer-then-cashout pattern 
# Re-sort by customer + step to check sequential behavior per account
df = df.sort_values(["nameOrig", "step"]).reset_index(drop=True)
df["next_type"] = df.groupby("nameOrig")["type"].shift(-1)
df["next_step"] = df.groupby("nameOrig")["step"].shift(-1)

df["transfer_then_cashout"] = (
    (df["type"] == "TRANSFER") &
    (df["next_type"] == "CASH_OUT")
).astype(int)

df["steps_to_cashout"] = df["next_step"] - df["step"]
df["steps_to_cashout"] = df["steps_to_cashout"].where(df["transfer_then_cashout"] == 1, -1)

# 5. Time-of-day feature (cyclical proxy — step is hourly, 24 steps = 1 day) 
df["hour_of_day"] = df["step"] % 24

# 6. Amount transformations 
df["amount_log"] = np.log1p(df["amount"])
df["is_round_amount"] = (df["amount"] % 1000 == 0).astype(int)

# 7. Amount percentile within transaction type 
df["amount_percentile_within_type"] = df.groupby("type")["amount"].rank(pct=True)

print("\nEngineered feature columns added.")

# Save a customer history lookup table BEFORE dropping IDs 
# This simulates a production transaction database — the backend will
# query this to look up a customer's latest known stats at request time.
lookup_cols = [
    "nameOrig", "step", "type", "amount",
    "cust_txn_count_so_far", "cust_hist_avg_amount", "cust_hist_max_amount",
    "recipient_received_count_so_far", "nameDest",
]
df[lookup_cols].to_csv("data/processed/customer_history_lookup.csv", index=False)
print("\nSaved customer history lookup table to data/processed/customer_history_lookup.csv")

# =====================================================================
# FINAL COLUMN SELECTION
# =====================================================================

# Drop leakage columns, raw IDs (kept only for feature construction, not as model inputs),
# and intermediate helper columns
DROP_COLS = [
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud", "nameOrig", "nameDest",
    "prev_step", "next_type", "next_step",
]
df_model = df.drop(columns=DROP_COLS)

print("\nFinal columns for modeling:", list(df_model.columns))
print("\nSample rows:")
print(df_model.head())

# Train/test split (stratified) 
X = df_model.drop(columns=["isFraud"])
y = df_model["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, "  Fraud rate:", y_train.mean() * 100)
print("Test shape:", X_test.shape, "  Fraud rate:", y_test.mean() * 100)

# Save 
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("\nSaved processed train/test splits with engineered features to data/processed/")