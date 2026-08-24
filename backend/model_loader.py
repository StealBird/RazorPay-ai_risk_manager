# backend/model_loader.py

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap

MODEL_PATH = "model/xgboost_model.json"
FEATURE_COLUMNS_PATH = "model/xgb_feature_columns.pkl"
LOOKUP_PATH = "data/processed/customer_history_lookup.csv"

# ---- Load once at import time ----
_model = xgb.XGBClassifier()
_model.load_model(MODEL_PATH)
_feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

# ---- Load the "database" simulation ----
_history_db = pd.read_csv(LOOKUP_PATH)


def lookup_customer_history(customer_id: str, current_step: int) -> dict:
    """
    Simulates a production database lookup: given a customer ID and the
    step (time) of the incoming transaction, find everything the system
    would already know about this customer BEFORE this transaction.

    Returns sensible defaults if the customer has no prior history
    (i.e., this is genuinely their first transaction).
    """
    # Only look at this customer's PAST transactions (strictly before current_step)
    # — never the current or future ones, same leakage rule as training.
    past_txns = _history_db[
        (_history_db["nameOrig"] == customer_id) &
        (_history_db["step"] < current_step)
    ].sort_values("step")

    if past_txns.empty:
        # No history — genuinely first transaction
        return {
            "cust_txn_count_so_far": 0,
            "cust_hist_avg_amount": -1.0,
            "cust_hist_max_amount": -1.0,
            "is_first_transaction": 1,
            "steps_since_last_txn": -1.0,
        }

    last_txn = past_txns.iloc[-1]
    return {
        "cust_txn_count_so_far": len(past_txns),
        "cust_hist_avg_amount": past_txns["amount"].mean(),
        "cust_hist_max_amount": past_txns["amount"].max(),
        "is_first_transaction": 0,
        "steps_since_last_txn": current_step - last_txn["step"],
    }


def lookup_recipient_history(recipient_id: str, current_step: int) -> dict:
    """Same idea, for the recipient side."""
    past_received = _history_db[
        (_history_db["nameDest"] == recipient_id) &
        (_history_db["step"] < current_step)
    ]
    return {
        "recipient_received_count_so_far": len(past_received)
    }


def build_features(customer_id: str, recipient_id: str, step: int, txn_type: str, amount: float) -> dict:
    """
    Takes the RAW inputs a real transaction would actually have
    (who, to whom, when, what type, how much) and derives every
    engineered feature the model needs — mirroring exactly what
    prepare_data.py did for training, but for one live transaction.
    """
    cust_hist = lookup_customer_history(customer_id, step)
    recipient_hist = lookup_recipient_history(recipient_id, step)

    amount_vs_hist_avg = (
        amount / cust_hist["cust_hist_avg_amount"]
        if cust_hist["cust_hist_avg_amount"] not in (-1.0, 0)
        else 1.0
    )
    amount_vs_hist_max = (
        amount / cust_hist["cust_hist_max_amount"]
        if cust_hist["cust_hist_max_amount"] not in (-1.0, 0)
        else 1.0
    )

    features = {
        "step": step,
        "type": txn_type,
        "amount": amount,
        "cust_txn_count_so_far": cust_hist["cust_txn_count_so_far"],
        "cust_hist_avg_amount": cust_hist["cust_hist_avg_amount"],
        "cust_hist_max_amount": cust_hist["cust_hist_max_amount"],
        "amount_vs_hist_median": amount_vs_hist_avg,
        "amount_vs_hist_max": amount_vs_hist_max,
        "is_first_transaction": cust_hist["is_first_transaction"],
        "steps_since_last_txn": cust_hist["steps_since_last_txn"],
        "recipient_received_count_so_far": recipient_hist["recipient_received_count_so_far"],
        "transfer_then_cashout": 0,
        "steps_to_cashout": -1.0,
        "hour_of_day": step % 24,
        "amount_log": np.log1p(amount),
        "is_round_amount": int(amount % 1000 == 0),
        "amount_percentile_within_type": 0.5,
    }

    # ---- Convert any numpy types to native Python types (JSON serialization requires this) ----
    features = {
        k: (v.item() if isinstance(v, np.generic) else v)
        for k, v in features.items()
    }

    return features


def preprocess(features: dict) -> pd.DataFrame:
    df = pd.DataFrame([features])
    df = pd.get_dummies(df, columns=["type"], drop_first=True)
    df = df.reindex(columns=_feature_columns, fill_value=0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(-1)
    return df


def predict_proba(features: dict) -> float:
    df = preprocess(features)
    dmatrix = xgb.DMatrix(df)
    prob = _model.get_booster().predict(dmatrix)[0]
    return float(prob)


def get_model():
    return _model


def get_feature_columns():
    return _feature_columns


# ---- Load SHAP explainer once at import time (expensive to rebuild per-request) ----
_explainer = shap.TreeExplainer(_model)

def get_explainer():
    return _explainer