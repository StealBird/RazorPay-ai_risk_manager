# backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from backend.model_loader import build_features, predict_proba, preprocess, get_explainer, get_feature_columns

app = FastAPI(title="AI Risk Manager API")


class TransactionRequest(BaseModel):
    customer_id: str
    recipient_id: str
    step: int
    type: Literal["TRANSFER", "CASH_OUT"]
    amount: float


@app.post("/score")
def score_transaction(txn: TransactionRequest):
    # Business-rule guardrail: bypass ML scoring for trivially small amounts
    # (addresses an out-of-distribution failure found during testing — see docs/failure_case_analysis.md)
    if txn.amount < 100:
        return {
            "risk_score": 0.0,
            "flagged": False,
            "risk_level": "LOW",
            "note": "Below minimum-amount threshold — auto-approved by business rule, not ML-scored",
        }

    features = build_features(
        txn.customer_id, txn.recipient_id, txn.step, txn.type, txn.amount
    )
    prob = predict_proba(features)
    return {
        "risk_score": round(prob, 4),
        "flagged": prob >= 0.5,
        "risk_level": "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW",
        "computed_features": features,
    }


@app.post("/explain")
def explain_transaction(txn: TransactionRequest):
    features = build_features(
        txn.customer_id, txn.recipient_id, txn.step, txn.type, txn.amount
    )
    feature_columns = get_feature_columns()
    df = preprocess(features)

    explainer = get_explainer()
    shap_values = explainer.shap_values(df)

    contributions = sorted(
        zip(feature_columns, shap_values[0]), key=lambda x: -abs(x[1])
    )
    return {
        "top_factors": [
            {"feature": feat, "contribution": round(float(val), 4)}
            for feat, val in contributions[:5]
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok"}