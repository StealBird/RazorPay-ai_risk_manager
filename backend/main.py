# backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from backend.model_loader import build_features, predict_proba, preprocess, get_model, get_feature_columns
import shap
from typing import Literal

app = FastAPI(title="AI Risk Manager API")


class TransactionRequest(BaseModel):
    customer_id: str
    recipient_id: str
    step: int
    type: Literal["TRANSFER", "CASH_OUT"]
    amount: float

@app.post("/score")
def score_transaction(txn: TransactionRequest):
    features = build_features(
        txn.customer_id, txn.recipient_id, txn.step, txn.type, txn.amount
    )
    prob = predict_proba(features)
    return {
        "risk_score": round(prob, 4),
        "flagged": prob >= 0.5,
        "risk_level": "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW",
        "computed_features": features,  # transparency — show what was derived
    }


@app.post("/explain")
def explain_transaction(txn: TransactionRequest):
    features = build_features(
        txn.customer_id, txn.recipient_id, txn.step, txn.type, txn.amount
    )
    model = get_model()
    feature_columns = get_feature_columns()
    df = preprocess(features)

    explainer = shap.TreeExplainer(model)
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