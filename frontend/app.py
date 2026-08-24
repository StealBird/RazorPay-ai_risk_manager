# frontend/app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AI Risk Manager", layout="wide")

BACKEND_URL = "http://127.0.0.1:8000"

# =====================================================================
# LOAD MODEL + TEST DATA (for the performance dashboard sections)
# =====================================================================

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("model/xgboost_model.json")
    feature_columns = joblib.load("model/xgb_feature_columns.pkl")
    return model, feature_columns


@st.cache_data
def load_test_data(_feature_columns):
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    X_test_enc = pd.get_dummies(X_test, columns=["type"], drop_first=True)
    X_test_enc = X_test_enc.reindex(columns=_feature_columns, fill_value=0)
    X_test_enc = X_test_enc.replace([np.inf, -np.inf], np.nan).fillna(-1)
    return X_test_enc, y_test


model, feature_columns = load_model()
X_test_enc, y_test = load_test_data(feature_columns)

dmatrix = xgb.DMatrix(X_test_enc)
y_probs = model.get_booster().predict(dmatrix)

avg_precision = average_precision_score(y_test, y_probs)

# =====================================================================
# HEADER
# =====================================================================

st.title("🛡️ AI Risk Manager")
st.caption("Fraud detection for TRANSFER and CASH_OUT transactions — built on PaySim1")

col1, col2, col3 = st.columns(3)
col1.metric("PR-AUC (model quality)", f"{avg_precision:.4f}", "13x vs. logistic regression baseline")
col2.metric("Transactions evaluated", f"{len(y_test):,}")
col3.metric("Fraud rate in test set", f"{y_test.mean()*100:.3f}%")

st.divider()

# =====================================================================
# SECTION 1: MODEL PERFORMANCE — PR CURVE + INTERACTIVE THRESHOLD
# =====================================================================

st.header("📊 Model Performance")

precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

left, right = st.columns([1, 1])

with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recalls, y=precisions, mode="lines", name="PR Curve"))
    fig.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        title=f"Precision-Recall Curve (AP = {avg_precision:.3f})",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Try a decision threshold")
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)

    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    m1, m2 = st.columns(2)
    m1.metric("Precision", f"{precision:.2%}")
    m2.metric("Recall", f"{recall:.2%}")

    st.write(f"**Fraud caught:** {tp} of {tp+fn} ({recall:.1%})")
    st.write(f"**False alarms:** {fp} legitimate transactions flagged")
    st.write(f"**Fraud missed:** {fn} cases")

    cm_fig = px.imshow(
        [[tn, fp], [fn, tp]],
        labels=dict(x="Predicted", y="Actual"),
        x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
        text_auto=True, color_continuous_scale="Blues",
    )
    cm_fig.update_layout(height=300, title="Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)

st.divider()

# =====================================================================
# SECTION 2: FEATURE IMPORTANCE
# =====================================================================

st.header("🔍 What Drives the Model")

importances = pd.Series(
    model.feature_importances_, index=feature_columns
).sort_values(ascending=True)

fig_imp = px.bar(
    importances, orientation="h",
    labels={"value": "Importance", "index": "Feature"},
    title="Feature Importance (XGBoost)",
)
fig_imp.update_layout(height=500, showlegend=False)
st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# =====================================================================
# SECTION 3: LIVE TRANSACTION SCORER
# =====================================================================

st.header("⚡ Live Transaction Scorer")
st.caption("Enter a transaction — the system looks up customer/recipient history automatically and scores it in real time.")

with st.form("score_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        customer_id = st.text_input("Customer ID", value="C1000000639")
        recipient_id = st.text_input("Recipient ID", value="M1979787155")
    with c2:
        step = st.number_input("Step (hour)", min_value=1, max_value=744, value=249)
        txn_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
    with c3:
        amount = st.number_input("Amount (₹)", min_value=1.0, value=100000.0, step=1000.0)

    submitted = st.form_submit_button("Score Transaction", type="primary")

if submitted:
    payload = {
        "customer_id": customer_id,
        "recipient_id": recipient_id,
        "step": int(step),
        "type": txn_type,
        "amount": float(amount),
    }
    try:
        score_resp = requests.post(f"{BACKEND_URL}/score", json=payload, timeout=5)
        explain_resp = requests.post(f"{BACKEND_URL}/explain", json=payload, timeout=5)

        if score_resp.status_code == 200:
            result = score_resp.json()
            risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[result["risk_level"]]

            st.subheader(f"{risk_color} Risk Level: {result['risk_level']}")
            st.metric("Risk Score", f"{result['risk_score']:.2%}")
            st.write("**Flagged for review:**", "Yes" if result["flagged"] else "No")

            if explain_resp.status_code == 200:
                factors = explain_resp.json()["top_factors"]
                st.subheader("Why this score — top contributing factors")
                factor_df = pd.DataFrame(factors)
                factor_df["direction"] = factor_df["contribution"].apply(
                    lambda x: "↑ toward fraud" if x > 0 else "↓ toward legitimate"
                )
                st.dataframe(factor_df, use_container_width=True, hide_index=True)
        else:
            st.error(f"API error: {score_resp.text}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Backend not running. Start it with: `uvicorn backend.main:app --reload`")

st.divider()

# =====================================================================
# SECTION 4: FAILURE CASE SHOWCASE
# =====================================================================

st.header("⚠️ Honest Failure Cases")
st.caption("No model is perfect — here's where ours struggles, and why.")

fc1, fc2 = st.columns(2)

with fc1:
    st.subheader("False Positive")
    st.write("₹2,72,879 TRANSFER, first transaction, new recipient — flagged at 99.9% confidence, but legitimate.")
    st.write("**Why:** Matches the exact behavioral fingerprint of real fraud (first-time, new recipient, large amount) without other signals like KYC status to disambiguate.")

with fc2:
    st.subheader("False Negative")
    st.write("₹71,867 CASH_OUT, smaller amount, recipient had some prior history — missed at 49.8% (right at threshold).")
    st.write("**Why:** This fraud case mimicked normal small-transaction behavior, avoiding the model's strongest signals (large amount, brand-new recipient).")

st.caption("Full analysis in `docs/failure_case_analysis.md`")
