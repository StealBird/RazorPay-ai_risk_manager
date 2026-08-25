# AI Risk Manager

A fraud/risk detection system built for the **Razorpay AI Buildathon (Track 2: AI Risk Manager)**. Detects fraudulent TRANSFER and CASH_OUT transactions using engineered behavioral features and XGBoost, with full explainability (SHAP) and an interactive dashboard.

## The problem

Fraud detection is a needle-in-a-haystack classification problem — in our dataset, only 0.3% of relevant transactions are fraudulent. A naive model can hit 99%+ "accuracy" by predicting "not fraud" every time, while catching zero actual fraud. This project focuses on **honest, measured performance** (precision/recall, not raw accuracy) and **explainability** (why a transaction was flagged, not just a score).

## Dataset

[PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic mobile money transaction simulator, ~6.3M transactions. Fraud only occurs in `TRANSFER` and `CASH_OUT` transaction types in this dataset; we filtered to those (~2.77M rows) to avoid diluting metrics with millions of trivially-safe transactions.

**Important data integrity note:** the dataset's balance columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) are reset when a transaction is flagged as fraud — using them would leak the outcome into the input. We excluded them entirely. Full reasoning in [`docs/session_notes.md`](docs/session_notes.md).

## Approach

1. **Feature engineering** — since raw fields are limited (`step`, `type`, `amount`), we engineered 14 additional features from customer transaction history, recipient behavior, timing patterns, and amount distributions — all computed using only *prior* data relative to each transaction (no lookahead/leakage).
2. **Baseline → XGBoost** — started with logistic regression as an honest baseline, then moved to XGBoost to capture feature interactions.
3. **Explainability** — SHAP TreeExplainer provides per-transaction explanations (which factors drove a specific score), not just global feature importance.
4. **Failure analysis** — documented real false positive, false negative, and out-of-distribution cases, with SHAP-backed explanations for each. See [`docs/failure_case_analysis.md`](docs/failure_case_analysis.md).

## Results

| Model | PR-AUC |
|---|---|
| Baseline (3 raw features, Logistic Regression) | 0.0218 |
| + Feature engineering (Logistic Regression) | 0.0462 |
| + XGBoost | **0.6208** |

At threshold 0.9: **28.9% precision, 70.1% recall** — a usable, honestly-measured operating point. Full precision-recall tradeoff curve in `docs/pr_curve_xgboost.png`.

## System architecture

```
Raw transaction (customer_id, recipient_id, step, type, amount)
        ↓
FastAPI backend — looks up customer/recipient history automatically
        ↓
Feature engineering (live) — mirrors training-time feature logic
        ↓
XGBoost model → risk score + SHAP explanation
        ↓
Streamlit dashboard — live scoring, PR curve, threshold explorer, feature importance
```

See `docs/architecture.png` for the full diagram.

## Project structure

```
├── data/               # data prep and loading scripts
├── model/              # training, evaluation, SHAP explainability
├── backend/            # FastAPI REST API
├── frontend/           # Streamlit dashboard
├── docs/               # analysis writeups, plots, architecture diagram
```

## Running locally

**1. Set up environment:**
```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

**2. Prepare data** (requires PaySim1 CSV in `data/raw/` — see dataset link above):
```bash
python data/prepare_data.py
```

**3. Train the model:**
```bash
python model/train_xgboost.py
python model/evaluate_xgboost.py   # optional — reproduces PR curve and metrics
```

**4. Run the backend (Terminal 1):**
```bash
uvicorn backend.main:app --reload
```

**5. Run the dashboard (Terminal 2):**
```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` to view the dashboard.

## Known limitations (see full analysis in `docs/failure_case_analysis.md`)

- Struggles to distinguish fraudulent vs. legitimate first-time large transactions (no KYC/account-age signals available in this dataset)
- Weaker on fraud that mimics small, routine transaction patterns
- Can be miscalibrated on inputs far outside the training distribution (e.g., trivially small amounts) — mitigated with a minimum-amount business-rule guardrail in the API
- `amount_percentile_within_type` is approximated (neutral default) in the live API, since a true percentile requires the full dataset distribution — a production system would maintain this via a rolling/approximate percentile in a feature store

## What we'd build next

- Recipient-pair history (sender-to-specific-recipient, not just aggregate recipient activity)
- Anomaly-detection ensemble to catch fraud that mimics normal behavior
- Account verification signals (KYC status, account age) if available
- Persistent feature store replacing the CSV-based lookup simulation used in this demo

---

Built for the Razorpay AI Buildathon, Track 2 (AI Risk Manager). See `docs/` for detailed writeups.
