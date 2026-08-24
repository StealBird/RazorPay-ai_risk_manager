# Failure Case Analysis — AI Risk Manager

Honest metrics are only half the story — understanding *why* a model fails matters as much as knowing that it works. This document walks through three real failure cases: a false positive and a false negative pulled from our test set, plus an out-of-distribution failure discovered live while testing the interactive dashboard — all explained using SHAP.

## Model summary (for context)

- Model: XGBoost classifier, trained on TRANSFER + CASH_OUT transactions from PaySim1
- PR-AUC: 0.6208 (vs. 0.0462 for a feature-engineered logistic regression baseline, and 0.0218 for a raw 3-feature baseline)
- Operating threshold used below: 0.5 (precision 5.5%, recall 87.2% at this threshold — see `docs/pr_curve_xgboost.png` for the full tradeoff curve)

---

## Case 1: False Positive (flagged as fraud, actually legitimate)

**Transaction:** ₹2,72,879 TRANSFER at hour 7, first-ever transaction from this customer, sent to a recipient who had never received a payment before.

**Model confidence:** 99.88% — the model was highly confident this was fraud, and it was wrong.

**Top SHAP drivers:**
| Feature | Contribution | Interpretation |
|---|---|---|
| `hour_of_day` | +3.40 | Time of transaction pushed strongly toward "fraud" |
| `recipient_received_count_so_far` | +2.50 | Brand-new recipient (0 prior transactions received) |
| `step` | +1.45 | Position in the simulation timeline |

**Why this happened:** This transaction has nearly the same behavioral fingerprint as genuine fraud in our dataset — a first-time transfer, to a first-time recipient, at a meaningful amount. Those are exactly the signals the model learned to associate with fraud (validated by our earlier SHAP analysis of a true fraud case, which showed the same top features). The problem is that this fingerprint is not unique to fraud — it also describes ordinary events like a customer's first payment to a new landlord, vendor, or service provider.

**What this reveals about the model:** Our feature set can't currently distinguish *fraudulent* first-contact transactions from *legitimate* first-contact transactions — both look statistically similar. In a production system, this gap would likely be closed with signals we don't have access to in this synthetic dataset: KYC/account verification status, account age, device fingerprinting, or IP/geolocation consistency. Without those, "unusual first-time behavior" is the best proxy available, and it necessarily produces false positives.

**Business framing:** This is the direct cost of prioritizing recall — catching 87% of real fraud at this threshold means accepting that some legitimate first-time large transactions will be flagged for review. A real deployment would route this to manual review rather than auto-block, keeping the false-positive cost as friction, not lost business.

---

## Case 2: False Negative (missed fraud — model said legitimate)

**Transaction:** ₹71,867 CASH_OUT at hour 23 (11pm), first transaction from this customer, sent to a recipient who had received one prior payment.

**Model confidence:** 49.84% — essentially a coin flip, sitting almost exactly on the 0.5 decision threshold.

**Top SHAP drivers:**
| Feature | Contribution | Interpretation |
|---|---|---|
| `hour_of_day` | +0.60 | Late-hour timing pushed toward "fraud" (correctly) |
| `amount_percentile_within_type` | −0.43 | Amount was unremarkable (24.7th percentile) — pushed toward "legitimate" |
| `recipient_received_count_so_far` | −0.39 | Recipient had *some* prior history — pushed toward "legitimate" |

**Why this happened:** This fraud case deliberately (or coincidentally) avoided the two strongest fraud signals our model relies on: it wasn't a large, attention-grabbing amount, and it wasn't sent to a completely new, unestablished recipient. By resembling smaller, more typical transaction behavior, it slipped just under the model's decision boundary.

**What this reveals about the model:** Our model is well-tuned to catch "large, flashy, first-contact" fraud — the pattern most common in this dataset — but is comparatively weaker against fraud that mimics smaller, more routine transaction patterns. This is a genuine limitation, not a bug: the model is doing exactly what it was trained to do, and this case simply falls in the harder, more ambiguous region of the feature space.

**Business framing:** At a probability of 0.4984, this was not a confident miss — it was a near-threshold case. Lowering the decision threshold (trading precision for recall, per our threshold analysis) would likely catch cases like this one, at the cost of more false positives elsewhere. This is a concrete, data-backed example of the threshold tradeoff in action, not an abstract concept.

---

## Case 3: Out-of-distribution failure (discovered live, via the dashboard)

**Transaction:** ₹1 TRANSFER at step 25 (hour 1 / 1am), first transaction from this customer.

**Model output:** Risk score 72.65%, flagged HIGH — for a one-rupee transfer.

**Top SHAP drivers:**
| Feature | Contribution | Interpretation |
|---|---|---|
| `hour_of_day` | +2.98 | 1am timing pushed strongly toward "fraud" |
| `recipient_received_count_so_far` | +0.95 | New recipient — pushed toward "fraud" |
| `amount` | −0.91 | Tiny amount pushed toward "legitimate," but not enough to offset the above |

**Why this happened:** This wasn't found by mining the test set — it surfaced while manually testing the live scorer, which is itself a useful reminder that interactive testing catches things static evaluation metrics don't. In the training data, odd-hour transactions were disproportionately fraud, but those fraud cases were almost always large amounts too — the combination of "odd hour" with a "trivially small amount" barely exists in the training distribution. Faced with an input outside what it learned, the model falls back hard on its strongest individual signal (`hour_of_day`) rather than correctly weighing the amount as an overriding factor.

**What this reveals about the model:** This is a classic **out-of-distribution** failure — the model was never meaningfully trained on this specific combination of feature values, so its behavior there isn't well-calibrated. It's a reminder that a model's average performance (PR-AUC 0.6208) doesn't guarantee sane behavior on every possible input, especially ones far from what it saw during training.

**Business framing / fix:** In production, this is exactly why ML risk scores are layered with simple business-rule guardrails rather than deployed alone — e.g., a minimum-amount floor (transactions under a small threshold, such as ₹100, bypass ML scoring entirely and are auto-approved) would prevent this specific failure mode without needing to retrain the model. This is a cheap, honest mitigation worth stating explicitly rather than claiming the model handles all cases well.

---

## What we'd improve next

1. **Add recipient-pair history** (has this *specific* sender ever sent to this *specific* recipient before), not just aggregate recipient activity — likely to sharpen the boundary between Case 1 and Case 2.
2. **Ensemble with an anomaly-detection model** (e.g., Isolation Forest) trained specifically on the "normal-looking" fraud pattern from Case 2, to catch fraud that mimics typical behavior.
3. **Incorporate account verification signals** if available in a real deployment — the single biggest gap exposed by Case 1.
4. **Add a minimum-amount business-rule floor** ahead of the ML model, directly addressing Case 3 — cheap, simple, and doesn't require retraining.

These three cases, together, show the model's real operating boundary: strong on high-signal fraud, weaker at the edges where fraud resembles normal behavior, and occasionally miscalibrated on inputs far outside its training distribution — an honest, expected set of limitations for a two-week build on synthetic data with a constrained feature set.