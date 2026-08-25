### Feature engineering results (logistic regression):
- Baseline (3 raw features): PR-AUC = 0.0218
- After feature engineering (17 features incl. customer history, velocity, 
  transfer-then-cashout pattern): PR-AUC = 0.0462 (~2.1x improvement)
- Fixed bugs along the way: 
  1. .gitignore typo (data.processed -> data/processed)
  2. temporal leakage risk in historical features (used expanding/shift to avoid 
     looking at future transactions)
  3. division-by-zero producing inf (not NaN) in ratio features - needed explicit 
     inf handling separate from fillna
  4. .expanding() too slow at scale - switched to vectorized cumsum/cummax
- Next: XGBoost (logistic regression is linear, can't capture feature interactions - 
  precision still low ~7.6% even at threshold 0.9)

 ### XGBoost results (major improvement):
- Logistic regression (engineered features): PR-AUC = 0.0462
- XGBoost (same 17 features): PR-AUC = 0.6208 (~13x improvement)
- At threshold=0.9: Precision 28.9%, Recall 70.1% (vs LogReg's 7.6%/31.0% at same threshold)
- Top features by importance: type_TRANSFER, recipient_received_count_so_far, 
  amount_percentile_within_type
- Honest finding: is_first_transaction, amount_vs_hist_median, steps_to_cashout 
  contributed ~0 importance - not every engineered feature idea panned out, 
  XGBoost found other features captured that signal more effectively
- Next: SHAP explainability, then failure case analysis

### Session end: failure case analysis complete
- docs/failure_case_analysis.md written and committed (FP + FN cases, SHAP-explained)
- Next session: FastAPI backend (/score, /explain, /metrics endpoints), then Streamlit frontend
- Deadline: Sep 5

### Backend debugging session:
- Fixed: duplicate /score route definition (leftover from merging a guardrail snippet)
- Fixed: import shap placed after use in model_loader.py
- Fixed: numpy.int64/float64 not JSON-serializable — added .item() conversion in build_features()
- Added: minimum-amount business-rule guardrail (amount < 100 bypasses ML scoring), 
  addressing the Case 3 out-of-distribution finding
- Backend + frontend now fully working end-to-end