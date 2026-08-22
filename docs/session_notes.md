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