# WORKPLAN.md

## Active Tasks

### Milestone 1: Project Setup & Data Acquisition
- ✅ M1-T1 — Write project proposal and literature review
- ✅ M1-T2 — Create VISION.md, WORKPLAN.md in admin/
- ✅ M1-T3 — Test `yfinance` install and run sample download
- ⏳ M1-T4 — Download and cache all 7 tickers (AAPL, MSFT, GOOGL, AMZN, NVDA, ^GSPC, ^IXIC) for 2015–2026; save to data/ as CSV
- [ ] M1-T5 — Validate downloaded data: check for missing dates, split/dividend adjustments, shape confirmation

### Milestone 2: Exploratory Data Analysis (EDA)
- [ ] M2-T1 — Plot price trends and volume over time for all 5 stocks
- [ ] M2-T2 — Compute and visualize return correlations between stocks; flag if >0.9
- [ ] M2-T3 — Analyze volatility (rolling std dev) and identify major market events in data
- [ ] M2-T4 — Document EDA findings and any data concerns in admin/EDA_NOTES.md

### Milestone 3: Feature Engineering
- [ ] M3-T1 — Implement lagged return features (1d, 7d, 30d) for all stocks
- [ ] M3-T2 — Implement moving average features (7, 30, 200-day) for all stocks
- [ ] M3-T3 — Implement technical indicators: RSI, rolling volatility
- [ ] M3-T4 — Add market context features: S&P 500 and Nasdaq daily returns
- [ ] M3-T5 — Assemble unified feature matrix per stock; confirm ~35 features

### Milestone 4: Modeling — ARIMA Baseline
- [ ] M4-T1 — Fit ARIMA models for AAPL and MSFT; evaluate on val set
- [ ] M4-T2 — Fit ARIMA models for GOOGL and AMZN; evaluate on val set
- [ ] M4-T3 — Fit ARIMA model for NVDA; evaluate on val set
- [ ] M4-T4 — Record ARIMA MAE, RMSE, directional accuracy per stock

### Milestone 5: Modeling — XGBoost
- [ ] M5-T1 — Train XGBoost for AAPL and MSFT with hyperparameter tuning; evaluate on val set
- [ ] M5-T2 — Train XGBoost for GOOGL and AMZN; evaluate on val set
- [ ] M5-T3 — Train XGBoost for NVDA; evaluate on val set
- [ ] M5-T4 — Record XGBoost metrics per stock

### Milestone 6: Modeling — LSTM
- [ ] M6-T1 — Build and train LSTM for AAPL and MSFT; evaluate on val set
- [ ] M6-T2 — Build and train LSTM for GOOGL and AMZN; evaluate on val set
- [ ] M6-T3 — Build and train LSTM for NVDA; evaluate on val set
- [ ] M6-T4 — Record LSTM metrics per stock

### Milestone 7: Model Selection & Ranking System
- [ ] M7-T1 — Select best model per stock based on validation MAE/directional accuracy
- [ ] M7-T2 — Implement weekly ranking function: takes 5 predictions, outputs ranked list with confidence scores
- [ ] M7-T3 — Generate buy/sell signals from rankings
- [ ] M7-T4 — Evaluate ranking accuracy on test set (2025–2026); compare to baselines

### Milestone 8: Final Evaluation & Reporting
- [ ] M8-T1 — Compare all models against baselines (random, equal-weight, S&P 500 buy-and-hold)
- [ ] M8-T2 — Create visualizations: ranking accuracy over time, predicted vs. actual, model comparison charts
- [ ] M8-T3 — Write final report
- [ ] M8-T4 — Prepare final presentation slides

---

## Changelog

### 2026-02-24
- 🆕 Initialized WORKPLAN.md with Milestones 1–8 covering full project lifecycle
- ✅ M1-T1 through M1-T3 marked complete based on proposal submission and repo setup
- ⏳ M1-T4 set as current active task (data download and caching)

