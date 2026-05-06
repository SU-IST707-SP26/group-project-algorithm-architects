# WORKPLAN.md

## Active Tasks

### Milestone 1: Project Setup & Data Acquisition
- ✅ M1-T1 — Write project proposal and literature review
- ✅ M1-T2 — Create VISION.md, WORKPLAN.md in admin/
- ✅ M1-T3 — Test `yfinance` install and run sample download
- ✅ M1-T4 — Download and cache all tickers (AAPL, MSFT, GOOGL, AMZN, NVDA, ^GSPC, ^IXIC) for 2015–2026; save to data/ as CSV
- ✅ M1-T5 — Validate downloaded data: check for missing dates, split/dividend adjustments, shape confirmation
- ✅ M1-T6 — Expand stock universe to 15 stocks across 5 sectors (META, TSLA, JPM, GS, JNJ, PFE, XOM, CVX, WMT, BA added)

### Milestone 2: Exploratory Data Analysis (EDA)
- ✅ M2-T1 — Plot price trends and volume over time for all stocks
- ✅ M2-T2 — Compute and visualize return correlations between stocks
- ✅ M2-T3 — Analyze volatility (rolling std dev) and identify major market events in data
- ✅ M2-T4 — Document EDA findings in EDA notebooks

### Milestone 3: Feature Engineering
- ✅ M3-T1 — Implement lagged return features (1d, 2d, 3d, 5d) for all stocks
- ✅ M3-T2 — Implement moving average ratios (5, 10, 20-day) for all stocks
- ✅ M3-T3 — Implement technical indicators: RSI, MACD, MACD signal, MACD histogram, rolling volatility
- ✅ M3-T4 — Add market context features: S&P 500 and Nasdaq daily returns
- ✅ M3-T5 — Add volume ratio and High-Low range features
- ✅ M3-T6 — Assemble unified feature matrix per stock (15 features)
- ✅ M3-T7 — Fix data leakage: removed Weekly_Return from feature set; recomputed returns via pct_change()

### Milestone 4: Modeling — ARIMA Baseline
- ✅ M4-T1 — Fit ARIMA models for AAPL and MSFT; corrected target to pct_change(5) weekly returns
- ✅ M4-T2 — Fit ARIMA models for GOOGL and AMZN
- ✅ M4-T3 — Extend ARIMAX to all 15 stocks using walk-forward validation
- ✅ M4-T4 — Record ARIMA MAE, RMSE, directional accuracy per stock
- ✅ M4-T5 — Implement BUY/SELL backtesting simulation vs. buy-and-hold baseline

### Milestone 5: Modeling — XGBoost
- ✅ M5-T1 — Train XGBoost for AAPL and MSFT with feature engineering
- ✅ M5-T2 — Train XGBoost for GOOGL and AMZN
- ✅ M5-T3 — Extend XGBoost to all 15 stocks
- ✅ M5-T4 — Correct XGBoost from XGBRegressor to XGBClassifier (multiclass: Down/Neutral/Up)
- ✅ M5-T5 — Add ±1% threshold for Neutral class to avoid treating near-zero returns as signals
- ✅ M5-T6 — Implement probability-based ranking score: P(Up) - P(Down)
- ✅ M5-T7 — Add naive baselines (always predict Up, majority class) for comparison
- ✅ M5-T8 — Record XGBoost metrics: accuracy, balanced accuracy, macro F1, confusion matrix per stock
- ✅ M5-T9 — Generate feature importance outputs for all 15 stocks

### Milestone 6: Modeling — LSTM
- ✅ M6-T1 — Build and train LSTM for all 5 original stocks; switched from regression to binary classification
- ✅ M6-T2 — Enhanced LSTM with 22 features including RSI, MACD, Bollinger Bands, ATR, volume ratios
- ✅ M6-T3 — Extend LSTM to all 15 stocks
- ✅ M6-T4 — Upgrade LSTM to 3-class classifier (Down/Neutral/Up) with ±1% threshold
- ✅ M6-T5 — Implement sector-based training: one LSTM per sector (Tech, Finance, Healthcare, Energy)
- ✅ M6-T6 — Fix train/test split bug: split per stock first, then pool by sector
- ✅ M6-T7 — Add balanced class weights to fix Up-majority bias
- ✅ M6-T8 — Add L2 regularization and reduce LSTM size to 16 units to fix overfitting
- ✅ M6-T9 — Record LSTM metrics: accuracy, macro F1, per-class F1 for all 15 stocks

### Milestone 7: Additional Models
- ✅ M7-T1 — Implement Elastic Net Logistic Regression as binary classifier across all 15 stocks
- ✅ M7-T2 — Implement Random Forest as binary classifier across all 15 stocks
- ✅ M7-T3 — Implement Gradient Boosting model
- ✅ M7-T4 — Evaluate all models using accuracy, precision, recall, F1, confusion matrix

### Milestone 8: Model Selection & Ranking System
- ✅ M8-T1 — Implement weekly ranking using XGBoost probability score: P(Up) - P(Down)
- ✅ M8-T2 — Generate buy/avoid signals from rankings
- ✅ M8-T3 — Evaluate ranking accuracy on test set (2024–2026)
- ✅ M8-T4 — Compare all models against baselines (random, equal-weight, S&P 500 buy-and-hold)

### Milestone 9: Backtesting & Portfolio Simulation
- ✅ M9-T1 — Implement portfolio backtesting using LSTM probability outputs
- ✅ M9-T2 — Simulate weekly rebalancing over test period; compute cumulative returns
- ✅ M9-T3 — Extend backtester to all models; add benchmark comparisons
- ✅ M9-T4 — Implement ranking backtest: weekly top-ranked stock selection and P&L simulation

### Milestone 10: Final Evaluation & Reporting
- ✅ M10-T1 — Compare all models against baselines with full metrics
- ✅ M10-T2 — Create visualizations: confusion matrices, accuracy charts, class distribution plots, loss curves
- ✅ M10-T3 — Write final report as submission.ipynb
- ✅ M10-T4 — Prepare and deliver final presentation
- ✅ M10-T5 — Incorporate professor feedback into LSTM and XGBoost models post-presentation
- ✅ M10-T6 — Submit final report with Google Drive data link and complete notebook index

---

## Changelog

### 2026-02-24
- 🆕 Initialized WORKPLAN.md with Milestones 1–8 covering full project lifecycle
- ✅ M1-T1 through M1-T3 marked complete based on proposal submission and repo setup
- ⏳ M1-T4 set as current active task (data download and caching)

### 2026-02-24
- ✅ M1-T4 — Downloaded all 5 tickers via yfinance and saved to data/
- ⏳ M1-T5 — Validation in progress across team

### 2026-02-25
- ✅ M1-T5 — Data validated: consistent date ranges, no missing trading days, column names standardized
- ✅ M1-T4 — Added ^GSPC and ^IXIC benchmark data (Poorvi)

### 2026-03-06
- ✅ M2-T1 through M2-T4 — EDA complete (Poorvi)
- ✅ M3-T1 through M3-T7 — Preprocessing pipeline complete (Shivani). Output: train.csv, val.csv, test.csv, scaler.pkl
- ⏳ M4-T1 — Initial ARIMA fitted on raw prices instead of returns; diagnosed error, MAE in hundreds (Priyanka)
- 🔄 M4-T1 — Fix: switch ARIMA target to pct_change(5) for weekly returns

### 2026-03-27
- ✅ M5-T1 — XGBoost trained for Google with 15 features, directional accuracy evaluated (Poorvi)
- ✅ M5-T2 — XGBoost trained for Apple using same pipeline (Poorvi)
- ✅ M4-T1 — ARIMA corrected to use returns; trained for Apple (Priyanka)
- ✅ M4-T2 — ARIMA trained for Amazon (Priyanka)

### 2026-04-05
- ✅ M6-T1 through M6-T3 — LSTM complete for all 5 stocks (Shivani)
- 🔄 M6-T2 — Switched from regression to binary classification; regression produced flat predictions for all stocks

### 2026-04-09
- (Shivani) Reviewed all model outputs. Identified gap: ARIMA and XGBoost produce individual stock signals, not cross-stock ranking inputs. Need standardized output format.
- ⏳ M4-T3, M4-T4 — ARIMA needs extension to remaining stocks + output standardization (Priyanka)
- ⏳ M5-T3, M5-T4 — XGBoost needs extension to remaining stocks + output standardization (Poorvi)

### 2026-04-15
- ✅ M6-T2 — Enhanced LSTM with 22 features; NVDA achieved 61.5% directional accuracy (Shivani)

### 2026-04-17
- ✅ M5-T3 — XGBoost extended to all 5 original stocks; combined predictions saved (Poorvi)
- ✅ M4-T3 — ARIMAX extended to multiple stocks with walk-forward validation (Priyanka)

### 2026-04-24
- ✅ M1-T6 — Stock universe expanded to 15 stocks across 5 sectors (Shivani)
- ✅ M7-T2 — Random Forest implemented across all 15 stocks (Shivani)

### 2026-04-25
- ✅ M7-T1 — Elastic Net Logistic Regression complete for all 15 stocks (Poorvi)

### 2026-04-26
- ✅ M7-T3 — Gradient Boosting model implemented (Priyanka)
- ✅ M4-T5 — ARIMAX with portfolio strategy implemented (Priyanka)

### 2026-04-27
- ✅ M6-T3 — LSTM extended to all 15 stocks (Shivani)

### 2026-05-04
- ✅ M5-T4 through M5-T9 — XGBoost corrected to multiclass classifier; feature importance, metrics, predictions for all 15 stocks (Poorvi)

### 2026-05-05
- ✅ M4-T3 — ARIMAX implemented and backtested for all 15 stocks (Priyanka)
- ✅ M6-T4 through M6-T9 — LSTM upgraded to sector-based 3-class classifier; all bugs fixed; all 15 stocks in results (Shivani)
- ✅ M10-T5 — Professor feedback incorporated into LSTM and XGBoost (Shivani, Poorvi)
- ✅ M10-T6 — Final report submitted as submission.ipynb (Shivani)
