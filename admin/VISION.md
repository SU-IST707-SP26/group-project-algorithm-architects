# VISION.md

## Current Vision

### Project Title
**Tech Stock Return Prediction and Ranking System**

### Team
- Shivani Kankatala
- Poorvi Nidsoshi
- Priyanka Pawar

### Stakeholders
- Active retail traders seeking data-driven weekly stock picks
- Portfolio rebalancers who need to know which positions to increase or decrease
- Learning investors who want explainable, comparative stock analysis

### Problem Statement
Retail investors have no free, accessible tool that simultaneously predicts and ranks multiple stocks by expected return. Professional research is expensive, analyst ratings are subjective, and institutional quant tools are inaccessible. This leaves individual investors guessing which tech stock to overweight each week.

### Envisioned Solution
A machine learning pipeline that:
1. Downloads historical daily price data (2015–2026) for five major tech stocks — AAPL, MSFT, GOOGL, AMZN, NVDA — plus S&P 500 and Nasdaq as market context, all via `yfinance`.
2. Engineers ~35 features per stock (lagged returns, moving averages, RSI, volatility, market indicators).
3. Trains three model types per stock — ARIMA (baseline), XGBoost, and LSTM — and selects the best-performing model per stock on a held-out validation set.
4. Produces a weekly ranking of all five stocks from highest to lowest predicted return (1-day, 7-day, and 30-day horizons), with confidence scores and buy/sell signals.
5. Evaluates performance against baselines: random selection, equal-weight portfolio, and buy-and-hold S&P 500.

### Success Criteria
- Ranking accuracy (top-ranked stock is actually the best performer) > 60%
- Directional accuracy > 55%
- MAE < $5, RMSE < $8
- Clear performance improvement over all three baselines

### Key Constraints
- Data sourced exclusively from Yahoo Finance (`yfinance`) — must be cached early to guard against API outages
- Strict temporal train/val/test split: 2015–2023 / 2024 / 2025–2026 (no shuffling)
- All data files < 100MB to comply with GitHub repository limits

---

## Version History
*(No prior versions — initial vision established February 2026)*
