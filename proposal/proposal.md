# Tech Stock Return Prediction and Ranking System
---
## Team

**Team Members:**
- **Shivani Kankatala**
- **Poorvi Nidsoshi** 
- **Priyanka Pawar**

---

## Introduction

### What are you trying to do?

Predict which of five major tech stocks (Apple, Microsoft, Google, Amazon, Nvidia) will generate the highest returns over the next 1-4 weeks, ranking them from best to worst investment opportunity.

### What is new in your approach?

1. **Comparative ranking:** Predict all 5 stocks simultaneously and rank them (NVDA #1, AAPL #2, etc.)
2. **Best-model-per-stock:** Train ARIMA, XGBoost, and LSTM for each stock; select best performer per stock
3. **Multiple applications:** Rankings power buy/sell signals, portfolio allocation, and risk alerts

**Why it will succeed:** ML achieves 55-60% directional accuracy on stocks (Fischer & Krauss 2018)[^1], we have 2,500+ observations per stock, and Yahoo Finance provides clean free data.

### Who cares? What difference will it make?

**For investors:** Identify best opportunities weekly, avoid underperformers. Example: Choosing top-ranked stock vs bottom-ranked = 6% performance difference.

**Impact:** 3-5% annual improvement on $10K portfolio = $300-500 extra profit from better stock selection.

---

## Literature Review

### How is it done today?

**Technical Analysis:** Retail investors use moving averages, RSI manually. **Limitation:** Subjective, backward-looking.

**Professional Analysts:** Goldman Sachs, Morgan Stanley publish target prices. **Limitation:** Expensive ($500+/report), long-term focus (6-12 months).

**Institutional Quant:** Renaissance Technologies use proprietary ML. **Limitation:** Completely inaccessible to retail.

**Academic Research:** Fischer & Krauss (2018)[^1] used LSTM for S&P 500 (55-60% accuracy), Sezer et al. (2020)[^2] reviewed deep learning for stocks. **Limitation:** Don't deploy usable tools.

**Retail Tools:** Robinhood shows analyst ratings ("Buy"/"Hold"). **Limitation:** Opinions not predictions, no ranking system.

**Gap:** No free ML-based tool that ranks multiple stocks simultaneously with confidence scores.

### Stakeholder Needs

**Active Traders:** Need weekly stock rankings, confidence scores, buy/sell signals. *Evidence:* Schwab survey[^3] shows 67% cite "difficulty identifying best picks."

**Portfolio Rebalancers:** Need to know which stocks to increase/decrease monthly. *Evidence:* Vanguard research[^4] shows 72% struggle with position adjustments.

**Learning Investors:** Need explainable predictions showing why NVDA ranked #1. *Evidence:* Financial Educators Council[^5] reports 81% "don't know how to compare stocks systematically."

---

## Data and Methods

### Data

**Source:** Yahoo Finance 
**Library:** https://finance.yahoo.com/?fr=sycsrp_catchall

**Stocks (5):** AAPL, MSFT, GOOGL, AMZN, NVDA  
**Indices (2):** S&P 500 (^GSPC), Nasdaq (^IXIC)

**Data per stock:**
- **Link example:** https://finance.yahoo.com/quote/AAPL/history
- **Rows:** ~2,500 daily observations (2015-2026)
- **Columns:** Date, Open, High, Low, Close, Volume (6 columns)
- **Total:** 7 tickers × 2,500 rows × 6 columns

**Provenance:** Official Nasdaq/NYSE exchange data, Yahoo Finance is licensed partner, used in millions of trades and thousands of research papers, adjusted for splits/dividends, zero missing trading days.

**Download:** `yf.download(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'], start="2015-01-01")`

---

### Methods

**Preprocessing:**
1. Download 7 tickers via yfinance (one command)
2. Extract Close prices (primary metric)
3. Align on trading dates

**Feature Engineering (~35 features per stock):**
- Lagged prices (1d, 7d, 30d)
- Moving averages (7, 30, 200-day)
- Technical indicators (RSI, volatility)
- Market context (S&P 500, Nasdaq returns)

**Modeling:**

Train 3 models for each of 5 stocks (15 total):
- **ARIMA:** Statistical baseline
- **XGBoost:** Gradient boosting
- **LSTM:** Deep learning

Select best model per stock on validation set.

**Targets:** 1-day, 7-day, 30-day returns

**Train/Val/Test:** 2015-2023 (80%) / 2024 (10%) / 2025-2026 (10%) - temporal split, no shuffling

**Evaluation:**
- **Primary:** Ranking accuracy (is #1 ranked stock actually best?) Target: >60%
- **Secondary:** MAE <$5, Directional accuracy >55%, RMSE <$8
- **Baselines:** Random selection, equal-weight, buy-and-hold S&P 500

---

## Project Plan

| Period | Activity | Milestone |
|--------|----------|-----------|
| **Feb 10-17** | Proposal, test yfinance, sample data download<br/>Create VISION/WORKPLAN/WORKLOG |  Proposal submittion | ✅ Project docs created |
| **Feb 17-Feb 24** | Download all 5 stocks + indices (2015-2026)<br/>Data validation<br/>EDA: trends, correlations, volatility 
---

## Risks

**Risk 1: Prediction Difficulty** - Markets may be too random. *Mitigation:* Realistic goals (55% accuracy), focus on ranking not absolute prices. *Fallback:* Model comparison study.

**Risk 2: Overfitting** - Models learn patterns that don't repeat. *Mitigation:* Regularization, temporal split, validation on different year. *Fallback:* Simpler models.

**Risk 3: High Stock Correlation** - All tech stocks move together. *Mitigation:* Check correlations in EDA, focus on magnitude differences. *Fallback:* Add non-tech stocks if >0.9 correlation.

**Risk 4: Data Access** - Yahoo Finance API could fail. *Mitigation:* Download and cache all data Week 2 (<10MB). *Fallback:* Use cached data.

**Risk 5: Team Coordination** - 15 models to train. *Mitigation:* Clear division (Shivani: AAPL/MSFT, Poorvi: GOOGL/AMZN, Priyanka: NVDA), shared templates. *Fallback:* Reduce to 2 models/stock.

---

## References

[^1]: Fischer, T., & Krauss, C. (2018). "Deep learning with long short-term memory networks for financial market predictions." *European Journal of Operational Research*, 270(2), 654-669.

[^2]: Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). "Financial time series forecasting with deep learning." *Applied Soft Computing*, 90, 106181.

