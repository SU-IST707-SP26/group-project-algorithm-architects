# WORKLOG.md

---
## 2026-02-25 — Data Cleaning & Validation (Team)

**Context:** Continued data prep by adding ^GSPC and ^IXIC and standardizing CSV formats for EDA.

**Solution Implemented:**
- (Poorvi) Downloaded and cleaned S&P 500 (^GSPC) and Nasdaq Composite (^IXIC) data by checking missing dates, nulls, and split/dividend adjustments.
- (Poorvi & Priyanka) Standardized the date format across all CSVs (e.g., 25 → 2025) and removed extra/unwanted rows to keep all datasets consistent for analysis.
- (Priyanka & Poorvi) Plotted historical Close prices for all stocks and analyzed their trends, growth phases, and major crashes.

**Impact:** Market index datasets (^GSPC, ^IXIC) are now added and validated, and all CSVs have consistent date formats and cleaned rows—datasets are ready for EDA with fewer format-related issues.

**Next Steps:**
- Save and organize final cleaned files into the agreed folder.
- Do a quick cross-check: same date range, no missing trading days, and consistent column names across all tickers.
- Begin EDA: compute daily returns, volatility, and correlations (stocks vs indices).

---

## 2026-02-24 — Data Acquisition & Project Documentation(Team)

**Context:** First working session. Set up all project management documents and began data pipeline.

**Solution Implemented:**
- Created VISION.md and WORKPLAN.md in admin/
- Downloaded AAPL historical data (2015-01-01 to 2025-12-31) via `yfinance`
- (Shivani) Cleaned AAPL data: checked for missing dates, nulls, and split/dividend adjustments
- (Poorvi) Cleaned NVDA, MSFT data: checked for missing dates, nulls, and split/dividend adjustments
- (Priyanka) Cleaned Amazon, Google data: checked for missing dates, nulls, and split/dividend adjustments
  
**Impact:** Project docs are in place and first stock data is ready for EDA. M1-T2 and partial M1-T4 complete.

**Next Steps:** Download remaining 6 tickers (^GSPC, ^IXIC) and validate all before EDA.

---

## 2026-02-18 — Project Initialization & Documentation(Team)

**Context:** Project officially kicked off with proposal submission. Need to establish admin docs before beginning data work.

**Solution Implemented:**
- Discussed VISION.md with problem statement, envisioned solution, success criteria, and constraints
- Planned WORKPLAN.md internally covering 8 milestones from data acquisition through final presentation
- Confirmed project scope: 5 tech stocks (AAPL, MSFT, GOOGL, AMZN, NVDA) + 2 indices, 2015–2026, via `yfinance`
- Confirmed modeling plan: ARIMA + XGBoost + LSTM per stock, best-model selection, weekly ranking output

**Impact:** Team has a shared reference for project goals and current task priorities. Ready to begin data acquisition (M1-T4).

**Next Steps:** Download and cache all 7 tickers to data/ as CSVs. Validate shape and completeness before EDA.

---
