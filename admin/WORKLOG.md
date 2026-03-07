# WORKLOG.md
---
## 2026-03-06 — EDA, Data Pipeline & ARIMA Modeling (Team)

**Context:** Continued project development by performing exploratory data analysis on the cleaned tech stock datasets, implementing a preprocessing pipeline for feature engineering, and building a baseline ARIMA forecasting model.

### Solution Implemented:

- **(Poorvi)** Conducted Exploratory Data Analysis (EDA) on Apple (AAPL), Microsoft (MSFT), Google (GOOGL), Amazon (AMZN), and NVIDIA (NVDA) datasets by loading the cleaned CSV files using pandas, numpy, matplotlib, and seaborn. Verified dataset structure, checked for missing values, and ensured the time-series format was correct. Combined stock closing prices into a single dataframe for comparison and plotted historical stock price trends (2015–2026) to analyze growth patterns. Calculated daily percentage returns and visualized return distributions using histograms. Generated a correlation heatmap to study relationships between stocks. Performed volatility analysis using annualized standard deviation of returns and created technical indicators such as 20-day and 50-day moving averages for further modeling.

- **(Shivani)** Implemented the data preprocessing and feature engineering pipeline. Fixed the missing MSFT entry for 2026-12-31 by forward-filling the Dec 30 value. Renamed incorrectly labeled files (MSCF.csv → MSFT.csv and NVIDIA .csv → NVIDIA.csv). Built `pipeline.py` in the `work/` directory to automate preprocessing tasks. Cleaned datasets by removing comma formatting from the Volume column and converting all fields to numeric format. Computed technical indicators including SMA, EMA, RSI-14, MACD, Bollinger Bands, and volume features. Resampled the data to weekly frequency and aligned all stocks on common dates. Generated weekly ranking labels (1 = best performing, 5 = worst performing). Split the dataset into training (2015–2021), validation (2022–2023), and test (2024–2026) sets. Applied `StandardScaler` to the training data and transformed validation and test datasets. Saved `train.csv`, `val.csv`, `test.csv`, and `scaler.pkl` into the `data/processed/` directory.

- **(Priyanka)** Implemented the ARIMA time-series forecasting model to analyze and predict stock price or return patterns using historical time-series data. Prepared the dataset by selecting the Close price series and performing a train–test split. Trained and fitted the ARIMA model to generate future stock predictions. Evaluated model performance using error metrics such as MAE and RMSE. Visualized actual versus predicted stock values to assess prediction accuracy. Documented the modeling process and results in the midterm project report.

### Impact:
EDA provided insights into stock price trends, volatility, and correlations among major technology companies. The preprocessing pipeline automated feature engineering and dataset preparation, producing standardized datasets for training, validation, and testing. The ARIMA model established a baseline forecasting method that can be used to compare future machine learning models.

### Next Steps:
- Complete ARIMA modeling for the **Apple (AAPL) dataset**, as the remaining datasets have already been trained and modeled.
- Compare ARIMA forecasts across all stocks to analyze prediction performance.
- Train additional machine learning models and integrate results into the portfolio optimization framework.
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
