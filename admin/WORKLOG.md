# WORKLOG.md
## 2026-04-11 — XGBoost Modeling and Combined Prediction File (Poorvi)

**Context:** After completing XGBoost modeling for all five assigned stocks, standardized outputs and prepared a unified prediction file for ranking and evaluation.

**Work Completed:**
- **(Poorvi)** Completed the XGBoost pipeline for GOOGL, AAPL, MSFT, AMZN, and NVDA, including preprocessing, feature engineering, target creation, train/validation/test split, model training, prediction generation, evaluation, signal generation, backtesting, and feature importance.
- **(Poorvi)** Standardized the Google and Apple XGBoost workflow to match the same structure used for Microsoft, Amazon, and NVIDIA.
- **(Poorvi)** Combined all five stock outputs into a single file: `xgboost_predictions.csv`.
- **(Poorvi)** Validated the combined prediction file by checking columns, stock coverage, row counts, and missing values.
- **(Poorvi)** Calculated overall XGBoost MAE, RMSE, and directional accuracy across all five stocks.
- **(Poorvi)** Calculated stock-wise MAE, RMSE, and directional accuracy for all five stocks.

**Impact:** The XGBoost modeling work is complete for all five stocks, and the combined prediction file is ready to support ranking, comparison, and later portfolio evaluation.

**Next Steps:** Create feature importance visualizations for XGBoost.
---

## 2026-04-08 — Portfolio Backtest Implementation (Shivani)

**Context:** After completing the LSTM classification model, built the backtesting framework to simulate portfolio performance using model predictions.

**Work Completed:**
- **(Shivani)** Implemented portfolio backtesting logic using LSTM probability outputs to rank stocks weekly and allocate portfolio weights based on rankings.
- **(Shivani)** Simulated weekly rebalancing over the test period and computed cumulative portfolio returns.
- **(Shivani)** Committed backtest code to GitHub.

**Impact:** Backtesting framework is functional and produces portfolio return series from LSTM predictions. This serves as the foundation for the full multi-model comparison once ARIMA and XGBoost outputs are standardized.

**Next Steps:** Extend backtester to accept inputs from all three models. Add benchmark comparisons (equal-weight, S&P 500, Nasdaq).

---

## 2026-04-06 — LSTM Classification Model Complete (Shivani)

**Context:** Initial LSTM regression approach collapsed to flat near-zero predictions for all stocks — a well-documented issue in financial ML where the model learns that predicting zero minimizes MSE.

**Problem Identified:** Regression LSTM predicted nearly identical values for all 5 stocks, making ranking impossible. The model was minimizing loss by predicting the mean return (~0) rather than learning directional patterns.

**Solution Implemented:**
- **(Shivani)** Switched from regression to binary classification (up vs down) with sigmoid output.
- Used minimal architecture: LSTM(8 units) → Dense(1, sigmoid) to prevent overfitting on ~350 training samples per stock.
- Used raw unscaled weekly returns as single input feature (4-week lookback window).
- Trained individual models per stock in a loop — each stock gets its own model weights.
- Probability scores (P(Up) = 0.0 to 1.0) serve as confidence levels for cross-stock ranking.
- Saved per-stock metrics, overall ranking accuracy, top-1 and top-2 accuracy to `lstm_summary.txt`.
- **(Shivani)** Committed LSTM model code to GitHub.

**Impact:** LSTM modeling complete for all 5 stocks. Probability-based ranking produces meaningful differentiation between stocks, unlike the flat regression outputs. Model achieves ranking accuracy above random baseline (20%).

**Next Steps:** Build ranking engine that consumes LSTM probability outputs alongside ARIMA and XGBoost predicted returns.

## 2026-03-27 — Model Training 

**Task:** Continued work on the stock prediction project by building and testing XGBoost models for Google and Apple using cleaned historical price data.

**Solution Implemented:**
- **(Poorvi)** Cleaned and prepared Google and Apple datasets by fixing formatting issues, converting columns properly, and organizing the data for modeling.Created important time-series features including lagged close prices, returns, moving averages, volatility, volume trends, and RSI.Defined the 7-day future return target and split the data into training, validation, and test sets based on time.Trained XGBoost models for both Google and Apple and evaluated their performance using MAE, RMSE, and directional accuracy.Converted predicted returns into Buy, Hold, and Avoid signals and compared the strategy results against buy-and-hold using a simple backtest. Reviewed feature importance and added interpretation to understand how useful the models are for investor decision-making.


- **(Priyanka)** Implemented the ARIMA model on Apple Inc. and Amazon.com, Inc. stock data using returns instead of raw prices.Performed parameter tuning (p,d,q) to improve model performance.Evaluated model performance using MAE, RMSE, and directional accuracy. Visualized actual vs predicted results to analyze model behavior and limitations. Identified that ARIMA predictions are smoother and do not fully capture market volatility. Developed a trading strategy using BUY/SELL signals based on model predictions. Implemented backtesting to compare strategy performance against buy-and-hold. Calculated cumulative returns and profit/loss for both strategies.

**Impact:** The Google and Apple models are now developed and evaluated from both forecasting and practical decision-making perspectives. This improves the project by showing not just prediction accuracy, but also how the predictions can support investor actions.

**Next Steps:**
- Compare the final performance of Google and Apple models.
- Improve the models through feature refinement and parameter tuning.
- Apply the same workflow to the other stocks in the project.
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
