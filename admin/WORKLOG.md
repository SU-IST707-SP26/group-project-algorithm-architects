# WORKLOG.md
---
## 2026-04-26 — Elastic Net Logistic Modeling, Evaluation, Visualization, and F1 Score Analysis (Poorvi)

**Context:**  
Completed end-to-end Elastic Net Logistic Classification modeling for multi-stock return direction prediction. Focused on building an interpretable baseline model, generating probability-based predictions, evaluating classification performance, and supporting ranking-based investment decision-making across all stocks.

Implementing a gradient boosting model aligned with project goals of stock ranking and decision-making. Focused on improving predictive performance beyond ARIMA.


**Work Completed:**
- **(Poorvi)** Built and executed the Elastic Net Logistic Classification pipeline across 15 stocks, including preprocessing, feature engineering, target creation, and time-based train/test split.
- **(Poorvi)** Standardized the modeling workflow across all stocks to ensure consistency in feature engineering, prediction outputs, and evaluation.
- **(Poorvi)** Generated predicted probabilities for positive 7-day returns and structured outputs for ranking-based analysis.
- **(Poorvi)** Implemented classification logic:
  - Positive return = 1
  - Non-positive return = 0
- **(Poorvi)** Evaluated model performance using accuracy, precision, recall, F1 score, confusion matrix, and classification behavior analysis.
- **(Poorvi)** Added F1 score analysis to better evaluate the balance between precision and recall for positive-return prediction.
- **(Poorvi)** Used F1 score to understand how well the model identifies positive-return opportunities instead of relying only on accuracy.
- **(Poorvi)** Identified model bias toward positive predictions through confusion matrix, recall behavior, and probability distribution analysis.
- **(Poorvi)** Developed ranking logic to identify top-performing stocks based on predicted probabilities.
- **(Poorvi)** Created visualization suite including:
  - Target direction distribution
  - Confusion matrix
  - Accuracy, precision, recall, and F1 score summary
  - Coefficient importance plots
  - Top-ranked stock frequency bar chart
  - Predicted probability trends
  - Subplot-based company comparison
  - Violin plot for cross-company probability comparison
- **(Poorvi)** Improved visualization clarity using color palettes, smoothing techniques, and subplot layouts for better interpretability.
- **(Poorvi)** Interpreted model outputs to explain prediction patterns, model bias, F1 score results, and limitations in stock differentiation.

- **(Priyanka)** Implemented XGBoost regression model using engineered features with lagged returns, moving averages, volatility. Defined 7-day return target to better align with ranking and investment decisions.Performed time-based train/validation/test split for realistic forecasting. Generated predictions and converted them into classification labels (BUY/SELL).Computed evaluation metrics including Accuracy and F1 Score for directional prediction. Built multi-stock prediction framework and calculated ranking accuracy across all stocks. Extracted dataset statistics including shape rows/columns and date range. Verified model performance using both statistical metrics and decision-based evaluation ranking.


**Impact:**  
The Elastic Net Logistic model provides an interpretable baseline for stock direction prediction and supports probability-based ranking of investment opportunities. Adding F1 score strengthened the evaluation because it helped measure how well the model balances precision and recall when identifying positive-return stocks. While the model is useful for understanding feature influence, classification behavior, and general probability trends, it also highlights limitations in capturing complex market behavior. This supports the need to compare Elastic Net Logistic Regression with more advanced models like XGBoost for stronger predictive performance.

Improved model capability to capture non-linear market patterns compared to ARIMA.
Enabled better stock selection and ranking performance, aligning with stakeholder needs.
Provided a stronger evaluation framework using F1 score and ranking accuracy, directly addressing professor feedback.
---


## 2026-04-17— XGBoost Modeling, Evaluation, and Ranking Support (Poorvi)

**Context:** Completed end-to-end XGBoost modeling for all five stocks, generated a unified prediction dataset, and extended the model outputs to support ranking-based investment decisions.
Focused on improving ARIMA-based stock prediction and aligning the model with project goals of ranking and investment decision-making.

**Work Completed:**
- **(Poorvi)** Built and executed the XGBoost pipeline for GOOGL, AAPL, MSFT, AMZN, and NVDA, including preprocessing, feature engineering, target creation, time-based train/test split, model training, and prediction generation.
- **(Poorvi)** Standardized the XGBoost workflow across all five stocks to ensure consistency in feature engineering, modeling, and output structure.
- **(Poorvi)** Combined predictions from all five stocks into a unified file: `xgboost_predictions.csv`.
- **(Poorvi)** Validated the combined dataset by checking column structure, stock coverage, missing values, and date ranges.
- **(Poorvi)** Evaluated model performance using MAE, RMSE, and directional accuracy at both overall and stock-wise levels.
- **(Poorvi)** Structured prediction outputs to support ranking-based investment decisions by enabling comparison of predicted returns across all stocks.
- **(Poorvi)** Developed ranking logic to identify top-performing stocks based on predicted returns for each time period.
- **(Poorvi)** Generated feature importance visualizations for all five XGBoost models to understand key predictive features.
- **(Poorvi)** Created actual vs predicted plots for all five stocks to visually assess model performance and trend alignment.
- **(Poorvi)** Interpreted model outputs to support investment recommendations based on predicted return rankings.

- **(Priyanka)** Implemented ARIMAX model with feature engineering (lag, moving averages, volatility) across multiple stocks including Apple, Google, Microsoft, and NVIDIA. Converted data to weekly frequency to reduce noise and improve model stability. ⁠Applied walk-forward validation for realistic time-series forecasting. ⁠Generated predictions for each stock and combined them into a unified dataframe for multi-stock analysis. Built a ranking system to select the best-performing stock at each time step. ⁠Implemented backtesting strategy to simulate investment decisions and portfolio growth. ⁠Calculated evaluation metrics including MAE, RMSE, and directional accuracy across all stocks. ⁠Created ranking accuracy metric and visualization to measure model effectiveness in stock selection. Debugged and resolved data alignment issues between prediction and actual datasets.


**Impact:**  
The XGBoost pipeline is fully complete with validated predictions, performance evaluation, and interpretability. The outputs are structured to support stock ranking and investment decision-making by identifying top-performing stocks based on predicted returns.

Successfully transformed ARIMA from a single-stock model into a multi-stock decision-making system. Established a baseline performance benchmark and identified limitations in handling volatility and ranking accuracy. ⁠Improved model evaluation by incorporating both statistical metrics and financial performance.


**Next Steps:**  
- Integrate XGBoost outputs into ranking engine and portfolio allocation framework  
- Compare XGBoost results with ARIMA and LSTM models .
- Improve ranking accuracy (>60%) using advanced models such as XGBoost and LSTM.
- ⁠Enhance trading strategy with confidence thresholds.
- ⁠Compare ARIMA performance with other models for final evaluation.

---
  
## 2026-04-15 — Enhanced LSTM with 22 Features (Shivani)

**Context:** Original LSTM used only 1 feature (weekly return) and produced near-static predictions. Upgraded to rich feature set and deeper architecture to capture more market dynamics.

**Work Completed:**
- **(Shivani)** Enhanced LSTM from single-feature to 22 features including multi-horizon returns (1/2/4/8 week), RSI-14, MACD histogram, Bollinger Bands (%B and width), ATR, rolling volatility (4w/8w), volume ratios, price-to-SMA, and cross-sectional rank/z-score features.
Upgraded architecture from LSTM(8) → Dense(1) to LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, ReLU) → BatchNorm → Dense(1, sigmoid).
Added ReduceLROnPlateau callback to halve learning rate when validation loss plateaus.
Observed that enhanced model produces more dynamic predictions for volatile stocks (AAPL std=0.115, NVDA std=0.101) while GOOGL and AMZN remained near-static despite richer features.
Documented findings in updated Section 16 interpretation.

**Impact:** Feature engineering improved prediction dynamics for 3 out of 5 stocks. NVDA achieved 61.5% directional accuracy in best run, exceeding the 60% project target. Identified that GOOGL and AMZN weekly movements are driven by events not captured in technical indicators.

**Next Steps:** Build Random Forest model with expanded stock universe. Finalize report and presentation.

---

## 2026-04-14 — LSTM vs XGBoost Portfolio Comparison (Shivani)

**Context:** Both LSTM and XGBoost models had completed predictions for all 5 stocks. Needed a unified comparison against benchmarks to evaluate portfolio-level performance.

**Work Completed:**
- **(Shivani)** Built comparison notebook combining LSTM and XGBoost portfolio returns with equal-weight, S&P 500, Nasdaq, and momentum benchmarks.
Computed Sharpe ratio, Sortino ratio, max drawdown, cumulative returns, and ranking accuracy for each strategy.
Found both ML portfolios outperformed S&P 500 by 23 percentage points and Nasdaq by 16 points with Sharpe ratios above 1.0.
Identified that ML-driven ranking provided no incremental value over equal-weight allocation within the 5-stock correlated universe.

**Impact:** Complete portfolio performance comparison showing stock selection drove returns while ML ranking did not add value over equal-weight. Provides the core results for the final report.

**Next Steps:** Enhance LSTM with richer features to improve ranking dynamics.

---

## 2026-04-12 — S&P 500 Benchmark Data (Shivani)

**Context:** Portfolio backtest needed market benchmarks for meaningful performance comparison.

**Work Completed:**
- **(Shivani)** Downloaded S&P 500 (GSPC) historical data via yfinance and saved to data directory.
Validated date range alignment with existing stock data (2015–2026).

**Impact:** S&P 500 benchmark data available for portfolio comparison in backtesting framework.

**Next Steps:** Integrate benchmark data into comparison notebook.

---


## 2026-04-12 — Arima modeling with validation on MSFT, Nvidia, Google (Priyanka)

**Context:** ARIMA-based time-series modeling pipeline to multiple stocks and evaluate its effectiveness in predicting returns and supporting investment decisions through backtesting.

**Work Completed:**
- **(Priyanka)** Implemented ARIMA models with walk-forward validation for Microsoft, NVIDIA, and Google stock data.
Implemented a walk-forward ARIMA model to predict stock returns using time-series data.
Performed model evaluation using MAE, RMSE, and directional accuracy metrics.
Visualized actual vs predicted returns to analyze model performance and limitations.
Developed a trading strategy (BUY/SELL signals) based on predicted returns.
Conducted backtesting to simulate real investment performance.
Calculated cumulative returns and profit/loss and compared results with a buy-and-hold strategy.
Identified limitations of ARIMA in capturing volatility and sharp market movements.

**Impact:** Identified that ARIMA captures general trends but fails to model volatility and sudden price movements, especially for high-volatility stocks like NVIDIA. Established a baseline performance benchmark for multiple stocks.
Demonstrated the ability to translate predictions into actionable trading decisions and profit/loss evaluation, addressing stakeholder needs.

**Next Steps:** Combine all stock into prediction system to select the best-performing stock dynamically.
---

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

## 2026-04-09 — Nasdaq and S&P 500 Benchmark Data (Shivani)

**Context:** Benchmark indices needed for portfolio performance evaluation as specified in the project proposal.

**Work Completed:**
- **(Shivani)** Downloaded Nasdaq Composite (IXIC) and S&P 500 data via yfinance.
Added both CSV files to the data directory for portfolio benchmarking.

**Impact:** Both market index benchmarks are now available for the backtesting pipeline.

**Next Steps:** Download GSPC separately and build comparison framework.


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
