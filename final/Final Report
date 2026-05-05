Tech Stock Return Prediction and Ranking System

Team

Shivani Kankatala - ShivaniKankatala 
Poorvi Nidsoshi - poorvinidsoshi01 
Priyanka Pawar - ppawar03-byte 

---

## Introduction

Retail investors face a persistent problem: which stocks are worth buying this week, and which should be avoided? Professional tools are either inaccessible, expensive, or offer only qualitative "Buy/Hold/Sell" ratings rather than quantitative, ranked guidance. This project builds a machine-learning-based system that predicts weekly returns for a universe of stocks, ranks them from best to worst opportunity, and translates those predictions into concrete buy/avoid decisions with an estimated profit-and-loss outcome.

Our primary stakeholder is an **individual retail investor** managing a small portfolio (e.g., $10,000) who wants weekly, data-driven guidance. The stakeholder need is clear: the investor needs to know not only *which* stock is likely to outperform, but also *what to do about it* and *what the financial consequence of that action is likely to be*. Knowing that "NVDA is ranked #1" is not enough — the investor needs to know whether to buy it, and what gain or loss they might expect if they follow that signal.

Our solution trains multiple model families — ARIMA, XGBoost, LSTM, and Elastic Net Logistic Regression — across 15 stocks spanning technology, energy, finance, healthcare, and consumer sectors. For each week, we rank all stocks by predicted 7-day return, generate a buy/avoid signal for each stock, and simulate portfolio performance to assess whether the model's signals are financially useful. The full pipeline from raw data to ranked predictions to profit-and-loss simulation is implemented and evaluated on held-out test data from 2024–2026.

---

## Literature Review

### Prior Work and Stakeholder Context

Stock return prediction has a long history in both academic research and industry practice. Fischer & Krauss (2018) demonstrated that LSTM networks applied to the S&P 500 achieve roughly 55–60% directional accuracy — a meaningful edge over a 50/50 baseline. Sezer et al. (2020) reviewed deep learning approaches to financial time series and confirmed that technical indicators combined with sequence models generally outperform simpler baselines on short-horizon return prediction.

Professional tools used by institutional quants are proprietary and completely inaccessible to retail investors. Retail-facing tools like Robinhood show analyst consensus ratings ("Buy"/"Hold") but do not produce quantitative predictions or rankings, do not update weekly, and do not provide expected return estimates. A Schwab survey cited in our proposal found that 67% of active traders report difficulty identifying the best stock picks, while Vanguard research found that 72% struggle with when to adjust portfolio positions. These findings motivate a tool that outputs not only a ranking but also actionable signals.

### Why These Methods?

We chose three complementary model families to capture different aspects of return predictability:

**ARIMA/ARIMAX** is the classical statistical baseline for time series. It models autocorrelation in returns and, with exogenous variables (lagged features), can capture short-term momentum. Its weakness is the assumption of stationarity and linearity, which may miss complex interactions.
**Gradient**
**XGBoost** 
**Elastic Net Logistic Regression** 
**LSTM** 
**Random Forest**

### The Gap We Fill

No existing free tool ranks multiple stocks simultaneously by predicted return, translates those rankings into buy/avoid decisions, and estimates the financial outcome of following those decisions. This project fills that gap.

---

## Data and Methods

### Data

**Source:** Yahoo Finance via `yfinance`

**Universe:** 15 stocks spanning five sectors:
- *Technology:* AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- *Finance:* JPM, GS
- *Healthcare:* JNJ, PFE
- *Energy:* XOM, CVX
- *Consumer/Industrial:* WMT, BA

**Market Indices:** S&P 500 (^GSPC), Nasdaq (^IXIC) — used as exogenous features for market context

**Coverage:** ~2,500 daily observations per stock (2015–2026), resampled to weekly (Friday close) for modeling

**Data quality:** Yahoo Finance is a licensed partner of NYSE/Nasdaq, data is adjusted for splits and dividends, and zero missing trading days were found in our download. The `yfinance` library is widely used in academic financial research, providing confidence in data integrity.

**Key observations from EDA:**
- Weekly returns for all stocks are approximately normally distributed with slight positive skew, centered near 0
- Stocks are positively correlated (tech stocks especially), but magnitude differences across weeks are sufficient to distinguish ranking positions
- NVDA shows the highest annualized volatility (~45%) and the highest average weekly return over the study period; WMT and JNJ show the lowest volatility
- Class balance (% of weeks with positive return) ranges from ~51% (BA) to ~58% (NVDA), confirming that a trivial "always predict up" baseline would score near 50–55% directional accuracy

### Methods

#### Preprocessing and Feature Engineering

All data was loaded from CSV files, date columns converted to datetime, and price/volume columns cleaned of comma-formatting. Daily data was resampled to weekly frequency using Friday closing prices to reduce noise.

The following feature families were constructed for each stock:



For ARIMA, features were limited to lagged returns, moving averages, and volatility, which serve as exogenous variables in SARIMAX.

#### Target Variable


#### Train/Validation/Test Split



#### Model Details

**ARIMA (single-stock and 15-stock):** Walk-forward ARIMAX with order (1,1,1). At each test step, the model is retrained on all available history and forecasts one step ahead, then the actual value is added to history. This is computationally expensive but realistic — it simulates what an investor would have access to each week. Exogenous variables are lagged return features.


#### Important Modeling Decisions and Issues Addressed


#### Supporting Files

| Notebook | Purpose |
|----------|---------|
| `arima_model-3.ipynb` | ARIMA/ARIMAX on AAPL; walk-forward, backtesting, trading signals |
| `Arima_For_15_Stocks.ipynb` | ARIMAX applied to all 15 stocks in a loop |

---

## Results

### Model Evaluation Framework

All models were evaluated on held-out test data (2024–2026). Regression models (ARIMA, XGBoost) are reported in terms of **MAE** and **RMSE** on the return scale (i.e., percentage points), as well as **directional accuracy**, **precision**, **recall**, and **F1 score** derived by thresholding predictions at zero. Classification models (LSTM, Elastic Net) are reported directly in classification terms with confusion matrices.

Baselines for comparison:
- **Random selection:** Expected ~50% directional accuracy
- **Equal-weight buy-and-hold:** Invest equal amounts in all stocks each period
- **S&P 500 buy-and-hold:** Full-period compounded return

### ARIMA Results (Single-Stock AAPL and 15 Stocks)

The walk-forward ARIMAX approach on AAPL produced:
- MAE and RMSE on test-set weekly returns (2025–2026)
- Directional accuracy: predictions were evaluated by comparing predicted vs. actual return signs
- A **backtesting simulation** starting from $1,000 using BUY/SELL signals (buy if predicted return > 0, otherwise sell/short) versus a buy-and-hold baseline

Directional accuracy for ARIMA is typically lower than XGBoost and LSTM because ARIMA assumes linearity. The walk-forward approach ensures no lookahead bias, at the cost of significant computation (re-fitting at each test step).

### Summary Comparison

| Model | Stocks | Target | Primary Metric | Simulation |
|-------|--------|--------|----------------|------------|
| ARIMA (walk-forward) | 15 | Weekly return (regression) | MAE + directional accuracy | BUY/SELL backtesting |


---

## Discussion

### Were the Goals Achieved?


### Connection to Stakeholder Need


---

## Limitations


**ARIMA scalability.** Walk-forward ARIMAX requires refitting at every test step, making it computationally prohibitive for large universes. The 15-stock ARIMA notebook is compute-intensive and was trained with limited hyperparameter tuning.


---

## Future Work


---

## References


---

## Supporting Notebook Index

**Data:** All data is downloaded from Yahoo Finance using `yfinance`. A Google Drive link with cached CSVs is available at: *(add link here)*
